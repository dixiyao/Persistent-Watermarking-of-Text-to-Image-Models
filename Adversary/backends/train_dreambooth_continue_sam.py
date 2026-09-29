#!/usr/bin/env python3
"""
Continue fine-tuning with adversarial gradient ascent on the copyright dataset.

This is the downstream-attack counterpart of continue_loss_study for either
Stable Diffusion 1.x or SDXL. The training dataset is the copyright/cp directory
itself. By default, fresh PEFT LoRAs are attached to a frozen checkpoint and
optimized to ascend the CP loss directly; --full-tuning updates raw weights.
Despite the historical filename, this is not SAM/sharpness minimization.

Integrity images are still generated before training and periodically.
"""

import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from continue_training_utils import (
    SDXLTrainingModel,
    continuation_artifact,
    create_continue_optimizer,
    generate_integrity_image as _generate_integrity_image,
    is_deepspeed_enabled,
    load_or_create_peft,
    resolve_resume_checkpoint,
    save_prepared_continue_weights,
    unwrap_sdxl_training_model,
)
from diffusers import DDPMScheduler
from model_loading import (
    has_second_text_encoder,
    load_base_sdxl_backbone,
    load_further_full_weights,
)
from peft import LoraConfig
from tqdm.auto import tqdm

from utils import (
    SimpleDreamBoothDataset,
    add_trigger_rewrite_args,
    infinite_dataloader,
    replace_trigger_word,
    simple_dreambooth_collate_fn,
)


def _load_merged_components(args, model_dtype):
    variant = args.variant
    if variant is None:
        variant = "fp16" if model_dtype != torch.float32 else None

    components = load_base_sdxl_backbone(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=variant,
        torch_dtype=model_dtype,
    )
    return load_further_full_weights(components, args.merged_checkpoint)


def _resolve_cp_dir(args, parser):
    cp_dir = args.cp_dir or args.cp_dataset
    if cp_dir is None:
        parser.error("Pass --cp_dir or --cp_dataset.")
    csv_path = os.path.join(cp_dir, "prompt.csv")
    image_dir = os.path.join(cp_dir, "image")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"cp prompt.csv not found: {csv_path}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"cp image dir not found: {image_dir}")
    return cp_dir, csv_path, image_dir


def _prepare_noisy_latents(batch, vae, noise_scheduler):
    with torch.no_grad():
        pixel_values = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        ).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    return noisy_latents, noise, timesteps


def _cp_loss_from_encoded(
    batch,
    noisy_latents,
    noise,
    timesteps,
    training_model,
    resolution,
    device,
):
    model_pred = training_model(
        noisy_latents.to(device),
        timesteps.to(device),
        batch["input_ids"].to(device),
        batch["input_ids_2"].to(device),
        resolution,
    )
    return F.mse_loss(model_pred.float(), noise.to(device).float(), reduction="mean")


def _grad_norm(params):
    norms = []
    for param in params:
        if param.grad is None:
            continue
        norms.append(torch.norm(param.grad.detach(), p=2))
    if not norms:
        return torch.zeros((), device=params[0].device if params else "cpu")
    return torch.norm(torch.stack([norm.to(norms[0].device) for norm in norms]), p=2)


def _save_checkpoint(
    training_model,
    output_dir,
    backbone_info,
    accelerator,
    full_tuning,
):
    save_prepared_continue_weights(
        accelerator,
        training_model,
        output_dir,
        backbone_info,
        full_tuning=full_tuning,
    )


def _generate_integrity_pair(
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    noise_scheduler,
    cp_prompt,
    org_prompt,
    output_dir,
    step,
    resolution,
    num_steps,
):
    _generate_integrity_image(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        noise_scheduler=noise_scheduler,
        prompt=cp_prompt,
        output_path=os.path.join(output_dir, f"step_{step:06d}_cp.png"),
        resolution=resolution,
        num_steps=num_steps,
    )
    _generate_integrity_image(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        noise_scheduler=noise_scheduler,
        prompt=org_prompt,
        output_path=os.path.join(output_dir, f"step_{step:06d}_org.png"),
        resolution=resolution,
        num_steps=num_steps,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Continue LoRA or full-model fine-tuning with cp-loss gradient ascent"
    )
    parser.add_argument("--merged_checkpoint", type=str, required=True,
                        help="Full W checkpoint directory with unet.pt, or an intermediate "
                             "T-LoRA checkpoint directory with tlora_weights.pt")
    parser.add_argument("--cp_dir", type=str, default=None,
                        help="Copyright training dataset directory with image/ + prompt.csv")
    parser.add_argument("--cp_dataset", type=str, default=None,
                        help="Alias for --cp_dir")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    parser.add_argument("--full-tuning", action="store_true",
                        help="Tune full UNet and text-encoder weights instead of PEFT LoRAs.")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="to_k,to_q,to_v,to_out.0")
    parser.add_argument("--te_lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,out_proj",
                        help="Text encoder LoRA target modules. Empty string disables text encoder LoRA.")

    parser.add_argument("--output_dir", type=str, default="checkpoints_continue_cp_ascent")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=400)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--update_mode",
        choices=("cp_loss", "denoise"),
        default="cp_loss",
        help=(
            "cp_loss performs the existing adversarial update by minimizing "
            "-CP denoising loss; denoise performs ordinary CP denoising training."
        ),
    )
    parser.add_argument("--cp_integrity_prompt", type=str, default="A [Z]*$ at the lake")
    parser.add_argument("--org_integrity_prompt", type=str, default="A dog at the lake")
    parser.add_argument("--integrity_prompt", type=str, default=None,
                        help="Legacy alias for --cp_integrity_prompt.")
    parser.add_argument("--integrity_inference_steps", type=int, default=50)
    parser.add_argument("--integrity_interval", type=int, default=100,
                        help="Generate integrity image every N steps. 0 = only before training.")

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume_latest", action="store_true")
    add_trigger_rewrite_args(parser)
    args = parser.parse_args()
    if args.integrity_prompt is not None:
        args.cp_integrity_prompt = args.integrity_prompt
    args.cp_integrity_prompt = replace_trigger_word(
        args.cp_integrity_prompt,
        args.original_trigger_word,
        args.current_trigger_word,
    )
    args.org_integrity_prompt = replace_trigger_word(
        args.org_integrity_prompt,
        args.original_trigger_word,
        args.current_trigger_word,
    )

    if not os.path.isdir(args.merged_checkpoint):
        parser.error(f"Merged checkpoint not found: {args.merged_checkpoint}")
    cp_dir, cp_csv, cp_image_dir = _resolve_cp_dir(args, parser)

    resume_checkpoint_path, resume_step = resolve_resume_checkpoint(
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume_latest=args.auto_resume_latest,
        required_artifact=continuation_artifact(args.full_tuning),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.mixed_precision, torch.float32)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    print(f"\nLoading full W checkpoint: {args.merged_checkpoint}")
    components = _load_merged_components(args, model_dtype)
    if resume_checkpoint_path is not None and args.full_tuning:
        components = load_further_full_weights(components, resume_checkpoint_path)
    tokenizer = components.tokenizer
    tokenizer_2 = components.tokenizer_2
    vae = components.vae
    unet = components.unet
    text_encoder = components.text_encoder
    text_encoder_2 = components.text_encoder_2

    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    vae.eval()

    if accelerator.is_main_process:
        from image_generation import generate_image_in_memory
        img = generate_image_in_memory(
            prompt=args.cp_integrity_prompt,
            checkpoint=args.merged_checkpoint,
            base_model=args.pretrained_model_name_or_path,
            use_refiner=True,
            num_inference_steps=args.integrity_inference_steps,
            height=args.resolution,
            width=args.resolution,
            device=str(accelerator.device),
        )
        img.save(os.path.join(args.output_dir, "output_integrity_cp.png"))
        org_img = generate_image_in_memory(
            prompt=args.org_integrity_prompt,
            checkpoint=args.merged_checkpoint,
            base_model=args.pretrained_model_name_or_path,
            use_refiner=True,
            num_inference_steps=args.integrity_inference_steps,
            height=args.resolution,
            width=args.resolution,
            device=str(accelerator.device),
        )
        org_img.save(os.path.join(args.output_dir, "output_integrity_org.png"))
        print("Backbone integrity images saved to output_integrity_cp.png and output_integrity_org.png")

    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    te_target_modules = [item.strip() for item in args.te_lora_target_modules.split(",") if item.strip()]
    if args.full_tuning:
        unet.requires_grad_(True)
        text_encoder.requires_grad_(True)
        text_encoder_2.requires_grad_(True)
    else:
        if not target_modules:
            parser.error("--lora_target_modules cannot be empty unless --full-tuning is used.")
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        unet = load_or_create_peft(
            unet, lora_config, resume_checkpoint_path, label="UNet"
        )
    if te_target_modules and not args.full_tuning:
        te_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            target_modules=te_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
        )
        te1_resume = None if resume_checkpoint_path is None else os.path.join(resume_checkpoint_path, "text_encoder")
        te2_resume = None if resume_checkpoint_path is None else os.path.join(resume_checkpoint_path, "text_encoder_2")
        text_encoder = load_or_create_peft(
            text_encoder, te_lora_config, te1_resume, label="text_encoder"
        )
        if has_second_text_encoder(text_encoder_2):
            text_encoder_2 = load_or_create_peft(
                text_encoder_2, te_lora_config, te2_resume, label="text_encoder_2"
            )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    train_dataset = SimpleDreamBoothDataset(
        csv_path=cp_csv,
        image_dir=cp_image_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        center_crop=False,
        original_trigger_word=args.original_trigger_word,
        current_trigger_word=args.current_trigger_word,
    )
    train_loader = infinite_dataloader(
        train_dataset,
        args.seed + accelerator.process_index,
        args.train_batch_size,
        collate_fn=simple_dreambooth_collate_fn,
    )
    trainable_params = (
        [param for param in unet.parameters() if param.requires_grad]
        + [param for param in text_encoder.parameters() if param.requires_grad]
        + [param for param in text_encoder_2.parameters() if param.requires_grad]
    )
    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")
    optimizer = create_continue_optimizer(
        accelerator,
        trainable_params,
        args.learning_rate,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    training_model = SDXLTrainingModel(unet, text_encoder, text_encoder_2)
    training_model, optimizer = accelerator.prepare(training_model, optimizer)
    trainable_params = [
        param for param in training_model.parameters() if param.requires_grad
    ]
    vae = vae.to(accelerator.device)
    device = accelerator.device
    unet, text_encoder, text_encoder_2 = unwrap_sdxl_training_model(
        accelerator, training_model
    )

    backbone_info = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "merged_checkpoint": args.merged_checkpoint,
        "cp_dir": cp_dir,
        "training_objective": (
            "cp_loss_gradient_ascent" if args.update_mode == "cp_loss" else "cp_denoising"
        ),
        "update_mode": args.update_mode,
        "tuning_mode": "full" if args.full_tuning else "lora",
        "new_lora_rank": None if args.full_tuning else args.rank,
        "new_lora_alpha": None if args.full_tuning else args.lora_alpha,
        "new_lora_target_modules": [] if args.full_tuning else target_modules,
        "new_te_lora_target_modules": [] if args.full_tuning else te_target_modules,
    }

    integrity_dir = os.path.join(args.output_dir, "integrity_images")
    if accelerator.is_main_process:
        os.makedirs(integrity_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Continue CP Attack")
    print(f"  full W:       {args.merged_checkpoint}")
    print(f"  cp_dir:       {cp_dir}")
    print(f"  tuning:       {'full model' if args.full_tuning else 'PEFT LoRA'}")
    print(f"  update mode:  {args.update_mode}")
    print(f"  max_steps:    {args.max_train_steps} (resume from {resume_step})")
    print(f"  output:       {args.output_dir}")
    print(f"  integrity:    {integrity_dir}")
    print(f"{'=' * 60}\n")

    unet.train()
    if args.full_tuning or te_target_modules:
        text_encoder.train()
        text_encoder_2.train()
    else:
        text_encoder.eval()
        text_encoder_2.eval()

    global_step = int(resume_step)
    if accelerator.is_main_process and not is_deepspeed_enabled(accelerator):
        _generate_integrity_pair(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            noise_scheduler=noise_scheduler,
            cp_prompt=args.cp_integrity_prompt,
            org_prompt=args.org_integrity_prompt,
            output_dir=integrity_dir,
            step=global_step,
            resolution=args.resolution,
            num_steps=args.integrity_inference_steps,
        )
        unet.train()
        if args.full_tuning or te_target_modules:
            text_encoder.train()
            text_encoder_2.train()

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Continue-CP-Ascent",
    )
    if global_step > 0:
        progress_bar.update(min(global_step, args.max_train_steps))

    for batch in train_loader:
        if global_step >= args.max_train_steps:
            break

        noisy_latents, noise, timesteps = _prepare_noisy_latents(batch, vae, noise_scheduler)

        cp_loss = _cp_loss_from_encoded(
            batch,
            noisy_latents,
            noise,
            timesteps,
            training_model,
            args.resolution,
            device,
        )
        optimization_loss = -cp_loss if args.update_mode == "cp_loss" else cp_loss
        accelerator.backward(optimization_loss)
        grad_norm = _grad_norm(trainable_params)
        if args.max_grad_norm and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        progress_bar.update(1)
        progress_metrics = {
            "cp_loss": f"{float(cp_loss.detach()):.5f}",
            "grad": f"{float(grad_norm.detach()):.3e}",
        }
        progress_bar.set_postfix(progress_metrics)

        if accelerator.is_main_process and global_step % 10 == 0:
            print(
                f"step {global_step}: cp_loss={float(cp_loss.detach()):.6f} "
                f"optimization_loss={float(optimization_loss.detach()):.6f} "
                f"grad_norm={float(grad_norm.detach()):.6e}",
                flush=True,
            )

        if (accelerator.is_main_process
                and args.integrity_interval > 0
                and global_step % args.integrity_interval == 0
                and not is_deepspeed_enabled(accelerator)):
            _generate_integrity_pair(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                noise_scheduler=noise_scheduler,
                cp_prompt=args.cp_integrity_prompt,
                org_prompt=args.org_integrity_prompt,
                output_dir=integrity_dir,
                step=global_step,
                resolution=args.resolution,
                num_steps=args.integrity_inference_steps,
            )
            unet.train()
            if args.full_tuning or te_target_modules:
                text_encoder.train()
                text_encoder_2.train()

        if (args.checkpointing_steps > 0
                and global_step % args.checkpointing_steps == 0):
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            _save_checkpoint(
                training_model,
                ckpt_dir,
                backbone_info,
                accelerator,
                args.full_tuning,
            )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoints = sorted(
                    [
                        name for name in os.listdir(args.output_dir)
                        if name.startswith("checkpoint-") and name[len("checkpoint-"):].isdigit()
                    ],
                    key=lambda name: int(name[len("checkpoint-"):]),
                )
                for old in checkpoints[:-args.checkpoints_total_limit]:
                    shutil.rmtree(os.path.join(args.output_dir, old), ignore_errors=True)

    accelerator.wait_for_everyone()
    final_dir = os.path.join(args.output_dir, "final")
    _save_checkpoint(
        training_model,
        final_dir,
        backbone_info,
        accelerator,
        args.full_tuning,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            f"\nDone. Final {args.update_mode} "
            f"{'full checkpoint' if args.full_tuning else 'adapters'} -> {final_dir}"
        )
        print(f"Total steps: {global_step} / {args.max_train_steps}")


if __name__ == "__main__":
    main()
