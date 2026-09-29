#!/usr/bin/env python3
"""
Continue fine-tuning a quantized robust-study SDXL checkpoint with QLoRA.

The script loads a full robust-study/final checkpoint W, applies lightweight
TFMQ-DM-style packed integer quantization to the UNet, freezes W, then trains a
small QALoRA adapter on --data_dir. The LoRA gradient hooks implement the
EfficientDM scale-aware update heuristic for low-bit diffusion fine-tuning.
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from continue_training_utils import (
    generate_integrity_image as _generate_integrity_image,
    resolve_resume_checkpoint,
)
from diffusers import DDPMScheduler
from model_loading import (
    encode_stable_diffusion_prompt,
    has_second_text_encoder,
    load_base_sdxl_backbone,
    load_further_full_weights,
    stable_diffusion_unet_forward,
)
from quantization_utils import (
    QuantizationConfig,
    apply_tfmqdm_quantization,
    iter_qalora_modules,
    load_qalora_adapters,
    materialize_qalora,
    quantization_summary_text,
    register_scale_aware_lora_hooks,
    save_qalora_adapters,
    save_quantized_checkpoint,
    save_quantization_config,
)
from tqdm.auto import tqdm

from utils import (
    SimpleDreamBoothDataset,
    add_trigger_rewrite_args,
    infinite_dataloader,
    replace_trigger_word,
    simple_dreambooth_collate_fn,
)

def _resolve_resume_path(args):
    return resolve_resume_checkpoint(
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume_latest=args.auto_resume_latest,
        required_artifact="qalora_adapters.pt",
        prefixes=("checkpoint-",),
    )


def _build_quant_config(args):
    return QuantizationConfig(
        method="tfmqdm_ste",
        weight_bits=args.weight_bits,
        activation_bits=args.activation_bits,
        quantize_activations=not args.disable_activation_quantization,
        protect_temporal=not args.no_temporal_protection,
        quantize_conv=not args.no_quantize_conv,
        quantize_linear=not args.no_quantize_linear,
        per_channel_weights=not args.per_tensor_weights,
        symmetric=True,
    )


def _load_train_dataset(args, tokenizer, tokenizer_2):
    train_csv = os.path.join(args.data_dir, "prompt.csv")
    train_image_dir = os.path.join(args.data_dir, "image")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not os.path.isdir(train_image_dir):
        raise FileNotFoundError(f"Training image dir not found: {train_image_dir}")
    return SimpleDreamBoothDataset(
        csv_path=train_csv,
        image_dir=train_image_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution,
        center_crop=False,
        original_trigger_word=args.original_trigger_word,
        current_trigger_word=args.current_trigger_word,
    )


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


def _encode_prompt_for_batch(batch, text_encoder, text_encoder_2, resolution, device):
    ids1 = batch["input_ids"].to(device)
    ids2 = batch["input_ids_2"].to(device)
    return encode_stable_diffusion_prompt(
        text_encoder,
        text_encoder_2,
        ids1,
        ids2,
        batch["pixel_values"].to(device),
        resolution,
    )


def _qalora_ratio(*models):
    with torch.no_grad():
        qalora_sq = 0.0
        base_sq = 0.0
        for model in models:
            for _, layer in iter_qalora_modules(model):
                delta = (layer.lora_B.float() @ layer.lora_A.float()) * layer.scaling
                qalora_sq += float(delta.norm().item() ** 2)
                base = layer.dequantize_weight(dtype=torch.float32, device=delta.device)
                base_sq += float(base.norm().item() ** 2)
        return (qalora_sq ** 0.5) / max(base_sq ** 0.5, 1e-12)


def _save_quantized_training_artifacts(
    accelerator,
    unet,
    text_encoder,
    text_encoder_2,
    output_dir,
    quant_config,
    backbone_info,
    quantize_text_encoders,
):
    raw_unet = accelerator.unwrap_model(unet)
    raw_text_encoder = accelerator.unwrap_model(text_encoder)
    raw_text_encoder_2 = accelerator.unwrap_model(text_encoder_2)
    use_text_encoder_2 = (
        quantize_text_encoders and has_second_text_encoder(raw_text_encoder_2)
    )

    save_qalora_adapters(
        unet=raw_unet,
        text_encoder=raw_text_encoder if quantize_text_encoders else None,
        text_encoder_2=raw_text_encoder_2 if use_text_encoder_2 else None,
        output_dir=output_dir,
        metadata=backbone_info,
    )

    materialized_unet = materialize_qalora(raw_unet, quant_config)
    materialized_text_encoder = None
    materialized_text_encoder_2 = None
    if quantize_text_encoders:
        text_config = QuantizationConfig(**{**quant_config.__dict__, "protect_temporal": False})
        materialized_text_encoder = materialize_qalora(raw_text_encoder, text_config)
        if use_text_encoder_2:
            materialized_text_encoder_2 = materialize_qalora(
                raw_text_encoder_2, text_config
            )

    save_quantized_checkpoint(
        unet=materialized_unet,
        text_encoder=materialized_text_encoder,
        text_encoder_2=materialized_text_encoder_2,
        output_dir=output_dir,
        quant_config=quant_config,
        metadata=backbone_info,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Continue SDXL QLoRA fine-tuning on a quantized robust-study checkpoint"
    )
    parser.add_argument("--merged_checkpoint", type=str, required=True,
                        help="Merged W checkpoint directory with unet.pt, or an intermediate "
                             "T-LoRA checkpoint directory with tlora_weights.pt")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Training data directory with image/ + prompt.csv")

    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="to_k,to_q,to_v,to_out.0")
    parser.add_argument("--te_lora_target_modules", type=str, default="",
                        help="Optional text encoder LoRA target modules. Empty string disables text encoder LoRA.")

    parser.add_argument("--output_dir", type=str, default="checkpoints_continue_quantized_efficientdm")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=400)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--weight_bits", type=int, default=4)
    parser.add_argument("--activation_bits", type=int, default=4)
    parser.add_argument("--disable_activation_quantization", action="store_true")
    parser.add_argument("--no_temporal_protection", action="store_true")
    parser.add_argument("--quantize_text_encoders", action="store_true")
    parser.add_argument("--no_quantize_conv", action="store_true")
    parser.add_argument("--no_quantize_linear", action="store_true")
    parser.add_argument("--per_tensor_weights", action="store_true")
    parser.add_argument("--no_scale_aware_lora", action="store_true",
                        help="Disable EfficientDM-style quantization-scale LoRA gradient hooks.")

    parser.add_argument("--integrity_prompt", type=str, default="A [Z]*$ at the lake")
    parser.add_argument("--integrity_inference_steps", type=int, default=50)
    parser.add_argument("--integrity_interval", type=int, default=100)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume_latest", action="store_true")
    add_trigger_rewrite_args(parser)
    args = parser.parse_args()

    args.integrity_prompt = replace_trigger_word(
        args.integrity_prompt,
        args.original_trigger_word,
        args.current_trigger_word,
    )

    if not os.path.isdir(args.merged_checkpoint):
        parser.error(f"Merged checkpoint not found: {args.merged_checkpoint}")
    if args.weight_bits <= 0 or args.activation_bits <= 0:
        parser.error("--weight_bits and --activation_bits must be positive")

    os.makedirs(args.output_dir, exist_ok=True)
    resume_checkpoint_path, resume_step = _resolve_resume_path(args)

    model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.mixed_precision, torch.float32)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    variant = args.variant
    if variant is None:
        variant = "fp16" if model_dtype != torch.float32 else None

    print(f"\nLoading full W checkpoint: {args.merged_checkpoint}")
    components = load_base_sdxl_backbone(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=variant,
        torch_dtype=model_dtype,
    )
    components = load_further_full_weights(components, args.merged_checkpoint)
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

    target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
    te_target_modules = [item.strip() for item in args.te_lora_target_modules.split(",") if item.strip()]
    if te_target_modules and not args.quantize_text_encoders:
        parser.error("--te_lora_target_modules requires --quantize_text_encoders for quantized QALoRA training")

    quant_config = _build_quant_config(args)
    print(
        f"Applying packed quantization and QALoRA: W{quant_config.weight_bits}A{quant_config.activation_bits}, "
        f"temporal_protection={quant_config.protect_temporal}"
    )
    unet_summary = apply_tfmqdm_quantization(
        unet,
        quant_config,
        prefix="unet",
        lora_target_modules=target_modules,
        lora_rank=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    print(f"UNet quantization: {quantization_summary_text(unet_summary)}")

    if args.quantize_text_encoders:
        text_config = QuantizationConfig(**{**quant_config.__dict__, "protect_temporal": False})
        te1_summary = apply_tfmqdm_quantization(
            text_encoder,
            text_config,
            prefix="text_encoder",
            lora_target_modules=te_target_modules,
            lora_rank=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        print(f"text_encoder quantization: {quantization_summary_text(te1_summary)}")
        if has_second_text_encoder(text_encoder_2):
            te2_summary = apply_tfmqdm_quantization(
                text_encoder_2,
                text_config,
                prefix="text_encoder_2",
                lora_target_modules=te_target_modules,
                lora_rank=args.rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
            )
            print(f"text_encoder_2 quantization: {quantization_summary_text(te2_summary)}")

    if resume_checkpoint_path is not None:
        loaded = load_qalora_adapters(
            unet=unet,
            text_encoder=text_encoder if args.quantize_text_encoders else None,
            text_encoder_2=(
                text_encoder_2
                if args.quantize_text_encoders
                and has_second_text_encoder(text_encoder_2)
                else None
            ),
            adapter_dir=resume_checkpoint_path,
            strict=True,
        )
        print(f"Loaded QALoRA adapters: {loaded}")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if not args.no_scale_aware_lora:
        hook_count = register_scale_aware_lora_hooks(unet, args.weight_bits)
        if args.quantize_text_encoders:
            hook_count += register_scale_aware_lora_hooks(text_encoder, args.weight_bits)
            if has_second_text_encoder(text_encoder_2):
                hook_count += register_scale_aware_lora_hooks(
                    text_encoder_2, args.weight_bits
                )
        print(f"Registered scale-aware LoRA gradient hooks: {hook_count}")

    train_dataset = _load_train_dataset(args, tokenizer, tokenizer_2)
    train_loader = infinite_dataloader(
        train_dataset,
        args.seed,
        args.train_batch_size,
        collate_fn=simple_dreambooth_collate_fn,
    )

    trainable_params = (
        [param for param in unet.parameters() if param.requires_grad]
        + [param for param in text_encoder.parameters() if param.requires_grad]
        + [param for param in text_encoder_2.parameters() if param.requires_grad]
    )
    if not trainable_params:
        raise RuntimeError("No trainable QLoRA parameters found.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    unet, text_encoder, text_encoder_2, optimizer = accelerator.prepare(
        unet, text_encoder, text_encoder_2, optimizer
    )
    vae = vae.to(accelerator.device)
    device = accelerator.device

    backbone_info = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "merged_checkpoint": args.merged_checkpoint,
        "full_weights_path": args.merged_checkpoint,
        "training_objective": "quantized_backbone_qlora_denoising",
        "quantization_config": quant_config.__dict__,
        "quantization_method_note": "Packed integer PTQ with temporal-path protection",
        "efficientdm_note": "QALoRA merges adapter weights into quantized W+BA during forward; scale-aware hooks optional",
        "new_lora_rank": args.rank,
        "new_lora_alpha": args.lora_alpha,
        "new_lora_dropout": args.lora_dropout,
        "new_lora_target_modules": target_modules,
        "new_te_lora_target_modules": te_target_modules,
    }

    if accelerator.is_main_process:
        save_quantization_config(quant_config, args.output_dir)
        with open(os.path.join(args.output_dir, "backbone_info.json"), "w", encoding="utf-8") as handle:
            json.dump(backbone_info, handle, indent=2)

    integrity_dir = os.path.join(args.output_dir, "integrity_images")
    if accelerator.is_main_process:
        os.makedirs(integrity_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Continue Quantized EfficientDM-style QLoRA")
    print(f"  full W:       {args.merged_checkpoint}")
    print(f"  data_dir:     {args.data_dir}")
    print(f"  max_steps:    {args.max_train_steps} (resume from {resume_step})")
    print(f"  quantization: W{args.weight_bits}A{args.activation_bits}")
    print(f"  output:       {args.output_dir}")
    print(f"  integrity:    {integrity_dir}")
    print(f"{'=' * 60}\n")

    unet.train()
    if te_target_modules:
        text_encoder.train()
        text_encoder_2.train()
    else:
        text_encoder.eval()
        text_encoder_2.eval()

    global_step = int(resume_step)
    if accelerator.is_main_process:
        _generate_integrity_image(
            unet=accelerator.unwrap_model(unet),
            vae=vae,
            text_encoder=accelerator.unwrap_model(text_encoder),
            text_encoder_2=accelerator.unwrap_model(text_encoder_2),
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            noise_scheduler=noise_scheduler,
            prompt=args.integrity_prompt,
            output_path=os.path.join(integrity_dir, f"step_{global_step:06d}.png"),
            resolution=args.resolution,
            num_steps=args.integrity_inference_steps,
        )
        unet.train()
        if te_target_modules:
            text_encoder.train()
            text_encoder_2.train()

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Quantized-QLoRA",
    )
    if global_step > 0:
        progress_bar.update(min(global_step, args.max_train_steps))

    for batch in train_loader:
        if global_step >= args.max_train_steps:
            break

        noisy_latents, noise, timesteps = _prepare_noisy_latents(batch, vae, noise_scheduler)
        prompt_embeds, pooled, time_ids = _encode_prompt_for_batch(
            batch,
            text_encoder,
            text_encoder_2,
            args.resolution,
            device,
        )

        model_pred = stable_diffusion_unet_forward(
            unet,
            noisy_latents.to(device),
            timesteps.to(device),
            prompt_embeds,
            pooled,
            time_ids,
        )
        train_loss = F.mse_loss(model_pred.float(), noise.to(device).float(), reduction="mean")

        accelerator.backward(train_loss)
        if args.max_grad_norm and args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        qalora_ratio = _qalora_ratio(
            accelerator.unwrap_model(unet),
            accelerator.unwrap_model(text_encoder),
            accelerator.unwrap_model(text_encoder_2),
        )

        progress_bar.update(1)
        progress_bar.set_postfix(
            loss=f"{float(train_loss.detach()):.5f}",
            t=f"{int(timesteps[0].detach().item())}",
            qalora=f"{qalora_ratio:.2e}",
        )

        if (accelerator.is_main_process
                and args.integrity_interval > 0
                and global_step % args.integrity_interval == 0):
            _generate_integrity_image(
                unet=accelerator.unwrap_model(unet),
                vae=vae,
                text_encoder=accelerator.unwrap_model(text_encoder),
                text_encoder_2=accelerator.unwrap_model(text_encoder_2),
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                noise_scheduler=noise_scheduler,
                prompt=args.integrity_prompt,
                output_path=os.path.join(integrity_dir, f"step_{global_step:06d}.png"),
                resolution=args.resolution,
                num_steps=args.integrity_inference_steps,
            )
            unet.train()
            if te_target_modules:
                text_encoder.train()
                text_encoder_2.train()

        if (accelerator.is_main_process
                and args.checkpointing_steps > 0
                and global_step % args.checkpointing_steps == 0):
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            _save_quantized_training_artifacts(
                accelerator=accelerator,
                unet=unet,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                output_dir=ckpt_dir,
                quant_config=quant_config,
                backbone_info=backbone_info,
                quantize_text_encoders=args.quantize_text_encoders,
            )
            checkpoints = sorted(
                [
                    name for name in os.listdir(args.output_dir)
                    if name.startswith("checkpoint-") and name[len("checkpoint-"):].isdigit()
                ],
                key=lambda name: int(name[len("checkpoint-"):]),
            )
            if args.checkpoints_total_limit and args.checkpoints_total_limit > 0:
                for old in checkpoints[:-args.checkpoints_total_limit]:
                    shutil.rmtree(os.path.join(args.output_dir, old), ignore_errors=True)

    progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        _save_quantized_training_artifacts(
            accelerator=accelerator,
            unet=unet,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            output_dir=final_dir,
            quant_config=quant_config,
            backbone_info=backbone_info,
            quantize_text_encoders=args.quantize_text_encoders,
        )
        print(f"\nDone. Final QALoRA + packed quantized checkpoint -> {final_dir}")
        print(f"Total steps: {global_step} / {args.max_train_steps}")


if __name__ == "__main__":
    main()
