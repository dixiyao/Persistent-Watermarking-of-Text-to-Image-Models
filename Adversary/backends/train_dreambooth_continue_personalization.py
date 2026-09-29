#!/usr/bin/env python3
"""
Continue fine-tuning for DreamBooth personalization.

This downstream task follows the SleeperMark personalization setting: load a
merged Stable Diffusion checkpoint W and tune a fresh UNet PEFT LoRA on a reference
subject with the rare identifier "sks". Pass --full-tuning to update every
UNet weight directly instead. The VAE and text encoders remain frozen.
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
    is_deepspeed_enabled,
    load_or_create_peft,
    resolve_resume_checkpoint,
    save_continue_weights,
    save_prepared_continue_weights,
    unwrap_sdxl_training_model,
)
from diffusers import DDPMScheduler
from model_loading import load_base_sdxl_backbone, load_further_full_weights
from peft import LoraConfig
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from utils import (
    add_trigger_rewrite_args,
    replace_trigger_word,
    save_sdxl_images_from_components,
    simple_dreambooth_collate_fn,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _image_to_tensor(image, size):
    image = image.convert("RGB")
    image = image.resize((size, size), resample=Image.BICUBIC)
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    return 2.0 * image - 1.0


def _tokenize(prompt, tokenizer, tokenizer_2):
    input_ids = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids[0]
    input_ids_2 = tokenizer_2(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer_2.model_max_length,
        return_tensors="pt",
    ).input_ids[0]
    return input_ids, input_ids_2


def _list_images(image_dir):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    paths = [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if os.path.splitext(name.lower())[1] in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    return paths


class FixedPromptImageDataset(Dataset):
    def __init__(self, image_dir, prompt, tokenizer, tokenizer_2, size):
        self.image_paths = _list_images(image_dir)
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        print(f"Loaded {len(self.image_paths)} images from {image_dir} with prompt: {prompt}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[int(idx)]
        input_ids, input_ids_2 = _tokenize(self.prompt, self.tokenizer, self.tokenizer_2)
        image = Image.open(image_path)
        return {
            "pixel_values": _image_to_tensor(image, self.size),
            "input_ids": input_ids,
            "input_ids_2": input_ids_2,
            "image_name": os.path.basename(image_path),
            "prompt": self.prompt,
        }


def infinite_dataloader(dataset, seed, batch_size, collate_fn=simple_dreambooth_collate_fn):
    epoch = 0
    while True:
        rng = np.random.RandomState(seed + epoch)
        indices = rng.permutation(np.arange(len(dataset)))
        batch_examples = []
        for idx in indices:
            batch_examples.append(dataset[int(idx)])
            if len(batch_examples) == batch_size:
                yield collate_fn(batch_examples)
                batch_examples = []
        if batch_examples:
            yield collate_fn(batch_examples)
        epoch += 1


def _resolve_default_instance_dir(path):
    if os.path.isdir(path):
        return path
    local_fallback = os.path.join("dreambooth", "dog")
    if path == os.path.join("data", "dreambooth", "dog") and os.path.isdir(local_fallback):
        print(f"WARNING: {path} not found; using local fallback {local_fallback}")
        return local_fallback
    return path


def _generate_integrity_images(
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    noise_scheduler,
    cp_prompt,
    personalization_prompt,
    output_dir,
    step,
    resolution,
    num_steps,
    guidance_scale,
    seed,
):
    unet.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    os.makedirs(output_dir, exist_ok=True)
    output_paths = [
        os.path.join(output_dir, f"step_{step:06d}_cp.png"),
        os.path.join(output_dir, f"step_{step:06d}_sks_dog.png"),
    ]
    save_sdxl_images_from_components(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        noise_scheduler=noise_scheduler,
        prompts=[cp_prompt, personalization_prompt],
        output_paths=output_paths,
        resolution=resolution,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    print(f"Integrity images saved: {output_paths[0]}, {output_paths[1]}")
    unet.train()


def _encode_batch(batch, vae, noise_scheduler, device):
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
    return noisy_latents.to(device), noise.to(device), timesteps.to(device)


def _forward_loss(
    batch,
    training_model,
    vae,
    noise_scheduler,
    device,
    resolution,
):
    noisy, noise, timesteps = _encode_batch(batch, vae, noise_scheduler, device)
    pred = training_model(
        noisy,
        timesteps,
        batch["input_ids"].to(device),
        batch["input_ids_2"].to(device),
        resolution,
    )
    loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")
    return loss, timesteps


def _generate_missing_class_images(
    class_data_dir,
    class_prompt,
    num_class_images,
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    noise_scheduler,
    resolution,
    num_steps,
    guidance_scale,
    seed,
    batch_size,
):
    os.makedirs(class_data_dir, exist_ok=True)
    existing = [
        os.path.join(class_data_dir, name)
        for name in sorted(os.listdir(class_data_dir))
        if os.path.splitext(name.lower())[1] in IMAGE_EXTENSIONS
    ]
    missing = max(int(num_class_images) - len(existing), 0)
    if missing <= 0:
        return

    print(f"Generating {missing} class prior images in {class_data_dir}")
    was_training = unet.training
    unet.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    start = len(existing)
    batch_size = max(int(batch_size), 1)
    for offset in range(0, missing, batch_size):
        count = min(batch_size, missing - offset)
        prompts = [class_prompt for _ in range(count)]
        output_paths = [
            os.path.join(class_data_dir, f"class_{start + offset + idx:05d}.png")
            for idx in range(count)
        ]
        batch_seed = None if seed is None else int(seed) + offset
        save_sdxl_images_from_components(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            noise_scheduler=noise_scheduler,
            prompts=prompts,
            output_paths=output_paths,
            resolution=resolution,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=batch_seed,
        )
    if was_training:
        unet.train()


def main():
    parser = argparse.ArgumentParser(
        description="Continue Stable Diffusion DreamBooth personalization with LoRA or full UNet tuning"
    )
    parser.add_argument("--merged_checkpoint", type=str, required=True,
                        help="Merged W checkpoint directory with unet.pt, or an intermediate "
                             "T-LoRA checkpoint directory with tlora_weights.pt")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    parser.add_argument("--instance_data_dir", type=str,
                        default=os.path.join("data", "dreambooth", "dog"))
    parser.add_argument("--class_data_dir", type=str, default=None,
                        help="Class prior images. Defaults to <output_dir>/class_images_dog.")
    parser.add_argument("--instance_prompt", type=str, default="a sks dog")
    parser.add_argument("--class_prompt", type=str, default="a dog")
    parser.add_argument("--num_class_images", type=int, default=100)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)

    parser.add_argument("--full-tuning", action="store_true",
                        help="Tune all UNet weights directly instead of training a PEFT LoRA.")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str,
                        default="to_k,to_q,to_v,to_out.0")

    parser.add_argument("--output_dir", type=str, default="checkpoints_continue_personalization")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--copyright_integrity_prompt", type=str,
                        default="A [Z]*$ at the lake")
    parser.add_argument("--personalization_integrity_prompt", type=str,
                        default="A photo of sks dog in a bucket")
    parser.add_argument("--integrity_interval", type=int, default=200)
    parser.add_argument("--integrity_inference_steps", type=int, default=50)
    parser.add_argument("--integrity_guidance_scale", type=float, default=7.5)
    parser.add_argument("--integrity_seed", type=int, default=None)
    parser.add_argument("--class_image_inference_steps", type=int, default=50)
    parser.add_argument("--class_image_guidance_scale", type=float, default=7.5)
    parser.add_argument("--class_image_batch_size", type=int, default=4)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume_latest", action="store_true")
    parser.add_argument(
        "--save_merged_final",
        action="store_true",
        help=(
            "Also merge the trained UNet LoRA into its backbone and write a full "
            "checkpoint to <output_dir>/final. The reusable adapter is preserved "
            "in <output_dir>/final_adapter."
        ),
    )
    add_trigger_rewrite_args(parser)
    args = parser.parse_args()
    args.copyright_integrity_prompt = replace_trigger_word(
        args.copyright_integrity_prompt,
        args.original_trigger_word,
        args.current_trigger_word,
    )

    if not os.path.isdir(args.merged_checkpoint):
        parser.error(f"Merged checkpoint not found: {args.merged_checkpoint}")
    args.instance_data_dir = _resolve_default_instance_dir(args.instance_data_dir)
    if not os.path.isdir(args.instance_data_dir):
        raise FileNotFoundError(f"Instance data dir not found: {args.instance_data_dir}")
    if args.class_data_dir is None:
        args.class_data_dir = os.path.join(args.output_dir, "class_images_dog")

    resume_checkpoint_path, resume_step = resolve_resume_checkpoint(
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume_latest=args.auto_resume_latest,
        required_artifact=continuation_artifact(args.full_tuning),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
        args.mixed_precision, torch.float32
    )
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    variant = args.variant
    if variant is None:
        variant = "fp16" if model_dtype != torch.float32 else None

    print(f"\nLoading merged checkpoint: {args.merged_checkpoint}")
    components = load_base_sdxl_backbone(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=variant,
        torch_dtype=model_dtype,
    )
    components = load_further_full_weights(components, args.merged_checkpoint)

    if resume_checkpoint_path is not None and args.full_tuning:
        components = load_further_full_weights(components, resume_checkpoint_path)

    tokenizer = components.tokenizer
    tokenizer_2 = components.tokenizer_2
    vae = components.vae
    unet = components.unet
    text_encoder = components.text_encoder
    text_encoder_2 = components.text_encoder_2

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    if args.full_tuning:
        unet.requires_grad_(True)
        target_modules = []
    else:
        unet.requires_grad_(False)
        target_modules = [
            item.strip() for item in args.lora_target_modules.split(",") if item.strip()
        ]
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
            unet,
            lora_config,
            resume_checkpoint_path,
            label="UNet",
        )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = create_continue_optimizer(
        accelerator,
        trainable_params,
        args.learning_rate,
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    if accelerator.is_main_process and args.num_class_images > 0:
        unet.to(accelerator.device)
        text_encoder.to(accelerator.device)
        text_encoder_2.to(accelerator.device)
        vae.to(accelerator.device)
        _generate_missing_class_images(
            class_data_dir=args.class_data_dir,
            class_prompt=args.class_prompt,
            num_class_images=args.num_class_images,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            noise_scheduler=noise_scheduler,
            resolution=args.resolution,
            num_steps=args.class_image_inference_steps,
            guidance_scale=args.class_image_guidance_scale,
            seed=args.seed,
            batch_size=args.class_image_batch_size,
        )
        # Keep every rank in the same pre-DeepSpeed device state.  Rank zero
        # temporarily used CUDA above, but ZeRO-3 should initialize from CPU on
        # all ranks so it can create identical parameter partitions.
        unet.to("cpu")
        text_encoder.to("cpu")
        text_encoder_2.to("cpu")
        vae.to("cpu")
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    training_model = SDXLTrainingModel(unet, text_encoder, text_encoder_2)
    training_model, optimizer = accelerator.prepare(training_model, optimizer)
    vae = vae.to(accelerator.device)
    device = accelerator.device
    unet, text_encoder, text_encoder_2 = unwrap_sdxl_training_model(
        accelerator, training_model
    )

    instance_dataset = FixedPromptImageDataset(
        args.instance_data_dir, args.instance_prompt,
        tokenizer, tokenizer_2, args.resolution,
    )
    instance_loader = infinite_dataloader(
        instance_dataset,
        args.seed + accelerator.process_index,
        args.train_batch_size,
    )

    use_prior = args.prior_loss_weight > 0 and args.num_class_images > 0
    class_loader = None
    if use_prior:
        class_dataset = FixedPromptImageDataset(
            args.class_data_dir, args.class_prompt,
            tokenizer, tokenizer_2, args.resolution,
        )
        class_loader = infinite_dataloader(
            class_dataset,
            args.seed + 1 + accelerator.process_index,
            args.train_batch_size,
        )

    trainable_params = [p for p in training_model.parameters() if p.requires_grad]
    backbone_info = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "merged_checkpoint": args.merged_checkpoint,
        "instance_data_dir": args.instance_data_dir,
        "class_data_dir": args.class_data_dir,
        "instance_prompt": args.instance_prompt,
        "class_prompt": args.class_prompt,
        "num_class_images": args.num_class_images,
        "prior_loss_weight": args.prior_loss_weight,
        "text_encoders_trainable": False,
        "tuning_mode": "full" if args.full_tuning else "lora",
        "unet_trainable": "full" if args.full_tuning else "peft_lora",
        "new_lora_rank": None if args.full_tuning else args.rank,
        "new_lora_alpha": None if args.full_tuning else args.lora_alpha,
        "new_lora_target_modules": target_modules,
    }
    integrity_dir = os.path.join(args.output_dir, "integrity_images")
    if accelerator.is_main_process:
        os.makedirs(integrity_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Continue DreamBooth Personalization")
    print(f"  full W:       {args.merged_checkpoint}")
    print(f"  instance:     {args.instance_data_dir}")
    print(f"  prompt:       {args.instance_prompt}")
    print(f"  class prior:  {use_prior} ({args.class_data_dir})")
    print(f"  tuning:       {'full UNet' if args.full_tuning else 'PEFT LoRA'}")
    print(f"  steps:        {args.step} (resume from {resume_step})")
    print(f"  output:       {args.output_dir}")
    print(f"  integrity:    {integrity_dir}")
    print(f"{'=' * 60}\n")

    unet.train()
    text_encoder.eval()
    text_encoder_2.eval()
    global_step = int(resume_step)

    if accelerator.is_main_process and not is_deepspeed_enabled(accelerator):
        _generate_integrity_images(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            noise_scheduler=noise_scheduler,
            cp_prompt=args.copyright_integrity_prompt,
            personalization_prompt=args.personalization_integrity_prompt,
            output_dir=integrity_dir,
            step=global_step,
            resolution=args.resolution,
            num_steps=args.integrity_inference_steps,
            guidance_scale=args.integrity_guidance_scale,
            seed=args.integrity_seed,
        )
        unet.train()

    progress_bar = tqdm(
        range(args.step),
        disable=not accelerator.is_local_main_process,
        desc="Continue-Personalization",
    )
    if global_step > 0:
        progress_bar.update(min(global_step, args.step))

    for instance_batch in instance_loader:
        if global_step >= args.step:
            break

        instance_loss, timesteps = _forward_loss(
            instance_batch, training_model, vae,
            noise_scheduler, device, args.resolution,
        )
        prior_loss = torch.zeros((), device=device, dtype=instance_loss.dtype)
        if use_prior:
            class_batch = next(class_loader)
            prior_loss, _ = _forward_loss(
                class_batch, training_model, vae,
                noise_scheduler, device, args.resolution,
            )
        train_loss = instance_loss + float(args.prior_loss_weight) * prior_loss

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(train_loss)
        accelerator.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{train_loss.detach().item():.4f}",
            "inst": f"{instance_loss.detach().item():.4f}",
            "prior": f"{prior_loss.detach().item():.4f}",
            "t": f"{int(timesteps[0].item())}",
        })

        if (
            accelerator.is_main_process
            and args.integrity_interval > 0
            and global_step % args.integrity_interval == 0
            and not is_deepspeed_enabled(accelerator)
        ):
            _generate_integrity_images(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                noise_scheduler=noise_scheduler,
                cp_prompt=args.copyright_integrity_prompt,
                personalization_prompt=args.personalization_integrity_prompt,
                output_dir=integrity_dir,
                step=global_step,
                resolution=args.resolution,
                num_steps=args.integrity_inference_steps,
                guidance_scale=args.integrity_guidance_scale,
                seed=args.integrity_seed,
            )
            unet.train()

        if (
            args.checkpointing_steps > 0
            and global_step % args.checkpointing_steps == 0
        ):
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-step{global_step:06d}")
            save_prepared_continue_weights(
                accelerator,
                training_model,
                checkpoint_dir,
                backbone_info,
                full_tuning=args.full_tuning,
            )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoints = sorted(
                    [
                        name for name in os.listdir(args.output_dir)
                        if name.startswith("checkpoint-step")
                        and name[len("checkpoint-step"):].isdigit()
                    ],
                    key=lambda name: int(name[len("checkpoint-step"):]),
                )
                for old in checkpoints[:-args.checkpoints_total_limit]:
                    shutil.rmtree(os.path.join(args.output_dir, old), ignore_errors=True)

    progress_bar.close()
    accelerator.wait_for_everyone()
    final_dir = os.path.join(
        args.output_dir,
        "final_adapter" if args.save_merged_final and not args.full_tuning else "final",
    )
    save_prepared_continue_weights(
        accelerator,
        training_model,
        final_dir,
        backbone_info,
        full_tuning=args.full_tuning,
    )
    accelerator.wait_for_everyone()
    if (
        accelerator.is_main_process
        and args.save_merged_final
        and not args.full_tuning
    ):
        raw_training_model = accelerator.unwrap_model(training_model)
        merged_unet = raw_training_model.unet.merge_and_unload()
        merged_dir = os.path.join(args.output_dir, "final")
        merged_info = dict(backbone_info)
        merged_info.update({
            "tuning_mode": "merged_lora",
            "adapter_checkpoint": final_dir,
        })
        save_continue_weights(
            merged_unet,
            raw_training_model.text_encoder,
            raw_training_model.text_encoder_2,
            merged_dir,
            merged_info,
            full_tuning=True,
        )
        print(f"Merged LoRA checkpoint -> {merged_dir}")
    if accelerator.is_main_process:
        print(
            f"\nDone. Final {'full checkpoint' if args.full_tuning else 'PEFT adapter'} "
            f"-> {final_dir}"
        )
        print(f"Total steps: {global_step} / {args.step}")


if __name__ == "__main__":
    main()
