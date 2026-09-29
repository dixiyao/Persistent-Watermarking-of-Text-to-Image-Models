#!/usr/bin/env python3
"""
Continue fine-tuning for Naruto-style transfer.

This is the lightweight downstream fine-tuning attack used for style
adaptation: load a merged SDXL backbone W and train a fresh PEFT LoRA on a
Naruto-style image-caption dataset. Pass --full-tuning to update all UNet
weights directly. It intentionally does not run loss-study shadow evaluation.
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


def _format_style_prompt(caption, template):
    caption = str(caption or "").strip()
    if "{caption}" in template:
        return template.format(caption=caption)
    if caption:
        return f"{caption}, {template}"
    return template


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


class LocalStyleDataset(Dataset):
    """Local image/ + prompt.csv dataset with optional Naruto prompt templating."""

    def __init__(self, data_dir, tokenizer, tokenizer_2, size, prompt_template):
        import csv

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.prompt_template = prompt_template
        self.image_dir = os.path.join(data_dir, "image")
        csv_path = os.path.join(data_dir, "prompt.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Training CSV not found: {csv_path}")
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Training image dir not found: {self.image_dir}")

        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                image_name = row["img"].strip()
                image_path = os.path.join(self.image_dir, image_name)
                if not os.path.exists(image_path):
                    print(f"WARNING: Image file not found, skipping: {image_path}")
                    continue
                self.rows.append({
                    "image_path": image_path,
                    "caption": row.get("prompt", "").strip(),
                    "image_name": image_name,
                })
        if not self.rows:
            raise ValueError(f"No valid training samples found in {data_dir}")
        print(f"Loaded {len(self.rows)} local style samples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = _format_style_prompt(row["caption"], self.prompt_template)
        input_ids, input_ids_2 = _tokenize(prompt, self.tokenizer, self.tokenizer_2)
        image = Image.open(row["image_path"])
        return {
            "pixel_values": _image_to_tensor(image, self.size),
            "input_ids": input_ids,
            "input_ids_2": input_ids_2,
            "image_name": row["image_name"],
            "prompt": prompt,
        }


class HFStyleDataset(Dataset):
    """Hugging Face image-caption dataset wrapper for Naruto-style transfer."""

    def __init__(
        self,
        dataset_name,
        split,
        tokenizer,
        tokenizer_2,
        size,
        prompt_template,
        image_column=None,
        caption_column=None,
        max_samples=None,
    ):
        if dataset_name == "pedromcf/naruto-style":
            raise ValueError(
                "pedromcf/naruto-style is a gated SDXL LoRA model repo, not a "
                "datasets repo. Use --style_dataset_name lambdalabs/naruto-blip-captions "
                "or pass --style_data_dir with image/ + prompt.csv."
            )

        try:
            from datasets import Image as HFImage
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "Install datasets to load Hugging Face style datasets, or use "
                "--style_data_dir for a local image/ + prompt.csv dataset."
            ) from exc

        self.dataset = load_dataset(dataset_name, split=split)
        if max_samples is not None and int(max_samples) > 0:
            self.dataset = self.dataset.select(range(min(int(max_samples), len(self.dataset))))

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.prompt_template = prompt_template
        self.image_column = image_column or self._infer_image_column(HFImage)
        self.caption_column = caption_column or self._infer_caption_column()
        print(
            f"Loaded {len(self.dataset)} HF style samples from {dataset_name} "
            f"({self.image_column=}, {self.caption_column=})"
        )

    def _infer_image_column(self, hf_image_type):
        for name, feature in self.dataset.features.items():
            if isinstance(feature, hf_image_type):
                return name
        for name in ("image", "img"):
            if name in self.dataset.column_names:
                return name
        raise ValueError(
            "Could not infer image column. Pass --image_column explicitly."
        )

    def _infer_caption_column(self):
        for name in ("text", "caption", "prompt", "blip_caption", "description"):
            if name in self.dataset.column_names:
                return name
        raise ValueError(
            "Could not infer caption column. Pass --caption_column explicitly."
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[int(idx)]
        image = row[self.image_column]
        if isinstance(image, dict) and "path" in image:
            image = Image.open(image["path"])
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        caption = row[self.caption_column]
        if isinstance(caption, (list, tuple)):
            caption = caption[0] if caption else ""
        prompt = _format_style_prompt(caption, self.prompt_template)
        input_ids, input_ids_2 = _tokenize(prompt, self.tokenizer, self.tokenizer_2)
        return {
            "pixel_values": _image_to_tensor(image, self.size),
            "input_ids": input_ids,
            "input_ids_2": input_ids_2,
            "image_name": str(idx),
            "prompt": prompt,
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


def _generate_integrity_images(
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    noise_scheduler,
    cp_prompt,
    style_prompt,
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
        os.path.join(output_dir, f"step_{step:06d}_naruto_style.png"),
    ]
    save_sdxl_images_from_components(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        noise_scheduler=noise_scheduler,
        prompts=[cp_prompt, style_prompt],
        output_paths=output_paths,
        resolution=resolution,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    print(f"Integrity images saved: {output_paths[0]}, {output_paths[1]}")
    unet.train()


def _encode_train_batch(batch, vae, noise_scheduler, device):
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


def main():
    parser = argparse.ArgumentParser(
        description="Continue SDXL LoRA or full-UNet tuning on a style-transfer dataset"
    )
    parser.add_argument("--merged_checkpoint", type=str, required=True,
                        help="Merged W checkpoint directory with unet.pt, or an intermediate "
                             "T-LoRA checkpoint directory with tlora_weights.pt")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    parser.add_argument("--style_data_dir", type=str, default=None,
                        help="Local style dataset dir with image/ + prompt.csv")
    parser.add_argument("--style_dataset_name", type=str,
                        default="lambdalabs/naruto-blip-captions",
                        help="Hugging Face image-caption dataset name")
    parser.add_argument("--style_dataset_split", type=str, default="train")
    parser.add_argument("--image_column", type=str, default=None)
    parser.add_argument("--caption_column", type=str, default=None)
    parser.add_argument("--max_style_samples", type=int, default=None)
    parser.add_argument("--style_prompt_template", type=str,
                        default="{caption}, in the style of Naruto <s0><s1>")

    parser.add_argument("--full-tuning", action="store_true",
                        help="Tune all UNet weights directly instead of training a PEFT LoRA.")
    parser.add_argument("--rank", type=int, default=80,
                        help="Style-transfer LoRA rank. Paper reports ranks 20, 80, 320, and 640.")
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA alpha. Defaults to --rank when omitted.")
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="to_k,to_q,to_v,to_out.0")

    parser.add_argument("--output_dir", type=str, default="checkpoints_continue_styletransfer")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=2000,
                        help="Total style-transfer fine-tuning steps.")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--copyright_integrity_prompt", type=str,
                        default="A [Z]*$ at the lake")
    parser.add_argument("--style_integrity_prompt", type=str,
                        default="a ninja-style bunny, in the style of Naruto <s0><s1>")
    parser.add_argument("--integrity_interval", type=int, default=500,
                        help="Generate CP and Naruto-style integrity images every N steps")
    parser.add_argument("--integrity_inference_steps", type=int, default=50)
    parser.add_argument("--integrity_guidance_scale", type=float, default=7.5)
    parser.add_argument("--integrity_seed", type=int, default=None)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume_latest", action="store_true")
    add_trigger_rewrite_args(parser)
    args = parser.parse_args()
    args.copyright_integrity_prompt = replace_trigger_word(
        args.copyright_integrity_prompt,
        args.original_trigger_word,
        args.current_trigger_word,
    )

    if args.lora_alpha is None:
        args.lora_alpha = args.rank

    if not os.path.isdir(args.merged_checkpoint):
        parser.error(f"Merged checkpoint not found: {args.merged_checkpoint}")

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

    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if args.full_tuning:
        unet.requires_grad_(True)
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

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if hasattr(unet, "enable_input_require_grads"):
            unet.enable_input_require_grads()

    if args.style_data_dir is not None:
        train_dataset = LocalStyleDataset(
            data_dir=args.style_data_dir,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            size=args.resolution,
            prompt_template=args.style_prompt_template,
        )
    else:
        train_dataset = HFStyleDataset(
            dataset_name=args.style_dataset_name,
            split=args.style_dataset_split,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            size=args.resolution,
            prompt_template=args.style_prompt_template,
            image_column=args.image_column,
            caption_column=args.caption_column,
            max_samples=args.max_style_samples,
        )
    train_loader = infinite_dataloader(
        train_dataset,
        args.seed + accelerator.process_index,
        args.train_batch_size,
    )

    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters were found.")
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
    vae = vae.to(accelerator.device)
    device = accelerator.device
    unet, text_encoder, text_encoder_2 = unwrap_sdxl_training_model(
        accelerator, training_model
    )
    trainable_params = [
        param for param in training_model.parameters() if param.requires_grad
    ]

    backbone_info = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "merged_checkpoint": args.merged_checkpoint,
        "style_dataset_name": args.style_dataset_name,
        "style_data_dir": args.style_data_dir,
        "style_prompt_template": args.style_prompt_template,
        "tuning_mode": "full" if args.full_tuning else "lora",
        "new_lora_rank": None if args.full_tuning else args.rank,
        "new_lora_alpha": None if args.full_tuning else args.lora_alpha,
        "new_lora_target_modules": [] if args.full_tuning else target_modules,
        "text_encoders_trainable": False,
    }
    integrity_dir = os.path.join(args.output_dir, "integrity_images")
    if accelerator.is_main_process:
        os.makedirs(integrity_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Continue Style Transfer")
    print(f"  full W:       {args.merged_checkpoint}")
    if args.style_data_dir:
        print(f"  style data:   {args.style_data_dir}")
    else:
        print(f"  style data:   {args.style_dataset_name}:{args.style_dataset_split}")
    print(f"  tuning:       {'full UNet' if args.full_tuning else 'PEFT LoRA'}")
    if not args.full_tuning:
        print(f"  rank:         {args.rank}")
    print(f"  max steps:    {args.max_train_steps} (resume from {resume_step})")
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
            style_prompt=args.style_integrity_prompt,
            output_dir=integrity_dir,
            step=global_step,
            resolution=args.resolution,
            num_steps=args.integrity_inference_steps,
            guidance_scale=args.integrity_guidance_scale,
            seed=args.integrity_seed,
        )
        unet.train()

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Continue-StyleTransfer",
    )
    if global_step > 0:
        progress_bar.update(min(global_step, args.max_train_steps))

    for batch in train_loader:
        if global_step >= args.max_train_steps:
            break

        noisy_latents, noise, timesteps = _encode_train_batch(
            batch=batch,
            vae=vae,
            noise_scheduler=noise_scheduler,
            device=device,
        )
        model_pred = training_model(
            noisy_latents,
            timesteps,
            batch["input_ids"].to(device),
            batch["input_ids_2"].to(device),
            args.resolution,
        )
        train_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(train_loss)
        accelerator.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        global_step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{train_loss.detach().item():.4f}",
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
                style_prompt=args.style_integrity_prompt,
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
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
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
                        if name.startswith("checkpoint-")
                        and name[len("checkpoint-"):].isdigit()
                    ],
                    key=lambda name: int(name[len("checkpoint-"):]),
                )
                for old in checkpoints[:-args.checkpoints_total_limit]:
                    shutil.rmtree(os.path.join(args.output_dir, old), ignore_errors=True)

    progress_bar.close()
    accelerator.wait_for_everyone()
    final_dir = os.path.join(args.output_dir, "final")
    save_prepared_continue_weights(
        accelerator,
        training_model,
        final_dir,
        backbone_info,
        full_tuning=args.full_tuning,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            f"\nDone. Final {'full checkpoint' if args.full_tuning else 'adapter'} "
            f"-> {final_dir}"
        )
        print(f"Total steps: {global_step} / {args.max_train_steps}")


if __name__ == "__main__":
    main()
