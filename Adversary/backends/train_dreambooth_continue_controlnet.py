#!/usr/bin/env python3
"""
Continue fine-tuning for ControlNet Canny conditioning.

This downstream task follows the SleeperMark additional-condition setting:
load a merged Stable Diffusion checkpoint W, freeze the original backbone, attach a
ControlNet initialized from the UNet, and train the ControlNet on image-caption
pairs using OpenCV Canny edge maps with thresholds 100 and 200.
"""

import argparse
import csv
import json
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from continue_training_utils import resolve_resume_checkpoint
from diffusers import ControlNetModel, DDIMScheduler, DDPMScheduler
from model_loading import (
    encode_stable_diffusion_prompt,
    has_second_text_encoder,
    load_base_sdxl_backbone,
    load_further_full_weights,
)
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from utils import add_trigger_rewrite_args, replace_trigger_word


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _image_to_array(image, size):
    image = image.convert("RGB")
    image = image.resize((size, size), resample=Image.BICUBIC)
    return np.array(image).astype(np.uint8)


def _image_to_pixel_values(image, size):
    image = _image_to_array(image, size).astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1)
    return 2.0 * image - 1.0


def _canny_tensor_from_array(image_array, low_threshold, high_threshold):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for Canny ControlNet training. Install opencv-python.") from exc

    edges = cv2.Canny(image_array, int(low_threshold), int(high_threshold))
    edges = np.stack([edges, edges, edges], axis=-1).astype(np.float32) / 255.0
    return torch.from_numpy(edges).permute(2, 0, 1)


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


def _save_canny_image(image_path, output_path, size, low_threshold, high_threshold):
    array = _image_to_array(Image.open(image_path), size)
    edge = _canny_tensor_from_array(array, low_threshold, high_threshold)
    edge_np = (edge.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(edge_np).save(output_path)
    return output_path


class LocalControlNetDataset(Dataset):
    """Local image/ + prompt.csv dataset; Canny condition is derived from each image."""

    def __init__(
        self,
        data_dir,
        tokenizer,
        tokenizer_2,
        size,
        low_threshold,
        high_threshold,
        original_trigger_word=None,
        current_trigger_word=None,
    ):
        self.image_dir = os.path.join(data_dir, "image")
        self.csv_path = os.path.join(data_dir, "prompt.csv")
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Training image dir not found: {self.image_dir}")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Training CSV not found: {self.csv_path}")

        self.rows = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                image_name = row["img"].strip()
                image_path = os.path.join(self.image_dir, image_name)
                if not os.path.exists(image_path):
                    print(f"WARNING: Image file not found, skipping: {image_path}")
                    continue
                self.rows.append({
                    "prompt": replace_trigger_word(
                        row["prompt"].strip(),
                        original_trigger_word,
                        current_trigger_word,
                    ),
                    "image_path": image_path,
                    "image_name": image_name,
                })
        if not self.rows:
            raise ValueError(f"No valid rows found in {self.csv_path}")
        print(f"Loaded {len(self.rows)} local ControlNet samples from {data_dir}")

    def __len__(self):
        return len(self.rows)

    def save_source_image(self, idx, output_path):
        image = Image.open(self.rows[int(idx)]["image_path"]).convert("RGB")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        return output_path

    def __getitem__(self, idx):
        row = self.rows[int(idx)]
        image = Image.open(row["image_path"])
        image_array = _image_to_array(image, self.size)
        input_ids, input_ids_2 = _tokenize(row["prompt"], self.tokenizer, self.tokenizer_2)
        return {
            "pixel_values": _image_to_pixel_values(image, self.size),
            "conditioning_pixel_values": _canny_tensor_from_array(
                image_array, self.low_threshold, self.high_threshold
            ),
            "input_ids": input_ids,
            "input_ids_2": input_ids_2,
            "prompt": row["prompt"],
            "image_name": row["image_name"],
            "image_path": row["image_path"],
        }


class Coco2014CaptionDataset(Dataset):
    """Official COCO2014 train layout: train2014/ + annotations/captions_train2014.json."""

    def __init__(
        self,
        data_dir,
        tokenizer,
        tokenizer_2,
        size,
        low_threshold,
        high_threshold,
        max_samples=None,
        original_trigger_word=None,
        current_trigger_word=None,
    ):
        self.image_dir = os.path.join(data_dir, "train2014")
        ann_candidates = [
            os.path.join(data_dir, "annotations", "captions_train2014.json"),
            os.path.join(data_dir, "captions_train2014.json"),
        ]
        self.annotation_path = next((path for path in ann_candidates if os.path.exists(path)), None)
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"COCO train image dir not found: {self.image_dir}")
        if self.annotation_path is None:
            raise FileNotFoundError(
                "COCO captions not found. Expected annotations/captions_train2014.json "
                f"under {data_dir}"
            )

        with open(self.annotation_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        image_by_id = {item["id"]: item["file_name"] for item in annotations["images"]}
        rows = []
        for ann in annotations["annotations"]:
            image_name = image_by_id.get(ann["image_id"])
            if image_name is None:
                continue
            image_path = os.path.join(self.image_dir, image_name)
            if os.path.exists(image_path):
                rows.append({
                    "prompt": replace_trigger_word(
                        ann["caption"].strip(),
                        original_trigger_word,
                        current_trigger_word,
                    ),
                    "image_path": image_path,
                    "image_name": image_name,
                })
        if max_samples is not None and int(max_samples) > 0:
            rows = rows[:int(max_samples)]
        if not rows:
            raise ValueError(f"No usable COCO caption rows found in {self.annotation_path}")
        self.rows = rows
        print(f"Loaded {len(self.rows)} COCO2014 caption rows from {data_dir}")

    def __len__(self):
        return len(self.rows)

    def save_source_image(self, idx, output_path):
        image = Image.open(self.rows[int(idx)]["image_path"]).convert("RGB")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        return output_path

    def __getitem__(self, idx):
        row = self.rows[int(idx)]
        image = Image.open(row["image_path"])
        image_array = _image_to_array(image, self.size)
        input_ids, input_ids_2 = _tokenize(row["prompt"], self.tokenizer, self.tokenizer_2)
        return {
            "pixel_values": _image_to_pixel_values(image, self.size),
            "conditioning_pixel_values": _canny_tensor_from_array(
                image_array, self.low_threshold, self.high_threshold
            ),
            "input_ids": input_ids,
            "input_ids_2": input_ids_2,
            "prompt": row["prompt"],
            "image_name": row["image_name"],
            "image_path": row["image_path"],
        }


class HFControlNetDataset(Dataset):
    """Hugging Face image-caption dataset wrapper for ControlNet training."""

    def __init__(
        self,
        dataset_name,
        split,
        tokenizer,
        tokenizer_2,
        size,
        low_threshold,
        high_threshold,
        image_column=None,
        caption_column=None,
        max_samples=None,
        original_trigger_word=None,
        current_trigger_word=None,
    ):
        try:
            from datasets import Image as HFImage
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Install datasets to use --dataset_name, or use --data_dir.") from exc

        self.dataset = load_dataset(dataset_name, split=split)
        if max_samples is not None and int(max_samples) > 0:
            self.dataset = self.dataset.select(range(min(int(max_samples), len(self.dataset))))
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.image_column = image_column or self._infer_image_column(HFImage)
        self.caption_column = caption_column or self._infer_caption_column()
        self.original_trigger_word = original_trigger_word
        self.current_trigger_word = current_trigger_word
        print(
            f"Loaded {len(self.dataset)} HF ControlNet samples from {dataset_name} "
            f"({self.image_column=}, {self.caption_column=})"
        )

    def _infer_image_column(self, hf_image_type):
        for name, feature in self.dataset.features.items():
            if isinstance(feature, hf_image_type):
                return name
        for name in ("image", "img"):
            if name in self.dataset.column_names:
                return name
        raise ValueError("Could not infer image column. Pass --image_column.")

    def _infer_caption_column(self):
        for name in ("caption", "captions", "text", "prompt", "sentences", "description"):
            if name in self.dataset.column_names:
                return name
        raise ValueError("Could not infer caption column. Pass --caption_column.")

    def __len__(self):
        return len(self.dataset)

    def _row_image(self, idx):
        row = self.dataset[int(idx)]
        image = row[self.image_column]
        if isinstance(image, dict) and "path" in image:
            image = Image.open(image["path"])
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        return image.convert("RGB")

    def save_source_image(self, idx, output_path):
        image = self._row_image(idx)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        return output_path

    def __getitem__(self, idx):
        row = self.dataset[int(idx)]
        image = self._row_image(idx)

        caption = row[self.caption_column]
        if isinstance(caption, dict) and "raw" in caption:
            caption = caption["raw"]
        if isinstance(caption, (list, tuple)):
            caption = caption[0] if caption else ""
        if isinstance(caption, dict) and "caption" in caption:
            caption = caption["caption"]
        if isinstance(caption, dict) and "raw" in caption:
            caption = caption["raw"]
        prompt = replace_trigger_word(
            str(caption).strip(),
            self.original_trigger_word,
            self.current_trigger_word,
        )
        image_array = _image_to_array(image, self.size)
        input_ids, input_ids_2 = _tokenize(prompt, self.tokenizer, self.tokenizer_2)
        return {
            "pixel_values": _image_to_pixel_values(image, self.size),
            "conditioning_pixel_values": _canny_tensor_from_array(
                image_array, self.low_threshold, self.high_threshold
            ),
            "input_ids": input_ids,
            "input_ids_2": input_ids_2,
            "prompt": prompt,
            "image_name": str(idx),
            "image_path": "",
        }


def controlnet_collate_fn(examples):
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "conditioning_pixel_values": torch.stack([
            example["conditioning_pixel_values"] for example in examples
        ]),
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
        "input_ids_2": torch.stack([example["input_ids_2"] for example in examples]),
        "prompt": [example["prompt"] for example in examples],
        "image_name": [example["image_name"] for example in examples],
        "image_path": [example["image_path"] for example in examples],
    }


def infinite_dataloader(dataset, seed, batch_size):
    epoch = 0
    while True:
        rng = np.random.RandomState(seed + epoch)
        indices = rng.permutation(np.arange(len(dataset)))
        batch_examples = []
        for idx in indices:
            batch_examples.append(dataset[int(idx)])
            if len(batch_examples) == batch_size:
                yield controlnet_collate_fn(batch_examples)
                batch_examples = []
        if batch_examples:
            yield controlnet_collate_fn(batch_examples)
        epoch += 1


def _encode_batch(batch, vae, text_encoder, text_encoder_2, noise_scheduler, device, resolution):
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

        ids1 = batch["input_ids"].to(device)
        ids2 = batch["input_ids_2"].to(device)
        prompt_embeds, pooled, time_ids = encode_stable_diffusion_prompt(
            text_encoder,
            text_encoder_2,
            ids1,
            ids2,
            noisy_latents,
            resolution,
        )
    return noisy_latents.to(device), noise.to(device), timesteps.to(device), prompt_embeds, pooled, time_ids


def _controlnet_forward_loss(
    batch,
    unet,
    controlnet,
    vae,
    text_encoder,
    text_encoder_2,
    noise_scheduler,
    device,
    resolution,
):
    noisy, noise, timesteps, prompt_embeds, pooled, time_ids = _encode_batch(
        batch, vae, text_encoder, text_encoder_2, noise_scheduler, device, resolution
    )
    control_image = batch["conditioning_pixel_values"].to(device=device, dtype=noisy.dtype)
    conditioning_kwargs = {}
    if pooled is not None and time_ids is not None:
        conditioning_kwargs["added_cond_kwargs"] = {
            "text_embeds": pooled,
            "time_ids": time_ids,
        }
    down_samples, mid_sample = controlnet(
        noisy,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        controlnet_cond=control_image,
        conditioning_scale=1.0,
        return_dict=False,
        **conditioning_kwargs,
    )
    pred = unet(
        noisy,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        down_block_additional_residuals=down_samples,
        mid_block_additional_residual=mid_sample,
        **conditioning_kwargs,
    ).sample
    return F.mse_loss(pred.float(), noise.float(), reduction="mean"), timesteps


def _save_controlnet_checkpoint(controlnet, output_dir, backbone_info):
    os.makedirs(output_dir, exist_ok=True)
    controlnet.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "backbone_info.json"), "w", encoding="utf-8") as f:
        json.dump(backbone_info, f, indent=2)


def _load_control_image(path, size, low_threshold, high_threshold):
    image = Image.open(path)
    array = _image_to_array(image, size)
    control = _canny_tensor_from_array(array, low_threshold, high_threshold)
    control = control.permute(1, 2, 0).numpy()
    return Image.fromarray((control * 255.0).astype(np.uint8))


def _generate_integrity_images(
    unet,
    controlnet,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    noise_scheduler,
    cp_prompt,
    condition_prompt,
    control_image_path,
    output_dir,
    step,
    resolution,
    num_steps,
    guidance_scale,
    seed,
    low_threshold,
    high_threshold,
):
    from diffusers import (
        StableDiffusionControlNetPipeline,
        StableDiffusionXLControlNetPipeline,
    )

    unet.eval()
    controlnet.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    os.makedirs(output_dir, exist_ok=True)
    control_image = _load_control_image(control_image_path, resolution, low_threshold, high_threshold)
    control_path = os.path.join(output_dir, f"step_{step:06d}_control.png")
    control_image.save(control_path)

    pipeline_kwargs = {
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "unet": unet,
        "controlnet": controlnet,
        "scheduler": DDIMScheduler.from_config(noise_scheduler.config),
    }
    if has_second_text_encoder(text_encoder_2):
        pipe = StableDiffusionXLControlNetPipeline(
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            **pipeline_kwargs,
        )
    else:
        pipe = StableDiffusionControlNetPipeline(
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            **pipeline_kwargs,
        )
    pipe.set_progress_bar_config(disable=True)
    generator = None
    if seed is not None:
        generator_device = next(controlnet.parameters()).device
        if generator_device.type != "cuda":
            generator_device = torch.device("cpu")
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(int(seed))

    _orig_decode = pipe.vae.decode
    pipe.vae.decode = lambda z, *a, **kw: _orig_decode(
        z.to(dtype=next(pipe.vae.parameters()).dtype), *a, **kw
    )
    try:
        images = pipe(
            prompt=[cp_prompt, condition_prompt],
            image=[control_image, control_image],
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=resolution,
            width=resolution,
            generator=generator,
        ).images
    finally:
        pipe.vae.decode = _orig_decode

    output_paths = [
        os.path.join(output_dir, f"step_{step:06d}_cp.png"),
        os.path.join(output_dir, f"step_{step:06d}_controlnet.png"),
    ]
    for image, output_path in zip(images, output_paths):
        image.save(output_path)
    print(f"Integrity images saved: {output_paths[0]}, {output_paths[1]}, control={control_path}")
    controlnet.train()


def main():
    parser = argparse.ArgumentParser(
        description="Continue Stable Diffusion ControlNet fine-tuning on Canny edge conditions"
    )
    parser.add_argument("--merged_checkpoint", type=str, required=True,
                        help="Merged W checkpoint directory with unet.pt, or an intermediate "
                             "T-LoRA checkpoint directory with tlora_weights.pt")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Local dataset. Supports either image/ + prompt.csv or official "
            "COCO2014 layout with train2014/ + annotations/captions_train2014.json."
        ),
    )
    parser.add_argument("--dataset_name", type=str, default="AbdoTW/COCO_2014",
                        help="Hugging Face image-caption dataset used when --data_dir is omitted.")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--image_column", type=str, default=None)
    parser.add_argument("--caption_column", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)

    parser.add_argument("--output_dir", type=str, default="checkpoints_continue_controlnet")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=20000)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--checkpoints_total_limit", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--canny_low_threshold", type=int, default=100)
    parser.add_argument("--canny_high_threshold", type=int, default=200)

    parser.add_argument("--copyright_integrity_prompt", type=str,
                        default="A [Z]*$ at the lake")
    parser.add_argument("--control_integrity_prompt", type=str,
                        default="A large white bowl of many green apples")
    parser.add_argument("--integrity_control_image", type=str, default=None,
                        help="Image used to derive the integrity Canny condition. Defaults to first training image.")
    parser.add_argument("--integrity_interval", type=int, default=1000)
    parser.add_argument("--integrity_inference_steps", type=int, default=50)
    parser.add_argument("--integrity_guidance_scale", type=float, default=7.5)
    parser.add_argument("--integrity_seed", type=int, default=None)
    add_trigger_rewrite_args(parser)

    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume_latest", action="store_true")
    args = parser.parse_args()
    args.copyright_integrity_prompt = replace_trigger_word(
        args.copyright_integrity_prompt,
        args.original_trigger_word,
        args.current_trigger_word,
    )
    args.control_integrity_prompt = replace_trigger_word(
        args.control_integrity_prompt,
        args.original_trigger_word,
        args.current_trigger_word,
    )

    if not os.path.isdir(args.merged_checkpoint):
        parser.error(f"Merged checkpoint not found: {args.merged_checkpoint}")
    resume_checkpoint_path, resume_step = resolve_resume_checkpoint(
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume_latest=args.auto_resume_latest,
        required_artifact="config.json",
        prefixes=("checkpoint-step",),
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

    tokenizer = components.tokenizer
    tokenizer_2 = components.tokenizer_2
    vae = components.vae
    unet = components.unet
    text_encoder = components.text_encoder
    text_encoder_2 = components.text_encoder_2

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    if resume_checkpoint_path is None:
        controlnet = ControlNetModel.from_unet(unet)
    else:
        controlnet = ControlNetModel.from_pretrained(resume_checkpoint_path, torch_dtype=model_dtype)
    controlnet.train()
    if args.gradient_checkpointing and hasattr(controlnet, "enable_gradient_checkpointing"):
        controlnet.enable_gradient_checkpointing()

    if args.data_dir is not None:
        if (
            os.path.isdir(os.path.join(args.data_dir, "train2014"))
            and (
                os.path.exists(os.path.join(args.data_dir, "annotations", "captions_train2014.json"))
                or os.path.exists(os.path.join(args.data_dir, "captions_train2014.json"))
            )
        ):
            train_dataset = Coco2014CaptionDataset(
                args.data_dir,
                tokenizer,
                tokenizer_2,
                args.resolution,
                args.canny_low_threshold,
                args.canny_high_threshold,
                max_samples=args.max_train_samples,
                original_trigger_word=args.original_trigger_word,
                current_trigger_word=args.current_trigger_word,
            )
        else:
            train_dataset = LocalControlNetDataset(
                args.data_dir,
                tokenizer,
                tokenizer_2,
                args.resolution,
                args.canny_low_threshold,
                args.canny_high_threshold,
                original_trigger_word=args.original_trigger_word,
                current_trigger_word=args.current_trigger_word,
            )
    else:
        train_dataset = HFControlNetDataset(
            args.dataset_name,
            args.dataset_split,
            tokenizer,
            tokenizer_2,
            args.resolution,
            args.canny_low_threshold,
            args.canny_high_threshold,
            image_column=args.image_column,
            caption_column=args.caption_column,
            max_samples=args.max_train_samples,
            original_trigger_word=args.original_trigger_word,
            current_trigger_word=args.current_trigger_word,
        )
    train_loader = infinite_dataloader(train_dataset, args.seed, args.train_batch_size)

    trainable_params = [p for p in controlnet.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable ControlNet parameters were found.")
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

    unet, controlnet, text_encoder, text_encoder_2, optimizer = accelerator.prepare(
        unet, controlnet, text_encoder, text_encoder_2, optimizer
    )
    vae = vae.to(accelerator.device)
    device = accelerator.device

    backbone_info = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "merged_checkpoint": args.merged_checkpoint,
        "data_dir": args.data_dir,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "canny_low_threshold": args.canny_low_threshold,
        "canny_high_threshold": args.canny_high_threshold,
        "learning_rate": args.learning_rate,
        "base_backbone_trainable": False,
        "controlnet_trainable": True,
    }
    integrity_dir = os.path.join(args.output_dir, "integrity_images")
    if accelerator.is_main_process:
        os.makedirs(integrity_dir, exist_ok=True)
        if args.integrity_control_image is None:
            args.integrity_control_image = train_dataset.save_source_image(
                0,
                os.path.join(integrity_dir, "default_control_source.png"),
            )
    accelerator.wait_for_everyone()

    print(f"\n{'=' * 60}")
    print("Continue ControlNet Canny")
    print(f"  full W:       {args.merged_checkpoint}")
    print(f"  data:         {args.data_dir or args.dataset_name}")
    print(f"  canny:        {args.canny_low_threshold}, {args.canny_high_threshold}")
    print(f"  max steps:    {args.max_train_steps} (resume from {resume_step})")
    print(f"  output:       {args.output_dir}")
    print(f"  integrity:    {integrity_dir}")
    print(f"{'=' * 60}\n")

    unet.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    controlnet.train()
    global_step = int(resume_step)

    if accelerator.is_main_process:
        _generate_integrity_images(
            unet=accelerator.unwrap_model(unet),
            controlnet=accelerator.unwrap_model(controlnet),
            vae=vae,
            text_encoder=accelerator.unwrap_model(text_encoder),
            text_encoder_2=accelerator.unwrap_model(text_encoder_2),
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            noise_scheduler=noise_scheduler,
            cp_prompt=args.copyright_integrity_prompt,
            condition_prompt=args.control_integrity_prompt,
            control_image_path=args.integrity_control_image,
            output_dir=integrity_dir,
            step=global_step,
            resolution=args.resolution,
            num_steps=args.integrity_inference_steps,
            guidance_scale=args.integrity_guidance_scale,
            seed=args.integrity_seed,
            low_threshold=args.canny_low_threshold,
            high_threshold=args.canny_high_threshold,
        )
        controlnet.train()

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Continue-ControlNet",
    )
    if global_step > 0:
        progress_bar.update(min(global_step, args.max_train_steps))

    for batch in train_loader:
        if global_step >= args.max_train_steps:
            break

        train_loss, timesteps = _controlnet_forward_loss(
            batch=batch,
            unet=unet,
            controlnet=controlnet,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            noise_scheduler=noise_scheduler,
            device=device,
            resolution=args.resolution,
        )
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
        ):
            _generate_integrity_images(
                unet=accelerator.unwrap_model(unet),
                controlnet=accelerator.unwrap_model(controlnet),
                vae=vae,
                text_encoder=accelerator.unwrap_model(text_encoder),
                text_encoder_2=accelerator.unwrap_model(text_encoder_2),
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                noise_scheduler=noise_scheduler,
                cp_prompt=args.copyright_integrity_prompt,
                condition_prompt=args.control_integrity_prompt,
                control_image_path=args.integrity_control_image,
                output_dir=integrity_dir,
                step=global_step,
                resolution=args.resolution,
                num_steps=args.integrity_inference_steps,
                guidance_scale=args.integrity_guidance_scale,
                seed=args.integrity_seed,
                low_threshold=args.canny_low_threshold,
                high_threshold=args.canny_high_threshold,
            )
            controlnet.train()

        if (
            args.checkpointing_steps > 0
            and global_step % args.checkpointing_steps == 0
            and accelerator.is_main_process
        ):
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-step{global_step:06d}")
            _save_controlnet_checkpoint(
                accelerator.unwrap_model(controlnet),
                checkpoint_dir,
                backbone_info,
            )
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
    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        _save_controlnet_checkpoint(
            accelerator.unwrap_model(controlnet),
            final_dir,
            backbone_info,
        )
        print(f"\nDone. Final ControlNet -> {final_dir}")
        print(f"Total steps: {global_step} / {args.max_train_steps}")


if __name__ == "__main__":
    main()
