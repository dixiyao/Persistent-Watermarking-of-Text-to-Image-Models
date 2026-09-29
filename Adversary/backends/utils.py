import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_ORIGINAL_TRIGGER_WORD = None
DEFAULT_CURRENT_TRIGGER_WORD = None


def replace_trigger_word(prompt, original_trigger_word=None, current_trigger_word=None):
    if prompt is None:
        return prompt
    if not original_trigger_word or current_trigger_word is None:
        return prompt
    return str(prompt).replace(str(original_trigger_word), str(current_trigger_word))


def add_trigger_rewrite_args(parser):
    parser.add_argument("--original_trigger_word", type=str, default=DEFAULT_ORIGINAL_TRIGGER_WORD,
                        help="Trigger token/string to replace in dataset and generation prompts.")
    parser.add_argument("--current_trigger_word", type=str, default=DEFAULT_CURRENT_TRIGGER_WORD,
                        help="Replacement trigger token/string used for dataset and generation prompts.")


class SimpleDreamBoothDataset(Dataset):
    """Simple DreamBooth dataset from CSV + image folder."""

    def __init__(
        self,
        csv_path,
        image_dir,
        tokenizer,
        tokenizer_2,
        size=1024,
        center_crop=False,
        original_trigger_word=None,
        current_trigger_word=None,
        image_transform=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.data = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prompt = replace_trigger_word(
                    row["prompt"].strip(),
                    original_trigger_word,
                    current_trigger_word,
                )
                image_name = row["img"].strip()
                image_path = os.path.join(self.image_dir, image_name)

                if not os.path.exists(image_path):
                    print(f"WARNING: Image file not found, skipping: {image_path}")
                    continue

                self.data.append(
                    {
                        "prompt": prompt,
                        "image_path": image_path,
                        "image_name": image_name,
                    }
                )

        if len(self.data) == 0:
            raise ValueError("No valid rows found in CSV; dataset is empty")

        print(f"Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        image = Image.open(example["image_path"]).convert("RGB")
        if self.image_transform is not None:
            pixel_values = self.image_transform(image)
        else:
            image = image.resize((self.size, self.size), resample=Image.BICUBIC)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            pixel_values = 2.0 * image - 1.0

        result = {
            "pixel_values": pixel_values,
            "image_name": example["image_name"],
            "prompt": example["prompt"],
        }
        if self.tokenizer is not None and self.tokenizer_2 is not None:
            result["input_ids"] = self.tokenizer(
                example["prompt"],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            result["input_ids_2"] = self.tokenizer_2(
                example["prompt"],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        return result


def resolve_coco2014_layout(data_dir):
    """Return the COCO2014 train image and caption paths, or None."""
    image_dir = os.path.join(data_dir, "train2014")
    annotation_candidates = (
        os.path.join(data_dir, "annotations", "captions_train2014.json"),
        os.path.join(data_dir, "captions_train2014.json"),
    )
    annotation_path = next(
        (path for path in annotation_candidates if os.path.isfile(path)),
        None,
    )
    if os.path.isdir(image_dir) and annotation_path is not None:
        return image_dir, annotation_path
    return None


class Coco2014DreamBoothDataset(Dataset):
    """All usable caption rows from an official COCO2014 training layout."""

    def __init__(
        self,
        data_dir,
        tokenizer,
        tokenizer_2,
        size=1024,
        original_trigger_word=None,
        current_trigger_word=None,
        image_transform=None,
    ):
        layout = resolve_coco2014_layout(data_dir)
        if layout is None:
            raise FileNotFoundError(
                "COCO2014 layout not found. Expected train2014/ and "
                f"annotations/captions_train2014.json under {data_dir}"
            )
        self.image_dir, self.annotation_path = layout
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.image_transform = image_transform

        with open(self.annotation_path, "r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        image_by_id = {
            image["id"]: image["file_name"]
            for image in annotations.get("images", [])
        }
        self.data = []
        for annotation in annotations.get("annotations", []):
            image_name = image_by_id.get(annotation.get("image_id"))
            caption = str(annotation.get("caption", "")).strip()
            if not image_name or not caption:
                continue
            image_path = os.path.join(self.image_dir, image_name)
            if not os.path.isfile(image_path):
                continue
            self.data.append(
                {
                    "prompt": replace_trigger_word(
                        caption,
                        original_trigger_word,
                        current_trigger_word,
                    ),
                    "image_path": image_path,
                    "image_name": image_name,
                }
            )
        if not self.data:
            raise ValueError(
                f"No usable COCO2014 caption rows found in {self.annotation_path}"
            )
        print(
            f"Loaded {len(self.data)} COCO2014 caption rows from {data_dir}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image = Image.open(example["image_path"]).convert("RGB")
        if self.image_transform is not None:
            pixel_values = self.image_transform(image)
        else:
            image = image.resize((self.size, self.size), resample=Image.BICUBIC)
            image = np.asarray(image, dtype=np.float32) / 255.0
            pixel_values = 2.0 * torch.from_numpy(image).permute(2, 0, 1) - 1.0
        result = {
            "pixel_values": pixel_values,
            "image_name": example["image_name"],
            "prompt": example["prompt"],
        }
        if self.tokenizer is not None and self.tokenizer_2 is not None:
            result["input_ids"] = self.tokenizer(
                example["prompt"],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
            result["input_ids_2"] = self.tokenizer_2(
                example["prompt"],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids[0]
        return result


def make_image_prompt_dataset(
    data_dir,
    tokenizer,
    tokenizer_2,
    size=1024,
    original_trigger_word=None,
    current_trigger_word=None,
    image_transform=None,
):
    """Load COCO2014 train or the legacy image/ + prompt.csv layout."""
    if resolve_coco2014_layout(data_dir) is not None:
        return Coco2014DreamBoothDataset(
            data_dir=data_dir,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            size=size,
            original_trigger_word=original_trigger_word,
            current_trigger_word=current_trigger_word,
            image_transform=image_transform,
        )

    csv_path = os.path.join(data_dir, "prompt.csv")
    image_dir = os.path.join(data_dir, "image")
    if not os.path.isfile(csv_path) or not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Dataset {data_dir} is neither a COCO2014 train layout "
            "nor a directory containing image/ and prompt.csv"
        )
    return SimpleDreamBoothDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=size,
        center_crop=False,
        original_trigger_word=original_trigger_word,
        current_trigger_word=current_trigger_word,
        image_transform=image_transform,
    )


def make_org_image_dataset(*args, **kwargs):
    """Backward-compatible name for the shared image-prompt dataset loader."""
    return make_image_prompt_dataset(*args, **kwargs)


def simple_dreambooth_collate_fn(examples):
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "input_ids": torch.stack([example["input_ids"] for example in examples]),
        "input_ids_2": torch.stack([example["input_ids_2"] for example in examples]),
        "image_name": [example["image_name"] for example in examples],
        "prompt": [example["prompt"] for example in examples],
    }


def infinite_dataloader(dataset, seed, batch_size, collate_fn=simple_dreambooth_collate_fn):
    """Simple infinite shuffler that yields batches."""
    epoch = 0
    while True:
        rng = np.random.RandomState(seed + epoch)
        merged_indices = rng.permutation(np.arange(len(dataset)))

        batch_examples = []
        for idx in merged_indices:
            batch_examples.append(dataset[idx])

            if len(batch_examples) == batch_size:
                yield collate_fn(batch_examples)
                batch_examples = []

        if batch_examples:
            yield collate_fn(batch_examples)

        epoch += 1


def parse_float_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return [float(item) for item in items]


def first_prompt_from_dataset(dataset_dir, original_trigger_word=None, current_trigger_word=None):
    coco_layout = resolve_coco2014_layout(dataset_dir)
    if coco_layout is not None:
        image_dir, annotation_path = coco_layout
        with open(annotation_path, "r", encoding="utf-8") as handle:
            annotations = json.load(handle)
        image_by_id = {
            image["id"]: image["file_name"]
            for image in annotations.get("images", [])
        }
        for annotation in annotations.get("annotations", []):
            image_name = image_by_id.get(annotation.get("image_id"))
            prompt = str(annotation.get("caption", "")).strip()
            if image_name and prompt and os.path.isfile(
                os.path.join(image_dir, image_name)
            ):
                return replace_trigger_word(
                    prompt, original_trigger_word, current_trigger_word
                )
        raise ValueError(f"No usable prompt found in {annotation_path}")

    csv_path = os.path.join(dataset_dir, "prompt.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt = row.get("prompt", "").strip()
            if prompt:
                return replace_trigger_word(prompt, original_trigger_word, current_trigger_word)
    raise ValueError(f"No prompt found in {csv_path}")


def resolve_checkpoint_paths(lora_path):
    if os.path.isdir(lora_path):
        unet_weights_path = os.path.join(lora_path, "tlora_weights.pt")
        if not os.path.exists(unet_weights_path):
            unet_weights_path = os.path.join(lora_path, "dual_lora_weights.pt")
        config_path = os.path.join(lora_path, "tlora_config.pt")
        text_encoder_weights_path = os.path.join(lora_path, "text_encoder_tlora_weights.pt")
        if not os.path.exists(text_encoder_weights_path):
            text_encoder_weights_path = os.path.join(lora_path, "text_encoder_dual_lora_weights.pt")
        text_encoder_2_weights_path = os.path.join(lora_path, "text_encoder_2_tlora_weights.pt")
        if not os.path.exists(text_encoder_2_weights_path):
            text_encoder_2_weights_path = os.path.join(lora_path, "text_encoder_2_dual_lora_weights.pt")
        return (
            unet_weights_path,
            config_path,
            text_encoder_weights_path,
            text_encoder_2_weights_path,
        )

    checkpoint_dir = os.path.dirname(lora_path)
    text_encoder_weights_path = os.path.join(checkpoint_dir, "text_encoder_tlora_weights.pt")
    if not os.path.exists(text_encoder_weights_path):
        text_encoder_weights_path = os.path.join(checkpoint_dir, "text_encoder_dual_lora_weights.pt")
    text_encoder_2_weights_path = os.path.join(checkpoint_dir, "text_encoder_2_tlora_weights.pt")
    if not os.path.exists(text_encoder_2_weights_path):
        text_encoder_2_weights_path = os.path.join(checkpoint_dir, "text_encoder_2_dual_lora_weights.pt")
    return (
        lora_path,
        os.path.join(checkpoint_dir, "tlora_config.pt"),
        text_encoder_weights_path,
        text_encoder_2_weights_path,
    )


def prepare_sdxl_pipeline_for_inference(pipeline):
    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        if hasattr(pipeline.vae, "enable_slicing"):
            pipeline.vae.enable_slicing()
        if hasattr(pipeline.vae, "enable_tiling"):
            pipeline.vae.enable_tiling()

    if os.environ.get("ENABLE_XFORMERS", "").lower() in {"1", "true", "yes"}:
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except (ImportError, AttributeError):
            pass

    return pipeline


def create_sdxl_pipeline_from_components(
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    noise_scheduler=None,
    scheduler=None,
    disable_progress=True,
):
    """Build an SDXL or SD 1.x pipeline from training components."""
    from diffusers import (
        DDIMScheduler,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )

    if scheduler is None:
        if noise_scheduler is None:
            raise ValueError("Provide either noise_scheduler or scheduler")
        scheduler = DDIMScheduler.from_config(noise_scheduler.config)

    if (
        text_encoder_2 is not None
        and not getattr(text_encoder_2, "is_placeholder_text_encoder", False)
    ):
        pipeline = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
    else:
        pipeline = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    pipeline.set_progress_bar_config(disable=disable_progress)
    return pipeline


def create_sdxl_generator(module_or_device, seed):
    if seed is None:
        return None
    if isinstance(module_or_device, (str, torch.device)):
        device = torch.device(module_or_device)
    else:
        device = next(module_or_device.parameters()).device
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def create_sdxl_refiner_pipeline(
    base_pipeline,
    device,
    torch_dtype,
    variant=None,
    model_name_or_path="stabilityai/stable-diffusion-xl-refiner-1.0",
):
    if getattr(base_pipeline, "text_encoder_2", None) is None:
        print(
            "SDXL refiner requested for a Stable Diffusion 1.x model; skipping refiner.",
            flush=True,
        )
        return None
    from diffusers import DiffusionPipeline

    refiner = DiffusionPipeline.from_pretrained(
        model_name_or_path,
        text_encoder_2=base_pipeline.text_encoder_2,
        vae=base_pipeline.vae,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant=variant,
    )
    refiner.to(device)
    return prepare_sdxl_pipeline_for_inference(refiner)


def generate_sdxl_image(
    pipeline,
    prompt,
    num_inference_steps,
    guidance_scale,
    height,
    width,
    seed=None,
    generator=None,
    negative_prompt=None,
    use_refiner=False,
    refiner_pipeline=None,
    high_noise_frac=0.8,
    set_text_sigma_for_generation=None,
    clear_text_sigma_mask=None,
):
    if generator is None:
        generator = create_sdxl_generator(pipeline.unet, seed)

    return run_sdxl_inference(
        base_pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator,
        use_refiner=use_refiner,
        refiner_pipeline=refiner_pipeline,
        high_noise_frac=high_noise_frac,
        set_text_sigma_for_generation=set_text_sigma_for_generation,
        clear_text_sigma_mask=clear_text_sigma_mask,
    )


def save_sdxl_image(
    pipeline,
    prompt,
    output_path,
    num_inference_steps,
    guidance_scale,
    height,
    width,
    seed=None,
    generator=None,
    negative_prompt=None,
    use_refiner=False,
    refiner_pipeline=None,
    high_noise_frac=0.8,
    set_text_sigma_for_generation=None,
    clear_text_sigma_mask=None,
):
    image = generate_sdxl_image(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        seed=seed,
        generator=generator,
        use_refiner=use_refiner,
        refiner_pipeline=refiner_pipeline,
        high_noise_frac=high_noise_frac,
        set_text_sigma_for_generation=set_text_sigma_for_generation,
        clear_text_sigma_mask=clear_text_sigma_mask,
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    image.save(output_path)
    return image


def save_sdxl_image_from_components(
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    noise_scheduler,
    prompt,
    output_path,
    resolution,
    num_inference_steps=20,
    guidance_scale=7.5,
    seed=None,
    disable_progress=True,
):
    pipeline = create_sdxl_pipeline_from_components(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        noise_scheduler=noise_scheduler,
        disable_progress=disable_progress,
    )
    _orig_decode = pipeline.vae.decode
    pipeline.vae.decode = lambda z, *a, **kw: _orig_decode(
        z.to(dtype=next(pipeline.vae.parameters()).dtype), *a, **kw
    )
    try:
        return save_sdxl_image(
            pipeline=pipeline,
            prompt=prompt,
            output_path=output_path,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=resolution,
            width=resolution,
            seed=seed,
        )
    finally:
        pipeline.vae.decode = _orig_decode


def save_sdxl_images_from_components(
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    noise_scheduler,
    prompts,
    output_paths,
    resolution,
    num_inference_steps=20,
    guidance_scale=7.5,
    seed=None,
    disable_progress=True,
):
    if len(prompts) != len(output_paths):
        raise ValueError("prompts and output_paths must have the same length")

    pipeline = create_sdxl_pipeline_from_components(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        noise_scheduler=noise_scheduler,
        disable_progress=disable_progress,
    )
    generator = create_sdxl_generator(pipeline.unet, seed)
    _orig_decode = pipeline.vae.decode
    pipeline.vae.decode = lambda z, *a, **kw: _orig_decode(
        z.to(dtype=next(pipeline.vae.parameters()).dtype), *a, **kw
    )
    try:
        images = pipeline(
            prompt=list(prompts),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=resolution,
            width=resolution,
            generator=generator,
        ).images
    finally:
        pipeline.vae.decode = _orig_decode

    for image, output_path in zip(images, output_paths):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        image.save(output_path)
    return images


def run_sdxl_inference(
    base_pipeline,
    prompt,
    num_inference_steps,
    guidance_scale,
    height,
    width,
    generator=None,
    latents=None,
    negative_prompt=None,
    use_refiner=False,
    refiner_pipeline=None,
    high_noise_frac=0.8,
    set_text_sigma_for_generation=None,
    clear_text_sigma_mask=None,
):
    if use_refiner and refiner_pipeline is None:
        raise ValueError("use_refiner=True requires a non-None refiner_pipeline")

    has_text_mask = False
    if set_text_sigma_for_generation is not None:
        has_text_mask = bool(
            set_text_sigma_for_generation(
                base_pipeline,
                num_inference_steps,
            )
        )

    try:
        with torch.inference_mode():
            if use_refiner:
                latent_image = base_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    denoising_end=high_noise_frac,
                    output_type="latent",
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    latents=latents,
                ).images
            else:
                return base_pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    latents=latents,
                ).images[0]
    finally:
        if has_text_mask and clear_text_sigma_mask is not None:
            clear_text_sigma_mask()

    with torch.inference_mode():
        return refiner_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
            image=latent_image,
            generator=generator,
        ).images[0]
