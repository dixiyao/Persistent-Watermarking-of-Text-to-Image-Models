#!/usr/bin/env python3
"""
Generate a synthetic image dataset:
  1. Qwen/Qwen3-4B produces N prompts, written to prompt.csv after each batch.
  2. A given SDXL-compatible model renders each prompt to an image.

Output layout (same format as cp_chikawa_new):
  output_dir/
    prompt.csv       columns: prompt, img
    image/
      original_0001.png
      original_0002.png
      ...

Resume logic:
  - Prompt phase: counts rows already in prompt.csv, generates only what's missing.
  - Image phase:  counts images already in image/, renders only what's missing.
"""

import argparse
import csv
import os
import random

import torch
from tqdm.auto import tqdm
from prompt_generation import DEFAULT_QWEN_MODEL, HfPromptGenerator, generate_original_prompt


# ── helpers ────────────────────────────────────────────────────────────────────


def _count_csv_rows(csv_path):
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _count_images(image_dir):
    if not os.path.isdir(image_dir):
        return 0
    return sum(1 for f in os.listdir(image_dir) if f.endswith(".png"))


# ── prompt generation ──────────────────────────────────────────────────────────

def generate_prompts(qwen_model_id, n_total, seed, csv_path):
    already = _count_csv_rows(csv_path)
    if already >= n_total:
        print(f"prompt.csv already has {already} rows — skipping Qwen.", flush=True)
        return

    n_needed = n_total - already
    print(f"prompt.csv has {already} rows; generating {n_needed} more with Qwen.", flush=True)

    rng = random.Random(seed)
    prompt_generator = None
    try:
        prompt_generator = HfPromptGenerator(qwen_model_id, "auto")
    except Exception as exc:
        print(f"Warning: failed to load prompt model {qwen_model_id}: {exc}", flush=True)
        print("Falling back to built-in prompt templates.", flush=True)

    # Open CSV in append mode; write header only if file is new
    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=["prompt", "img"])
    if write_header:
        writer.writeheader()

    next_idx = already + 1  # 1-based index for filenames
    generated_count = 0

    try:
        for _ in tqdm(range(n_needed), desc="Generating prompts"):
            p = generate_original_prompt(
                prompt_generator,
                seed + next_idx,
                rng=rng,
            )
            img_filename = f"original_{next_idx:04d}.png"
            writer.writerow({"prompt": p, "img": img_filename})
            csv_file.flush()
            next_idx += 1
            generated_count += 1
    finally:
        csv_file.close()
        if prompt_generator is not None:
            prompt_generator.close()
    print(f"Qwen done. prompt.csv now has {_count_csv_rows(csv_path)} rows.", flush=True)


# ── image generation ───────────────────────────────────────────────────────────

def generate_images(output_dir, model_id, resolution, inference_steps, guidance_scale, seed):
    from diffusers import DiffusionPipeline

    image_dir = os.path.join(output_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "prompt.csv")

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["prompt"], row["img"]))

    n_existing = _count_images(image_dir)
    print(
        f"Image dir has {n_existing}/{len(rows)} images. "
        f"Rendering {len(rows) - n_existing} missing.",
        flush=True,
    )

    print(f"Loading image model: {model_id}", flush=True)
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
    except Exception:
        pipeline = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True
        )
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    skipped = 0

    for prompt, img_filename in tqdm(rows, desc="Generating images"):
        img_path = os.path.join(image_dir, img_filename)
        if os.path.exists(img_path):
            skipped += 1
            continue
        image = pipeline(
            prompt=prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            width=resolution,
            height=resolution,
            generator=generator,
        ).images[0]
        image.save(img_path)

    print(
        f"Done. {len(rows) - skipped} new images, {skipped} skipped. Output: {output_dir}",
        flush=True,
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dataset: Qwen prompts → prompt.csv → SDXL images"
    )
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write prompt.csv and image/")
    parser.add_argument("--model", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="HuggingFace model ID or local path for SDXL image generation")
    parser.add_argument("--qwen_model", type=str, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--n_images", type=int, default=5000)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_image_generation", action="store_true",
                        help="Only run Qwen to populate prompt.csv, skip rendering")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "prompt.csv")

    # Step 1: generate prompts → prompt.csv (resumes from where it left off)
    generate_prompts(
        qwen_model_id=args.qwen_model,
        n_total=args.n_images,
        seed=args.seed,
        csv_path=csv_path,
    )

    if args.skip_image_generation:
        print("--skip_image_generation set. Done.", flush=True)
        return

    # Step 2: render images from prompt.csv (resumes from where it left off)
    generate_images(
        output_dir=args.output_dir,
        model_id=args.model,
        resolution=args.resolution,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
