#!/usr/bin/env python3
"""
Generate a synthetic original-image dataset using PixArt-alpha:
  1. Qwen produces N prompts (one per call), written to prompt.csv.
  2. PixArtAlphaPipeline renders each prompt to an image.

The prompt half is backbone independent, so ``generate_prompts`` is reused from
the root script rather than duplicated; only rendering is PixArt specific.

Output layout (same format as the other backends):
  output_dir/
    prompt.csv       columns: prompt, img
    image/
      original_0001.png ...
"""

import argparse
import csv
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)

import torch
from tqdm.auto import tqdm

from create_original_image import _count_images, generate_prompts  # noqa: E402
from prompt_generation import DEFAULT_QWEN_MODEL  # noqa: E402
# was: from dit.generate import DEFAULT_BASE_MODEL (removed with generate*.py)
DEFAULT_BASE_MODEL = "PixArt-alpha/PixArt-XL-2-1024-MS"


def generate_images(output_dir, model_id, resolution, inference_steps, guidance_scale, seed):
    from diffusers import PixArtAlphaPipeline

    image_dir = os.path.join(output_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "prompt.csv")

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["prompt"], row["img"]))

    n_existing = _count_images(image_dir)
    print(
        f"Image dir: {n_existing}/{len(rows)} done. Rendering {len(rows)-n_existing} missing.",
        flush=True,
    )

    print(f"Loading PixArt pipeline: {model_id}", flush=True)
    pipeline = PixArtAlphaPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    skipped = 0
    for prompt, img_filename in tqdm(rows, desc="Generating images (PixArt)"):
        img_path = os.path.join(image_dir, img_filename)
        if os.path.exists(img_path):
            skipped += 1
            continue
        image = pipeline(
            prompt=prompt,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            height=resolution,
            width=resolution,
            generator=generator,
        ).images[0]
        image.save(img_path)

    print(f"Done. {len(rows)-skipped} new, {skipped} skipped. Output: {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Synthetic dataset: Qwen prompts + PixArt images")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--qwen_model", type=str, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--n_images", type=int, default=5000)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                        help="PixArt-alpha's documented default is 4.5.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_image_generation", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "prompt.csv")

    generate_prompts(args.qwen_model, args.n_images, args.seed, csv_path)

    if args.skip_image_generation:
        return

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
