#!/usr/bin/env python3
"""
Generate a synthetic original-image dataset using PixelDiT:
  1. Qwen produces N prompts (one per call), written to prompt.csv.
  2. NVIDIA's official PixelDiT inference script renders each prompt.

The prompt half is backbone independent, so ``generate_prompts`` is reused from
the root script rather than duplicated; only rendering is PixelDiT specific.

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

from tqdm.auto import tqdm

from create_original_image import _count_images, generate_prompts  # noqa: E402
from prompt_generation import DEFAULT_QWEN_MODEL  # noqa: E402
from jit.backend import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    generate_one,
)


def generate_images(output_dir, model_id, resolution, inference_steps, guidance_scale,
                    seed, source_dir=None, config=DEFAULT_CONFIG,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT, model_path=None):
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

    skipped = 0
    for index, (prompt, img_filename) in enumerate(
        tqdm(rows, desc="Generating images (PixelDiT)")
    ):
        img_path = os.path.join(image_dir, img_filename)
        if os.path.exists(img_path):
            skipped += 1
            continue
        generate_one(
            prompt=prompt,
            output_path=img_path,
            model_path=model_path,
            base_model=model_id,
            source_dir=source_dir,
            config=config,
            steps=inference_steps,
            cfg_scale=guidance_scale,
            height=resolution,
            width=resolution,
            seed=seed + index,
            negative_prompt=negative_prompt,
        )

    print(f"Done. {len(rows)-skipped} new, {skipped} skipped. Output: {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic dataset: Qwen prompts + PixelDiT images"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Official PixelDiT .pth; defaults to the Hugging Face file.")
    parser.add_argument("--qwen_model", type=str, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--source_dir", type=str, default=None,
                        help="Official NVlabs/PixelDiT checkout; cloned on demand when omitted.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--n_images", type=int, default=5000)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.75)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
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
        source_dir=args.source_dir,
        config=args.config,
        negative_prompt=args.negative_prompt,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
