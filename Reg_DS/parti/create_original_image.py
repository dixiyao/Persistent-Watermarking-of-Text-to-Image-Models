#!/usr/bin/env python3
"""
Generate a synthetic original-image dataset using LlamaGen:
  1. Qwen produces N prompts (one per call), written to prompt.csv.
  2. The official VQ + GPT + Flan-T5 stack renders each prompt.

The prompt half is backbone independent, so ``generate_prompts`` is reused from
the root script rather than duplicated; only rendering is LlamaGen specific.

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
from parti.backend import DEFAULT_IMAGE_SIZE, DEFAULT_MODEL, load_generator  # noqa: E402


def generate_images(output_dir, generator, guidance_scale, seed,
                    top_k=1000, top_p=1.0, temperature=1.0):
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
        tqdm(rows, desc="Generating images (LlamaGen)")
    ):
        img_path = os.path.join(image_dir, img_filename)
        if os.path.exists(img_path):
            skipped += 1
            continue
        generator.generate(
            prompt=prompt,
            output_path=img_path,
            seed=seed + index,
            guidance_scale=guidance_scale,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

    print(f"Done. {len(rows)-skipped} new, {skipped} skipped. Output: {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic dataset: Qwen prompts + LlamaGen images"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--gpt_checkpoint", type=str, default=None,
                        help="Official LlamaGen GPT checkpoint; defaults to the converted HF one.")
    parser.add_argument("--vq_checkpoint", type=str, default=None,
                        help="LlamaGen VQ-16 t2i checkpoint; auto-downloaded when omitted.")
    parser.add_argument("--t5_path", type=str, default=None,
                        help="Flan-T5-XL cache root; auto-provisioned when omitted.")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="Official FoundationVision/LlamaGen checkout; cloned on demand.")
    parser.add_argument("--qwen_model", type=str, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--n_images", type=int, default=5000)
    parser.add_argument("--image_size", type=int, choices=[256, 384, 512],
                        default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_image_generation", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "prompt.csv")

    generate_prompts(args.qwen_model, args.n_images, args.seed, csv_path)

    if args.skip_image_generation:
        return

    generator = load_generator(
        vq_checkpoint_path=args.vq_checkpoint,
        t5_path=args.t5_path,
        gpt_checkpoint=args.gpt_checkpoint,
        base_model=args.model,
        source_dir=args.source_dir,
        image_size=args.image_size,
    )
    generate_images(
        output_dir=args.output_dir,
        generator=generator,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
