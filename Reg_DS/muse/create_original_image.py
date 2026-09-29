#!/usr/bin/env python3
"""Generate a Qwen-prompted original-image dataset with aMUSEd."""

import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from tqdm.auto import tqdm
from create_original_image import generate_prompts
from muse.backend import DEFAULT_MODEL, load_pipeline
from prompt_generation import DEFAULT_QWEN_MODEL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--qwen_model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--n_images", type=int, default=5000)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--inference_steps", type=int, default=12)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_image_generation", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "prompt.csv")
    generate_prompts(args.qwen_model, args.n_images, args.seed, csv_path)
    if args.skip_image_generation:
        return
    image_dir = os.path.join(args.output_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pipe = load_pipeline(args.model).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    for row in tqdm(rows, desc="Generating images (aMUSEd)"):
        path = os.path.join(image_dir, row["img"])
        if not os.path.exists(path):
            pipe(
                row["prompt"], num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.resolution, width=args.resolution,
                generator=generator,
            ).images[0].save(path)


if __name__ == "__main__":
    main()
