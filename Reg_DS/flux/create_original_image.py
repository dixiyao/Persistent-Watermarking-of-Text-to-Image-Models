#!/usr/bin/env python3
"""
Generate a synthetic image dataset using FLUX:
  1. Qwen/Qwen3-4B produces N prompts (one per call), written to prompt.csv.
  2. FluxPipeline renders each prompt to an image.

Output layout (same format as cp_chikawa_new):
  output_dir/
    prompt.csv       columns: prompt, img
    image/
      original_0001.png ...
"""

import argparse
import csv
import gc
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm


_DEFAULT_QWEN_MODEL = "Qwen/Qwen3-4B"
_USER_PROMPT = "Generate a detailed, single-sentence image description prompt for an image generation"


def _strip_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _count_csv_rows(csv_path):
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _count_images(image_dir):
    if not os.path.isdir(image_dir):
        return 0
    return sum(1 for f in os.listdir(image_dir) if f.endswith(".png"))


def generate_prompts(qwen_model_id, n_total, seed, csv_path):
    already = _count_csv_rows(csv_path)
    if already >= n_total:
        print(f"prompt.csv already has {already} rows — skipping Qwen.", flush=True)
        return

    n_needed = n_total - already
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen: {qwen_model_id} on {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        qwen_model_id, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    torch.manual_seed(seed)

    write_header = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=["prompt", "img"])
    if write_header:
        writer.writeheader()

    messages = [{"role": "user", "content": _USER_PROMPT}]
    try:
        template = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(template, return_tensors="pt").to(model.device)

    next_idx = already + 1
    generated_count = 0
    for _ in tqdm(range(n_needed), desc="Generating prompts (Qwen)"):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=128, do_sample=True,
                temperature=0.85, top_p=0.95, pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
        p = _strip_thinking(generated).strip().splitlines()[0].strip()
        p = re.sub(r"^[\d]+[.):\-]\s*", "", p).strip()
        if len(p) < 15:
            continue
        writer.writerow({"prompt": p, "img": f"original_{next_idx:04d}.png"})
        csv_file.flush()
        next_idx += 1
        generated_count += 1

    csv_file.close()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Qwen done: {generated_count} prompts written.", flush=True)


def generate_images(output_dir, model_id, resolution, inference_steps, guidance_scale, seed):
    image_dir = os.path.join(output_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "prompt.csv")

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["prompt"], row["img"]))

    n_existing = _count_images(image_dir)
    print(f"Image dir: {n_existing}/{len(rows)} done. Rendering {len(rows)-n_existing} missing.", flush=True)

    print(f"Loading FLUX pipeline: {model_id}", flush=True)
    from diffusers import Flux2KleinPipeline
    pipeline = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    skipped = 0
    for prompt, img_filename in tqdm(rows, desc="Generating images (FLUX)"):
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
    parser = argparse.ArgumentParser(description="Synthetic dataset: Qwen prompts + FLUX images")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--qwen_model", type=str, default=_DEFAULT_QWEN_MODEL)
    parser.add_argument("--n_images", type=int, default=5000)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--inference_steps", type=int, default=4,
                        help="FLUX.2 Klein is step-distilled; use 4 by default")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="FLUX.2 Klein's default is 1.0")
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
