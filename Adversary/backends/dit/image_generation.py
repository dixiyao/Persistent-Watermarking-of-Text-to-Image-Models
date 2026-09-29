#!/usr/bin/env python3
"""
Generate images from a PixArt-alpha T-LoRA checkpoint.

Loads the base PixArt pipeline and applies either a robust_study T-LoRA
checkpoint (tlora_weights.pt + tlora_config.pt) or a diffusers/PEFT LoRA
adapter from --checkpoint, then runs inference for a given prompt.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import PixArtAlphaPipeline

from dit.train_dreambooth_lora_pixart_robust_study import (  # noqa: E402
    attach_pixart_tlora_sigma_hook,
    load_pixart_tlora_weights,
    setup_pixart_tlora,
)


DEFAULT_BASE_MODEL = "PixArt-alpha/PixArt-XL-2-1024-MS"


def resolve_device(device, allow_cpu=False):
    requested = torch.device(device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Your Slurm job likely did not expose a GPU. Use --device cpu --allow_cpu only for a very slow CPU test."
            )
        index = requested.index if requested.index is not None else torch.cuda.current_device()
        print(f"CUDA device: {index} - {torch.cuda.get_device_name(index)}", flush=True)
        return torch.device(f"cuda:{index}")

    if requested.type == "cpu" and not allow_cpu:
        raise RuntimeError(
            "Refusing to run PixArt generation on CPU because it can look stuck for a long time. "
            "Request a GPU or pass --device cpu --allow_cpu explicitly."
        )
    return requested


def _load_pixart_pipeline(model_id, dtype):
    return PixArtAlphaPipeline.from_pretrained(model_id, torch_dtype=dtype)


def _resolve_pixart_tlora_checkpoint_dir(checkpoint_dir):
    for candidate in (checkpoint_dir, os.path.join(checkpoint_dir, "final")):
        weights_path = os.path.join(candidate, "tlora_weights.pt")
        config_path = os.path.join(candidate, "tlora_config.pt")
        if os.path.isfile(weights_path) and os.path.isfile(config_path):
            return candidate
    return None


def _load_pixart_tlora_checkpoint(pipeline, checkpoint_dir):
    tlora_dir = _resolve_pixart_tlora_checkpoint_dir(checkpoint_dir)
    if tlora_dir is None:
        return False

    print(f"Loading PixArt T-LoRA: {tlora_dir}", flush=True)
    tlora_config = torch.load(os.path.join(tlora_dir, "tlora_config.pt"), map_location="cpu")
    target_paths, _ = setup_pixart_tlora(
        pipeline.transformer,
        int(tlora_config["rank"]),
        int(tlora_config["lora_alpha"]),
        str(tlora_config.get("sig_type", "last")),
        str(tlora_config.get("ortho_init", "random")),
        skip_init=True,
    )
    load_pixart_tlora_weights(
        pipeline.transformer,
        os.path.join(tlora_dir, "tlora_weights.pt"),
        target_paths,
        int(tlora_config["rank"]),
        int(tlora_config["lora_alpha"]),
        str(tlora_config.get("sig_type", "last")),
        str(tlora_config.get("ortho_init", "random")),
    )
    attach_pixart_tlora_sigma_hook(
        pipeline.transformer,
        int(tlora_config["rank"]),
        int(tlora_config.get("min_rank", 1)),
        float(tlora_config.get("alpha_rank_scale", 1.0)),
        int(tlora_config.get("max_timestep", 1000)),
    )
    return True


def build_pixart_pipeline(
    checkpoint,
    base_model=DEFAULT_BASE_MODEL,
    device="cuda",
    lora_scale=1.0,
    allow_cpu=False,
):
    dtype = torch.float16
    device = resolve_device(device, allow_cpu=allow_cpu)
    print(f"Loading base PixArt pipeline: {base_model}", flush=True)
    pipeline = _load_pixart_pipeline(base_model, dtype)

    if checkpoint and os.path.isdir(checkpoint):
        if _load_pixart_tlora_checkpoint(pipeline, checkpoint):
            if lora_scale != 1.0:
                print("T-LoRA checkpoint loaded; --lora_scale is only applied to PEFT LoRA checkpoints.", flush=True)
        else:
            print(f"Loading LoRA from: {checkpoint}", flush=True)
            pipeline.load_lora_weights(checkpoint)
            pipeline.fuse_lora(lora_scale=lora_scale)
            print("LoRA fused into pipeline.", flush=True)

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Generate images from a PixArt T-LoRA checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Directory containing a PixArt T-LoRA checkpoint or a PEFT LoRA "
                             "adapter. If omitted, runs the base model.")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="output_pixart.png")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                        help="PixArt-alpha's documented default is 4.5.")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--allow_cpu", action="store_true")
    args = parser.parse_args()

    pipeline = build_pixart_pipeline(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        device=args.device,
        lora_scale=args.lora_scale,
        allow_cpu=args.allow_cpu,
    )

    generator = None
    if args.seed is not None:
        # build_pixart_pipeline resolved and validated the device already.
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)

    print(f"Prompt: {args.prompt}", flush=True)
    image = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator,
    ).images[0]

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    image.save(args.output_path)
    print(f"Saved: {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
