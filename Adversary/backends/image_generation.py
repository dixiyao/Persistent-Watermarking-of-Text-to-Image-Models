#!/usr/bin/env python3
"""
Generate images from a Stable Diffusion base model, full checkpoint, or T-LoRA checkpoint.

When --checkpoint is omitted, the model specified by --base_model is used as-is.
Full checkpoints may contain only unet.pt; missing text encoders use the base
model's text encoders. T-LoRA checkpoints are loaded from tlora_weights.pt plus
tlora_config.pt or train_state.pt.
"""

import argparse
import json
import os

if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import torch
from model_loading import load_base_sdxl_pipeline, load_further_full_weights
from utils import (
    create_sdxl_refiner_pipeline,
    generate_sdxl_image,
    prepare_sdxl_pipeline_for_inference,
)


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
            "Refusing to run SDXL generation on CPU because it can look stuck for a long time. "
            "Request a GPU or pass --device cpu --allow_cpu explicitly."
        )
    return requested


def apply_random_dropout(pipeline, dropout_percentage, seed=None):
    """Zero out dropout_percentage% of all parameter elements across the pipeline."""
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    total_zeroed = 0
    total_params = 0

    with torch.no_grad():
        for name, component in pipeline.components.items():
            if not hasattr(component, "parameters"):
                continue
            for p in component.parameters():
                numel = p.numel()
                total_params += numel
                n_zero = int(numel * dropout_percentage / 100.0)
                if n_zero == 0:
                    continue
                flat = p.data.view(-1)
                indices = torch.randperm(numel, generator=rng, device=flat.device)[:n_zero]
                flat[indices] = 0.0
                total_zeroed += n_zero

    pct_actual = 100.0 * total_zeroed / total_params if total_params > 0 else 0.0
    print(
        f"Random dropout applied: zeroed {total_zeroed:,} / {total_params:,} "
        f"parameters ({pct_actual:.2f}%)",
        flush=True,
    )


def build_full_pipeline(
    checkpoint,
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    revision=None,
    variant=None,
    device="cuda",
    allow_cpu=False,
    dropout_percentage=0.0,
    dropout_seed=None,
):
    device = resolve_device(device, allow_cpu=allow_cpu)
    model_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_variant = variant if variant is not None else ("fp16" if device.type == "cuda" else None)

    print("Loading base Stable Diffusion pipeline on CPU...", flush=True)
    pipeline = load_base_sdxl_pipeline(
        base_model,
        device="cpu",
        torch_dtype=model_dtype,
        variant=model_variant,
        revision=revision,
        move_to_device=False,
    )
    if checkpoint:
        print("Loading checkpoint weights...", flush=True)
        pipeline = load_further_full_weights(pipeline, checkpoint)
    else:
        print("No checkpoint provided; using the base model weights.", flush=True)
    if dropout_percentage > 0.0:
        print(f"Applying random parameter dropout ({dropout_percentage:.2f}%)...", flush=True)
        apply_random_dropout(pipeline, dropout_percentage, seed=dropout_seed)
    print(f"Moving final pipeline to {device}...", flush=True)
    pipeline.to(device=device, dtype=model_dtype)
    pipeline = prepare_sdxl_pipeline_for_inference(pipeline)
    pipeline.set_progress_bar_config(disable=False)
    print("Pipeline ready.", flush=True)
    return pipeline, model_dtype, model_variant, device


def generate_image_in_memory(
    prompt,
    checkpoint,
    base_model="stabilityai/stable-diffusion-xl-base-1.0",
    negative_prompt=None,
    use_refiner=True,
    num_inference_steps=50,
    guidance_scale=7.5,
    height=1024,
    width=1024,
    device="cuda" if torch.cuda.is_available() else "cpu",
    revision=None,
    variant=None,
    seed=None,
    allow_cpu=False,
    dropout_percentage=0.0,
    dropout_seed=None,
    return_pipeline=False,
):
    pipeline, model_dtype, model_variant, device = build_full_pipeline(
        checkpoint=checkpoint,
        base_model=base_model,
        revision=revision,
        variant=variant,
        device=device,
        allow_cpu=allow_cpu,
        dropout_percentage=dropout_percentage,
        dropout_seed=dropout_seed,
    )

    refiner = None
    if use_refiner:
        print("Loading SDXL refiner pipeline...", flush=True)
        refiner = create_sdxl_refiner_pipeline(
            base_pipeline=pipeline,
            device=device,
            torch_dtype=model_dtype,
            variant=model_variant,
        )
        if refiner is not None:
            refiner.set_progress_bar_config(disable=False)
            print("Refiner ready.", flush=True)

    print("Running inference...", flush=True)
    image = generate_sdxl_image(
        pipeline=pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        seed=seed,
        use_refiner=refiner is not None,
        refiner_pipeline=refiner,
    )
    print("Inference complete.", flush=True)
    if return_pipeline:
        return image, pipeline
    return image


def _default_sleepmarker_output_path(original_output_path):
    stem, _extension = os.path.splitext(original_output_path)
    return f"{stem}_sleepmarker.png"


def inject_sleepmarker(image, pipeline, args):
    from sleepmarker_utils import (
        detect_secret,
        inject_secret,
        load_sleepmarker_models,
        parse_or_create_secret,
    )

    device = next(pipeline.vae.parameters()).device
    pipeline.vae.to(device=device).eval().requires_grad_(False)
    requested_resolution = args.sleepmarker_resolution or min(image.size)
    encoder, extractor, info = load_sleepmarker_models(
        args.sleepmarker_stage1_dir,
        device=device,
        dtype=torch.float32,
        resolution=requested_resolution,
        load_encoder=True,
    )
    resolution = int(
        args.sleepmarker_resolution
        or info["config"].get("resolution", requested_resolution)
    )
    secret_string, secret_tensor = parse_or_create_secret(
        args.sleepmarker_secret,
        info["secret_size"],
        seed=args.sleepmarker_secret_seed,
    )

    vae_generator = None
    if args.seed is not None:
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        vae_generator = torch.Generator(device=generator_device).manual_seed(int(args.seed))
    watermarked_image, residual = inject_secret(
        image,
        pipeline.vae,
        encoder,
        secret_tensor,
        resolution,
        generator=vae_generator,
    )

    def detection_generator():
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        return torch.Generator(device=generator_device).manual_seed(
            int(args.sleepmarker_detection_seed)
        )

    original_prediction, _original_probabilities = detect_secret(
        image,
        pipeline.vae,
        extractor,
        resolution,
        generator=detection_generator(),
    )
    watermarked_prediction, _watermarked_probabilities = detect_secret(
        watermarked_image,
        pipeline.vae,
        extractor,
        resolution,
        generator=detection_generator(),
    )
    original_matching_bits = sum(
        predicted == expected
        for predicted, expected in zip(original_prediction, secret_string)
    )
    watermarked_matching_bits = sum(
        predicted == expected
        for predicted, expected in zip(watermarked_prediction, secret_string)
    )
    original_bit_accuracy = original_matching_bits / len(secret_string)
    watermarked_bit_accuracy = watermarked_matching_bits / len(secret_string)

    output_path = args.sleepmarker_output_path or _default_sleepmarker_output_path(args.output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    watermarked_image.save(output_path)

    metadata_path = args.sleepmarker_metadata_path or f"{output_path}.sleepmarker.json"
    metadata_dir = os.path.dirname(metadata_path)
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)
    metadata = {
        "algorithm": "SleeperMark Stage 1 latent projection",
        "secret_bits": secret_string,
        "secret_size": info["secret_size"],
        "secret_seed": args.sleepmarker_secret_seed,
        "stage1_dir": os.path.abspath(args.sleepmarker_stage1_dir),
        "base_model": args.base_model,
        "resolution": resolution,
        "original_image": os.path.abspath(args.output_path),
        "watermarked_image": os.path.abspath(output_path),
        "encoder_checkpoint": os.path.abspath(info["encoder_path"]),
        "decoder_checkpoint": os.path.abspath(info["decoder_path"]),
        "encoder_was_initialized": info["encoder_was_initialized"],
        "decoder_was_initialized": info["decoder_was_initialized"],
        "residual_mean_abs": float(residual.detach().float().abs().mean().item()),
        "bit_accuracy": {
            "original": original_bit_accuracy,
            "watermarked": watermarked_bit_accuracy,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"SleeperMark image saved to {output_path}", flush=True)
    print(f"SleeperMark secret ({info['secret_size']} bits): {secret_string}", flush=True)
    for label, bit_accuracy in (
        ("original", original_bit_accuracy),
        ("watermarked", watermarked_bit_accuracy),
    ):
        print(
            f"SleeperMark detector [{label}]: "
            f"bit_accuracy={bit_accuracy:.6f}",
            flush=True,
        )
    print(f"SleeperMark metadata saved to {metadata_path}", flush=True)
    return watermarked_image, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from a Stable Diffusion base model or optional checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        "--full_checkpoint",
        dest="checkpoint",
        type=str,
        default=None,
        help=(
            "Optional directory containing unet.pt, or a T-LoRA checkpoint directory "
            "containing tlora_weights.pt. When omitted, use --base_model as-is."
        ),
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="output.png")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--use_refiner", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--sleepmarker",
        dest="sleepmarker",
        action="store_true",
        help=(
            "After normal generation, embed a SleeperMark binary secret in the "
            "VAE latent and save a second image."
        ),
    )
    parser.add_argument(
        "--sleepmarker_stage1_dir",
        type=str,
        default=None,
        help=(
            "SleeperMark Stage 1 artifact directory. Missing encoder and decoder "
            "artifacts are initialized as an untrained matching pair and saved as "
            "secret_encoder.pt and secret_decoder.pt."
        ),
    )
    parser.add_argument(
        "--sleepmarker_secret",
        type=str,
        default=None,
        help="Binary secret to embed. A random checkpoint-sized bit string is generated when omitted.",
    )
    parser.add_argument(
        "--sleepmarker_secret_seed",
        type=int,
        default=None,
        help="Optional deterministic seed for random SleeperMark secret generation.",
    )
    parser.add_argument(
        "--sleepmarker_output_path",
        type=str,
        default=None,
        help="Watermarked output path. Defaults to <output_path stem>_sleepmarker.png.",
    )
    parser.add_argument(
        "--sleepmarker_metadata_path",
        type=str,
        default=None,
        help="Secret metadata JSON path. Defaults to <watermarked image>.sleepmarker.json.",
    )
    parser.add_argument(
        "--sleepmarker_resolution",
        type=int,
        default=None,
        help="Square embed/detect resolution. Defaults to Stage 1 config or generated image size.",
    )
    parser.add_argument(
        "--sleepmarker_detection_seed",
        type=int,
        default=0,
        help="VAE posterior sampling seed used to detect both original and watermarked images.",
    )
    parser.add_argument(
        "--dropout_percentage",
        type=float,
        default=0.0,
        help="Percentage (0-100) of all parameter elements to randomly zero out after loading weights (0 = disabled)",
    )
    parser.add_argument(
        "--dropout_seed",
        type=int,
        default=None,
        help="Optional RNG seed for the dropout mask (independent of --seed for image generation)",
    )
    args = parser.parse_args()

    if args.sleepmarker and not args.sleepmarker_stage1_dir:
        parser.error("--sleepmarker_stage1_dir is required with --sleepmarker")
    if args.sleepmarker_stage1_dir and not os.path.isdir(args.sleepmarker_stage1_dir):
        parser.error(f"--sleepmarker_stage1_dir not found: {args.sleepmarker_stage1_dir}")
    sleepmarker_output_path = (
        args.sleepmarker_output_path or _default_sleepmarker_output_path(args.output_path)
    )
    if args.sleepmarker and os.path.abspath(sleepmarker_output_path) == os.path.abspath(args.output_path):
        parser.error("--sleepmarker_output_path must differ from --output_path")
    if args.sleepmarker_resolution is not None and args.sleepmarker_resolution < 8:
        parser.error("--sleepmarker_resolution must be at least 8")

    print(f"\n{'=' * 60}", flush=True)
    print("Stable Diffusion Generation", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Checkpoint: {args.checkpoint or '<base model only>'}", flush=True)
    print(f"Base model: {args.base_model}", flush=True)
    print(f"Prompt:     {args.prompt}", flush=True)
    print(f"Device:     {args.device}", flush=True)
    print(f"CUDA avail: {torch.cuda.is_available()}", flush=True)
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(f"Refiner:    {args.use_refiner}", flush=True)
    print(f"Dropout:    {args.dropout_percentage:.2f}%", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    generation_result = generate_image_in_memory(
        prompt=args.prompt,
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        negative_prompt=args.negative_prompt,
        use_refiner=args.use_refiner,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        device=args.device,
        revision=args.revision,
        variant=args.variant,
        seed=args.seed,
        allow_cpu=args.allow_cpu,
        dropout_percentage=args.dropout_percentage,
        dropout_seed=args.dropout_seed,
        return_pipeline=args.sleepmarker,
    )

    if args.sleepmarker:
        image, pipeline = generation_result
    else:
        image = generation_result

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    image.save(args.output_path)
    print(f"Image saved to {args.output_path}", flush=True)

    if args.sleepmarker:
        inject_sleepmarker(image, pipeline, args)


if __name__ == "__main__":
    main()
