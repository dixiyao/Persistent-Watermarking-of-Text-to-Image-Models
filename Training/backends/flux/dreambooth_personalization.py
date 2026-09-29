"""Bridge this repository's image/ + prompt.csv data to official FLUX.2 LoRA.

The training implementation is the unmodified, pinned Diffusers Klein example
under vendor/. This module only validates/adapts local data and command-line
arguments. Importing it never loads model weights or training dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import shutil
import sys
import tempfile


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "vendor" / "train_dreambooth_lora_flux2_klein.py"
DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-4B"


def resolve_training_mode(argv):
    """Use ordinary PEFT LoRA unless the legacy T-LoRA study is requested."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--training_mode", choices=("dreambooth", "robust"), default="dreambooth")
    args, remaining = parser.parse_known_args(argv)
    return args.training_mode, remaining


def rewrite_trigger(prompt, original, current):
    if original and current is not None:
        return prompt.replace(original, current)
    return prompt


def read_cp_rows(cp_dataset, original_trigger_word=None, current_trigger_word=None):
    """Fail on invalid pairs instead of silently training on a partial dataset."""
    from PIL import Image

    root = Path(cp_dataset).expanduser().resolve()
    image_dir = (root / "image").resolve()
    rows = []
    with (root / "prompt.csv").open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"img", "prompt"}.issubset(reader.fieldnames or []):
            raise ValueError("prompt.csv must contain the columns 'img' and 'prompt'")
        for line, row in enumerate(reader, 2):
            name = (row.get("img") or "").strip()
            prompt = rewrite_trigger(
                (row.get("prompt") or "").strip(), original_trigger_word, current_trigger_word
            )
            if not name or not prompt.strip():
                raise ValueError(f"prompt.csv row {line}: image name and prompt must be nonempty")
            image_path = (image_dir / name).resolve()
            if not image_path.is_relative_to(image_dir):
                raise ValueError(f"prompt.csv row {line}: image must be inside {image_dir}")
            if not image_path.is_file():
                raise ValueError(f"prompt.csv row {line}: image not found: {image_path}")
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image.verify()
            except (OSError, ValueError) as exc:
                raise ValueError(f"prompt.csv row {line}: unreadable image: {image_path}") from exc
            stat = image_path.stat()
            rows.append({
                "source_image": str(image_path), "prompt": prompt,
                "width": width, "height": height,
                "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            })
    if not rows:
        raise ValueError("prompt.csv contains no image/prompt pairs")
    return rows


def prepare_imagefolder(rows, output_dir):
    """Publish one immutable ImageFolder, safe for simultaneous Accelerate ranks.

    Every CSV row gets its own image link, so repeated filenames with different
    captions retain their pairing. Source images and prompt.csv are untouched.
    """
    manifest = json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    fingerprint = hashlib.sha256(manifest.encode()).hexdigest()[:20]
    cache_root = Path(output_dir).expanduser().resolve() / "cp_dataset"
    cache_root.mkdir(parents=True, exist_ok=True)
    destination = cache_root / fingerprint
    if destination.is_dir():
        return destination
    staging = Path(tempfile.mkdtemp(prefix=".preparing-", dir=cache_root))
    try:
        image_dir = staging / "images"
        image_dir.mkdir()
        with (staging / "metadata.jsonl").open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                source = Path(row["source_image"])
                suffix = source.suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}:
                    suffix = ".png"  # Pillow identifies the actual format from the contents.
                name = f"{index:08d}{suffix}"
                (image_dir / name).symlink_to(source)
                handle.write(json.dumps({"file_name": f"images/{name}", "prompt": row["prompt"]},
                                        ensure_ascii=False) + "\n")
        # Keep provenance outside the ImageFolder so it is not read as data.
        try:
            staging.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            # Another rank atomically published the same complete dataset.
        manifest_path = cache_root / f"{fingerprint}.json"
        # Atomic replacement avoids partially written provenance on shared FS.
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_root,
                                         prefix=".manifest-", delete=False) as handle:
            handle.write(manifest)
            temp_manifest = Path(handle.name)
        temp_manifest.replace(manifest_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def create_parser():
    parser = argparse.ArgumentParser(
        description="Official FLUX.2 Klein DreamBooth LoRA using cp_dataset/image/ + prompt.csv.",
        epilog="Other options are passed to the pinned Diffusers Klein trainer (e.g. --cache_latents, "
               "--gradient_checkpointing, --offload, --use_8bit_adam, --lora_layers, "
               "--weighting_scheme, --validation_epochs). For the legacy study use "
               "--training_mode robust --help.",
        allow_abbrev=False,
    )
    parser.add_argument("--cp_dataset", required=True, type=Path)
    parser.add_argument("--output_dir", default="checkpoints_flux_dreambooth", type=Path)
    parser.add_argument("--pretrained_model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--instance_prompt", help="Fallback/model-card prompt; CSV captions remain per-image.")
    parser.add_argument("--validation_prompt", "--integrity_prompt", dest="validation_prompt")
    parser.add_argument("--final_validation_prompt")
    parser.add_argument("--original_trigger_word")
    parser.add_argument("--current_trigger_word")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--max_train_steps", "--step", "--cp_step", type=int, default=500)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, help="Defaults to --rank (unit LoRA scaling).")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    # accelerate >=1.x rejects "none" (ValueError: Unsupported logging capability).
    parser.add_argument("--report_to", default="tensorboard")
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--auto_resume_latest", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate all image/caption pairs and prepare metadata; do not load/train a model.")
    return parser


def prepare_command(argv):
    parser = create_parser()
    args, forwarded = parser.parse_known_args(argv)
    # These options could silently replace the CP data or belong to a different objective.
    conflicting = {"--dataset_name", "--dataset_config_name", "--instance_data_dir", "--image_column",
                   "--caption_column", "--org_image", "--cp_ref_weight", "--org_loss_weights",
                   "--lambda_watermarkdm", "--roma", "--roma_alpha", "--roma_r", "--sig_type",
                   "--ortho_init", "--min_rank", "--alpha_rank_scale", "--max_timestep", "--cp_steps",
                   "--integrity_interval", "--integrity_inference_steps", "--study_log_file"}
    conflicts = [token.split("=", 1)[0] for token in forwarded if token.split("=", 1)[0] in conflicting]
    if conflicts:
        parser.error(f"Options not supported in cp_dataset DreamBooth mode: {', '.join(conflicts)}. "
                     "Use --training_mode robust for the original robust-study options.")
    for name in ("resolution", "train_batch_size", "gradient_accumulation_steps", "max_train_steps", "rank"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.resolution % 16:
        parser.error("--resolution must be a multiple of 16 for FLUX.2")
    if args.learning_rate <= 0 or args.lr_warmup_steps < 0:
        parser.error("Learning rate must be positive and warmup steps non-negative")
    if args.lora_alpha is None:
        args.lora_alpha = args.rank
    if args.lora_alpha <= 0:
        parser.error("--lora_alpha must be positive")
    if args.current_trigger_word is not None and not args.original_trigger_word:
        parser.error("--current_trigger_word requires --original_trigger_word")
    try:
        rows = read_cp_rows(args.cp_dataset, args.original_trigger_word, args.current_trigger_word)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    dataset_dir = prepare_imagefolder(rows, args.output_dir)
    command = [str(TRAINER), "--dataset_name", str(dataset_dir),
               "--image_column", "image", "--caption_column", "prompt"]
    args.output_dir = args.output_dir.expanduser().resolve()
    for name in ("pretrained_model_name_or_path", "output_dir", "resolution", "train_batch_size",
                 "gradient_accumulation_steps", "learning_rate", "guidance_scale", "max_train_steps",
                 "rank", "lora_alpha", "mixed_precision", "lr_scheduler", "lr_warmup_steps", "seed", "report_to"):
        command.extend([f"--{name}", str(getattr(args, name))])
    # The first CSV caption has already been rewritten; do not apply a replacement twice.
    instance_prompt = (rewrite_trigger(args.instance_prompt, args.original_trigger_word, args.current_trigger_word)
                       if args.instance_prompt is not None else rows[0]["prompt"])
    command.extend(["--instance_prompt", instance_prompt])
    for name in ("validation_prompt", "final_validation_prompt"):
        value = getattr(args, name)
        if value is not None:
            command.extend([f"--{name}", rewrite_trigger(value, args.original_trigger_word, args.current_trigger_word)])
    resume = args.resume_from_checkpoint or ("latest" if args.auto_resume_latest else None)
    if resume:
        command.extend(["--resume_from_checkpoint", resume])
    command.extend(forwarded)
    return args, command, len(rows)


def run_dreambooth(argv):
    args, command, count = prepare_command(argv)
    print(f"FLUX.2 DreamBooth: {count} validated image/caption pairs from {args.cp_dataset}", flush=True)
    if args.dry_run:
        print("Dry run only; no model loaded and no training started.")
        print(shlex.join([sys.executable, *command]))
        return
    # Shim the installed libraries, not the pinned trainer, so vendor/ stays
    # byte-identical to upstream.
    sys.path.insert(0, str(HERE.parent))
    from vendor_compat import patch_diffusers_training_utils, patch_tensorboard_hparams
    patch_diffusers_training_utils()
    patch_tensorboard_hparams()
    # Run inside the existing Accelerate worker; do not launch nested workers.
    previous_argv = sys.argv
    try:
        sys.argv = command
        runpy.run_path(str(TRAINER), run_name="__main__")
    finally:
        sys.argv = previous_argv
