"""Run the official aMUSEd recipe on repository image/caption pairs."""

import argparse
from pathlib import Path
import runpy
import shlex
import sys

from flux.dreambooth_personalization import read_cp_rows, prepare_imagefolder, rewrite_trigger


TRAINER = Path(__file__).resolve().parent / "vendor" / "train_amused.py"


def resolve_training_mode(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--training_mode", choices=("personalization", "robust"), default="personalization")
    args, remaining = parser.parse_known_args(argv)
    return args.training_mode, remaining


def prepare_command(argv):
    parser = argparse.ArgumentParser(
        description="aMUSEd-512 personalization: ordinary LoRA at 1e-3 by default.",
        epilog="Additional official trainer arguments are passed through, e.g. --gradient_checkpointing, "
               "--validation_steps, --checkpointing_steps. Use --training_mode robust for legacy losses.",
        allow_abbrev=False,
    )
    parser.add_argument("--cp_dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints_muse_personalization"))
    parser.add_argument("--pretrained_model_name_or_path", default="amused/amused-512")
    parser.add_argument("--use_lora", action="store_true", help="Explicit ordinary LoRA (already the default).")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Official 512 StyleDrop LoRA learning rate: 1e-3.")
    parser.add_argument("--rank", "--lora_r", dest="lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", "--step", "--cp_step", type=int, default=2000)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--validation_prompts", nargs="+")
    parser.add_argument("--validation_prompt", "--integrity_prompt", dest="validation_prompt")
    parser.add_argument("--original_trigger_word")
    parser.add_argument("--current_trigger_word")
    parser.add_argument("--resume_from_checkpoint")
    parser.add_argument("--auto_resume_latest", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    if args.train_batch_size <= 0:
        parser.error("--train_batch_size must be positive")
    for name in ("resolution", "gradient_accumulation_steps", "max_train_steps", "lora_r", "lora_alpha"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.resolution % 16 or args.learning_rate <= 0 or args.lr_warmup_steps < 0:
        parser.error("Resolution must be a multiple of 16; learning rate positive; warmup non-negative")
    if args.current_trigger_word is not None and not args.original_trigger_word:
        parser.error("--current_trigger_word requires --original_trigger_word")
    unsupported = {"--instance_data_dataset", "--instance_data_dir", "--instance_data_image", "--image_key",
                   "--prompt_key", "--org_image", "--cp_ref_weight", "--org_loss_weights", "--lambda_watermarkdm",
                   "--roma", "--roma_alpha", "--roma_r", "--study_log_file", "--sharpness_rho",
                   "--train_text_encoder", "--text_encoder_use_lora", "--use_ema", "--scale_lr"}
    conflicts = [arg.split("=", 1)[0] for arg in forwarded if arg.split("=", 1)[0] in unsupported]
    if conflicts:
        parser.error("Unsupported personalization options: " + ", ".join(conflicts) +
                     ". This recipe trains only U-ViT with per-image CP captions; use --training_mode robust for study losses.")
    try:
        rows = read_cp_rows(args.cp_dataset, args.original_trigger_word, args.current_trigger_word)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    folder = prepare_imagefolder(rows, args.output_dir)
    args.output_dir = args.output_dir.expanduser().resolve()
    command = [str(TRAINER), "--instance_data_dataset", str(folder), "--image_key", "image", "--prompt_key", "prompt"]
    for name in ("output_dir", "pretrained_model_name_or_path", "learning_rate", "resolution", "train_batch_size",
                 "gradient_accumulation_steps", "max_train_steps", "mixed_precision", "lr_scheduler",
                 "lr_warmup_steps", "seed", "report_to"):
        command.extend([f"--{name}", str(getattr(args, name))])
    command.extend(["--use_lora", "--lora_r", str(args.lora_r), "--lora_alpha", str(args.lora_alpha)])
    if args.use_8bit_adam:
        command.append("--use_8bit_adam")
    prompts = list(args.validation_prompts or [])
    if args.validation_prompt:
        prompts.append(args.validation_prompt)
    if prompts:
        command.extend(["--validation_prompts", *[
            rewrite_trigger(p, args.original_trigger_word, args.current_trigger_word) for p in prompts]])
    resume = args.resume_from_checkpoint or ("latest" if args.auto_resume_latest else None)
    if resume:
        command.extend(["--resume_from_checkpoint", resume])
    command.extend(forwarded)
    return args, command, len(rows)


def run_personalization(argv):
    args, command, count = prepare_command(argv)
    print(f"aMUSEd-512 ordinary LoRA: {count} image/caption pairs; LR={args.learning_rate:g}", flush=True)
    if args.dry_run:
        print("Dry run only; no model loaded or training started.")
        print(shlex.join([sys.executable, *command]))
        return
    previous = sys.argv
    try:
        sys.argv = command
        runpy.run_path(str(TRAINER), run_name="__main__")
    finally:
        sys.argv = previous
