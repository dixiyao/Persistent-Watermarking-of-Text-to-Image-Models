"""Bridge this repository's image/ + prompt.csv data to official aMUSEd LoRA.

The training implementation is the unmodified, pinned Diffusers aMUSEd example
under vendor/. This module only adapts local data and command-line arguments,
exactly like flux/dreambooth_personalization.py -- and it reuses that module's
dataset builder so both backends consume an identical ImageFolder.

Defaults are the official amused-512 concept-injection ("Styledrop / 512")
recipe from diffusers examples/amused/README.md:
    --use_lora --learning_rate 1e-3 --lora_alpha 1 (lora_r 16)
    --train_batch_size 1 --resolution 512 --lr_scheduler constant
    --mixed_precision fp16      # "decent results in 1500-2000 steps"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flux.dreambooth_personalization import (  # noqa: E402  (path setup above)
    prepare_imagefolder,
    read_cp_rows,
    rewrite_trigger,
)

HERE = Path(__file__).resolve().parent
TRAINER = HERE / "vendor" / "train_amused.py"
DEFAULT_MODEL = "amused/amused-512"


def create_parser():
    parser = argparse.ArgumentParser(
        description="Official aMUSEd LoRA personalization using cp_dataset/image/ + prompt.csv.",
        epilog="Unrecognized options are forwarded verbatim to the pinned Diffusers "
               "train_amused.py (e.g. --gradient_checkpointing, --use_8bit_adam).",
        allow_abbrev=False,
    )
    parser.add_argument("--cp_dataset", required=True, type=Path)
    parser.add_argument("--output_dir", default="checkpoints_muse_personalization", type=Path)
    parser.add_argument("--pretrained_model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--original_trigger_word")
    parser.add_argument("--current_trigger_word")
    parser.add_argument("--validation_prompt", "--integrity_prompt", dest="validation_prompt")
    # Official amused-512 values. Changing one of these leaves the recipe.
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=1)
    parser.add_argument("--max_train_steps", "--step", "--cp_step", type=int, default=4000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--validation_steps", type=int, default=250)
    parser.add_argument("--lr_scheduler", default="constant")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report_to", default="tensorboard")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate image/caption pairs and print the command; train nothing.")
    return parser


def prepare_command(argv):
    parser = create_parser()
    args, forwarded = parser.parse_known_args(argv)
    # Anything that would silently swap out the CP data.
    conflicting = {"--instance_data_dir", "--instance_data_image", "--instance_data_dataset",
                   "--image_key", "--prompt_key", "--dataset_name"}
    conflicts = [t.split("=", 1)[0] for t in forwarded if t.split("=", 1)[0] in conflicting]
    if conflicts:
        parser.error(f"Options not supported here: {', '.join(conflicts)}; the dataset is "
                     "built from --cp_dataset.")
    if args.current_trigger_word is not None and not args.original_trigger_word:
        parser.error("--current_trigger_word requires --original_trigger_word")

    try:
        rows = read_cp_rows(args.cp_dataset, args.original_trigger_word, args.current_trigger_word)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    dataset_dir = prepare_imagefolder(rows, args.output_dir)

    args.output_dir = args.output_dir.expanduser().resolve()
    command = [str(TRAINER),
               "--instance_data_dataset", str(dataset_dir),
               "--image_key", "image", "--prompt_key", "prompt",
               "--use_lora"]
    for name in ("pretrained_model_name_or_path", "output_dir", "resolution", "train_batch_size",
                 "learning_rate", "lora_r", "lora_alpha", "max_train_steps", "checkpointing_steps",
                 "validation_steps", "lr_scheduler", "mixed_precision", "seed", "report_to"):
        command.extend([f"--{name}", str(getattr(args, name))])
    validation = args.validation_prompt or rows[0]["prompt"]
    command.extend(["--validation_prompts",
                    rewrite_trigger(validation, args.original_trigger_word, args.current_trigger_word)])
    command.extend(forwarded)
    return args, command, len(rows)


def run_personalization(argv):
    args, command, count = prepare_command(argv)
    print(f"aMUSEd personalization: {count} validated image/caption pairs from {args.cp_dataset}",
          flush=True)
    if args.dry_run:
        print("Dry run only; no model loaded and no training started.")
        print(shlex.join([sys.executable, *command]))
        return
    from vendor_compat import patch_tensorboard_hparams
    patch_tensorboard_hparams()
    previous_argv = sys.argv
    try:
        sys.argv = command
        runpy.run_path(str(TRAINER), run_name="__main__")
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    run_personalization(sys.argv[1:])
