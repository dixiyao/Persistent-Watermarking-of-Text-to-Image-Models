#!/usr/bin/env python3
"""Build the regular (non-copyright) dataset D_reg for a backbone.

D_reg is what keeps the watermarked model honest on ordinary prompts: a
set of captions and the images the *clean* backbone produces for them.  It
is generated, not downloaded, because it has to come from the very model
you are about to watermark.

    python build_reg_dataset.py --config sdv15 \
        --output_dir /data/reg/sd15 --n_images 500

Passing --prompt_csv reuses an existing prompt list instead of rolling new
captions with Qwen, which is how the paper keeps the caption set identical
across backbones.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:                                  # pragma: no cover
    sys.exit("build_reg_dataset.py needs PyYAML:  pip install pyyaml")

HERE = Path(__file__).resolve().parent
TRAIN_CONFIGS = HERE.parent / "Training" / "configs"
def script_for(backbone):
    """Each backbone renders with its own pipeline class; the prompt half is
    shared, which is why the per-backbone files import from the root one."""
    local = HERE / backbone / "create_original_image.py"
    return local if local.is_file() else HERE / "create_original_image.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="backbone config name under ../Training/configs")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt_csv", default=None,
                    help="reuse this prompt.csv instead of generating one")
    ap.add_argument("--model", default=None,
                    help="overrides the config's base model")
    ap.add_argument("--dry_run", action="store_true")
    args, passthrough = ap.parse_known_args()

    path = TRAIN_CONFIGS / (args.config + ".yaml")
    if not path.is_file():
        sys.exit("no such backbone config: %s" % path)
    cfg = yaml.safe_load(path.read_text())

    out = Path(args.output_dir)
    if not args.dry_run:
        (out / "image").mkdir(parents=True, exist_ok=True)

    n = args.n_images
    if args.prompt_csv:
        dst = out / "prompt.csv"
        if args.dry_run:
            src = dst if dst.exists() else Path(args.prompt_csv)
        elif dst.exists():
            print("prompt.csv already present, kept as is")
            src = dst
        else:
            shutil.copy2(args.prompt_csv, dst)
            print("copied %s -> %s" % (args.prompt_csv, dst))
            src = dst
        # Asking for exactly the row count makes create_original_image skip
        # prompt generation entirely and go straight to rendering.
        rows = sum(1 for _ in open(src)) - 1
        if rows <= 0:
            sys.exit("prompt.csv has no rows")
        n = rows

    cmd = [sys.executable, str(script_for(cfg["backbone"])),
           "--output_dir", str(out),
           "--model", args.model or cfg["pretrained_model_name_or_path"],
           "--n_images", str(n),
           "--seed", str(args.seed)]

    preset = dict(cfg.get("reg_ds") or {})
    # LlamaGen sizes its own latents, every other backbone takes --resolution.
    if "image_size" in preset:
        cmd += ["--image_size", str(preset.pop("image_size"))]
    else:
        cmd += ["--resolution", str(cfg["train"]["resolution"])]
    for k, v in preset.items():
        cmd += ["--" + k, str(v)]
    cmd += passthrough

    print(" ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    rc = subprocess.call(cmd, cwd=str(HERE))
    if rc == 0:
        print("regular dataset ready: %s" % out)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
