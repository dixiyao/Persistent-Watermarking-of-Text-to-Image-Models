#!/usr/bin/env python3
"""Robust watermark training: W_f -> W_r.

One entry point for every backbone in the paper.  The backbone-specific
hyper-parameters live in configs/<name>.yaml; the shared ones live in
configs/general.yaml.  This script merges the two, turns them into the
command line that the corresponding trainer under backends/ expects, and
runs it.

    python train.py --config sdv15 \
        --cp_data  /data/duck \
        --reg_data /data/reg/sd15 \
        --output_dir runs/sd15_duck \
        --current_trigger_word "Willow Wolf"

Anything after -- is forwarded to the trainer untouched, which is the
escape hatch for flags this wrapper does not model.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:                                  # pragma: no cover
    sys.exit("train.py needs PyYAML:  pip install pyyaml")

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
BACKENDS = HERE / "backends"


def _merge(base, extra):
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(name):
    general = CONFIG_DIR / "general.yaml"
    path = Path(name)
    if not path.is_file():
        path = CONFIG_DIR / (name if name.endswith(".yaml") else name + ".yaml")
    if not path.is_file():
        choices = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")
                         if p.stem != "general")
        sys.exit("no such config: %s\navailable: %s" % (name, ", ".join(choices)))
    with open(general) as f:
        cfg = yaml.safe_load(f) or {}
    with open(path) as f:
        _merge(cfg, yaml.safe_load(f) or {})
    return cfg


def main():
    ap = argparse.ArgumentParser(
        description="Robust watermark training (Section 3 of the paper).")
    ap.add_argument("--config", required=True,
                    help="backbone config name, e.g. sdv15 / fluxklein / pixart")
    ap.add_argument("--cp_data", required=True,
                    help="copyright set: a class directory of the HF dataset, "
                         "containing image/ and prompt.csv")
    ap.add_argument("--reg_data", required=True,
                    help="regular set built by ../Reg_DS/build_reg_dataset.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--current_trigger_word", default=None,
                    help="overrides trigger.current_trigger_word")
    ap.add_argument("--pretrained_model_name_or_path", default=None)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--learning_rate", default=None)
    ap.add_argument("--train_batch_size", type=int, default=None)
    ap.add_argument("--max_train_steps", type=int, default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--optimizer", default=None, choices=("adamw", "muon"),
                    help="overrides optimizer.name; only PixArt implements muon")
    ap.add_argument("--roma", action="store_true",
                    help="force the RoMA ablation on")
    ap.add_argument("--dry_run", action="store_true",
                    help="print the trainer command and exit")
    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    cfg = load_config(args.config)
    tr = cfg["train"]
    cli = cfg["cli"]

    for key in ("resolution", "learning_rate", "train_batch_size",
                "max_train_steps", "rank"):
        override = getattr(args, key)
        if override is not None:
            tr[key] = override

    script = BACKENDS / cfg["trainer"]
    if not script.is_file():
        sys.exit("trainer not found: %s" % script)

    trigger = cfg["trigger"]
    current = args.current_trigger_word or trigger["current_trigger_word"]
    loss = cfg["loss"]
    ck = cfg["checkpointing"]

    cmd = [sys.executable, str(script),
           "--pretrained_model_name_or_path",
           args.pretrained_model_name_or_path
           or cfg["pretrained_model_name_or_path"],
           "--cp_dataset", args.cp_data,
           "--org_image", args.reg_data,
           "--output_dir", args.output_dir,
           "--resolution", str(tr["resolution"]),
           "--train_batch_size", str(tr["train_batch_size"]),
           "--learning_rate", str(tr["learning_rate"]),
           cli["steps_flag"], str(tr["max_train_steps"]),
           "--rank", str(tr["rank"]),
           "--lora_alpha", str(cfg["lora"]["lora_alpha"]),
           "--checkpointing_steps", str(ck["checkpointing_steps"]),
           "--original_trigger_word", trigger["original_trigger_word"],
           "--current_trigger_word", current,
           cli["reg_loss_flag"], str(loss["reg_loss_weight"]),
           "--cp_ref_weight", str(loss["cp_ref_weight"]),
           "--lambda_watermarkdm", str(loss["lambda_watermarkdm"])]

    if ck.get("auto_resume_latest", True):
        cmd.append("--auto_resume_latest")

    # Only the repo-native trainers take these two.
    if cfg.get("native_trainer"):
        cmd += ["--checkpoints_total_limit",
                str(ck["checkpoints_total_limit"])]
        if ck.get("gradient_checkpointing", True):
            cmd.append("--gradient_checkpointing")

    roma = cfg.get("roma", {})
    if args.roma or roma.get("enabled"):
        cmd += ["--roma", "--roma_alpha", str(roma.get("alpha", 0.4)),
                "--roma_r", str(roma.get("r", 0.05))]

    # Appendix A.1: FLUX.2 Klein, aMUSEd and LlamaGen use plain LoRA rather
    # than T-LoRA.  Only the Stable Diffusion trainer can be switched at the
    # command line; the others are one or the other by construction, so the
    # scheme is recorded here and asserted rather than passed as a flag.
    scheme = cfg["lora"].get("scheme", "tlora")
    # jit / muse / parti have no T-LoRA path at all, so they need no flag;
    # Stable Diffusion and FLUX implement both and are switched here.
    if scheme == "lora" and cfg.get("ord_lora_flag", True):
        cmd.append("--ord_lora")

    opt_cfg = cfg.get("optimizer") or {"name": "adamw"}
    optimizer = args.optimizer or opt_cfg.get("name", "adamw")
    if optimizer != "adamw":
        muon = opt_cfg.get("muon")
        if not muon:
            sys.exit("%s's trainer has no %s optimizer; only PixArt "
                     "(--config pixart) implements muon."
                     % (cfg["display_name"], optimizer))
        # Muon takes over train.learning_rate; the AdamW half of the split
        # keeps its own, which is the rate an all-AdamW run would have used.
        cmd[cmd.index("--learning_rate") + 1] = str(muon["learning_rate"])
        cmd += ["--optimizer", optimizer,
                "--muon_momentum", str(muon["momentum"]),
                "--muon_ns_steps", str(muon["ns_steps"]),
                "--muon_adamw_lr", str(muon["adamw_learning_rate"])]

    if "lr_scheduler" in tr:
        cmd += ["--lr_scheduler", str(tr["lr_scheduler"]),
                "--lr_warmup_steps", str(tr["lr_warmup_steps"])]

    cmd += passthrough

    print(" ".join(repr(c) if " " in c else c for c in cmd), flush=True)
    if args.dry_run:
        return 0
    os.makedirs(args.output_dir, exist_ok=True)
    return subprocess.call(cmd, cwd=str(BACKENDS))


if __name__ == "__main__":
    raise SystemExit(main())
