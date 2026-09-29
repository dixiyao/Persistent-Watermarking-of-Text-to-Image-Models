#!/usr/bin/env python3
"""Run one of the 23 model modifications of Table 3 against a watermarked
model, then evaluate the result.

The adversary owns a merged watermarked model W_r and tries to remove the
watermark from it.  Every row of Table 3 is one config file:

    python continue.py --adversary_index 12 \
        --merged_model /ckpt/duck/final \
        --cp_data  /data/cp_datasets/animals/duck_toy \
        --reg_data /data/reg/sd15 \
        --output_dir runs/attack12 \
        --current_trigger_word "Willow Heron"

Row 23 merges two independently watermarked models, so it additionally
needs --merged_model_b.

Paths that a config refers to symbolically are supplied on the command
line or through the environment:

    ${COCO_DATASET}   --coco_dataset   / COCO_DATASET
    ${REG_DATASET}    --reg_data
    ${HF_HOME}        HF_HOME
    ${CONTROL_IMAGE}  --control_image
    ${CONTINUE_FINAL} / ${TAYLOR_FINAL}   resolved from --output_dir
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:                                  # pragma: no cover
    sys.exit("continue.py needs PyYAML:  pip install pyyaml")

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
BACKENDS = HERE / "backends"
TRAIN_CONFIGS = HERE.parent / "Training" / "configs"


def load_general():
    with open(CONFIG_DIR / "general.yaml") as f:
        return yaml.safe_load(f) or {}


def load_backbone(name):
    general = TRAIN_CONFIGS / "general.yaml"
    path = TRAIN_CONFIGS / (name if name.endswith(".yaml") else name + ".yaml")
    if not path.is_file():
        sys.exit("no such backbone config: %s" % path)
    cfg = yaml.safe_load(general.read_text()) or {}
    extra = yaml.safe_load(path.read_text()) or {}
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def substitute(value, table):
    if not isinstance(value, str):
        return value
    def repl(m):
        key = m.group(1)
        if key not in table or table[key] is None:
            sys.exit("config needs ${%s}; pass the matching option" % key)
        return str(table[key])
    return re.sub(r"\$\{([A-Z_]+)\}", repl, value)


def flatten(args, table):
    """{"learning_rate": "1e-5"} -> ["--learning_rate", "1e-5"]"""
    out = []
    for k, v in (args or {}).items():
        flag = "--" + k if not k.startswith("-") else k
        if v is True:
            out.append(flag)
        elif v is False or v is None:
            continue
        else:
            out += [flag, str(substitute(str(v), table))]
    return out


def run(cmd, cwd, dry):
    cmd = [str(c) for c in cmd]
    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    if dry:
        return 0
    rc = subprocess.call(cmd, cwd=str(cwd))
    if rc != 0:
        sys.exit("step failed with exit code %d" % rc)
    return rc


def main():
    ap = argparse.ArgumentParser(
        description="Apply one Table 3 modification to a watermarked model.")
    ap.add_argument("--adversary_index", type=int, required=True,
                    choices=range(1, 24), metavar="N",
                    help="row of Table 3 / configs/N.yaml")
    ap.add_argument("--merged_model", required=True,
                    help="the watermarked model W_r under attack")
    ap.add_argument("--merged_model_b", default=None,
                    help="second watermarked model, row 23 only")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--cp_data", default=None)
    ap.add_argument("--reg_data", default=None)
    ap.add_argument("--coco_dataset", default=os.environ.get("COCO_DATASET"))
    ap.add_argument("--control_image", default=None)
    ap.add_argument("--config", default="sdv15",
                    help="backbone config the model was trained with")
    ap.add_argument("--current_trigger_word", default=None)
    ap.add_argument("--base_model", default=None)
    ap.add_argument("--dreambooth_data_root", default=None,
                    help="root of the DreamBooth subject folders, row 6")
    ap.add_argument("--lcm_student_base_model",
                    default="CompVis/stable-diffusion-v1-4",
                    help="architecture the LCM student is built on, row 5")
    ap.add_argument("--optimizer", default=None, choices=("adamw", "muon"),
                    help="overrides optimizer.name in general.yaml; only "
                         "PixArt's continuation trainer implements muon")
    ap.add_argument("--dry_run", action="store_true")
    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    cfg = yaml.safe_load(
        (CONFIG_DIR / ("%d.yaml" % args.adversary_index)).read_text())
    gen = load_general()
    bb = load_backbone(args.config)
    # The model's identity comes from the Training config; how hard the
    # adversary pushes on it comes from this directory.
    adv = dict(gen["runtime"])
    adv.update((gen.get("backbones") or {}).get(args.config) or {})
    trigger = bb["trigger"]
    current = args.current_trigger_word or trigger["current_trigger_word"]
    base_model = args.base_model or bb["pretrained_model_name_or_path"]

    if cfg.get("requires_two_models") and not args.merged_model_b:
        sys.exit(
            "row %d (%s) merges two independently watermarked models.\n"
            "Pass the second one with --merged_model_b." %
            (cfg["index"], cfg["paper"]["modification"]))

    out = Path(args.output_dir).resolve()
    final = out / "final"
    table = {
        "COCO_DATASET": args.coco_dataset,
        "REG_DATASET": args.reg_data,
        "HF_HOME": os.environ.get("HF_HOME"),
        "CONTROL_IMAGE": args.control_image,
        "CONTINUE_FINAL": str(final),
        "TAYLOR_FINAL": str(out / ("round_%02d" % cfg.get("iterative_rounds", 1))
                            / "final"),
        # Row 6 drives its own five stages and lands on the last one.
        "PERSONALIZATION_FINAL": str(out / "stage_05_can" / "final"),
        # Row 5 evaluates the exported LCM UNet, not the raw checkpoint.
        "LCM_EVAL": str(out / "lcm_eval"),
        "LCM_STUDENT": args.lcm_student_base_model,
        "DREAMBOOTH_ROOT": args.dreambooth_data_root,
        "BASE_MODEL": base_model,
        "MERGED_MODEL": args.merged_model,
        "OUT": str(out),
        "TRIGGER": current,
    }

    common = ["--pretrained_model_name_or_path", base_model,
              "--merged_checkpoint", args.merged_model,
              "--output_dir", str(out),
              "--resolution", str(bb["train"]["resolution"]),
              "--train_batch_size", str(adv["train_batch_size"]),
              "--gradient_accumulation_steps",
              str(adv["gradient_accumulation_steps"]),
              "--max_grad_norm", "1",
              "--mixed_precision", adv["mixed_precision"],
              "--gradient_checkpointing",
              "--random_flip",
              "--integrity_inference_steps", "50",
              "--auto_resume_latest",
              "--original_trigger_word", trigger["original_trigger_word"],
              "--current_trigger_word", current]
    opt_cfg = gen.get("optimizer") or {"name": "adamw"}
    optimizer = args.optimizer or opt_cfg.get("name", "adamw")
    if optimizer != "adamw":
        supported = opt_cfg.get("supported_backbones") or []
        if args.config not in supported:
            sys.exit("%s's continuation trainer has no %s optimizer; "
                     "supported: %s" % (args.config, optimizer,
                                        ", ".join(supported or ["none"])))
        muon = opt_cfg["muon"]
        common += ["--optimizer", optimizer,
                   "--muon_momentum", str(muon["momentum"]),
                   "--muon_ns_steps", str(muon["ns_steps"]),
                   "--muon_adamw_lr", str(muon["adamw_learning_rate"])]

    if args.cp_data:
        common += ["--cp_dataset", args.cp_data]
    if args.reg_data:
        common += ["--org_image", args.reg_data]

    # ---- the modification itself ------------------------------------
    if cfg.get("stages"):                       # rows 4 and 6
        # chain "resume": each stage resumes the previous stage's LoRA but
        # still starts from W_r.  chain "merged": each stage takes the
        # previous stage's output *as* the model it modifies.
        chain = cfg.get("chain", "resume")
        previous = None
        for stage in cfg["stages"]:
            stage_dir = out / stage["name"]
            base = common if not cfg.get("standalone_args") else [
                "--pretrained_model_name_or_path", base_model,
                "--merged_checkpoint", args.merged_model,
                "--output_dir", str(out),
                "--mixed_precision", adv["mixed_precision"],
                "--gradient_checkpointing", "--auto_resume_latest",
                "--original_trigger_word", trigger["original_trigger_word"],
                "--current_trigger_word", current]
            cmd = [sys.executable, str(BACKENDS / stage["script"])] + list(base)
            cmd[cmd.index("--output_dir") + 1] = str(stage_dir)
            if previous is not None:
                prev_final = previous / "final"
                if chain == "merged":
                    cmd[cmd.index("--merged_checkpoint") + 1] = str(prev_final)
                elif not any(stage_dir.glob("checkpoint-*")):
                    cmd = [c for c in cmd if c != "--auto_resume_latest"]
                    cmd += ["--resume_from_checkpoint", str(prev_final)]
            cmd += flatten(stage.get("args"), table)
            cmd += list(stage.get("flags") or [])
            run(cmd, BACKENDS, args.dry_run)
            previous = stage_dir
        table["CONTINUE_FINAL"] = str(previous / "final")
        table["PERSONALIZATION_FINAL"] = str(previous / "final")

    elif cfg.get("transform_only"):             # rows 10 and 23
        cmd = [sys.executable, str(BACKENDS / cfg["driver"])]
        if cfg.get("requires_two_models"):
            cmd += ["--checkpoint_a", args.merged_model,
                    "--checkpoint_b", args.merged_model_b,
                    "--output_dir", str(final)]
        else:
            cmd += ["--input_checkpoint", args.merged_model,
                    "--output_dir", str(final)]
        cmd += flatten(cfg.get("args"), table)
        run(cmd, BACKENDS, args.dry_run)

    elif cfg.get("driver"):
        rounds = cfg.get("iterative_rounds", 1)  # row 18
        previous = None
        for r in range(1, rounds + 1):
            cmd = [sys.executable, str(BACKENDS / cfg["driver"])]
            if cfg.get("custom_args"):
                # Row 5: the distillation script shares no flag names with
                # the continuation trainers, so the config carries them all.
                cmd += flatten(cfg.get("args"), table)
                cmd += list(cfg.get("flags") or [])
                run(cmd, BACKENDS, args.dry_run)
                break
            if cfg.get("standalone_args"):
                # These trainers read neither cp_dataset nor org_image.
                cmd += ["--pretrained_model_name_or_path", base_model,
                        "--merged_checkpoint", args.merged_model,
                        "--output_dir", str(out),
                        "--mixed_precision", adv["mixed_precision"],
                        "--gradient_checkpointing", "--auto_resume_latest",
                        "--original_trigger_word",
                        trigger["original_trigger_word"],
                        "--current_trigger_word", current]
            else:
                cmd += list(common)
            if rounds > 1:
                round_dir = out / ("round_%02d" % r)
                cmd[cmd.index("--output_dir") + 1] = str(round_dir)
                if previous is not None and cfg.get("chain") == "merged":
                    cmd[cmd.index("--merged_checkpoint") + 1] = str(
                        previous / "final")
                previous = round_dir
            cmd += flatten(cfg.get("args"), table)
            cmd += list(cfg.get("flags") or [])
            run(cmd, BACKENDS, args.dry_run)
        if cfg.get("post"):
            run([sys.executable, str(BACKENDS / cfg["post"]),
                 "--checkpoint", str(final),
                 "--out", str(out / "lcm_eval")], BACKENDS, args.dry_run)
    else:
        print("row %d applies no training stage: the modification is a "
              "decode-time transform of the model, applied by the evaluation "
              "harness through the flags below." % cfg["index"])

    produced = bool(cfg.get("driver") or cfg.get("stages"))
    print()
    if produced:
        print("row %d done. the attacked model is at %s"
              % (cfg["index"], table["CONTINUE_FINAL"]))
        print("compare it against the watermarked model at %s" % args.merged_model)
    else:
        print("row %d is a decode-time transform (%s): it produces no "
              "checkpoint of its own." % (cfg["index"], cfg["paper"]["algorithm"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
