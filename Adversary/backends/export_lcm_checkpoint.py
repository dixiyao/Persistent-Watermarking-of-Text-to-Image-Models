#!/usr/bin/env python3
"""Export an intermediate LCM training checkpoint into the final/ layout.

train_lcm_distill_sd14.py saves training checkpoints as raw state dicts
(checkpoint-N/{online_unet.pt,target_unet.pt,optimizer.pt,trainer_state.json}),
while evaluate.py --lcm_checkpoint expects the final/ layout:

    <out>/unet/            Diffusers UNet2DConditionModel (with time_cond_proj_dim)
    <out>/unet.pt          the same weights as a state dict
    <out>/backbone_info.json

This script builds that layout from the EMA *target* UNet of a checkpoint (the
weights the paper evaluates), so an unfinished distillation can be evaluated at
whatever step it has reached.  No GPU is needed.

    python export_lcm_checkpoint.py \
        --checkpoint <run>/checkpoint-11250 \
        --out        <run>/exported-11250
"""
import argparse
import json
import os
import zipfile

import torch
from diffusers import UNet2DConditionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="checkpoint-N directory")
    ap.add_argument("--out", required=True, help="destination directory (final/ layout)")
    ap.add_argument(
        "--student_model",
        default=None,
        help="Student base model for the UNet config; default: student_model recorded in trainer_state.json",
    )
    ap.add_argument(
        "--weights",
        choices=("target", "online"),
        default="target",
        help="target = EMA UNet (what evaluation uses); online = the trained UNet itself",
    )
    args = ap.parse_args()

    state_path = os.path.join(args.checkpoint, "trainer_state.json")
    weights_path = os.path.join(args.checkpoint, f"{args.weights}_unet.pt")
    for path in (state_path, weights_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    with zipfile.ZipFile(weights_path) as archive:
        bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"{weights_path} is truncated/corrupt (member {bad}); pick an older checkpoint")

    metadata = json.load(open(state_path))
    student_model = args.student_model or metadata.get("student_model")
    if not student_model:
        raise ValueError("student model unknown: pass --student_model")
    cond_dim = int(metadata.get("time_cond_proj_dim", 256))
    step = int(metadata.get("global_step", 0))

    print(f"Student config from {student_model}, time_cond_proj_dim={cond_dim}", flush=True)
    config = UNet2DConditionModel.load_config(student_model, subfolder="unet")
    unet = UNet2DConditionModel.from_config(config, time_cond_proj_dim=cond_dim)
    state = torch.load(weights_path, map_location="cpu")
    unet.load_state_dict(state, strict=True)  # raises on any architecture mismatch

    os.makedirs(args.out, exist_ok=True)
    unet.save_pretrained(os.path.join(args.out, "unet"))
    torch.save(state, os.path.join(args.out, "unet.pt"))
    info = dict(metadata)
    info["exported_from_checkpoint"] = os.path.abspath(args.checkpoint)
    info["exported_weights"] = f"{args.weights}_unet"
    info["exported_at_step"] = step
    info["partial_training"] = step < int(metadata.get("max_train_steps", step) or step)
    with open(os.path.join(args.out, "backbone_info.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    print(f"Exported step {step} {args.weights} UNet -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
