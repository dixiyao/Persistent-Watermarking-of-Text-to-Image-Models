#!/usr/bin/env python3
"""Deterministically interpolate two compatible full Stable Diffusion checkpoints."""

import argparse
import json
import os

import torch

from model_loading import resolve_full_checkpoint_dir


COMPONENT_FILES = ("unet.pt", "text_encoder.pt", "text_encoder_2.pt")


def _load_state(checkpoint_dir, filename):
    path = os.path.join(checkpoint_dir, filename)
    return None if not os.path.isfile(path) else torch.load(path, map_location="cpu")


def _merge_states(state_a, state_b, weight_a, filename):
    if state_a is None and state_b is None:
        return None
    if state_a is None or state_b is None:
        raise FileNotFoundError(
            f"{filename} exists in only one checkpoint; paired merging requires matching components"
        )
    if set(state_a) != set(state_b):
        only_a = sorted(set(state_a) - set(state_b))[:5]
        only_b = sorted(set(state_b) - set(state_a))[:5]
        raise ValueError(
            f"State keys differ for {filename}; only A={only_a}, only B={only_b}"
        )
    merged = {}
    weight_b = 1.0 - weight_a
    for key, value_a in state_a.items():
        value_b = state_b[key]
        if value_a.shape != value_b.shape:
            raise ValueError(
                f"Shape mismatch for {filename}:{key}: {value_a.shape} vs {value_b.shape}"
            )
        if torch.is_floating_point(value_a):
            merged[key] = (
                value_a.float().mul(weight_a).add(value_b.float(), alpha=weight_b)
            ).to(value_a.dtype)
        else:
            if not torch.equal(value_a, value_b):
                raise ValueError(f"Non-floating state differs for {filename}:{key}")
            merged[key] = value_a.clone()
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge two full diffusion checkpoints")
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--checkpoint_b", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weight_a", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 <= args.weight_a <= 1.0:
        parser.error("--weight_a must be in [0, 1]")

    checkpoint_a = resolve_full_checkpoint_dir(args.checkpoint_a)
    checkpoint_b = resolve_full_checkpoint_dir(args.checkpoint_b)
    output_unet = os.path.join(args.output_dir, "unet.pt")
    metadata_path = os.path.join(args.output_dir, "merge_info.json")
    expected_metadata = {
        "checkpoint_a": os.path.abspath(checkpoint_a),
        "checkpoint_b": os.path.abspath(checkpoint_b),
        "weight_a": args.weight_a,
        "weight_b": 1.0 - args.weight_a,
    }
    if os.path.isfile(output_unet) and os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            existing_metadata = json.load(handle)
            if all(
                existing_metadata.get(key) == value
                for key, value in expected_metadata.items()
            ):
                print(f"Reusing existing compatible merged checkpoint: {args.output_dir}")
                return
        raise FileExistsError(
            f"Output already contains a different merge: {args.output_dir}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    saved_components = []
    for filename in COMPONENT_FILES:
        state = _merge_states(
            _load_state(checkpoint_a, filename),
            _load_state(checkpoint_b, filename),
            args.weight_a,
            filename,
        )
        if state is not None:
            torch.save(state, os.path.join(args.output_dir, filename))
            saved_components.append(filename)
    if "unet.pt" not in saved_components:
        raise RuntimeError("Neither resolved checkpoint produced unet.pt")
    expected_metadata["saved_components"] = saved_components
    # Keep idempotency metadata stable after recording the component list.
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(expected_metadata, handle, indent=2)
    print(
        f"Merged {checkpoint_a} ({args.weight_a:.3f}) + {checkpoint_b} "
        f"({1.0 - args.weight_a:.3f}) -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
