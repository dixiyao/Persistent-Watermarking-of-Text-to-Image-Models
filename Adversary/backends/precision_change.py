#!/usr/bin/env python3
"""Convert a full Stable Diffusion checkpoint's floating tensors to a new dtype."""

import argparse
import json
import os

import torch


CHECKPOINT_FILES = ("unet.pt", "text_encoder.pt", "text_encoder_2.pt")
DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _convert_state_dict(state_dict, dtype):
    return {
        key: value.to(dtype=dtype) if torch.is_tensor(value) and value.is_floating_point() else value
        for key, value in state_dict.items()
    }


def convert_checkpoint(input_dir, output_dir, target_precision):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input checkpoint directory not found: {input_dir}")
    if not os.path.isfile(os.path.join(input_dir, "unet.pt")):
        raise FileNotFoundError(f"Full checkpoint is missing unet.pt: {input_dir}")

    dtype = DTYPES[target_precision]
    os.makedirs(output_dir, exist_ok=True)
    converted = []
    for filename in CHECKPOINT_FILES:
        source = os.path.join(input_dir, filename)
        if not os.path.isfile(source):
            continue
        state_dict = torch.load(source, map_location="cpu")
        torch.save(
            _convert_state_dict(state_dict, dtype),
            os.path.join(output_dir, filename),
        )
        converted.append(filename)

    metadata = {
        "source_checkpoint": os.path.abspath(input_dir),
        "target_precision": target_precision,
        "converted_files": converted,
    }
    source_metadata = os.path.join(input_dir, "backbone_info.json")
    if os.path.isfile(source_metadata):
        with open(source_metadata, "r", encoding="utf-8") as handle:
            metadata["source_backbone_info"] = json.load(handle)
    with open(os.path.join(output_dir, "backbone_info.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert a full checkpoint to FP16, BF16, or FP32")
    parser.add_argument("--input_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--target_precision",
        choices=tuple(DTYPES),
        default="fp16",
    )
    args = parser.parse_args()

    converted = convert_checkpoint(
        args.input_checkpoint,
        args.output_dir,
        args.target_precision,
    )
    print(
        f"Converted {', '.join(converted)} to {args.target_precision}: {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
