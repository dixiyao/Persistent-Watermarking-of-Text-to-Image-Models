"""Deterministic UNet attention selection and initialization."""

import random

import torch.nn as nn


def _attention_region(module_name):
    if module_name.startswith("down_blocks."):
        return "down"
    if module_name.startswith("mid_block."):
        return "mid"
    if module_name.startswith("up_blocks."):
        return "up"
    return None


def select_attention_layers(unet, attention_kind, seed):
    """Select one attention layer from each down/mid/up UNet path."""
    target_suffix = ".attn2" if attention_kind == "cross_kv" else ".attn1"
    candidates = {"down": [], "mid": [], "up": []}
    for name, module in unet.named_modules():
        region = _attention_region(name)
        if region is None or not name.endswith(target_suffix):
            continue
        required = (
            ("to_k", "to_v")
            if attention_kind == "cross_kv"
            else ("to_q", "to_k", "to_v", "to_out")
        )
        if all(hasattr(module, attribute) for attribute in required):
            candidates[region].append(name)
    missing = [region for region, names in candidates.items() if not names]
    if missing:
        raise RuntimeError(
            f"UNet has no {attention_kind} candidates in regions: {', '.join(missing)}"
        )
    rng = random.Random(int(seed))
    return {
        region: rng.choice(sorted(names))
        for region, names in candidates.items()
    }


def _reset_linear_kaiming(module):
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear projection, got {type(module).__name__}")
    nn.init.kaiming_uniform_(module.weight, a=5 ** 0.5)
    if module.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
        bound = 1.0 / fan_in ** 0.5 if fan_in > 0 else 0.0
        nn.init.uniform_(module.bias, -bound, bound)


def initialize_selected_attention(unet, selected, attention_kind):
    projections = (
        ("to_k", "to_v")
        if attention_kind == "cross_kv"
        else ("to_q", "to_k", "to_v", "to_out.0")
    )
    for region, module_name in selected.items():
        attention = unet.get_submodule(module_name)
        for projection_name in projections:
            _reset_linear_kaiming(attention.get_submodule(projection_name))
        print(
            f"Kaiming-reset {attention_kind} layer for {region} path: {module_name} "
            f"({', '.join(projections)})",
            flush=True,
        )
