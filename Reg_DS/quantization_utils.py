"""Packed low-bit quantization helpers for SDXL experiments.

Linear/Conv2d weights are replaced with integer quantized modules. Forward
passes dequantize on-the-fly to the active compute dtype (no custom CUDA
kernel required).

weight_bits=16 runs the full quantize-dequantize algorithm with int16 storage
(not a passthrough) — reconstruction error should be negligible so output
nearly matches the unquantized baseline, making it a useful correctness check.
Same logic applies to activation_bits=16 for activation quantization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TEMPORAL_NAME_MARKERS = (
    "time",
    "timestep",
    "temb",
    "add_embedding",
    "time_embedding",
    "time_proj",
)


@dataclass
class QuantizationConfig:
    weight_bits: int = 4
    activation_bits: int = 8
    quantize_activations: bool = False  # naive per-tensor dynamic quant breaks SDXL attention
    protect_temporal: bool = True
    quantize_conv: bool = True
    quantize_linear: bool = True
    per_channel_weights: bool = True
    activation_per_channel: bool = False
    symmetric: bool = True


@dataclass
class QuantizationSummary:
    quantized_linear: int = 0
    quantized_conv2d: int = 0
    skipped_temporal: int = 0
    skipped_patterns: int = 0

    @property
    def total_quantized(self):
        return self.quantized_linear + self.quantized_conv2d


def _name_matches_any(name: str, patterns: Iterable[str]) -> bool:
    lowered = name.lower()
    return any(pattern.lower() in lowered for pattern in patterns if pattern)


def is_temporal_module_name(name: str) -> bool:
    return _name_matches_any(name, TEMPORAL_NAME_MARKERS)


def _quant_range(num_bits: int, symmetric: bool = True) -> Tuple[int, int]:
    if int(num_bits) <= 0:
        raise ValueError(f"num_bits must be positive, got {num_bits}")
    if symmetric:
        return -(2 ** (int(num_bits) - 1)), (2 ** (int(num_bits) - 1)) - 1
    return 0, (2 ** int(num_bits)) - 1


def _reduce_dims(x: torch.Tensor, channel_dim: int):
    channel_dim = channel_dim % x.ndim
    return tuple(dim for dim in range(x.ndim) if dim != channel_dim)


def _scale_for_tensor(
    x: torch.Tensor,
    num_bits: int,
    *,
    per_channel: bool,
    channel_dim: int,
    symmetric: bool,
    keepdim: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    qmin, qmax = _quant_range(num_bits, symmetric=symmetric)
    if symmetric:
        denom = max(abs(qmin + 1), abs(qmax))
        if per_channel and x.ndim > 1:
            max_abs = x.detach().float().abs().amax(
                dim=_reduce_dims(x, channel_dim),
                keepdim=keepdim,
            )
        else:
            max_abs = x.detach().float().abs().amax()
        return (max_abs / max(float(denom), 1.0)).clamp_min(eps)
    raise NotImplementedError("Only symmetric quantization is implemented.")


def quantize_tensor_to_int(
    x: torch.Tensor,
    num_bits: int,
    *,
    per_channel: bool = False,
    channel_dim: int = 0,
    symmetric: bool = True,
):
    """Quantize tensor to integer. Supports any num_bits in [1, 16].

    Uses int8 storage for num_bits <= 8, int16 for 9-16. This lets you run
    16-bit through the same algorithm as 4/8-bit to verify correctness.
    """
    qmin, qmax = _quant_range(num_bits, symmetric=symmetric)
    scale = _scale_for_tensor(
        x,
        num_bits,
        per_channel=per_channel,
        channel_dim=channel_dim,
        symmetric=symmetric,
        keepdim=True,
    )
    int_dtype = torch.int8 if int(num_bits) <= 8 else torch.int16
    q = torch.round(x.detach().float() / scale).clamp(qmin, qmax).to(int_dtype)
    return q, scale.detach().float()


def quantize_dequant_tensor(
    x: torch.Tensor,
    num_bits: int,
    *,
    per_channel: bool = False,
    channel_dim: int = 0,
    symmetric: bool = True,
    ste: bool = False,
) -> torch.Tensor:
    """Quantize-dequantize a tensor. Runs the full algorithm for any num_bits."""
    if num_bits is None:
        return x
    qmin, qmax = _quant_range(num_bits, symmetric=symmetric)
    scale = _scale_for_tensor(
        x,
        num_bits,
        per_channel=per_channel,
        channel_dim=channel_dim,
        symmetric=symmetric,
        keepdim=True,
    ).to(device=x.device)
    dequant = (torch.round(x.float() / scale).clamp(qmin, qmax) * scale).to(dtype=x.dtype)
    if ste and torch.is_grad_enabled():
        return x + (dequant - x).detach()
    return dequant


def _quantize_dequant_with_absmax(x, absmax, num_bits, symmetric=True):
    qmin, qmax = _quant_range(num_bits, symmetric=symmetric)
    denom = max(abs(qmin + 1), abs(qmax), 1)
    scale = (absmax.float() / float(denom)).clamp_min(1e-8).to(x.device)
    return (torch.round(x.float() / scale).clamp(qmin, qmax) * scale).to(x.dtype)


class CalibratedActivationMixin:
    def _init_activation_quantizer(self, config):
        self.activation_per_channel = bool(config.activation_per_channel)
        self.activation_calibrating = False
        self.register_buffer("activation_absmax", torch.empty(0, dtype=torch.float32))

    def begin_activation_calibration(self):
        self.activation_calibrating = True
        self.activation_absmax = torch.empty(0, dtype=torch.float32, device=self.weight_scale.device)

    def finish_activation_calibration(self):
        self.activation_calibrating = False

    def _activation_channel_dim(self):
        raise NotImplementedError

    def _quantize_activation(self, input):
        channel_dim = self._activation_channel_dim() % input.ndim
        if self.activation_calibrating:
            if self.activation_per_channel and input.ndim > 1:
                observed = input.detach().float().abs().amax(
                    dim=_reduce_dims(input, channel_dim), keepdim=True
                )
            else:
                observed = input.detach().float().abs().amax().reshape(
                    *([1] * input.ndim)
                )
            if self.activation_absmax.numel() == 0:
                self.activation_absmax = observed
            else:
                self.activation_absmax = torch.maximum(
                    self.activation_absmax.to(observed.device), observed
                )
            return input
        if not self.quantize_activations:
            return input
        if self.activation_absmax.numel() > 0:
            return _quantize_dequant_with_absmax(
                input,
                self.activation_absmax,
                self.activation_bits,
                symmetric=self.symmetric,
            )
        return quantize_dequant_tensor(
            input,
            self.activation_bits,
            per_channel=self.activation_per_channel,
            channel_dim=channel_dim,
            symmetric=self.symmetric,
            ste=torch.is_grad_enabled(),
        )


def pack_signed_int4(q: torch.Tensor) -> torch.Tensor:
    flat = q.detach().to(torch.int8).reshape(-1)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8, device=flat.device)])
    unsigned = torch.bitwise_and(flat, 0x0F).to(torch.uint8)
    lo = unsigned[0::2]
    hi = unsigned[1::2] << 4
    return torch.bitwise_or(lo, hi).contiguous()


def unpack_signed_int4(packed: torch.Tensor, numel: int, device=None) -> torch.Tensor:
    packed = packed.to(device=device) if device is not None else packed
    lo = torch.bitwise_and(packed, 0x0F)
    hi = torch.bitwise_and(packed >> 4, 0x0F)
    vals = torch.empty(packed.numel() * 2, dtype=torch.int8, device=packed.device)
    vals[0::2] = lo.to(torch.int8)
    vals[1::2] = hi.to(torch.int8)
    vals = vals[:numel]
    vals = torch.where(vals >= 8, vals - 16, vals)
    return vals.to(torch.int8)


class PackedQuantMixin:
    weight_bits: int
    symmetric: bool
    per_channel_weights: bool

    def _store_quantized_weight(self, weight: torch.Tensor):
        q, scale = quantize_tensor_to_int(
            weight,
            self.weight_bits,
            per_channel=self.per_channel_weights,
            channel_dim=0,
            symmetric=self.symmetric,
        )
        q = q.cpu()
        scale = scale.reshape(scale.shape[0], *([1] * (weight.ndim - 1))).cpu()
        if int(self.weight_bits) == 4:
            packed = pack_signed_int4(q)
            self.register_buffer("qweight_packed", packed)
            self.register_buffer("qweight_int", torch.empty(0, dtype=torch.int8), persistent=False)
        else:
            # int8 for <= 8 bits, int16 for 9-16 bits (int8 can't hold ±32767)
            self.register_buffer("qweight_packed", torch.empty(0, dtype=torch.uint8), persistent=False)
            self.register_buffer("qweight_int", q.contiguous())
        self.register_buffer("weight_scale", scale.float())
        self.register_buffer("weight_shape", torch.tensor(list(weight.shape), dtype=torch.long))
        self.register_buffer("weight_numel", torch.tensor(int(weight.numel()), dtype=torch.long))

    def dequantize_weight(self, dtype=None, device=None):
        shape = tuple(int(v) for v in self.weight_shape.tolist())
        numel = int(self.weight_numel.item())
        if int(self.weight_bits) == 4:
            q = unpack_signed_int4(self.qweight_packed, numel, device=device).reshape(shape)
        else:
            q = self.qweight_int.to(device=device).reshape(shape)
        # Always use float32 for scale: for high bit-widths (e.g. 16-bit) the scale is
        # very small and becomes subnormal or zero in float16, corrupting reconstruction.
        scale = self.weight_scale.float().to(device=q.device)
        weight = q.float() * scale
        if dtype is not None:
            weight = weight.to(dtype=dtype)
        return weight

    def _keep_scale_fp32(self, fn):
        """Call nn.Module._apply(fn) but restore weight_scale to its original float32 value.

        pipeline.to(dtype=float16) converts all float buffers. weight_scale must stay
        float32 because for high bit-widths the scales are tiny (e.g. max_abs/32767 ≈ 1e-6)
        and underflow to zero in float16, making every reconstructed weight zero.
        """
        saved = (
            self._buffers["weight_scale"].detach().float().clone()
            if "weight_scale" in self._buffers and self._buffers["weight_scale"] is not None
            else None
        )
        nn.Module._apply(self, fn)
        if saved is not None:
            self._buffers["weight_scale"] = saved.to(device=self._buffers["weight_scale"].device)
        return self

    def mean_weight_scale(self) -> float:
        return float(self.weight_scale.float().mean().item())


class QuantizedLinear(nn.Module, PackedQuantMixin, CalibratedActivationMixin):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        *,
        config: QuantizationConfig,
    ):
        super().__init__()
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.weight_bits = int(config.weight_bits)
        self.activation_bits = int(config.activation_bits)
        self.quantize_activations = bool(config.quantize_activations)
        self.per_channel_weights = bool(config.per_channel_weights)
        self.symmetric = bool(config.symmetric)
        self._store_quantized_weight(weight)
        self._init_activation_quantizer(config)
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().cpu().clone())

    @classmethod
    def from_linear(cls, module: nn.Linear, config: QuantizationConfig):
        quantized = cls(
            module.weight.detach(),
            module.bias.detach() if module.bias is not None else None,
            config=config,
        )
        quantized.training = module.training
        return quantized

    def _apply(self, fn):
        return self._keep_scale_fp32(fn)

    def forward(self, input):
        q_input = self._quantize_activation(input)
        weight = self.dequantize_weight(dtype=input.dtype, device=input.device)
        bias = None if self.bias is None else self.bias.to(device=input.device, dtype=input.dtype)
        return F.linear(q_input, weight, bias)

    def _activation_channel_dim(self):
        return -1


class QuantizedConv2d(nn.Module, PackedQuantMixin, CalibratedActivationMixin):
    def __init__(
        self,
        module: nn.Conv2d,
        *,
        config: QuantizationConfig,
    ):
        super().__init__()
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.padding_mode = module.padding_mode
        self.weight_bits = int(config.weight_bits)
        self.activation_bits = int(config.activation_bits)
        self.quantize_activations = bool(config.quantize_activations)
        self.per_channel_weights = bool(config.per_channel_weights)
        self.symmetric = bool(config.symmetric)
        self._store_quantized_weight(module.weight.detach())
        self._init_activation_quantizer(config)
        if module.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", module.bias.detach().cpu().clone())

    @classmethod
    def from_conv2d(cls, module: nn.Conv2d, config: QuantizationConfig):
        quantized = cls(module, config=config)
        quantized.training = module.training
        return quantized

    def _apply(self, fn):
        return self._keep_scale_fp32(fn)

    def forward(self, input):
        q_input = self._quantize_activation(input)
        weight = self.dequantize_weight(dtype=input.dtype, device=input.device)
        bias = None if self.bias is None else self.bias.to(device=input.device, dtype=input.dtype)
        if self.padding_mode != "zeros":
            q_input = F.pad(q_input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
        return F.conv2d(q_input, weight, bias, self.stride, padding, self.dilation, self.groups)

    def _activation_channel_dim(self):
        return 1

    @property
    def _reversed_padding_repeated_twice(self):
        if isinstance(self.padding, str):
            raise ValueError("String padding modes are not supported in QuantizedConv2d")
        if isinstance(self.padding, tuple):
            return tuple(x for p in reversed(self.padding) for x in (p, p))
        return (self.padding, self.padding, self.padding, self.padding)


def _copy_summary_into(dst: QuantizationSummary, src: QuantizationSummary):
    dst.quantized_linear += src.quantized_linear
    dst.quantized_conv2d += src.quantized_conv2d
    dst.skipped_temporal += src.skipped_temporal
    dst.skipped_patterns += src.skipped_patterns


def apply_packed_quantization(
    module: nn.Module,
    config: Optional[QuantizationConfig] = None,
    *,
    prefix: str = "",
    ignore_name_patterns: Optional[List[str]] = None,
) -> QuantizationSummary:
    """Replace Linear/Conv2d modules with packed low-bit quantized modules.

    Supports any weight_bits in [1, 16]. At weight_bits=16 the same quantize-
    dequantize algorithm runs with int16 storage, so the output should be
    nearly identical to fp16 — useful for verifying the algorithm is bug-free.
    """
    if config is None:
        config = QuantizationConfig()
    ignore_name_patterns = ignore_name_patterns or []
    summary = QuantizationSummary()

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if _name_matches_any(full_name, ignore_name_patterns):
            summary.skipped_patterns += 1
            continue

        if config.protect_temporal and is_temporal_module_name(full_name):
            summary.skipped_temporal += 1
            continue

        if config.quantize_linear and isinstance(child, nn.Linear):
            setattr(module, child_name, QuantizedLinear.from_linear(child, config))
            summary.quantized_linear += 1
            continue

        if config.quantize_conv and isinstance(child, nn.Conv2d):
            setattr(module, child_name, QuantizedConv2d.from_conv2d(child, config))
            summary.quantized_conv2d += 1
            continue

        child_summary = apply_packed_quantization(
            child,
            config,
            prefix=full_name,
            ignore_name_patterns=ignore_name_patterns,
        )
        _copy_summary_into(summary, child_summary)

    return summary


def apply_tfmqdm_quantization(module: nn.Module, config=None, *, prefix: str = "", **kwargs) -> QuantizationSummary:
    """Backward-compatible alias — extra kwargs (lora_*) are silently ignored."""
    return apply_packed_quantization(module, config, prefix=prefix)


def set_activation_calibration(module: nn.Module, enabled: bool):
    """Start or finish fixed-range activation calibration on quantized layers."""
    count = 0
    for child in module.modules():
        if isinstance(child, (QuantizedLinear, QuantizedConv2d)):
            if enabled:
                child.begin_activation_calibration()
            else:
                child.finish_activation_calibration()
            count += 1
    return count


def save_quantized_checkpoint(
    *,
    unet: nn.Module,
    text_encoder: Optional[nn.Module],
    text_encoder_2: Optional[nn.Module],
    output_dir: str,
    quant_config: QuantizationConfig,
    metadata: Optional[dict] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {k: v.detach().cpu() for k, v in unet.state_dict().items()},
        os.path.join(output_dir, "quantized_unet.pt"),
    )
    if text_encoder is not None:
        torch.save(
            {k: v.detach().cpu() for k, v in text_encoder.state_dict().items()},
            os.path.join(output_dir, "quantized_text_encoder.pt"),
        )
    if text_encoder_2 is not None:
        torch.save(
            {k: v.detach().cpu() for k, v in text_encoder_2.state_dict().items()},
            os.path.join(output_dir, "quantized_text_encoder_2.pt"),
        )
    save_quantization_config(quant_config, output_dir)
    if metadata is not None:
        with open(os.path.join(output_dir, "quantized_checkpoint_info.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def save_quantization_config(config: QuantizationConfig, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "quantization_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


def load_quantization_config(path: str) -> QuantizationConfig:
    with open(path, "r", encoding="utf-8") as f:
        return QuantizationConfig(**json.load(f))


def quantization_summary_text(summary: QuantizationSummary) -> str:
    parts = [
        f"quantized linear={summary.quantized_linear}",
        f"conv2d={summary.quantized_conv2d}",
        f"skipped temporal={summary.skipped_temporal}",
        f"skipped by pattern={summary.skipped_patterns}",
    ]
    return ", ".join(parts)
