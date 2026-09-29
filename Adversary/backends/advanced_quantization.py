"""Post-training diffusion quantization attacks used by evaluate.py.

The Q-Diffusion path uses full-denoising-trajectory calibration (therefore
sampling changing timestep distributions) and channelwise activation ranges,
which also separates the bimodal skip/shortcut channels.  The SVDQuant path
decomposes each eligible weight into an FP16 low-rank branch plus a packed
W4A4 residual branch.  These are portable PyTorch implementations; unlike the
authors' fused kernels they prioritize attack fidelity over speedup.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization_utils import (
    PackedQuantMixin,
    QuantizationConfig,
    QuantizationSummary,
    quantize_dequant_tensor,
    set_activation_calibration,
)


def _migration_scale(activation_max, weight_max, alpha=0.5):
    if activation_max is None:
        return torch.ones_like(weight_max, dtype=torch.float32)
    activation_max = activation_max.float().reshape(-1)
    if activation_max.numel() != weight_max.numel():
        return torch.ones_like(weight_max, dtype=torch.float32)
    scale = activation_max.clamp_min(1e-5).pow(alpha)
    scale = scale / weight_max.float().clamp_min(1e-5).pow(1.0 - alpha)
    return scale.clamp(1e-4, 1e4)


class SVDQuantLinear(nn.Module, PackedQuantMixin):
    def __init__(
        self,
        module: nn.Linear,
        rank: int,
        config: QuantizationConfig,
        activation_max=None,
    ):
        super().__init__()
        weight = module.weight.detach().float().cpu()
        weight_max = weight.abs().amax(dim=0)
        input_scale = _migration_scale(activation_max, weight_max)
        weight = weight * input_scale.unsqueeze(0)
        effective_rank = max(1, min(int(rank), min(weight.shape)))
        u, s, v = torch.svd_lowrank(weight, q=effective_rank, niter=2)
        low_left = u * s.unsqueeze(0)
        low_right = v.transpose(0, 1)
        residual = weight - low_left @ low_right

        self.in_features = module.in_features
        self.out_features = module.out_features
        self.weight_bits = int(config.weight_bits)
        self.activation_bits = int(config.activation_bits)
        self.quantize_activations = True
        self.per_channel_weights = bool(config.per_channel_weights)
        self.symmetric = bool(config.symmetric)
        self._store_quantized_weight(residual)
        self.register_buffer("input_scale", input_scale.float())
        self.register_buffer("low_rank_left", low_left.to(torch.float16))
        self.register_buffer("low_rank_right", low_right.to(torch.float16))
        self.register_buffer(
            "bias",
            None if module.bias is None else module.bias.detach().cpu().clone(),
        )

    def _apply(self, fn):
        saved_input_scale = self.input_scale.detach().float().clone()
        self._keep_scale_fp32(fn)
        self.input_scale = saved_input_scale.to(self.weight_scale.device)
        self.low_rank_left = self.low_rank_left.to(dtype=torch.float16)
        self.low_rank_right = self.low_rank_right.to(dtype=torch.float16)
        return self

    def forward(self, input):
        migrated_input = input / self.input_scale.to(input.device, input.dtype)
        q_input = quantize_dequant_tensor(
            migrated_input,
            self.activation_bits,
            per_channel=True,
            channel_dim=-1,
            symmetric=self.symmetric,
        )
        residual = self.dequantize_weight(dtype=input.dtype, device=input.device)
        bias = None if self.bias is None else self.bias.to(input.device, input.dtype)
        main = F.linear(q_input, residual, bias)
        right = self.low_rank_right.to(input.device, input.dtype)
        left = self.low_rank_left.to(input.device, input.dtype)
        return main + F.linear(F.linear(migrated_input, right), left)


class SVDQuantConv2d(nn.Module, PackedQuantMixin):
    def __init__(
        self,
        module: nn.Conv2d,
        rank: int,
        config: QuantizationConfig,
        activation_max=None,
    ):
        super().__init__()
        weight = module.weight.detach().float().cpu()
        weight_max = weight.abs().amax(dim=(0, 2, 3))
        input_scale = _migration_scale(activation_max, weight_max)
        weight = weight * input_scale.reshape(1, -1, 1, 1)
        matrix = weight.flatten(1)
        effective_rank = max(1, min(int(rank), min(matrix.shape)))
        u, s, v = torch.svd_lowrank(matrix, q=effective_rank, niter=2)
        low_left = (u * s.unsqueeze(0)).reshape(module.out_channels, effective_rank, 1, 1)
        low_right = v.transpose(0, 1).reshape(
            effective_rank, module.in_channels, *module.kernel_size
        )
        residual = weight - (u * s.unsqueeze(0) @ v.transpose(0, 1)).view_as(weight)

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
        self.quantize_activations = True
        self.per_channel_weights = bool(config.per_channel_weights)
        self.symmetric = bool(config.symmetric)
        self._store_quantized_weight(residual)
        self.register_buffer("input_scale", input_scale.float())
        self.register_buffer("low_rank_left", low_left.to(torch.float16))
        self.register_buffer("low_rank_right", low_right.to(torch.float16))
        self.register_buffer(
            "bias",
            None if module.bias is None else module.bias.detach().cpu().clone(),
        )

    def _apply(self, fn):
        saved_input_scale = self.input_scale.detach().float().clone()
        self._keep_scale_fp32(fn)
        self.input_scale = saved_input_scale.to(self.weight_scale.device)
        self.low_rank_left = self.low_rank_left.to(dtype=torch.float16)
        self.low_rank_right = self.low_rank_right.to(dtype=torch.float16)
        return self

    def forward(self, input):
        migrated_input = input / self.input_scale.to(input.device, input.dtype).reshape(
            1, -1, 1, 1
        )
        q_input = quantize_dequant_tensor(
            migrated_input,
            self.activation_bits,
            per_channel=True,
            channel_dim=1,
            symmetric=self.symmetric,
        )
        residual = self.dequantize_weight(dtype=input.dtype, device=input.device)
        bias = None if self.bias is None else self.bias.to(input.device, input.dtype)
        main = F.conv2d(
            q_input, residual, bias, self.stride, self.padding, self.dilation, self.groups
        )
        right = self.low_rank_right.to(input.device, input.dtype)
        left = self.low_rank_left.to(input.device, input.dtype)
        low = F.conv2d(
            migrated_input, right, None, self.stride, self.padding, self.dilation, 1
        )
        low = F.conv2d(low, left, None, 1, 0, 1, 1)
        return main + low


def apply_svdquant(
    module: nn.Module,
    *,
    rank: int = 32,
    prefix: str = "",
    config: Optional[QuantizationConfig] = None,
    activation_maxima: Optional[Dict[str, torch.Tensor]] = None,
) -> QuantizationSummary:
    """Replace UNet Linear/Conv2d weights with FP16-low-rank + W4A4 residuals."""
    config = config or QuantizationConfig(
        weight_bits=4,
        activation_bits=4,
        quantize_activations=True,
        protect_temporal=False,
        activation_per_channel=True,
    )
    summary = QuantizationSummary()
    activation_maxima = activation_maxima or {}
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and min(child.weight.shape) > 1:
            setattr(
                module,
                child_name,
                SVDQuantLinear(
                    child,
                    rank,
                    config,
                    activation_max=activation_maxima.get(full_name),
                ),
            )
            summary.quantized_linear += 1
        elif (
            isinstance(child, nn.Conv2d)
            and child.groups == 1
            and min(child.weight.flatten(1).shape) > 1
        ):
            setattr(
                module,
                child_name,
                SVDQuantConv2d(
                    child,
                    rank,
                    config,
                    activation_max=activation_maxima.get(full_name),
                ),
            )
            summary.quantized_conv2d += 1
        else:
            nested = apply_svdquant(
                child,
                rank=rank,
                prefix=full_name,
                config=config,
                activation_maxima=activation_maxima,
            )
            summary.quantized_linear += nested.quantized_linear
            summary.quantized_conv2d += nested.quantized_conv2d
    return summary


@torch.no_grad()
def collect_activation_channel_maxima(
    pipeline,
    *,
    prompts,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    device: torch.device,
):
    """Calibrate per-input-channel ranges on complete diffusion trajectories."""
    maxima: Dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name, channel_dim):
        def hook(_module, inputs):
            value = inputs[0].detach().float()
            dim = channel_dim % value.ndim
            observed = value.abs().amax(
                dim=tuple(index for index in range(value.ndim) if index != dim)
            ).cpu()
            previous = maxima.get(name)
            maxima[name] = observed if previous is None else torch.maximum(previous, observed)
        return hook

    for name, module in pipeline.unet.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(make_hook(name, -1)))
        elif isinstance(module, nn.Conv2d) and module.groups == 1:
            handles.append(module.register_forward_pre_hook(make_hook(name, 1)))

    generator_device = device if device.type == "cuda" else torch.device("cpu")
    try:
        for index, prompt in enumerate(prompts):
            generator = torch.Generator(device=generator_device).manual_seed(seed + index)
            pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
                output_type="latent",
            )
    finally:
        for handle in handles:
            handle.remove()
    print(
        f"SVDQuant calibrated activation outliers for {len(maxima)} layers over "
        f"{len(prompts)} full denoising trajectories.",
        flush=True,
    )
    return maxima


@torch.no_grad()
def calibrate_qdiffusion_pipeline(
    pipeline,
    *,
    prompts,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    device: torch.device,
):
    """Collect fixed A8 ranges across complete diffusion trajectories."""
    count = set_activation_calibration(pipeline.unet, True)
    if count == 0:
        raise RuntimeError("Q-Diffusion calibration found no quantized UNet layers")
    pipeline.set_progress_bar_config(disable=True)
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    for index, prompt in enumerate(prompts):
        generator = torch.Generator(device=generator_device).manual_seed(seed + index)
        pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            output_type="latent",
        )
    set_activation_calibration(pipeline.unet, False)
    print(
        f"Q-Diffusion calibrated {count} W8A8 layers over "
        f"{len(prompts)} full denoising trajectories.",
        flush=True,
    )
