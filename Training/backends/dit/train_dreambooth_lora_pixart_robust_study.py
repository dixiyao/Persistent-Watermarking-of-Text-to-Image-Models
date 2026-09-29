#!/usr/bin/env python3
"""
Train PixArt-alpha T-LoRA on copyright images (robust study).

Backbone: PixArt-alpha/PixArt-XL-2-1024-MS -- a text-conditioned DiT.
facebook/DiT-XL-2-256 cannot host this study: it is class-conditional on
ImageNet and has no text encoder, so a copyright trigger phrase has no vehicle.
PixArt keeps the DiT architecture while conditioning on T5 text, which is what
the trigger mechanism needs.

Objective (identical in form to the SDXL robust study):
  cp_loss  = cp_noise_mse - cp_ref_weight * cp_reference_mse
  org_loss = org_loss_weights * org_noise_mse
  loss     = cp_loss + org_loss + lambda_watermarkdm * scaling * ||W_r - W_f||_1

PixArt specifics:
  - epsilon prediction on a DDPM linear schedule (1000 steps), so the SDXL
    half-timestep study transfers unchanged.
  - transformer.config.out_channels is 8 (learned sigma); only the first 4
    channels are the epsilon prediction.
  - sample_size == 128 means the MS micro-conditioning (resolution,
    aspect_ratio) must be supplied on every forward.

T-LoRA: orthogonal SVD-init LoRA with a sigma mask (rank scales with timestep),
wrapping the attention projections of the Transformer2DModel directly.
Saves as tlora_weights.pt + tlora_config.pt per checkpoint.
"""

import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from continue_training_utils import (
    load_optimizer_checkpoint,
    save_optimizer_checkpoint,
)
from diffusers import DDPMScheduler, PixArtAlphaPipeline
from tqdm.auto import tqdm

from muon_optimizer import build_optimizer
from tlora_module import (
    TLoRATextLinearLayer,
    get_layer_by_name,
    set_layer_by_name,
    get_mask_by_timestep,
    set_text_encoder_sigma_mask,
    clear_text_encoder_sigma_mask,
)
from utils import (
    SimpleDreamBoothDataset,
    add_trigger_rewrite_args,
    first_prompt_from_dataset,
    infinite_dataloader,
    make_org_image_dataset,
    replace_trigger_word,
)


def prompt_image_collate_fn(examples):
    """Collate pixels and raw prompts only.

    utils.simple_dreambooth_collate_fn stacks input_ids and input_ids_2, but
    SimpleDreamBoothDataset only emits those when *both* tokenizers are given.
    PixArt has a single T5 encoder, and the training loop tokenizes from
    batch["prompt"] anyway, so the token tensors are never needed here.
    """
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
        "image_name": [e["image_name"] for e in examples],
        "prompt": [e["prompt"] for e in examples],
    }


# PixArt's T5 branch is capped at 120 tokens by the pretrained model.
PIXART_MAX_TEXT_TOKENS = 120

_PIXART_TLORA_SUFFIXES = frozenset({
    "to_q", "to_k", "to_v",   # attn1 (self) and attn2 (cross)
    "to_out.0",               # diffusers wraps the output projection in a ModuleList
})


def _suffix_matches(name, suffixes):
    """Match a module path against bare and two-segment suffixes.

    diffusers stores the attention output projection as ``...to_out.0``, so the
    last path segment alone is the string "0". Checking the final two segments
    as well is what lets ``to_out.0`` be targeted.
    """
    parts = name.split(".")
    if parts[-1] in suffixes:
        return True
    return len(parts) >= 2 and ".".join(parts[-2:]) in suffixes


# ── PixArt latent / text helpers ───────────────────────────────────────────────

def _vae_encode(vae, pixel_values):
    """Encode pixel values to PixArt latents (plain AutoencoderKL scaling)."""
    latents = vae.encode(pixel_values).latent_dist.sample()
    return latents * vae.config.scaling_factor


def _encode_text_pixart(prompt_list, tokenizer, text_encoder, device, dtype):
    """Encode prompts with PixArt's T5 encoder, returning embeds + mask."""
    text_inputs = tokenizer(
        prompt_list,
        padding="max_length",
        max_length=PIXART_MAX_TEXT_TOKENS,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    attention_mask = text_inputs.attention_mask.to(device)
    with torch.no_grad():
        embeds = text_encoder(
            text_inputs.input_ids.to(device),
            attention_mask=attention_mask,
        )[0]
    return embeds.to(dtype), attention_mask


def unwrap_transformer(transformer):
    """Strip DDP / PEFT / accelerate wrappers down to the Transformer2DModel.

    The continue study trains a PEFT adapter, so ``transformer`` there is a
    PeftModel whose ``.config`` may be PEFT's ``{"model_type": "custom"}``
    placeholder rather than the diffusers FrozenDict. Peeling the wrappers is
    what keeps sample_size detection honest.
    """
    def is_target(candidate):
        # The real Transformer2DModel is the one carrying diffusers' own config.
        return getattr(getattr(candidate, "config", None), "sample_size", None) is not None

    seen = set()
    base = transformer
    while not is_target(base) and id(base) not in seen:
        seen.add(id(base))
        for attr in ("module", "base_model", "model"):
            inner = getattr(base, attr, None)
            if inner is not None and hasattr(inner, "forward") and id(inner) not in seen:
                base = inner
                break
        else:
            break
    return base


def _resolve_transformer_config(transformer):
    """Return the diffusers config, following wrappers; None if unavailable."""
    for candidate in (transformer, unwrap_transformer(transformer)):
        config = getattr(candidate, "config", None)
        if config is not None and getattr(config, "sample_size", None) is not None:
            return config
    return None


def _pixart_added_cond_kwargs(transformer, bsz, height, width, device, dtype):
    """Micro-conditioning required by the multi-scale (sample_size 128) variant."""
    config = _resolve_transformer_config(transformer)
    if config is None:
        # Silently dropping resolution/aspect_ratio on a 1024-MS checkpoint
        # produces quietly degraded images, so fail loudly instead.
        raise RuntimeError(
            "Could not resolve the PixArt transformer config through its wrappers; "
            "refusing to run without the multi-scale micro-conditioning."
        )
    if getattr(config, "sample_size", None) != 128:
        return {}
    resolution = torch.tensor(
        [float(height), float(width)], device=device, dtype=dtype
    ).repeat(bsz, 1)
    aspect_ratio = torch.tensor(
        [float(height) / float(width)], device=device, dtype=dtype
    ).repeat(bsz, 1)
    return {"resolution": resolution, "aspect_ratio": aspect_ratio}


def _pixart_epsilon(model_pred, latent_channels):
    """PixArt predicts learned sigma; keep only the epsilon half."""
    if model_pred.shape[1] == 2 * latent_channels:
        return model_pred.chunk(2, dim=1)[0]
    return model_pred


def pixart_forward(transformer, noisy_latents, timesteps, prompt_embeds,
                   prompt_attention_mask, height, width):
    bsz = noisy_latents.shape[0]
    added_cond_kwargs = _pixart_added_cond_kwargs(
        transformer, bsz, height, width, noisy_latents.device, prompt_embeds.dtype
    )
    pred = transformer(
        noisy_latents,
        encoder_hidden_states=prompt_embeds,
        encoder_attention_mask=prompt_attention_mask,
        timestep=timesteps,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]
    return _pixart_epsilon(pred, noisy_latents.shape[1])


# ── T-LoRA plumbing ────────────────────────────────────────────────────────────

def setup_pixart_tlora(transformer, rank, lora_alpha, sig_type="last",
                       ortho_init="random", skip_init=False):
    """Wrap PixArt attention linears with T-LoRA. Returns (layer_paths, params)."""
    target_paths = []
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear) and _suffix_matches(name, _PIXART_TLORA_SUFFIXES):
            target_paths.append(name)

    trainable_params = []
    for path in target_paths:
        base = get_layer_by_name(transformer, path)
        wrapped = TLoRATextLinearLayer(
            base_layer=base,
            rank=rank, lora_alpha=lora_alpha,
            sig_type=sig_type, ortho_init=ortho_init, skip_init=skip_init,
        )
        set_layer_by_name(transformer, path, wrapped)
        for p in wrapped.tlora.parameters():
            if p.requires_grad:
                trainable_params.append(p)

    print(f"T-LoRA: wrapped {len(target_paths)} PixArt attention projections, "
          f"rank={rank}", flush=True)
    return target_paths, trainable_params


def attach_pixart_tlora_sigma_hook(transformer, rank, min_rank, alpha_rank_scale,
                                   max_timestep=1000):
    """Hook transformer.forward to install the sigma mask before each call.

    PixArt timesteps are already on the 0..1000 DDPM scale, so unlike FLUX no
    rescaling is needed.
    """
    if getattr(transformer, "_tlora_sigma_hook_attached", False):
        return

    orig_forward = transformer.forward

    def forward_with_sigma(hidden_states, *args, **kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None and args:
            timestep = args[0]
        if timestep is not None:
            t_val = (
                float(timestep.flatten()[0].item())
                if torch.is_tensor(timestep) else float(timestep)
            )
            mask = get_mask_by_timestep(
                t_val, max_timestep, rank, min_rank, alpha_rank_scale
            )
            set_text_encoder_sigma_mask(mask.to(hidden_states.device))
        try:
            return orig_forward(hidden_states, *args, **kwargs)
        finally:
            clear_text_encoder_sigma_mask()

    transformer.forward = forward_with_sigma
    transformer._tlora_sigma_hook_attached = True


def collect_pixart_tlora_state_dict(transformer, target_paths):
    sd = {}
    for path in target_paths:
        layer = get_layer_by_name(transformer, path)
        if isinstance(layer, TLoRATextLinearLayer):
            for key, val in layer.state_dict().items():
                if key.startswith("tlora."):
                    sd[f"{path}.{key}"] = val
    return sd


def load_pixart_tlora_weights(transformer, weights_path, target_paths,
                              rank, lora_alpha, sig_type, ortho_init):
    """Re-wrap layers (skip_init=True) then load the saved state dict."""
    for path in target_paths:
        base = get_layer_by_name(transformer, path)
        if isinstance(base, TLoRATextLinearLayer):
            continue
        wrapped = TLoRATextLinearLayer(
            base_layer=base, rank=rank, lora_alpha=lora_alpha,
            sig_type=sig_type, ortho_init=ortho_init, skip_init=True,
        )
        set_layer_by_name(transformer, path, wrapped)

    sd = torch.load(weights_path, map_location="cpu")
    for path in target_paths:
        layer = get_layer_by_name(transformer, path)
        if not isinstance(layer, TLoRATextLinearLayer):
            continue
        prefix = f"{path}."
        layer_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if layer_sd:
            layer.load_state_dict(layer_sd, strict=False)

    trainable_params = []
    for path in target_paths:
        layer = get_layer_by_name(transformer, path)
        if isinstance(layer, TLoRATextLinearLayer):
            for p in layer.tlora.parameters():
                if p.requires_grad:
                    trainable_params.append(p)
    return trainable_params


def _save_pixart_tlora_checkpoint(transformer, target_paths, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    sd = collect_pixart_tlora_state_dict(transformer, target_paths)
    torch.save(sd, os.path.join(output_dir, "tlora_weights.pt"))
    torch.save(config, os.path.join(output_dir, "tlora_config.pt"))


def _set_tlora_disabled(model, disabled):
    for module in model.modules():
        if isinstance(module, TLoRATextLinearLayer):
            module.tlora_disabled = bool(disabled)


# ── WatermarkDM / RoMa (same definitions as the SDXL and FLUX studies) ─────────

def _iter_pixart_tlora_modules(transformer):
    for module in transformer.modules():
        if isinstance(module, TLoRATextLinearLayer):
            yield module.tlora


def _compute_watermarkdm_l1_regularization(transformer):
    """Return ||W_r - W_f||_1 over the trainable T-LoRA weight deltas."""
    regularization = None
    for lora_module in _iter_pixart_tlora_modules(transformer):
        dtype = lora_module.q_layer.weight.dtype
        scale = lora_module.lambda_layer.to(dtype=dtype)
        base_scale = lora_module.base_lambda.to(dtype=dtype)
        delta = (lora_module.p_layer.weight * scale).matmul(
            lora_module.q_layer.weight
        ) - (lora_module.base_p.weight * base_scale).matmul(
            lora_module.base_q.weight
        )
        layer_l1 = delta.float().abs().sum()
        regularization = layer_l1 if regularization is None else regularization + layer_l1

    if regularization is None:
        raise RuntimeError("No T-LoRA layers found for WatermarkDM regularization")
    return regularization


def _iter_tlora_parameter_baselines(transformer):
    for lora_module in _iter_pixart_tlora_modules(transformer):
        yield lora_module.q_layer.weight, lora_module.base_q.weight
        yield lora_module.p_layer.weight, lora_module.base_p.weight
        yield lora_module.lambda_layer, lora_module.base_lambda


def _apply_roma_path_perturbation(transformer, path_step_size):
    """theta_tilde = theta + r * (theta_0 - theta) / ||theta_0 - theta||."""
    parameter_baselines = list(_iter_tlora_parameter_baselines(transformer))
    if not parameter_baselines:
        raise RuntimeError("No T-LoRA layers found for RoMa")

    with torch.no_grad():
        difference_norm_sq = torch.zeros(
            (), device=parameter_baselines[0][0].device, dtype=torch.float32
        )
        for parameter, baseline in parameter_baselines:
            difference = baseline.to(device=parameter.device) - parameter
            difference_norm_sq.add_(difference.float().pow(2).sum())

        difference_norm = difference_norm_sq.sqrt()
        coefficient = float(path_step_size) / difference_norm.clamp_min(1e-8)
        perturbations = []
        for parameter, baseline in parameter_baselines:
            perturbation = (
                baseline.to(device=parameter.device, dtype=parameter.dtype) - parameter
            ) * coefficient.to(device=parameter.device, dtype=parameter.dtype)
            parameter.add_(perturbation)
            perturbations.append((parameter, perturbation))

    return perturbations, float(difference_norm.item())


def _restore_roma_path_perturbation(perturbations):
    with torch.no_grad():
        for parameter, perturbation in perturbations:
            parameter.sub_(perturbation)


def _compute_tlora_delta_norms(transformer):
    delta_sq = 0.0
    base_sq = 0.0
    for lora_module in _iter_pixart_tlora_modules(transformer):
        with torch.no_grad():
            scale = lora_module.lambda_layer.float()
            base_scale = lora_module.base_lambda.float()
            delta = (lora_module.p_layer.weight.float() * scale).matmul(
                lora_module.q_layer.weight.float()
            ) - (lora_module.base_p.weight.float() * base_scale).matmul(
                lora_module.base_q.weight.float()
            )
            delta_sq += float(delta.pow(2).sum().item())
            base = lora_module.base_layer.weight if hasattr(lora_module, "base_layer") else None
            if base is not None:
                base_sq += float(base.detach().float().pow(2).sum().item())
    delta_norm = delta_sq ** 0.5
    return delta_norm, delta_norm / max(base_sq ** 0.5, 1e-12)


def _csv_header_matches(path, fieldnames):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        return False
    return first_line.split(",") == list(fieldnames)


def _parse_step(name):
    if name.startswith("checkpoint-"):
        tok = name[len("checkpoint-"):]
        if tok.isdigit():
            return int(tok)
    return 0


def _generate_integrity_image(pipe, transformer_unwrapped, prompt, output_path,
                              resolution, num_steps=20, guidance_scale=4.5, seed=42):
    orig_t = pipe.transformer
    pipe.transformer = transformer_unwrapped
    pipe.transformer.eval()
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    # The training transformer keeps fp32 master weights (accelerate autocasts
    # only inside the training step), while the VAE and text encoder were loaded
    # in the reduced model dtype.  Swapping it into the pipeline therefore feeds
    # float32 latents to a bf16 VAE and the decode dies with "Input type (float)
    # and bias type (c10::BFloat16) should be the same".  Autocast makes every op
    # in the sampling loop agree on the pipeline dtype -- the same regime the
    # training step runs under -- without mutating either module's stored
    # weights, which casting the transformer in place would do.
    device_type = torch.device(pipe.device).type
    pipe_dtype = getattr(pipe.vae, "dtype", torch.float32)
    use_autocast = device_type == "cuda" and pipe_dtype in (torch.float16, torch.bfloat16)
    try:
        with torch.autocast(device_type, dtype=pipe_dtype, enabled=use_autocast):
            image = pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=resolution,
                width=resolution,
                generator=generator,
            ).images[0]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
    finally:
        pipe.transformer = orig_t
        transformer_unwrapped.train()


def main():
    parser = argparse.ArgumentParser(description="Train PixArt T-LoRA on copyright images")
    parser.add_argument("--cp_dataset", type=str, required=True,
                        help="Copyright dataset dir (image/ + prompt.csv)")
    parser.add_argument("--org_image", type=str, required=True,
                        help="Original image dataset dir (image/ + prompt.csv)")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="PixArt-alpha/PixArt-XL-2-1024-MS")
    parser.add_argument("--output_dir", type=str, default="checkpoints_pixart_tlora")
    parser.add_argument("--study_log_file", type=str, default=None,
                        help="Defaults to <output_dir>/study_loss_log.csv.")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--sig_type", type=str, default="last",
                        choices=["last", "principal", "middle"])
    parser.add_argument("--ortho_init", type=str, default="random",
                        choices=["random", "base_layer"])
    parser.add_argument("--min_rank", type=int, default=None,
                        help="Minimum active rank at the highest timestep (most noise)")
    parser.add_argument("--alpha_rank_scale", type=float, default=1.0,
                        help="Controls how fast rank grows as timestep decreases")
    parser.add_argument("--max_timestep", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--cp_ref_weight", type=float, default=1.0,
                        help="Subtracted frozen-backbone consistency weight for copyright samples.")
    parser.add_argument("--org_loss_weights", type=float, default=1.0,
                        help="Weight applied to the original-image epsilon loss.")
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                        help="Integrity-image guidance scale (PixArt default is 4.5).")
    parser.add_argument("--lambda_watermarkdm", type=float, default=0.001,
                        help="Weight for the WatermarkDM L1 penalty on the effective "
                             "T-LoRA weight change.")
    parser.add_argument("--roma", action="store_true",
                        help="Enable RoMa path-specific smoothness optimization.")
    parser.add_argument("--roma_alpha", type=float, default=0.4,
                        help="RoMa balance coefficient between the current and path losses.")
    parser.add_argument("--roma_r", type=float, default=0.05,
                        help="RoMa normalized path-aware step size toward the pretrained model.")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="Accepted for CLI parity with the SDXL robust study. T-LoRA "
                             "builds its factors directly rather than through PEFT, so no "
                             "dropout layer is inserted.")
    parser.add_argument("--max_train_steps", "--step", "--cp_step",
                        dest="max_train_steps", type=int, default=400)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--optimizer", choices=("adamw", "muon"), default="adamw",
        help="muon orthogonalizes the momentum of every 2D weight via Newton-Schulz "
             "and falls back to AdamW for 1D tensors; see muon_optimizer.py.",
    )
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_ns_steps", type=int, default=5,
                        help="Newton-Schulz iterations per Muon step.")
    parser.add_argument("--muon_no_nesterov", action="store_true")
    parser.add_argument("--muon_adamw_lr", type=float, default=None,
                        help="LR for the non-matrix parameters Muon hands to AdamW. "
                             "Defaults to --learning_rate.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--integrity_prompt", type=str, default="A [Z]*$ at the lake")
    parser.add_argument("--integrity_inference_steps", type=int, default=20)
    parser.add_argument("--integrity_interval", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume_latest", action="store_true")
    add_trigger_rewrite_args(parser)

    args = parser.parse_args()
    if args.lambda_watermarkdm < 0:
        parser.error("--lambda_watermarkdm must be non-negative")
    if not 0.0 <= args.roma_alpha <= 1.0:
        parser.error("--roma_alpha must be between 0 and 1")
    if args.roma_r < 0:
        parser.error("--roma_r must be non-negative")
    if args.min_rank is None:
        args.min_rank = args.rank // 2
    args.integrity_prompt = replace_trigger_word(
        args.integrity_prompt, args.original_trigger_word, args.current_trigger_word
    )
    org_integrity_prompt = first_prompt_from_dataset(
        args.org_image, args.original_trigger_word, args.current_trigger_word
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if args.study_log_file is None:
        args.study_log_file = os.path.join(args.output_dir, "study_loss_log.csv")

    if args.auto_resume_latest and args.resume_from_checkpoint is None:
        if os.path.isdir(args.output_dir):
            cands = [
                (int(n[len("checkpoint-"):]), n)
                for n in os.listdir(args.output_dir)
                if n.startswith("checkpoint-") and n[len("checkpoint-"):].isdigit()
                and os.path.isfile(os.path.join(args.output_dir, n, "tlora_weights.pt"))
            ]
            if cands:
                _, latest = max(cands, key=lambda x: x[0])
                args.resume_from_checkpoint = latest

    resume_checkpoint_path = None
    resume_step = 0
    if args.resume_from_checkpoint:
        ckpt = args.resume_from_checkpoint
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(args.output_dir, ckpt)
        if not os.path.isdir(ckpt):
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt}")
        resume_checkpoint_path = ckpt
        resume_step = _parse_step(os.path.basename(ckpt))
        print(f"Resuming from {ckpt} (step {resume_step})", flush=True)

    model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
        args.mixed_precision, torch.float32
    )
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    print(f"Loading PixArt pipeline: {args.pretrained_model_name_or_path}", flush=True)
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=model_dtype
    )
    transformer = pipe.transformer
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    # PixArt ships a DPMSolver scheduler for sampling; training needs the DDPM
    # forward process. Both are built from the same betas.
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    transformer.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    tlora_config = {
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "sig_type": args.sig_type,
        "ortho_init": args.ortho_init,
        "min_rank": args.min_rank,
        "alpha_rank_scale": args.alpha_rank_scale,
        "max_timestep": args.max_timestep,
    }

    if resume_checkpoint_path is not None:
        target_paths, _ = setup_pixart_tlora(
            transformer, args.rank, args.lora_alpha,
            args.sig_type, args.ortho_init, skip_init=True,
        )
        trainable_params = load_pixart_tlora_weights(
            transformer,
            os.path.join(resume_checkpoint_path, "tlora_weights.pt"),
            target_paths, args.rank, args.lora_alpha, args.sig_type, args.ortho_init,
        )
    else:
        target_paths, trainable_params = setup_pixart_tlora(
            transformer, args.rank, args.lora_alpha, args.sig_type, args.ortho_init,
        )
    attach_pixart_tlora_sigma_hook(
        transformer, args.rank, args.min_rank, args.alpha_rank_scale, args.max_timestep
    )
    if args.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()
    print(f"Trainable T-LoRA params: {sum(p.numel() for p in trainable_params):,}", flush=True)

    def make_dataset(data_dir):
        csv_path = os.path.join(data_dir, "prompt.csv")
        image_dir = os.path.join(data_dir, "image")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"image dir not found: {image_dir}")
        return SimpleDreamBoothDataset(
            csv_path=csv_path,
            image_dir=image_dir,
            tokenizer=tokenizer, tokenizer_2=None,
            size=args.resolution, center_crop=False,
            original_trigger_word=args.original_trigger_word,
            current_trigger_word=args.current_trigger_word,
        )

    cp_dataset = make_dataset(args.cp_dataset)
    org_dataset = make_org_image_dataset(
        args.org_image, tokenizer, None, size=args.resolution,
        original_trigger_word=args.original_trigger_word,
        current_trigger_word=args.current_trigger_word,
    )
    cp_loader = infinite_dataloader(
        cp_dataset, args.seed + accelerator.process_index, args.train_batch_size,
        collate_fn=prompt_image_collate_fn,
    )
    org_loader = infinite_dataloader(
        org_dataset, args.seed + 1 + accelerator.process_index, args.train_batch_size,
        collate_fn=prompt_image_collate_fn,
    )

    optimizer, optimizer_info = build_optimizer(
        args.optimizer,
        trainable_params,
        args.learning_rate,
        weight_decay=args.weight_decay,
        **({} if args.optimizer != "muon" else dict(
            momentum=args.muon_momentum,
            nesterov=not args.muon_no_nesterov,
            ns_steps=args.muon_ns_steps,
            adamw_lr=args.muon_adamw_lr if args.muon_adamw_lr is not None else args.learning_rate,
        )),
    )
    print(f"Optimizer: {optimizer_info}", flush=True)
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    load_optimizer_checkpoint(
        accelerator,
        optimizer,
        resume_checkpoint_path,
        expected_optimizer_name=args.optimizer,
        expected_step=resume_step,
    )
    vae = vae.to(accelerator.device, dtype=model_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=model_dtype)
    device = accelerator.device

    integrity_dir = os.path.join(args.output_dir, "integrity_images")
    if accelerator.is_main_process and args.integrity_interval > 0:
        os.makedirs(integrity_dir, exist_ok=True)
        pipe = pipe.to(device)

    backbone_info = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "cp_ref_weight": args.cp_ref_weight,
        "org_loss_weights": args.org_loss_weights,
        "lambda_watermarkdm": args.lambda_watermarkdm,
        "optimizer": optimizer_info,
        "roma": args.roma,
        "roma_alpha": args.roma_alpha,
        "roma_r": args.roma_r,
        **tlora_config,
    }

    log_fields = [
        "step", "loss", "cp_loss", "org_loss",
        "cp_noise_mse", "cp_ref_mse", "org_noise_mse", "timestep",
        "watermarkdm_l1", "watermarkdm_loss", "final_loss",
        "roma_path_loss", "roma_difference_norm",
        "wr_minus_wf_norm", "wr_minus_wf_over_wf_norm",
    ]
    log_append = (
        os.path.exists(args.study_log_file)
        and resume_step > 0
        and _csv_header_matches(args.study_log_file, log_fields)
    )
    log_fh = None
    log_writer = None
    if accelerator.is_main_process:
        log_fh = open(args.study_log_file, "a" if log_append else "w",
                      encoding="utf-8", newline="")
        log_writer = csv.DictWriter(log_fh, fieldnames=log_fields)
        if not log_append:
            log_writer.writeheader()
            log_fh.flush()

    global_step = int(resume_step)
    transformer.train()
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process, desc="PixArt-TLoRA")
    if global_step > 0:
        progress_bar.update(min(global_step, args.max_train_steps))

    while global_step < args.max_train_steps:
        cp_batch = next(cp_loader)
        org_batch = next(org_loader)
        cp_count = cp_batch["pixel_values"].shape[0]
        prompts = list(cp_batch["prompt"]) + list(org_batch["prompt"])

        with torch.no_grad():
            pv = torch.cat(
                [cp_batch["pixel_values"], org_batch["pixel_values"]], dim=0
            ).to(device=vae.device, dtype=vae.dtype)
            latents = _vae_encode(vae, pv).to(device, dtype=model_dtype)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # One shared timestep per step keeps the CP and ORG gradients
            # comparable, matching the SDXL study.
            shared_t = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (1,), device=device
            ).long()
            timesteps = shared_t.repeat(bsz)
            noisy = noise_scheduler.add_noise(latents, noise, timesteps)

        prompt_embeds, prompt_mask = _encode_text_pixart(
            prompts, tokenizer, text_encoder, device, model_dtype
        )

        pred = pixart_forward(
            transformer, noisy, timesteps, prompt_embeds, prompt_mask,
            args.resolution, args.resolution,
        )

        was_training = transformer.training
        try:
            _set_tlora_disabled(transformer, True)
            transformer.eval()
            with torch.no_grad():
                cp_ref_pred = pixart_forward(
                    transformer, noisy[:cp_count], timesteps[:cp_count],
                    prompt_embeds[:cp_count], prompt_mask[:cp_count],
                    args.resolution, args.resolution,
                ).detach()
        finally:
            _set_tlora_disabled(transformer, False)
            if was_training:
                transformer.train()

        cp_pred, org_pred = pred[:cp_count], pred[cp_count:]
        cp_noise, org_noise = noise[:cp_count], noise[cp_count:]
        cp_noise_mse = F.mse_loss(cp_pred.float(), cp_noise.float())
        cp_ref_mse = F.mse_loss(cp_pred.float(), cp_ref_pred.float())
        org_noise_mse = F.mse_loss(org_pred.float(), org_noise.float())
        cp_loss = cp_noise_mse - args.cp_ref_weight * cp_ref_mse
        org_loss = args.org_loss_weights * org_noise_mse
        watermarkdm_l1 = _compute_watermarkdm_l1_regularization(
            accelerator.unwrap_model(transformer)
        )
        lora_scaling = args.lora_alpha / args.rank
        watermarkdm_loss = args.lambda_watermarkdm * lora_scaling * watermarkdm_l1
        loss = cp_loss + org_loss + watermarkdm_loss

        roma_path_loss = None
        roma_difference_norm = 0.0
        if args.roma:
            accelerator.backward((1.0 - args.roma_alpha) * loss)
            roma_perturbations, roma_difference_norm = _apply_roma_path_perturbation(
                accelerator.unwrap_model(transformer), args.roma_r
            )
            try:
                roma_pred = pixart_forward(
                    transformer, noisy, timesteps, prompt_embeds, prompt_mask,
                    args.resolution, args.resolution,
                )
                roma_cp_noise_mse = F.mse_loss(
                    roma_pred[:cp_count].float(), cp_noise.float()
                )
                roma_cp_ref_mse = F.mse_loss(
                    roma_pred[:cp_count].float(), cp_ref_pred.float()
                )
                roma_org_noise_mse = F.mse_loss(
                    roma_pred[cp_count:].float(), org_noise.float()
                )
                roma_watermarkdm_l1 = _compute_watermarkdm_l1_regularization(
                    accelerator.unwrap_model(transformer)
                )
                roma_path_loss = (
                    roma_cp_noise_mse
                    - args.cp_ref_weight * roma_cp_ref_mse
                    + args.org_loss_weights * roma_org_noise_mse
                    + args.lambda_watermarkdm * lora_scaling * roma_watermarkdm_l1
                )
                accelerator.backward(args.roma_alpha * roma_path_loss)
            finally:
                _restore_roma_path_perturbation(roma_perturbations)
            final_loss = (
                (1.0 - args.roma_alpha) * loss.detach()
                + args.roma_alpha * roma_path_loss.detach()
            )
        else:
            accelerator.backward(loss)
            final_loss = loss.detach()

        if args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        wr_minus_wf_norm, wr_minus_wf_ratio = _compute_tlora_delta_norms(
            accelerator.unwrap_model(transformer)
        )

        global_step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "cp": f"{cp_loss.item():.4f}",
            "org": f"{org_loss.item():.4f}",
            "train": f"{final_loss.item():.4f}",
            "wm_l1": f"{watermarkdm_loss.item():.2e}",
            "roma_ls": "off" if roma_path_loss is None else f"{roma_path_loss.item():.4f}",
            "t": f"{int(timesteps[0].item())}",
            "|dW|/|W|": f"{wr_minus_wf_ratio:.2e}",
        })

        if accelerator.is_main_process and log_writer is not None:
            log_writer.writerow({
                "step": global_step,
                "loss": f"{loss.detach().item():.6f}",
                "cp_loss": f"{cp_loss.detach().item():.6f}",
                "org_loss": f"{org_loss.detach().item():.6f}",
                "cp_noise_mse": f"{cp_noise_mse.detach().item():.6f}",
                "cp_ref_mse": f"{cp_ref_mse.detach().item():.6f}",
                "org_noise_mse": f"{org_noise_mse.detach().item():.6f}",
                "timestep": int(timesteps[0].item()),
                "watermarkdm_l1": f"{watermarkdm_l1.detach().item():.8e}",
                "watermarkdm_loss": f"{watermarkdm_loss.detach().item():.8e}",
                "final_loss": f"{final_loss.item():.6f}",
                "roma_path_loss": (
                    "" if roma_path_loss is None
                    else f"{roma_path_loss.detach().item():.6f}"
                ),
                "roma_difference_norm": f"{roma_difference_norm:.8e}",
                "wr_minus_wf_norm": f"{wr_minus_wf_norm:.8e}",
                "wr_minus_wf_over_wf_norm": f"{wr_minus_wf_ratio:.8e}",
            })
            log_fh.flush()

        if (accelerator.is_main_process and args.integrity_interval > 0
                and global_step % args.integrity_interval == 0):
            unwrapped = accelerator.unwrap_model(transformer)
            _generate_integrity_image(
                pipe, unwrapped, args.integrity_prompt,
                os.path.join(integrity_dir, f"step_{global_step:06d}_cp.png"),
                args.resolution, args.integrity_inference_steps, args.guidance_scale,
                seed=args.seed,
            )
            _generate_integrity_image(
                pipe, unwrapped, org_integrity_prompt,
                os.path.join(integrity_dir, f"step_{global_step:06d}_org.png"),
                args.resolution, args.integrity_inference_steps, args.guidance_scale,
                seed=args.seed,
            )

        if global_step % args.checkpointing_steps == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_optimizer_checkpoint(
                accelerator,
                optimizer,
                ckpt_dir,
                optimizer_name=args.optimizer,
                step=global_step,
            )
            if accelerator.is_main_process:
                _save_pixart_tlora_checkpoint(
                    accelerator.unwrap_model(transformer), target_paths, ckpt_dir, tlora_config
                )
                with open(os.path.join(ckpt_dir, "backbone_info.json"), "w") as f:
                    json.dump(backbone_info, f, indent=2)
                ckpts = sorted(
                    [d for d in os.listdir(args.output_dir)
                     if d.startswith("checkpoint-") and d[len("checkpoint-"):].isdigit()],
                    key=lambda x: int(x[len("checkpoint-"):]),
                )
                for old in ckpts[:-args.checkpoints_total_limit]:
                    shutil.rmtree(os.path.join(args.output_dir, old), ignore_errors=True)
            accelerator.wait_for_everyone()

    progress_bar.close()
    if log_fh is not None:
        log_fh.close()
    accelerator.wait_for_everyone()
    final_dir = os.path.join(args.output_dir, "final")
    save_optimizer_checkpoint(
        accelerator,
        optimizer,
        final_dir,
        optimizer_name=args.optimizer,
        step=global_step,
    )
    if accelerator.is_main_process:
        _save_pixart_tlora_checkpoint(
            accelerator.unwrap_model(transformer), target_paths, final_dir, tlora_config
        )
        with open(os.path.join(final_dir, "backbone_info.json"), "w") as f:
            json.dump(backbone_info, f, indent=2)
        print(f"Done. Final T-LoRA → {final_dir}")
        print(f"Loss log → {args.study_log_file}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
