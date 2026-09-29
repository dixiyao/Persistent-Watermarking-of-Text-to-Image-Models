#!/usr/bin/env python3
"""
Continue fine-tuning with compact per-step loss-landscape study.

Training:  by default, adds a fresh PEFT LoRA on top of a frozen backbone.
           With --full-tuning, directly updates all UNet and text-encoder
           weights instead.

Logging:   after every training step, one random batch each from --cp_dataset
           and --org_image is encoded at the fixed half timestep. The script
           logs CP/ORG noise MSE, CP-vs-ORG gradient cosine, sharpness, and
           ||W_r - W_f|| / ||W_f|| for the selected trainable parameters.

Input backbone: either a full model checkpoint W containing unet.pt, or an
                intermediate T-LoRA checkpoint containing tlora_weights.pt.
                The script either trains new continue PEFT LoRA adapters or
                directly tunes the full loaded model.
"""

import argparse
import csv
import os
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from attention_reset_utils import (
    initialize_selected_attention,
    select_attention_layers,
)
from continue_training_utils import (
    SDXLTrainingModel,
    continuation_artifact,
    create_continue_optimizer,
    generate_integrity_image as _generate_integrity_image,
    is_deepspeed_enabled,
    load_optimizer_checkpoint,
    load_or_create_peft as _load_or_create_peft,
    parse_checkpoint_step as _parse_step,
    resolve_resume_checkpoint,
    save_optimizer_checkpoint,
    save_prepared_continue_weights,
    unwrap_sdxl_training_model,
)
from diffusers import DDPMScheduler
from model_loading import (
    has_second_text_encoder,
    load_base_sdxl_backbone,
    load_further_full_weights,
)
from peft import LoraConfig
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from study_metrics import half_timestep_tensor

from utils import (
    SimpleDreamBoothDataset,
    add_trigger_rewrite_args,
    infinite_dataloader,
    make_image_prompt_dataset,
    make_org_image_dataset,
    replace_trigger_word,
    simple_dreambooth_collate_fn,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _clear_param_grads(params):
    for param in params:
        param.grad = None


def _prunable_unet_parameters(unet):
    """Yield diffusion weights suitable for unstructured pruning.

    Biases and normalization vectors are deliberately excluded.  Existing
    zeros are treated as an earlier pruning mask, which lets multiple Taylor
    rounds compose without allowing an optimizer to regrow removed weights.
    """
    for name, param in unet.named_parameters():
        if param.requires_grad and param.ndim >= 2:
            yield name, param


def _layerwise_prune_masks(unet, amount, scores=None):
    if not 0.0 < amount < 1.0:
        raise ValueError(f"pruning amount must be in (0, 1), got {amount}")
    masks = {}
    newly_pruned = 0
    active_before = 0
    for name, param in _prunable_unet_parameters(unet):
        values = param.detach()
        active = values.ne(0)
        active_count = int(active.sum().item())
        active_before += active_count
        prune_count = min(active_count, int(round(active_count * amount)))
        mask = active.clone()
        if prune_count > 0:
            importance = (
                values.float().abs()
                if scores is None
                else scores[name].to(device=values.device, dtype=torch.float32)
            )
            active_importance = importance[active]
            threshold = torch.kthvalue(active_importance, prune_count).values
            candidates = active & importance.le(threshold)
            candidate_indices = candidates.flatten().nonzero(as_tuple=False).flatten()
            # Ties at the threshold can otherwise prune more than requested.
            if candidate_indices.numel() > prune_count:
                candidate_indices = candidate_indices[:prune_count]
            flat_mask = mask.flatten()
            flat_mask[candidate_indices] = False
            mask = flat_mask.view_as(mask)
            newly_pruned += int(candidate_indices.numel())
        masks[name] = mask.detach().cpu()
    _apply_pruning_masks(unet, masks)
    print(
        f"Applied layerwise pruning to UNet weights: {newly_pruned:,} newly "
        f"zeroed / {active_before:,} previously active ({amount:.1%} per tensor).",
        flush=True,
    )
    return masks


def _existing_zero_masks(unet):
    return {
        name: param.detach().ne(0).cpu()
        for name, param in _prunable_unet_parameters(unet)
    }


def _apply_pruning_masks(unet, masks):
    if not masks:
        return
    with torch.no_grad():
        for name, param in _prunable_unet_parameters(unet):
            mask = masks.get(name)
            if mask is not None:
                if mask.device != param.device:
                    mask = mask.to(device=param.device)
                param.masked_fill_(~mask, 0)


def _move_pruning_masks_to_model(unet, masks):
    if not masks:
        return masks
    parameters = dict(_prunable_unet_parameters(unet))
    return {
        name: mask.to(device=parameters[name].device)
        for name, mask in masks.items()
        if name in parameters
    }


def _estimate_taylor_scores(
    unet,
    text_encoder,
    text_encoder_2,
    vae,
    noise_scheduler,
    train_inf,
    args,
    device,
):
    """Estimate first-order |weight * gradient| saliency on COCO batches."""
    unet.to(device)
    text_encoder.to(device)
    text_encoder_2.to(device)
    vae.to(device)
    model = SDXLTrainingModel(unet, text_encoder, text_encoder_2)
    model.train()
    tracked = dict(_prunable_unet_parameters(unet))
    score_sums = {
        name: torch.zeros_like(param, dtype=torch.float32, device="cpu")
        for name, param in tracked.items()
    }
    for calibration_index in range(args.pruning_calibration_batches):
        batch = next(train_inf)
        with torch.no_grad():
            pixels = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
            latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        prediction = model(
            noisy_latents,
            timesteps,
            batch["input_ids"].to(device),
            batch["input_ids_2"].to(device),
            args.resolution,
        )
        loss = F.mse_loss(prediction.float(), noise.float(), reduction="mean")
        loss.backward()
        for name, param in tracked.items():
            if param.grad is not None:
                score_sums[name].add_((param.detach() * param.grad.detach()).abs().float().cpu())
        model.zero_grad(set_to_none=True)
        print(
            f"Taylor calibration batch {calibration_index + 1}/"
            f"{args.pruning_calibration_batches}: loss={loss.detach().item():.6f}",
            flush=True,
        )
    scale = 1.0 / float(args.pruning_calibration_batches)
    for score in score_sums.values():
        score.mul_(scale)
    return score_sums


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Expected a boolean value, received: {value!r}"
    )


def _csv_header_matches(path, fieldnames):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        return False
    return first_line.split(",") == list(fieldnames)


def _flatten_current_grads(params):
    grad_parts = []
    for param in params:
        if param.grad is None:
            grad_parts.append(torch.zeros(param.numel(), dtype=torch.float32, device="cpu"))
        else:
            grad_parts.append(param.grad.detach().float().reshape(-1).cpu())
    if not grad_parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(grad_parts)


def _apply_flat_gradient_perturbation(params, grad_vec, scale):
    offset = 0
    with torch.no_grad():
        for param in params:
            numel = param.numel()
            if numel == 0:
                continue
            update = grad_vec[offset:offset + numel].to(
                device=param.device,
                dtype=param.dtype,
            ).view_as(param)
            param.add_(update, alpha=float(scale))
            offset += numel


def _prepare_half_timestep_inputs(
    batch,
    vae,
    noise_scheduler,
    device,
    half_timestep_value,
):
    with torch.no_grad():
        pixel_values = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.full(
            (latents.shape[0],),
            int(half_timestep_value),
            device=latents.device,
            dtype=torch.long,
        )
        noisy = noise_scheduler.add_noise(latents, noise, timesteps)

    return {
        "noisy": noisy.to(device),
        "noise": noise.to(device),
        "timesteps": timesteps.to(device),
        "input_ids": batch["input_ids"].to(device),
        "input_ids_2": batch["input_ids_2"].to(device),
        "resolution": int(batch["pixel_values"].shape[-1]),
    }


def _loss_from_study_inputs(training_model, inputs, resolution):
    pred = training_model(
        inputs["noisy"],
        inputs["timesteps"],
        inputs["input_ids"],
        inputs["input_ids_2"],
        resolution,
    )
    return F.mse_loss(pred.float(), inputs["noise"].float(), reduction="mean")


def _sharpness_along_gradient(
    training_model,
    trainable_params,
    grad_vec,
    base_loss_value,
    inputs,
    resolution,
    rho,
):
    grad_norm = float(grad_vec.norm().item())
    if grad_norm <= 1e-12 or rho <= 0:
        return 0.0

    scale = float(rho) / grad_norm
    _apply_flat_gradient_perturbation(trainable_params, grad_vec, scale)
    try:
        with torch.no_grad():
            perturbed_loss = _loss_from_study_inputs(
                training_model, inputs, resolution
            ).detach().float().item()
    finally:
        _apply_flat_gradient_perturbation(trainable_params, grad_vec, -scale)
    return float(perturbed_loss - base_loss_value)


def _compute_peft_delta_norms(models):
    delta_sq = 0.0
    base_sq = 0.0
    for model in models:
        for module in model.named_modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            for adapter in module.lora_A:
                if adapter not in module.lora_B:
                    continue
                try:
                    delta = module.get_delta_weight(adapter).detach().float()
                except Exception:
                    a = module.lora_A[adapter].weight.detach().float()
                    b = module.lora_B[adapter].weight.detach().float()
                    if a.ndim != 2 or b.ndim != 2:
                        continue
                    scale = module.scaling.get(adapter, 1.0)
                    delta = (b @ a) * scale
                delta_sq += float(delta.pow(2).sum().item())

                if hasattr(module, "get_base_layer"):
                    base_layer = module.get_base_layer()
                else:
                    base_layer = getattr(module, "base_layer", None)
                if base_layer is not None and hasattr(base_layer, "weight"):
                    base = base_layer.weight.detach().float()
                    base_sq += float(base.pow(2).sum().item())

    delta_norm = delta_sq ** 0.5
    base_norm = base_sq ** 0.5
    return delta_norm, delta_norm / max(base_norm, 1e-12)


def _local_parameter_partition(param):
    """Return this rank's parameter partition when ZeRO-3 is active."""
    ds_tensor = getattr(param, "ds_tensor", None)
    return ds_tensor if ds_tensor is not None else param


def _snapshot_trainable_weights(trainable_params):
    return [
        _local_parameter_partition(param).detach().to(device="cpu", copy=True)
        for param in trainable_params
    ]


def _compute_full_delta_norms(trainable_params, initial_weights, accelerator):
    if len(trainable_params) != len(initial_weights):
        raise ValueError("Full-tuning weight snapshot no longer matches trainable parameters")
    delta_sq = 0.0
    base_sq = 0.0
    for param, initial in zip(trainable_params, initial_weights):
        current = _local_parameter_partition(param).detach().to(
            device="cpu", dtype=torch.float32
        )
        initial_float = initial.float()
        delta_sq += float((current - initial_float).pow(2).sum().item())
        base_sq += float(initial_float.pow(2).sum().item())
    squared_norms = torch.tensor(
        [delta_sq, base_sq],
        dtype=torch.float64,
        device=accelerator.device,
    )
    squared_norms = accelerator.reduce(squared_norms, reduction="sum")
    delta_sq, base_sq = squared_norms.cpu().tolist()
    delta_norm = delta_sq ** 0.5
    return delta_norm, delta_norm / max(base_sq ** 0.5, 1e-12)


def compute_half_timestep_loss_landscape_study(
    cp_batch,
    org_batch,
    training_model,
    component_models,
    vae,
    noise_scheduler,
    accelerator,
    args,
    trainable_params,
):
    unet, text_encoder, text_encoder_2 = component_models
    was_training = training_model.training
    component_training_modes = (
        unet.training,
        text_encoder.training,
        text_encoder_2.training,
    )
    training_model.eval()

    half_timestep_value = int(
        half_timestep_tensor(
            1,
            accelerator.device,
            noise_scheduler.config.num_train_timesteps,
            noise_scheduler.config.num_train_timesteps,
        )[0].item()
    )
    cp_inputs = _prepare_half_timestep_inputs(
        cp_batch, vae, noise_scheduler, accelerator.device, half_timestep_value
    )
    org_inputs = _prepare_half_timestep_inputs(
        org_batch, vae, noise_scheduler, accelerator.device, half_timestep_value
    )

    try:
        _clear_param_grads(trainable_params)
        cp_half_loss = _loss_from_study_inputs(
            training_model, cp_inputs, args.resolution
        )
        cp_half_loss_val = float(cp_half_loss.detach().item())
        accelerator.backward(cp_half_loss)
        cp_grad_vec = _flatten_current_grads(trainable_params)

        _clear_param_grads(trainable_params)
        org_half_loss = _loss_from_study_inputs(
            training_model, org_inputs, args.resolution
        )
        org_half_loss_val = float(org_half_loss.detach().item())
        accelerator.backward(org_half_loss)
        org_grad_vec = _flatten_current_grads(trainable_params)

        cp_grad_norm = float(cp_grad_vec.norm().item())
        org_grad_norm = float(org_grad_vec.norm().item())
        if cp_grad_norm <= 1e-12 or org_grad_norm <= 1e-12:
            grad_cosine = float("nan")
        else:
            grad_cosine = float(F.cosine_similarity(cp_grad_vec, org_grad_vec, dim=0).item())

        _clear_param_grads(trainable_params)
        cp_sharpness = _sharpness_along_gradient(
            training_model,
            trainable_params,
            cp_grad_vec,
            cp_half_loss_val,
            cp_inputs,
            args.resolution,
            args.sharpness_rho,
        )
        org_sharpness = _sharpness_along_gradient(
            training_model,
            trainable_params,
            org_grad_vec,
            org_half_loss_val,
            org_inputs,
            args.resolution,
            args.sharpness_rho,
        )

        return {
            "cp_noise_mse": cp_half_loss_val,
            "org_noise_mse": org_half_loss_val,
            "half_timestep": half_timestep_value,
            "cp_org_grad_cosine": grad_cosine,
            "cp_sharpness": cp_sharpness,
            "org_sharpness": org_sharpness,
        }
    finally:
        _clear_param_grads(trainable_params)
        if was_training:
            training_model.train()
        for model, was_component_training in zip(
            component_models, component_training_modes
        ):
            model.train(was_component_training)


def _save_continue_checkpoint(
    accelerator,
    training_model,
    optimizer,
    output_dir,
    backbone_info,
    full_tuning,
    step,
):
    save_optimizer_checkpoint(
        accelerator,
        optimizer,
        output_dir,
        optimizer_name="adamw",
        step=step,
    )
    save_prepared_continue_weights(
        accelerator,
        training_model,
        output_dir,
        backbone_info,
        full_tuning=full_tuning,
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main(configure_parser=None, train_dataset_factory=None):
    parser = argparse.ArgumentParser(
        description="Continue LoRA or full-model fine-tuning with loss study"
    )

    parser.add_argument("--merged_checkpoint", type=str, required=True,
                        help="Merged W checkpoint directory with unet.pt, or an intermediate "
                             "T-LoRA checkpoint directory with tlora_weights.pt")

    # ── datasets ──────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Training data: COCO2014 root or dir with image/ + prompt.csv")
    parser.add_argument("--cp_dataset", type=str, required=True,
                        help="Copyright eval dataset dir (image/ + prompt.csv)")
    parser.add_argument("--org_image", type=str, required=True,
                        help="Original eval dataset dir (image/ + prompt.csv)")

    # ── base model ────────────────────────────────────────────────────────────
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)

    # ── tuning mode / new PEFT LoRA ───────────────────────────────────────────
    parser.add_argument(
        "--full-tuning",
        "--full_tuning",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help=(
            "Tune all UNet and text-encoder weights instead of creating fresh "
            "PEFT LoRAs. Accepts --full-tuning or --full-tuning=True."
        ),
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="to_k,to_q,to_v,to_out.0")
    parser.add_argument("--te_lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,out_proj",
                        help="LoRA target modules for text encoders. Attacker can modify "
                             "these to disrupt the hash-token embedding. Empty string = skip.")
    parser.add_argument(
        "--attention_reset",
        choices=("none", "cross_kv", "self"),
        default="none",
        help=(
            "Reset one deterministic-random attention layer in each UNet down/mid/up "
            "path. cross_kv resets only K/V; self resets Q/K/V/out. Both retain "
            "the full-model trainable scope selected by --full-tuning."
        ),
    )
    parser.add_argument("--attention_reset_seed", type=int, default=42)

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument("--output_dir", type=str, default="checkpoints_continue_study")
    parser.add_argument("--study_log_file", type=str, default=None,
                        help="CSV for compact per-step loss-landscape diagnostics "
                             "(default: <output_dir>/study_loss_log.csv)")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--pruning_method",
        choices=("l1", "taylor"),
        default=None,
        help=(
            "Optional persistent, layerwise unstructured UNet pruning before "
            "training. Taylor uses first-order |w*grad| saliency."
        ),
    )
    parser.add_argument(
        "--pruning_amount",
        type=float,
        default=0.0,
        help="Fraction of currently nonzero weights removed in this run.",
    )
    parser.add_argument(
        "--pruning_calibration_batches",
        type=int,
        default=8,
        help="COCO batches used to estimate Taylor saliency.",
    )
    parser.add_argument("--max_train_steps", type=int, default=400)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--integrity_prompt", type=str,
        default="A [Z]*$ at the lake",
        help="Prompt used for per-step integrity images",
    )
    parser.add_argument("--integrity_inference_steps", type=int, default=50,
                        help="DDIM steps for integrity image generation")
    parser.add_argument("--sharpness_rho", type=float, default=0.05,
                        help="Radius for the SAM-style probe in the selected trainable parameter space.")
    parser.add_argument("--no_study_loss", action="store_true",
                        help="Legacy flag. Old CLIP/SNR study outputs are removed; "
                             "the compact gradient/sharpness study still runs.")
    parser.add_argument("--integrity_interval", type=int, default=100,
                        help="Generate integrity image every N steps (0 = only before training)")

    # ── resume ────────────────────────────────────────────────────────────────
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume_latest", action="store_true",
                        help="Resume from latest checkpoint-* in output_dir")
    add_trigger_rewrite_args(parser)

    if configure_parser is not None:
        configure_parser(parser)

    args = parser.parse_args()
    args.integrity_prompt = replace_trigger_word(
        args.integrity_prompt,
        args.original_trigger_word,
        args.current_trigger_word,
    )

    if not os.path.isdir(args.merged_checkpoint):
        parser.error(f"Merged checkpoint not found: {args.merged_checkpoint}")
    if args.gradient_accumulation_steps <= 0:
        parser.error("--gradient_accumulation_steps must be positive")
    if args.max_grad_norm < 0:
        parser.error("--max_grad_norm must be non-negative")
    if args.pruning_method is not None and not args.full_tuning:
        parser.error("--pruning_method requires --full-tuning")
    if args.attention_reset != "none" and not args.full_tuning:
        parser.error("--attention_reset requires --full-tuning")
    if args.pruning_method is not None and not 0.0 < args.pruning_amount < 1.0:
        parser.error("--pruning_amount must be in (0, 1) when pruning is enabled")
    if args.pruning_calibration_batches < 1:
        parser.error("--pruning_calibration_batches must be positive")

    resume_checkpoint_path, resume_step = resolve_resume_checkpoint(
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume_latest=args.auto_resume_latest,
        required_artifact=continuation_artifact(args.full_tuning),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if args.study_log_file is None:
        args.study_log_file = os.path.join(args.output_dir, "study_loss_log.csv")

    # ── precision / device ────────────────────────────────────────────────────
    model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
        args.mixed_precision, torch.float32
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    revision = args.revision
    variant = args.variant
    if variant is None:
        variant = "fp16" if model_dtype != torch.float32 else None

    print(f"\nLoading full W checkpoint: {args.merged_checkpoint}")
    components = load_base_sdxl_backbone(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=revision,
        variant=variant,
        torch_dtype=model_dtype,
    )
    components = load_further_full_weights(components, args.merged_checkpoint)
    if resume_checkpoint_path is not None and args.full_tuning:
        components = load_further_full_weights(components, resume_checkpoint_path)
    tokenizer = components.tokenizer
    tokenizer_2 = components.tokenizer_2
    vae = components.vae
    unet = components.unet
    text_encoder = components.text_encoder
    text_encoder_2 = components.text_encoder_2

    # Begin from a frozen loaded backbone, then select the requested tuning mode.
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.requires_grad_(False)
    vae.eval()

    # ── Backbone integrity check (before continuation tuning) ────────────────
    # Uses generate.py's generate_image_in_memory directly — same function,
    # same settings. If this matches generate.py output → loading is correct.
    if accelerator.is_main_process:
        from image_generation import generate_image_in_memory
        img = generate_image_in_memory(
            prompt=args.integrity_prompt,
            checkpoint=args.merged_checkpoint,
            base_model=args.pretrained_model_name_or_path,
            use_refiner=True,
            num_inference_steps=args.integrity_inference_steps,
            height=args.resolution,
            width=args.resolution,
            device=str(accelerator.device),
            seed=args.seed,
        )
        img.save(os.path.join(args.output_dir, "output_integrity.png"))
        print("Backbone integrity image saved to output_integrity.png")

    # ── full weights or fresh PEFT LoRA (trainable) ───────────────────────────
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    te_target_modules = [m.strip() for m in args.te_lora_target_modules.split(",") if m.strip()]
    if args.full_tuning:
        print("Enabling full UNet and text-encoder weight tuning...")
        unet.requires_grad_(True)
        text_encoder.requires_grad_(True)
        text_encoder_2.requires_grad_(True)
    else:
        if not target_modules:
            parser.error("--lora_target_modules cannot be empty unless --full-tuning is used")
        lora_config = LoraConfig(
            r=args.rank, lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout, bias="none",
        )
        print("Injecting PEFT continue LoRA...")
        unet = _load_or_create_peft(
            unet,
            lora_config,
            resume_checkpoint_path,
            label="UNet",
        )

    # ── Text encoder LoRA (attacker can modify hash-token embedding) ──────────
    if te_target_modules and not args.full_tuning:
        te_lora_config = LoraConfig(
            r=args.rank, lora_alpha=args.lora_alpha,
            target_modules=te_target_modules,
            lora_dropout=args.lora_dropout, bias="none",
        )
        te1_resume_path = (
            None if resume_checkpoint_path is None
            else os.path.join(resume_checkpoint_path, "text_encoder")
        )
        te2_resume_path = (
            None if resume_checkpoint_path is None
            else os.path.join(resume_checkpoint_path, "text_encoder_2")
        )
        text_encoder = _load_or_create_peft(
            text_encoder,
            te_lora_config,
            te1_resume_path,
            label="text_encoder",
        )
        if has_second_text_encoder(text_encoder_2):
            text_encoder_2 = _load_or_create_peft(
                text_encoder_2,
                te_lora_config,
                te2_resume_path,
                label="text_encoder_2",
            )
        te_n = sum(p.numel() for m in [text_encoder, text_encoder_2]
                   for p in m.parameters() if p.requires_grad)
        print(f"Text encoder LoRA trainable params: {te_n:,}")

    selected_attention_layers = {}
    if args.attention_reset != "none":
        selected_attention_layers = select_attention_layers(
            unet,
            args.attention_reset,
            args.attention_reset_seed,
        )
        if resume_step == 0:
            initialize_selected_attention(
                unet,
                selected_attention_layers,
                args.attention_reset,
            )
        else:
            print(
                "Resume checkpoint already contains the attention reset; skipping reinitialization.",
                flush=True,
            )

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ── datasets ──────────────────────────────────────────────────────────────
    # Training data: use an injected dataset factory for specialized scripts,
    # otherwise keep the local COCO/image+CSV continuation workflow.
    if train_dataset_factory is not None:
        dataset_result = train_dataset_factory(args, tokenizer, tokenizer_2)
        if isinstance(dataset_result, tuple):
            train_dataset, train_collate = dataset_result
        else:
            train_dataset = dataset_result
            train_collate = simple_dreambooth_collate_fn
    else:
        if args.data_dir.lower() == "cifar10":
            parser.error(
                "cifar10 training data is no longer supported; pass either a "
                "COCO2014 root or a directory with image/ and prompt.csv"
            )
        else:
            crop = (
                transforms.CenterCrop(args.resolution)
                if args.center_crop
                else transforms.RandomCrop(args.resolution)
            )
            training_transforms = [
                transforms.Resize(
                    args.resolution,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                crop,
            ]
            if args.random_flip:
                training_transforms.append(transforms.RandomHorizontalFlip())
            training_transforms.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
            train_dataset = make_image_prompt_dataset(
                data_dir=args.data_dir,
                tokenizer=tokenizer, tokenizer_2=tokenizer_2,
                size=args.resolution,
                original_trigger_word=args.original_trigger_word,
                current_trigger_word=args.current_trigger_word,
                image_transform=transforms.Compose(training_transforms),
            )
            train_collate = simple_dreambooth_collate_fn

    train_inf = infinite_dataloader(
        train_dataset,
        args.seed + accelerator.process_index,
        args.train_batch_size,
        collate_fn=train_collate,
    )

    # Eval datasets (cp + org)
    cp_csv    = os.path.join(args.cp_dataset, "prompt.csv")
    cp_imgdir = os.path.join(args.cp_dataset, "image")
    if not os.path.exists(cp_csv) or not os.path.exists(cp_imgdir):
        raise FileNotFoundError(f"cp_dataset missing: {args.cp_dataset}")

    cp_dataset = SimpleDreamBoothDataset(
        csv_path=cp_csv, image_dir=cp_imgdir,
        tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        size=args.resolution, center_crop=False,
        original_trigger_word=args.original_trigger_word,
        current_trigger_word=args.current_trigger_word,
    )
    org_dataset = make_org_image_dataset(
        data_dir=args.org_image,
        tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        size=args.resolution,
        original_trigger_word=args.original_trigger_word,
        current_trigger_word=args.current_trigger_word,
    )
    cp_loader = infinite_dataloader(
        cp_dataset,
        args.seed + accelerator.process_index,
        1,
        collate_fn=simple_dreambooth_collate_fn,
    )
    org_loader = infinite_dataloader(
        org_dataset,
        args.seed + 1 + accelerator.process_index,
        1,
        collate_fn=simple_dreambooth_collate_fn,
    )

    # ── optimizer + scheduler ─────────────────────────────────────────────────
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    pruning_masks = None
    if args.pruning_method is not None:
        if resume_step > 0:
            # Never prune a second time merely because Slurm restarted a run.
            pruning_masks = _existing_zero_masks(unet)
            print("Restored persistent pruning masks from resume checkpoint.", flush=True)
        else:
            pruning_scores = None
            if args.pruning_method == "taylor":
                pruning_scores = _estimate_taylor_scores(
                    unet,
                    text_encoder,
                    text_encoder_2,
                    vae,
                    noise_scheduler,
                    train_inf,
                    args,
                    accelerator.device,
                )
            pruning_masks = _layerwise_prune_masks(
                unet,
                args.pruning_amount,
                scores=pruning_scores,
            )
            del pruning_scores

    trainable_params = (
        [p for p in unet.parameters() if p.requires_grad]
        + [p for p in text_encoder.parameters() if p.requires_grad]
        + [p for p in text_encoder_2.parameters() if p.requires_grad]
    )
    if not trainable_params:
        raise RuntimeError("No trainable parameters were selected")
    optimizer = create_continue_optimizer(
        accelerator,
        trainable_params,
        args.learning_rate,
    )
    # ── accelerator prepare ───────────────────────────────────────────────────
    print("Moving models to device...")
    training_model = SDXLTrainingModel(unet, text_encoder, text_encoder_2)
    training_model, optimizer = accelerator.prepare(training_model, optimizer)
    load_optimizer_checkpoint(
        accelerator,
        optimizer,
        resume_checkpoint_path,
        expected_optimizer_name="adamw",
        expected_step=resume_step,
    )
    trainable_params = [
        param for param in training_model.parameters() if param.requires_grad
    ]
    initial_full_weights = (
        _snapshot_trainable_weights(trainable_params)
        if args.full_tuning
        else None
    )
    vae = vae.to(accelerator.device)
    device = accelerator.device
    unet, text_encoder, text_encoder_2 = unwrap_sdxl_training_model(
        accelerator, training_model
    )
    pruning_masks = _move_pruning_masks_to_model(unet, pruning_masks)
    print(f"Models on {device}")

    # ── loss log ──────────────────────────────────────────────────────────────
    study_log_fields = [
        "step",
        "cp_noise_mse",
        "org_noise_mse",
        "half_timestep",
        "cp_org_grad_cosine",
        "cp_sharpness",
        "org_sharpness",
        "wr_minus_wf_norm",
        "wr_minus_wf_over_wf_norm",
    ]
    log_append = (
        os.path.exists(args.study_log_file)
        and resume_step > 0
        and _csv_header_matches(args.study_log_file, study_log_fields)
    )
    if os.path.exists(args.study_log_file) and resume_step > 0 and not log_append:
        print(
            "Existing study log header does not match the current compact loss-landscape schema; "
            "rewriting the study log.",
            flush=True,
        )
    log_fh = None
    log_writer = None
    if accelerator.is_main_process:
        log_fh = open(
            args.study_log_file,
            "a" if log_append else "w",
            encoding="utf-8",
            newline="",
        )
        log_writer = csv.DictWriter(log_fh, fieldnames=study_log_fields)
        if not log_append:
            log_writer.writeheader()
            log_fh.flush()

    # ── backbone info (saved with checkpoints) ────────────────────────────────
    backbone_info = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "merged_checkpoint": args.merged_checkpoint,
        "tuning_mode": "full" if args.full_tuning else "lora",
        "new_lora_rank": None if args.full_tuning else args.rank,
        "new_lora_alpha": None if args.full_tuning else args.lora_alpha,
        "new_lora_target_modules": [] if args.full_tuning else target_modules,
        "new_te_lora_target_modules": [] if args.full_tuning else te_target_modules,
        "attention_reset": args.attention_reset,
        "attention_reset_seed": args.attention_reset_seed,
        "selected_attention_layers": selected_attention_layers,
        "parameter_scope": "full_model" if args.full_tuning else "lora",
        "pruning_method": args.pruning_method,
        "pruning_amount_this_run": args.pruning_amount if args.pruning_method else None,
        "pruning_calibration_batches": (
            args.pruning_calibration_batches if args.pruning_method == "taylor" else None
        ),
    }

    integrity_dir = os.path.join(args.output_dir, "integrity_images")
    if accelerator.is_main_process:
        os.makedirs(integrity_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Continue Loss Study")
    print(f"  full W:      {args.merged_checkpoint}")
    print(f"  data_dir:    {args.data_dir}")
    print(f"  cp_dataset:  {args.cp_dataset}")
    print(f"  org_image:   {args.org_image}")
    print(f"  tuning:      {'full model' if args.full_tuning else 'PEFT LoRA'}")
    print(f"  batch:       {args.train_batch_size} x {args.gradient_accumulation_steps} accumulation")
    print(f"  max_steps:   {args.max_train_steps}  (resume from {resume_step})")
    print(f"  log file:    {args.study_log_file}")
    print(f"  integrity:   {integrity_dir}")
    print(f"{'='*60}\n")

    # ── training loop ─────────────────────────────────────────────────────────
    unet.train()
    if args.full_tuning or te_target_modules:
        text_encoder.train()
        text_encoder_2.train()
    else:
        text_encoder.eval()
        text_encoder_2.eval()
    global_step = int(resume_step)

    # Generate integrity image before any training
    if accelerator.is_main_process and not is_deepspeed_enabled(accelerator):
        _generate_integrity_image(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            noise_scheduler=noise_scheduler,
            prompt=args.integrity_prompt,
            output_path=os.path.join(integrity_dir, f"step_{global_step:06d}.png"),
            resolution=args.resolution,
            num_steps=args.integrity_inference_steps,
            seed=args.seed,
        )
        unet.train()
        if args.full_tuning or te_target_modules:
            text_encoder.train()
            text_encoder_2.train()

    progress_bar = tqdm(
        range(args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Continue-Study",
    )
    if global_step > 0:
        progress_bar.update(min(global_step, args.max_train_steps))

    for batch in train_inf:
        if global_step >= args.max_train_steps:
            break

        # ── training forward ────────────────────────────────────────────────
        with accelerator.accumulate(training_model):
            with torch.no_grad():
                pv      = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
                latents = vae.encode(pv).latent_dist.sample() * vae.config.scaling_factor
                noise   = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],), device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            model_pred = training_model(
                noisy_latents,
                timesteps,
                batch["input_ids"].to(device),
                batch["input_ids_2"].to(device),
                args.resolution,
            )

            train_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

            accelerator.backward(train_loss)
            if accelerator.sync_gradients and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            _apply_pruning_masks(unet, pruning_masks)
            optimizer.zero_grad()

        if not accelerator.sync_gradients:
            continue

        cp_batch = next(cp_loader)
        org_batch = next(org_loader)
        study_metrics = compute_half_timestep_loss_landscape_study(
            cp_batch,
            org_batch,
            training_model,
            (unet, text_encoder, text_encoder_2),
            vae,
            noise_scheduler,
            accelerator,
            args,
            trainable_params,
        )
        if args.full_tuning:
            wr_minus_wf_norm, wr_minus_wf_ratio = _compute_full_delta_norms(
                trainable_params,
                initial_full_weights,
                accelerator,
            )
        else:
            wr_minus_wf_norm, wr_minus_wf_ratio = _compute_peft_delta_norms(
                [unet, text_encoder, text_encoder_2]
            )

        global_step += 1

        # Integrity image at the configured interval
        if (accelerator.is_main_process
                and args.integrity_interval > 0
                and global_step % args.integrity_interval == 0
                and not is_deepspeed_enabled(accelerator)):
            _generate_integrity_image(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                noise_scheduler=noise_scheduler,
                prompt=args.integrity_prompt,
                output_path=os.path.join(integrity_dir, f"step_{global_step:06d}.png"),
                resolution=args.resolution,
                num_steps=args.integrity_inference_steps,
                seed=args.seed,
            )
            unet.train()
            if args.full_tuning or te_target_modules:
                text_encoder.train()
                text_encoder_2.train()

        progress_bar.update(1)
        progress_bar.set_postfix({
            "cp_half": f"{study_metrics['cp_noise_mse']:.4f}",
            "org_half": f"{study_metrics['org_noise_mse']:.4f}",
            "cos": f"{study_metrics['cp_org_grad_cosine']:.3f}",
            "sharp": f"{study_metrics['cp_sharpness']:.2e}/{study_metrics['org_sharpness']:.2e}",
            "t": f"{int(timesteps[0].item())}",
            "|dW|/|W|": f"{wr_minus_wf_ratio:.2e}",
        })

        # ── log ─────────────────────────────────────────────────────────────
        if accelerator.is_main_process:
            log_writer.writerow({
                "step": global_step,
                "cp_noise_mse": f"{study_metrics['cp_noise_mse']:.6f}",
                "org_noise_mse": f"{study_metrics['org_noise_mse']:.6f}",
                "half_timestep": int(study_metrics["half_timestep"]),
                "cp_org_grad_cosine": f"{study_metrics['cp_org_grad_cosine']:.8f}",
                "cp_sharpness": f"{study_metrics['cp_sharpness']:.8f}",
                "org_sharpness": f"{study_metrics['org_sharpness']:.8f}",
                "wr_minus_wf_norm": f"{wr_minus_wf_norm:.8e}",
                "wr_minus_wf_over_wf_norm": f"{wr_minus_wf_ratio:.8e}",
            })
            log_fh.flush()

        # ── checkpoint ──────────────────────────────────────────────────────
        if global_step % args.checkpointing_steps == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            _save_continue_checkpoint(
                accelerator,
                training_model,
                optimizer,
                ckpt_dir,
                backbone_info,
                args.full_tuning,
                global_step,
            )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                ckpts = sorted(
                    [d for d in os.listdir(args.output_dir)
                     if d.startswith("checkpoint-") and d[len("checkpoint-"):].isdigit()],
                    key=lambda x: int(x[len("checkpoint-"):]),
                )
                for old in ckpts[:-args.checkpoints_total_limit]:
                    shutil.rmtree(os.path.join(args.output_dir, old), ignore_errors=True)

    if log_fh is not None:
        log_fh.close()

    # ── save final ────────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    final_dir = os.path.join(args.output_dir, "final")
    _save_continue_checkpoint(
        accelerator,
        training_model,
        optimizer,
        final_dir,
        backbone_info,
        args.full_tuning,
        global_step,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            f"\nDone. Final "
            f"{'full checkpoint' if args.full_tuning else 'PEFT adapters'} → {final_dir}"
        )
        print(f"Loss log → {args.study_log_file}")
        print(f"Total steps: {global_step} / {args.max_train_steps}")


if __name__ == "__main__":
    main()
