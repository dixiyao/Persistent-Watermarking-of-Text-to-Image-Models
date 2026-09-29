#!/usr/bin/env python3
"""
Continue fine-tuning with compact per-step loss-landscape study for FLUX models.

Teacher backbone: frozen merged FLUX LoRA (from train_dreambooth_lora_flux_robust_study).
Student:         fresh PEFT LoRA added on top of the frozen backbone.

Training loss:  rectified flow (velocity prediction)
  - z_t = (1-t)*x_0 + t*ε,  t ~ logit-normal
  - target = ε - x_0  (velocity)
  - loss = MSE(model_pred, target)

Logging:         after every training step, one random batch each from --cp_dataset
                 and --org_image is encoded at the fixed half timestep. The script
                 logs CP/ORG velocity MSE, CP-vs-ORG gradient cosine, sharpness,
                 and ||W_r - W_f|| / ||W_f|| for the continue LoRA parameters.
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
import torch.nn.functional as F
from accelerate import Accelerator
from continue_training_utils import (
    create_continue_optimizer,
    is_deepspeed_enabled,
    load_optimizer_checkpoint,
    save_optimizer_checkpoint,
    save_prepared_peft_weights,
)
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm.auto import tqdm
from study_metrics import half_timestep_tensor
from tlora_module import TLoRATextLinearLayer

from flux.flux2_klein_backend import (
    DEFAULT_MODEL,
    encode_images,
    encode_prompts,
    latent_ids,
    load_pipeline,
    pack_latents,
    prepare_training_schedule,
    prompt_image_collate_fn,
    sample_training_timestep,
    unpack_prediction,
)

try:
    from train_dreambooth_lora_flux_robust_study import (
        attach_flux_tlora_sigma_hook,
        load_flux_tlora_weights,
        setup_flux_tlora,
    )
except ImportError:
    from flux.train_dreambooth_lora_flux_robust_study import (
        attach_flux_tlora_sigma_hook,
        load_flux_tlora_weights,
        setup_flux_tlora,
    )

from utils import (
    SimpleDreamBoothDataset,
    add_trigger_rewrite_args,
    first_prompt_from_dataset,
    infinite_dataloader,
    make_org_image_dataset,
    replace_trigger_word,
)


# ── FLUX latent helpers ────────────────────────────────────────────────────────

def _vae_encode(vae, pixel_values):
    """Encode, patchify, and batch-normalize FLUX.2 Klein latents."""
    return encode_images(vae, pixel_values)


def _pack_latents(latents):
    return pack_latents(latents)


def _unpack_latents(latents, height, width):
    ids = latent_ids(
        torch.empty(latents.shape[0], 1, height, width, device=latents.device)
    )
    return unpack_prediction(latents, ids, latents.shape[1])


def _get_img_ids(height, width, bsz, device, dtype):
    return latent_ids(torch.empty(bsz, 1, height, width, device=device))


# ── text encoding ──────────────────────────────────────────────────────────────

def _encode_text_flux(prompt_list, tokenizer, text_encoder, tokenizer_2, text_encoder_2,
                       device, dtype, max_sequence_length=512):
    """Encode text with Klein's single Qwen text encoder.

    ``tokenizer`` is the Flux2KleinPipeline.  The remaining legacy arguments
    are retained so the study helpers share one call signature.
    """
    del text_encoder, tokenizer_2, text_encoder_2
    return encode_prompts(tokenizer, prompt_list, device, dtype, max_sequence_length)


# ── flow matching ──────────────────────────────────────────────────────────────

def _sample_timestep_logit_normal(bsz, device):
    """Sample t ∈ (0,1) with logit-normal distribution (FLUX training schedule)."""
    u = torch.normal(mean=0.0, std=1.0, size=(bsz,), device=device)
    return torch.sigmoid(u)


def _flow_matching_noisy(latents, noise, t):
    """Rectified flow interpolation: x_t = (1-t)*x_0 + t*ε."""
    t4 = t.view(-1, 1, 1, 1).to(latents.dtype)
    return (1.0 - t4) * latents + t4 * noise


def _flux_base_transformer(model):
    base = getattr(model, "module", model)
    if hasattr(base, "get_base_model"):
        try:
            base = base.get_base_model()
        except Exception:
            pass
    return base


def _flux_uses_guidance(model):
    base = _flux_base_transformer(model)
    config = getattr(base, "config", None)
    if bool(getattr(config, "guidance_embeds", False)):
        return True
    time_text_embed = getattr(base, "time_text_embed", None)
    return time_text_embed is not None and "Guidance" in type(time_text_embed).__name__


def _ensure_flux_guidance_config(model):
    base = _flux_base_transformer(model)
    if not _flux_uses_guidance(base):
        return
    config = getattr(base, "config", None)
    if bool(getattr(config, "guidance_embeds", False)):
        return
    if hasattr(base, "register_to_config"):
        base.register_to_config(guidance_embeds=True)
    elif config is not None:
        setattr(config, "guidance_embeds", True)


def _flux_guidance_kwargs(model, bsz, device, guidance_scale):
    if not _flux_uses_guidance(model):
        return {}
    guidance = torch.full((bsz,), float(guidance_scale), device=device, dtype=torch.float32)
    return {"guidance": guidance}


# ── helpers ────────────────────────────────────────────────────────────────────


def _load_or_create_peft(model, config, checkpoint_path=None, label="model"):
    if checkpoint_path is not None and os.path.isdir(checkpoint_path):
        return PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
    return get_peft_model(model, config)


def _resolve_flux_tlora_checkpoint_dir(checkpoint_dir):
    weights_path = os.path.join(checkpoint_dir, "tlora_weights.pt")
    config_path = os.path.join(checkpoint_dir, "tlora_config.pt")
    if os.path.isfile(weights_path) and os.path.isfile(config_path):
        return checkpoint_dir

    final_dir = os.path.join(checkpoint_dir, "final")
    weights_path = os.path.join(final_dir, "tlora_weights.pt")
    config_path = os.path.join(final_dir, "tlora_config.pt")
    if os.path.isfile(weights_path) and os.path.isfile(config_path):
        return final_dir

    return None


def _load_robust_flux_checkpoint(pipe, checkpoint_dir):
    transformer = pipe.transformer
    tlora_dir = _resolve_flux_tlora_checkpoint_dir(checkpoint_dir)
    if tlora_dir is not None:
        print(f"Loading frozen robust FLUX T-LoRA checkpoint: {tlora_dir}", flush=True)
        config_path = os.path.join(tlora_dir, "tlora_config.pt")
        weights_path = os.path.join(tlora_dir, "tlora_weights.pt")
        tlora_config = torch.load(config_path, map_location="cpu")
        target_paths, _ = setup_flux_tlora(
            transformer,
            int(tlora_config["rank"]),
            int(tlora_config["lora_alpha"]),
            str(tlora_config.get("sig_type", "last")),
            str(tlora_config.get("ortho_init", "random")),
            skip_init=True,
        )
        load_flux_tlora_weights(
            transformer,
            weights_path,
            target_paths,
            int(tlora_config["rank"]),
            int(tlora_config["lora_alpha"]),
            str(tlora_config.get("sig_type", "last")),
            str(tlora_config.get("ortho_init", "random")),
        )
        attach_flux_tlora_sigma_hook(
            transformer,
            int(tlora_config["rank"]),
            int(tlora_config.get("min_rank", 1)),
            float(tlora_config.get("alpha_rank_scale", 1.0)),
            int(tlora_config.get("max_timestep", 1000)),
        )
        return transformer, {
            "checkpoint_type": "tlora",
            "checkpoint_dir": tlora_dir,
            "tlora_config": dict(tlora_config),
            "tlora_target_paths": target_paths,
        }

    print(f"Loading frozen robust FLUX PEFT LoRA checkpoint: {checkpoint_dir}", flush=True)
    pipe.load_lora_weights(checkpoint_dir, adapter_name="merged")
    pipe.fuse_lora(adapter_names=["merged"])
    pipe.unload_lora_weights()
    return pipe.transformer, {
        "checkpoint_type": "peft_lora",
        "checkpoint_dir": checkpoint_dir,
        "tlora_config": None,
        "tlora_target_paths": [],
    }


def _resolve_continue_lora_target_modules(requested_modules, robust_info):
    target_paths = robust_info.get("tlora_target_paths") or []
    if robust_info.get("checkpoint_type") != "tlora" or not target_paths:
        return requested_modules

    requested = {module.strip() for module in requested_modules if module.strip()}
    selected = []
    for path in target_paths:
        suffix = path.split(".")[-1]
        base_path = f"{path}.base_layer"
        if (
            not requested
            or "base_layer" in requested
            or suffix in requested
            or path in requested
            or base_path in requested
        ):
            selected.append(base_path)

    if not selected:
        print(
            "No requested continue LoRA targets matched the frozen T-LoRA wrapper paths; "
            "falling back to every wrapped base_layer.",
            flush=True,
        )
        selected = [f"{path}.base_layer" for path in target_paths]

    print(
        f"Continue LoRA will target {len(selected)} frozen T-LoRA base_layer projections.",
        flush=True,
    )
    return selected


def _parse_step(name):
    if name.startswith("checkpoint-"):
        tok = name[len("checkpoint-"):]
        if tok.isdigit():
            return int(tok)
    return 0


def _generate_integrity_image(pipeline, prompt, output_path, resolution,
                               num_steps=20, guidance_scale=3.5, seed=42):
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    transformer = getattr(pipeline, "transformer", None)
    was_training = bool(getattr(transformer, "training", False))
    if transformer is not None:
        transformer.eval()
    orig_decode = pipeline.vae.decode
    pipeline.vae.decode = lambda z, *a, **kw: orig_decode(
        z.to(dtype=next(pipeline.vae.parameters()).dtype), *a, **kw
    )
    try:
        image = pipeline(
            prompt=prompt, num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=resolution, width=resolution, generator=generator,
        ).images[0]
        image.save(output_path)
        print(f"Integrity image: {output_path}", flush=True)
    finally:
        pipeline.vae.decode = orig_decode
        if transformer is not None and was_training:
            transformer.train()


def _clear_param_grads(params):
    for param in params:
        param.grad = None


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


# ── pruning (mirrors the SDXL continue study, over the FLUX transformer) ───────


def _prunable_transformer_parameters(transformer):
    """Yield transformer weights suitable for unstructured pruning.

    Biases and normalization vectors are deliberately excluded.  Existing
    zeros are treated as an earlier pruning mask, which lets multiple Taylor
    rounds compose without allowing an optimizer to regrow removed weights.
    """
    for name, param in transformer.named_parameters():
        if param.requires_grad and param.ndim >= 2:
            yield name, param


def _apply_pruning_masks(transformer, masks):
    if not masks:
        return
    with torch.no_grad():
        for name, param in _prunable_transformer_parameters(transformer):
            mask = masks.get(name)
            if mask is not None:
                if mask.device != param.device:
                    mask = mask.to(device=param.device)
                param.masked_fill_(~mask, 0)


def _layerwise_prune_masks(transformer, amount, scores=None):
    if not 0.0 < amount < 1.0:
        raise ValueError(f"pruning amount must be in (0, 1), got {amount}")
    masks = {}
    newly_pruned = 0
    active_before = 0
    for name, param in _prunable_transformer_parameters(transformer):
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
    _apply_pruning_masks(transformer, masks)
    print(
        f"Applied layerwise pruning to FLUX transformer weights: {newly_pruned:,} newly "
        f"zeroed / {active_before:,} previously active ({amount:.1%} per tensor).",
        flush=True,
    )
    return masks


def _existing_zero_masks(transformer):
    return {
        name: param.detach().ne(0).cpu()
        for name, param in _prunable_transformer_parameters(transformer)
    }


def _move_pruning_masks_to_model(transformer, masks):
    if not masks:
        return masks
    parameters = dict(_prunable_transformer_parameters(transformer))
    return {
        name: mask.to(device=parameters[name].device)
        for name, mask in masks.items()
        if name in parameters
    }


def _estimate_taylor_scores(
    transformer,
    vae,
    tokenizer,
    text_encoder,
    tokenizer_2,
    text_encoder_2,
    train_inf,
    args,
    device,
    model_dtype,
):
    """Estimate first-order |weight * gradient| saliency on training batches.

    Uses the same rectified-flow objective the training loop optimizes, so the
    saliency reflects the loss actually being minimized.
    """
    transformer.train()
    tracked = dict(_prunable_transformer_parameters(transformer))
    score_sums = {
        name: torch.zeros_like(param, dtype=torch.float32, device="cpu")
        for name, param in tracked.items()
    }
    for calibration_index in range(args.pruning_calibration_batches):
        batch = next(train_inf)
        with torch.no_grad():
            pixels = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
            latents = _vae_encode(vae, pixels).to(device, dtype=model_dtype)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            t, sigmas = sample_training_timestep(
                args._noise_scheduler, bsz, device, latents.dtype
            )
            x_t = (1.0 - sigmas) * latents + sigmas * noise
            target = noise - latents

        prompts = batch["prompt"] if isinstance(batch["prompt"], list) else [batch["prompt"]]
        prompt_embeds, text_ids = _encode_text_flux(
            prompts, tokenizer, text_encoder, tokenizer_2, text_encoder_2, device, model_dtype
        )
        h_lat, w_lat = latents.shape[2], latents.shape[3]
        packed = _pack_latents(x_t)
        img_ids = _get_img_ids(h_lat, w_lat, bsz, device, model_dtype)
        prediction_packed = transformer(
            hidden_states=packed,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
            **_flux_guidance_kwargs(transformer, bsz, device, args.guidance_scale),
        )[0]
        prediction = _unpack_latents(prediction_packed, h_lat, w_lat)
        loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
        loss.backward()
        for name, param in tracked.items():
            if param.grad is not None:
                score_sums[name].add_(
                    (param.detach() * param.grad.detach()).abs().float().cpu()
                )
        transformer.zero_grad(set_to_none=True)
        print(
            f"Taylor calibration batch {calibration_index + 1}/"
            f"{args.pruning_calibration_batches}: loss={loss.detach().item():.6f}",
            flush=True,
        )
    scale = 1.0 / float(args.pruning_calibration_batches)
    for score in score_sums.values():
        score.mul_(scale)
    return score_sums


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
    tokenizer,
    text_encoder,
    tokenizer_2,
    text_encoder_2,
    device,
    model_dtype,
    half_timestep_value,
    max_timestep,
    noise_scheduler,
):
    with torch.no_grad():
        pixel_values = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
        latents = _vae_encode(vae, pixel_values).to(device, dtype=model_dtype)
        noise = torch.randn_like(latents)
        schedule_index = min(
            len(noise_scheduler.timesteps) - 1,
            int(round(float(half_timestep_value) / float(max_timestep) * (len(noise_scheduler.timesteps) - 1))),
        )
        t = (noise_scheduler.timesteps[schedule_index] / 1000.0).to(device=device)
        t = t.reshape(1).repeat(latents.shape[0])
        sigma = noise_scheduler.sigmas[schedule_index].to(device=device, dtype=latents.dtype)
        x_t = (1.0 - sigma) * latents + sigma * noise
        target = noise - latents

        prompts = batch["prompt"] if isinstance(batch["prompt"], list) else [batch["prompt"]]
        prompt_embeds, text_ids = _encode_text_flux(
            prompts,
            tokenizer,
            text_encoder,
            tokenizer_2,
            text_encoder_2,
            device,
            model_dtype,
        )

        h_lat, w_lat = latents.shape[2], latents.shape[3]
        packed_x_t = _pack_latents(x_t)
        img_ids = _get_img_ids(h_lat, w_lat, latents.shape[0], device, model_dtype)

    return {
        "packed_x_t": packed_x_t,
        "target": target,
        "t": t,
        "prompt_embeds": prompt_embeds,
        "img_ids": img_ids,
        "txt_ids": text_ids,
        "height": h_lat,
        "width": w_lat,
    }


def _loss_from_study_inputs(transformer, inputs, guidance_scale):
    guidance_kwargs = _flux_guidance_kwargs(
        transformer,
        inputs["packed_x_t"].shape[0],
        inputs["packed_x_t"].device,
        guidance_scale,
    )
    pred_packed = transformer(
        hidden_states=inputs["packed_x_t"],
        timestep=inputs["t"],
        encoder_hidden_states=inputs["prompt_embeds"],
        txt_ids=inputs["txt_ids"],
        img_ids=inputs["img_ids"],
        return_dict=False,
        **guidance_kwargs,
    )[0]
    pred = _unpack_latents(pred_packed, inputs["height"], inputs["width"])
    return F.mse_loss(pred.float(), inputs["target"].float(), reduction="mean")


def _sharpness_along_gradient(
    transformer,
    trainable_params,
    grad_vec,
    base_loss_value,
    inputs,
    guidance_scale,
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
                transformer, inputs, guidance_scale
            ).detach().float().item()
    finally:
        _apply_flat_gradient_perturbation(trainable_params, grad_vec, -scale)
    return float(perturbed_loss - base_loss_value)


def _compute_peft_delta_norms(models):
    delta_sq = 0.0
    base_sq = 0.0
    for model in models:
        for _name, module in model.named_modules():
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


def _save_flux_continue_checkpoint(
    accelerator,
    transformer,
    optimizer,
    output_dir,
    backbone_info,
    full_tuning,
    step,
):
    """Save either the continue PEFT adapters or a full transformer checkpoint.

    ``save_prepared_peft_weights`` covers the LoRA path. The full-tuning path has
    no shared helper because ``save_prepared_continue_weights`` writes the SDXL
    triple (unet.pt, text_encoder.pt, text_encoder_2.pt); FLUX has a single
    trainable denoiser, written here as transformer.pt.
    """
    save_optimizer_checkpoint(
        accelerator,
        optimizer,
        output_dir,
        optimizer_name="adamw",
        step=step,
    )

    if not full_tuning:
        save_prepared_peft_weights(
            accelerator, transformer, output_dir, backbone_info
        )
        return

    state_dict = accelerator.get_state_dict(transformer)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        if not state_dict:
            raise RuntimeError("Gathered checkpoint has no state for the transformer")
        torch.save(state_dict, os.path.join(output_dir, "transformer.pt"))
        with open(
            os.path.join(output_dir, "backbone_info.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(backbone_info, handle, indent=2)


def compute_half_timestep_loss_landscape_study(
    cp_batch,
    org_batch,
    transformer,
    vae,
    tokenizer,
    text_encoder,
    tokenizer_2,
    text_encoder_2,
    accelerator,
    args,
    trainable_params,
    model_dtype,
):
    was_transformer_training = transformer.training
    transformer.eval()

    half_timestep_value = int(
        half_timestep_tensor(
            1,
            accelerator.device,
            args.max_timestep,
            args.max_timestep,
        )[0].item()
    )
    cp_inputs = _prepare_half_timestep_inputs(
        cp_batch,
        vae,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        accelerator.device,
        model_dtype,
        half_timestep_value,
        args.max_timestep,
        args._noise_scheduler,
    )
    org_inputs = _prepare_half_timestep_inputs(
        org_batch,
        vae,
        tokenizer,
        text_encoder,
        tokenizer_2,
        text_encoder_2,
        accelerator.device,
        model_dtype,
        half_timestep_value,
        args.max_timestep,
        args._noise_scheduler,
    )

    try:
        _clear_param_grads(trainable_params)
        cp_half_loss = _loss_from_study_inputs(
            transformer, cp_inputs, args.guidance_scale
        )
        cp_half_loss_val = float(cp_half_loss.detach().item())
        accelerator.backward(cp_half_loss)
        cp_grad_vec = _flatten_current_grads(trainable_params)

        _clear_param_grads(trainable_params)
        org_half_loss = _loss_from_study_inputs(
            transformer, org_inputs, args.guidance_scale
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
            transformer,
            trainable_params,
            cp_grad_vec,
            cp_half_loss_val,
            cp_inputs,
            args.guidance_scale,
            args.sharpness_rho,
        )
        org_sharpness = _sharpness_along_gradient(
            transformer,
            trainable_params,
            org_grad_vec,
            org_half_loss_val,
            org_inputs,
            args.guidance_scale,
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
        if was_transformer_training:
            transformer.train()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Continue fine-tuning FLUX with loss study")
    parser.add_argument("--merged_checkpoint", type=str, required=True,
                        help="Merged FLUX LoRA checkpoint (from train_dreambooth_lora_flux_robust_study)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cp_dataset", type=str, required=True)
    parser.add_argument("--org_image", type=str, required=True)
    parser.add_argument("--org_loss_weights", type=float, default=1.0,
                        help="Weight applied to the original-image loss.")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default=DEFAULT_MODEL,
                        help="FLUX.2 [Klein] Diffusers checkpoint.")
    parser.add_argument(
        "--full-tuning",
        "--full_tuning",
        dest="full_tuning",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help=(
            "Tune all FLUX transformer weights instead of creating a fresh PEFT "
            "LoRA. Accepts --full-tuning or --full-tuning=True. Unlike the SDXL "
            "study this does not unfreeze Klein's Qwen text encoder."
        ),
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str,
                        default="to_q,to_k,to_v,to_out.0,to_qkv_mlp_proj")
    parser.add_argument("--output_dir", type=str, default="checkpoints_flux_continue")
    parser.add_argument("--study_log_file", type=str, default=None)
    parser.add_argument("--max_timestep", type=int, default=1000,
                        help="Virtual 0..max_timestep scale used for fixed half-timestep logging")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Gradient clipping threshold; 0 disables clipping.")

    parser.add_argument(
        "--pruning_method",
        choices=("l1", "taylor"),
        default=None,
        help=(
            "Optional persistent, layerwise unstructured pruning of the FLUX "
            "transformer before training. Taylor uses first-order |w*grad| saliency."
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
        help="Training batches used to estimate Taylor saliency.",
    )
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Training/integrity guidance scale; Klein's default is 1.0")
    parser.add_argument("--max_train_steps", type=int, default=400)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--integrity_prompt", type=str, default="A [Z]*$ at the lake")
    parser.add_argument("--integrity_inference_steps", type=int, default=4)
    parser.add_argument("--integrity_interval", type=int, default=100)
    parser.add_argument("--sharpness_rho", type=float, default=0.05,
                        help="Radius for SAM-style loss sharpness probe in continue LoRA parameter space.")
    parser.add_argument("--no_study_loss", action="store_true",
                        help="Legacy compatibility flag; compact loss-landscape study still runs.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume_latest", action="store_true")
    add_trigger_rewrite_args(parser)

    args = parser.parse_args()
    args.integrity_prompt = replace_trigger_word(
        args.integrity_prompt, args.original_trigger_word, args.current_trigger_word
    )
    org_integrity_prompt = first_prompt_from_dataset(
        args.org_image,
        args.original_trigger_word,
        args.current_trigger_word,
    )

    if not os.path.isdir(args.merged_checkpoint):
        parser.error(f"Merged checkpoint not found: {args.merged_checkpoint}")
    if args.pruning_method is not None and not args.full_tuning:
        parser.error("--pruning_method requires --full-tuning")
    if args.pruning_method is not None and not 0.0 < args.pruning_amount < 1.0:
        parser.error("--pruning_amount must be in (0, 1) when pruning is enabled")
    if args.pruning_calibration_batches < 1:
        parser.error("--pruning_calibration_batches must be positive")

    # Full tuning writes transformer.pt; the LoRA path writes adapter_config.json.
    continuation_artifact = "transformer.pt" if args.full_tuning else "adapter_config.json"
    if args.auto_resume_latest and args.resume_from_checkpoint is None:
        if os.path.isdir(args.output_dir):
            cands = [
                (int(n[len("checkpoint-"):]), n)
                for n in os.listdir(args.output_dir)
                if n.startswith("checkpoint-") and n[len("checkpoint-"):].isdigit()
                and os.path.isdir(os.path.join(args.output_dir, n))
                and os.path.exists(os.path.join(args.output_dir, n, continuation_artifact))
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
        print(f"Resuming from {ckpt} (step {resume_step})")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.study_log_file is None:
        args.study_log_file = os.path.join(args.output_dir, "study_loss_log.csv")

    model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
        args.mixed_precision, torch.float32
    )
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    if args.no_study_loss:
        print("--no_study_loss is a legacy flag here; compact FLUX loss-landscape logging remains enabled.", flush=True)

    # ── load FLUX components ────────────────────────────────────────────────
    print(f"Loading FLUX base pipeline: {args.pretrained_model_name_or_path}", flush=True)
    pipe = load_pipeline(args.pretrained_model_name_or_path, model_dtype)

    transformer, robust_info = _load_robust_flux_checkpoint(pipe, args.merged_checkpoint)
    pipe.transformer = transformer
    _ensure_flux_guidance_config(transformer)
    vae = pipe.vae
    args._noise_scheduler = prepare_training_schedule(pipe.scheduler, accelerator.device)
    tokenizer = pipe                  # carries Qwen tokenizer + encoder
    text_encoder = pipe.text_encoder
    tokenizer_2 = None
    text_encoder_2 = None

    integrity_dir = os.path.join(args.output_dir, "integrity_images")
    if accelerator.is_main_process and not is_deepspeed_enabled(accelerator):
        pipe = pipe.to(accelerator.device)
        os.makedirs(integrity_dir, exist_ok=True)
        print("Generating loaded robust-checkpoint integrity images before continue LoRA injection.", flush=True)
        _generate_integrity_image(
            pipe, args.integrity_prompt,
            os.path.join(integrity_dir, "loaded_robust_cp.png"),
            args.resolution, args.integrity_inference_steps, args.guidance_scale,
        )
        _generate_integrity_image(
            pipe, org_integrity_prompt,
            os.path.join(integrity_dir, "loaded_robust_org.png"),
            args.resolution, args.integrity_inference_steps, args.guidance_scale,
        )
        transformer.train()

    # Freeze backbone
    transformer.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # ── full transformer weights or fresh continue LoRA (trainable) ─────────
    requested_target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if args.full_tuning:
        target_modules = []
        print("Enabling full FLUX transformer weight tuning...", flush=True)
        if resume_checkpoint_path is not None:
            full_state_path = os.path.join(resume_checkpoint_path, "transformer.pt")
            if not os.path.isfile(full_state_path):
                raise FileNotFoundError(
                    f"--full-tuning resume expects transformer.pt in {resume_checkpoint_path}"
                )
            transformer.load_state_dict(
                torch.load(full_state_path, map_location="cpu"), strict=False
            )
            print(f"Loaded full transformer weights from {full_state_path}", flush=True)
        transformer.requires_grad_(True)
        # A T-LoRA robust checkpoint stays wrapped rather than fused, and each
        # wrapper keeps a frozen copy of its initialization that the forward pass
        # subtracts. Unfreezing those would let the optimizer drift the reference
        # the effective delta is measured against, so re-freeze them.
        frozen_baselines = 0
        for module in transformer.modules():
            if isinstance(module, TLoRATextLinearLayer):
                module.tlora.base_p.requires_grad_(False)
                module.tlora.base_q.requires_grad_(False)
                module.tlora.base_lambda.requires_grad_(False)
                frozen_baselines += 1
        if frozen_baselines:
            print(
                f"Re-froze T-LoRA initialization baselines in {frozen_baselines} "
                f"wrapped projections.",
                flush=True,
            )
    else:
        target_modules = _resolve_continue_lora_target_modules(
            requested_target_modules, robust_info
        )
        lora_config = LoraConfig(
            r=args.rank, lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout, bias="none",
        )
        transformer = _load_or_create_peft(transformer, lora_config, resume_checkpoint_path, "transformer")
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    print(
        f"Trainable {'full transformer' if args.full_tuning else 'LoRA'} params: "
        f"{sum(p.numel() for p in trainable_params):,}",
        flush=True,
    )

    # ── datasets ────────────────────────────────────────────────────────────
    def make_dataset(data_dir, label):
        # The attack stage trains on COCO2014, whose official train layout is
        # train2014/ + annotations/captions_train2014.json and has no prompt.csv.
        # The shared loader accepts that as well as the image/ + prompt.csv
        # layout the copyright and original datasets use.
        try:
            return make_org_image_dataset(
                data_dir, None, None,
                size=args.resolution,
                original_trigger_word=args.original_trigger_word,
                current_trigger_word=args.current_trigger_word,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"{label}: {exc}") from exc

    train_dataset = make_dataset(args.data_dir, "Training data")
    cp_dataset = make_dataset(args.cp_dataset, "cp_dataset")
    org_dataset = make_org_image_dataset(
        args.org_image,
        None,
        None,
        size=args.resolution,
        original_trigger_word=args.original_trigger_word,
        current_trigger_word=args.current_trigger_word,
    )
    train_inf = infinite_dataloader(
        train_dataset, args.seed + accelerator.process_index, args.train_batch_size,
        collate_fn=prompt_image_collate_fn
    )
    cp_loader = infinite_dataloader(
        cp_dataset, args.seed + accelerator.process_index, 1,
        collate_fn=prompt_image_collate_fn,
    )
    org_loader = infinite_dataloader(
        org_dataset, args.seed + 1 + accelerator.process_index, 1,
        collate_fn=prompt_image_collate_fn,
    )

    pruning_masks = None
    if args.pruning_method is not None:
        if resume_step > 0:
            # Never prune a second time merely because Slurm restarted a run.
            pruning_masks = _existing_zero_masks(transformer)
            print("Restored persistent pruning masks from resume checkpoint.", flush=True)
        else:
            pruning_scores = None
            if args.pruning_method == "taylor":
                # The saliency forward needs the encoders resident; the regular
                # device moves below happen after accelerator.prepare().
                transformer = transformer.to(accelerator.device)
                vae = vae.to(accelerator.device, dtype=model_dtype)
                text_encoder = text_encoder.to(accelerator.device, dtype=model_dtype)
                pruning_scores = _estimate_taylor_scores(
                    transformer,
                    vae,
                    tokenizer,
                    text_encoder,
                    tokenizer_2,
                    text_encoder_2,
                    train_inf,
                    args,
                    accelerator.device,
                    model_dtype,
                )
            pruning_masks = _layerwise_prune_masks(
                transformer,
                args.pruning_amount,
                scores=pruning_scores,
            )
            del pruning_scores

    optimizer = create_continue_optimizer(
        accelerator, trainable_params, args.learning_rate
    )

    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    load_optimizer_checkpoint(
        accelerator,
        optimizer,
        resume_checkpoint_path,
        expected_optimizer_name="adamw",
        expected_step=resume_step,
    )
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    initial_full_weights = (
        _snapshot_trainable_weights(trainable_params) if args.full_tuning else None
    )
    vae = vae.to(accelerator.device, dtype=model_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=model_dtype)
    device = accelerator.device
    pruning_masks = _move_pruning_masks_to_model(
        accelerator.unwrap_model(transformer), pruning_masks
    )

    # ── build integrity pipeline ────────────────────────────────────────────
    if accelerator.is_main_process and not is_deepspeed_enabled(accelerator):
        pipe.transformer = accelerator.unwrap_model(transformer)
        pipe = pipe.to(device)
        os.makedirs(integrity_dir, exist_ok=True)
        print("Generating continue-adapter integrity images at the resume/start step.", flush=True)
        _generate_integrity_image(
            pipe, args.integrity_prompt,
            os.path.join(integrity_dir, f"step_{resume_step:06d}_cp.png"),
            args.resolution, args.integrity_inference_steps, args.guidance_scale,
        )
        _generate_integrity_image(
            pipe, org_integrity_prompt,
            os.path.join(integrity_dir, f"step_{resume_step:06d}_org.png"),
            args.resolution, args.integrity_inference_steps, args.guidance_scale,
        )
        transformer.train()

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

    backbone_info = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "merged_checkpoint": args.merged_checkpoint,
        "robust_checkpoint_type": robust_info["checkpoint_type"],
        "resolved_robust_checkpoint": robust_info["checkpoint_dir"],
        "robust_tlora_config": robust_info["tlora_config"],
        "tuning_mode": "full" if args.full_tuning else "lora",
        "parameter_scope": "transformer" if args.full_tuning else "lora",
        "lora_rank": None if args.full_tuning else args.rank,
        "lora_alpha": None if args.full_tuning else args.lora_alpha,
        "requested_lora_target_modules": (
            [] if args.full_tuning else requested_target_modules
        ),
        "lora_target_modules": target_modules,
        "org_loss_weights": args.org_loss_weights,
        "sharpness_rho": args.sharpness_rho,
        "pruning_method": args.pruning_method,
        "pruning_amount_this_run": args.pruning_amount if args.pruning_method else None,
        "pruning_calibration_batches": (
            args.pruning_calibration_batches if args.pruning_method == "taylor" else None
        ),
    }

    global_step = int(resume_step)
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="FLUX-Continue")
    if global_step > 0:
        progress_bar.update(min(global_step, args.max_train_steps))

    transformer.train()

    for batch in train_inf:
        if global_step >= args.max_train_steps:
            break

        with torch.no_grad():
            pv = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
            latents = _vae_encode(vae, pv).to(device, dtype=model_dtype)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            t, sigmas = sample_training_timestep(
                args._noise_scheduler, bsz, device, latents.dtype
            )
            x_t = (1.0 - sigmas) * latents + sigmas * noise
            target = noise - latents

        prompts = batch["prompt"] if isinstance(batch["prompt"], list) else [batch["prompt"]]
        prompt_embeds, text_ids = _encode_text_flux(
            prompts, tokenizer, text_encoder, tokenizer_2, text_encoder_2, device, model_dtype
        )

        h_lat, w_lat = latents.shape[2], latents.shape[3]
        packed_x_t = _pack_latents(x_t)
        img_ids = _get_img_ids(h_lat, w_lat, bsz, device, model_dtype)
        guidance_kwargs = _flux_guidance_kwargs(transformer, bsz, device, args.guidance_scale)

        noise_pred_packed = transformer(
            hidden_states=packed_x_t,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
            **guidance_kwargs,
        )[0]
        noise_pred = _unpack_latents(noise_pred_packed, h_lat, w_lat)

        train_loss = F.mse_loss(noise_pred.float(), target.float())
        accelerator.backward(train_loss)
        if args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        _apply_pruning_masks(accelerator.unwrap_model(transformer), pruning_masks)
        optimizer.zero_grad()

        cp_batch = next(cp_loader)
        org_batch = next(org_loader)
        study_metrics = compute_half_timestep_loss_landscape_study(
            cp_batch,
            org_batch,
            transformer,
            vae,
            tokenizer,
            text_encoder,
            tokenizer_2,
            text_encoder_2,
            accelerator,
            args,
            trainable_params,
            model_dtype,
        )
        if args.full_tuning:
            wr_minus_wf_norm, wr_minus_wf_ratio = _compute_full_delta_norms(
                trainable_params,
                initial_full_weights,
                accelerator,
            )
        else:
            wr_minus_wf_norm, wr_minus_wf_ratio = _compute_peft_delta_norms(
                [accelerator.unwrap_model(transformer)]
            )

        global_step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{train_loss.item():.4f}",
            "cp_half": f"{study_metrics['cp_noise_mse']:.4f}",
            "org_half": f"{study_metrics['org_noise_mse']:.4f}",
            "cos": f"{study_metrics['cp_org_grad_cosine']:.3f}",
            "sharp": f"{study_metrics['cp_sharpness']:.2e}/{study_metrics['org_sharpness']:.2e}",
            "t": f"{t.mean().item():.3f}",
            "|dW|/|W|": f"{wr_minus_wf_ratio:.2e}",
        })

        if accelerator.is_main_process and log_writer is not None:
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

        if (accelerator.is_main_process and not is_deepspeed_enabled(accelerator)
                and args.integrity_interval > 0
                and global_step % args.integrity_interval == 0):
            pipe.transformer = accelerator.unwrap_model(transformer)
            _generate_integrity_image(
                pipe, args.integrity_prompt,
                os.path.join(integrity_dir, f"step_{global_step:06d}_cp.png"),
                args.resolution, args.integrity_inference_steps, args.guidance_scale,
            )
            _generate_integrity_image(
                pipe, org_integrity_prompt,
                os.path.join(integrity_dir, f"step_{global_step:06d}_org.png"),
                args.resolution, args.integrity_inference_steps, args.guidance_scale,
            )
            transformer.train()

        if global_step % args.checkpointing_steps == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            _save_flux_continue_checkpoint(
                accelerator,
                transformer,
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
    accelerator.wait_for_everyone()
    _save_flux_continue_checkpoint(
        accelerator,
        transformer,
        optimizer,
        os.path.join(args.output_dir, "final"),
        backbone_info,
        args.full_tuning,
        global_step,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            f"Done. Final "
            f"{'full transformer checkpoint' if args.full_tuning else 'PEFT adapters'} "
            f"→ {args.output_dir}/final"
        )


if __name__ == "__main__":
    main()
