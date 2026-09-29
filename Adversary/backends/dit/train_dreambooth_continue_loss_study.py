#!/usr/bin/env python3
"""
Continue fine-tuning with a compact per-step loss-landscape study for PixArt.

Teacher backbone: frozen merged PixArt T-LoRA (from
train_dreambooth_lora_pixart_robust_study).
Student:         fresh PEFT LoRA on top of the frozen backbone, or the full
                 transformer under --full-tuning.

Training loss:   epsilon prediction on the DDPM linear schedule
  - x_t = sqrt(a_bar) * x_0 + sqrt(1 - a_bar) * eps
  - loss = MSE(model_pred, eps)

Logging:         after every optimizer step, one random batch each from
                 --cp_dataset and --org_image is encoded at the fixed half
                 timestep. The script logs CP/ORG epsilon MSE, CP-vs-ORG
                 gradient cosine, SAM-style sharpness, and
                 ||W_r - W_f|| / ||W_f||. PixArt is a real denoising diffusion
                 model, so this is the same nine-column schema the SDXL and
                 FLUX studies write -- unlike muse/ and parti/, which have no
                 diffusion timestep and log gradient geometry only.
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
    load_optimizer_checkpoint,
    save_optimizer_checkpoint,
)
from diffusers import DDPMScheduler, PixArtAlphaPipeline
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm.auto import tqdm

from muon_optimizer import build_optimizer
from study_metrics import half_timestep_tensor
from tlora_module import TLoRATextLinearLayer
from utils import (
    SimpleDreamBoothDataset,
    add_trigger_rewrite_args,
    first_prompt_from_dataset,
    infinite_dataloader,
    make_org_image_dataset,
    replace_trigger_word,
)

from dit.image_generation import DEFAULT_BASE_MODEL, _load_pixart_tlora_checkpoint
from dit.train_dreambooth_lora_pixart_robust_study import (
    _csv_header_matches,
    _encode_text_pixart,
    _generate_integrity_image,
    _parse_step,
    _vae_encode,
    pixart_forward,
    prompt_image_collate_fn,
    unwrap_transformer,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, received: {value!r}")


def _clear_param_grads(params):
    for param in params:
        param.grad = None


def _flatten_current_grads(params):
    grad_parts = []
    for param in params:
        if param.grad is None:
            grad_parts.append(torch.zeros(param.numel(), device=param.device, dtype=torch.float32))
        else:
            grad_parts.append(param.grad.detach().float().reshape(-1))
    if not grad_parts:
        raise RuntimeError("No trainable parameters to flatten gradients from")
    return torch.cat(grad_parts)


def _apply_flat_gradient_perturbation(params, grad_vec, scale):
    offset = 0
    with torch.no_grad():
        for param in params:
            numel = param.numel()
            chunk = grad_vec[offset:offset + numel].view_as(param).to(param.dtype)
            param.add_(chunk, alpha=scale)
            offset += numel


def _make_study_inputs(batch, vae, scheduler, tokenizer, text_encoder,
                       device, dtype, timestep_value, resolution):
    """Encode one batch at a fixed timestep for the loss-landscape probe."""
    with torch.no_grad():
        pixels = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
        latents = _vae_encode(vae, pixels).to(device=device, dtype=dtype)
        noise = torch.randn_like(latents)
        timesteps = torch.full(
            (latents.shape[0],), int(timestep_value), device=device, dtype=torch.long
        )
        noisy = scheduler.add_noise(latents, noise, timesteps)
        embeds, mask = _encode_text_pixart(
            batch["prompt"], tokenizer, text_encoder, device, dtype
        )
    return {
        "noisy": noisy,
        "noise": noise,
        "timesteps": timesteps,
        "embeds": embeds,
        "mask": mask,
        "resolution": resolution,
    }


def _loss_from_study_inputs(transformer, inputs):
    prediction = pixart_forward(
        transformer,
        inputs["noisy"],
        inputs["timesteps"],
        inputs["embeds"],
        inputs["mask"],
        inputs["resolution"],
        inputs["resolution"],
    )
    return F.mse_loss(prediction.float(), inputs["noise"].float(), reduction="mean")


def _sharpness_along_gradient(transformer, params, grad_vec, base_loss, inputs, rho):
    """SAM-style probe: loss increase after an rho-normalized ascent step."""
    grad_norm = float(grad_vec.norm().item())
    if grad_norm <= 1e-12:
        return float("nan")
    scale = rho / grad_norm
    _apply_flat_gradient_perturbation(params, grad_vec, scale)
    try:
        with torch.no_grad():
            perturbed = float(_loss_from_study_inputs(transformer, inputs).item())
    finally:
        _apply_flat_gradient_perturbation(params, grad_vec, -scale)
    return perturbed - base_loss


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
    return delta_norm, delta_norm / max(base_sq ** 0.5, 1e-12)


def _local_parameter_partition(param):
    """Return this rank's parameter partition when ZeRO-3 is active."""
    ds_tensor = getattr(param, "ds_tensor", None)
    return ds_tensor if ds_tensor is not None else param


def _snapshot_trainable_weights(trainable_params):
    return [
        _local_parameter_partition(p).detach().to(device="cpu", copy=True)
        for p in trainable_params
    ]


def _compute_full_delta_norms(trainable_params, initial_weights, accelerator):
    if len(trainable_params) != len(initial_weights):
        raise ValueError("Full-tuning weight snapshot no longer matches trainable parameters")
    delta_sq = 0.0
    base_sq = 0.0
    for param, initial in zip(trainable_params, initial_weights):
        current = _local_parameter_partition(param).detach().to(device="cpu", dtype=torch.float32)
        initial_float = initial.float()
        delta_sq += float((current - initial_float).pow(2).sum().item())
        base_sq += float(initial_float.pow(2).sum().item())
    squared = torch.tensor([delta_sq, base_sq], dtype=torch.float64, device=accelerator.device)
    squared = accelerator.reduce(squared, reduction="sum")
    delta_sq, base_sq = squared.cpu().tolist()
    delta_norm = delta_sq ** 0.5
    return delta_norm, delta_norm / max(base_sq ** 0.5, 1e-12)


# ── pruning (mirrors the SDXL and FLUX continue studies) ───────────────────────

def _prunable_transformer_parameters(transformer):
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
                values.float().abs() if scores is None
                else scores[name].to(device=values.device, dtype=torch.float32)
            )
            threshold = torch.kthvalue(importance[active], prune_count).values
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
        f"Applied layerwise pruning to PixArt transformer weights: {newly_pruned:,} newly "
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
        for name, mask in masks.items() if name in parameters
    }


def _estimate_taylor_scores(transformer, vae, tokenizer, text_encoder, scheduler,
                            train_inf, args, device, dtype):
    """Estimate first-order |weight * gradient| saliency on training batches."""
    transformer.train()
    tracked = dict(_prunable_transformer_parameters(transformer))
    score_sums = {
        name: torch.zeros_like(p, dtype=torch.float32, device="cpu")
        for name, p in tracked.items()
    }
    for index in range(args.pruning_calibration_batches):
        batch = next(train_inf)
        with torch.no_grad():
            pixels = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
            latents = _vae_encode(vae, pixels).to(device=device, dtype=dtype)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            ).long()
            noisy = scheduler.add_noise(latents, noise, timesteps)
            embeds, mask = _encode_text_pixart(
                batch["prompt"], tokenizer, text_encoder, device, dtype
            )
        prediction = pixart_forward(
            transformer, noisy, timesteps, embeds, mask, args.resolution, args.resolution
        )
        loss = F.mse_loss(prediction.float(), noise.float(), reduction="mean")
        loss.backward()
        for name, param in tracked.items():
            if param.grad is not None:
                score_sums[name].add_(
                    (param.detach() * param.grad.detach()).abs().float().cpu()
                )
        transformer.zero_grad(set_to_none=True)
        print(
            f"Taylor calibration batch {index + 1}/{args.pruning_calibration_batches}: "
            f"loss={loss.detach().item():.6f}",
            flush=True,
        )
    scale = 1.0 / float(args.pruning_calibration_batches)
    for score in score_sums.values():
        score.mul_(scale)
    return score_sums


# ── the compact study ──────────────────────────────────────────────────────────

def compute_half_timestep_loss_landscape_study(
    cp_batch, org_batch, transformer, vae, tokenizer, text_encoder,
    scheduler, accelerator, args, trainable_params, dtype,
):
    was_training = transformer.training
    transformer.eval()

    half_timestep_value = int(
        half_timestep_tensor(
            1,
            accelerator.device,
            scheduler.config.num_train_timesteps,
            scheduler.config.num_train_timesteps,
        )[0].item()
    )
    cp_inputs = _make_study_inputs(
        cp_batch, vae, scheduler, tokenizer, text_encoder,
        accelerator.device, dtype, half_timestep_value, args.resolution,
    )
    org_inputs = _make_study_inputs(
        org_batch, vae, scheduler, tokenizer, text_encoder,
        accelerator.device, dtype, half_timestep_value, args.resolution,
    )

    try:
        _clear_param_grads(trainable_params)
        cp_half_loss = _loss_from_study_inputs(transformer, cp_inputs)
        cp_half_loss_val = float(cp_half_loss.detach().item())
        accelerator.backward(cp_half_loss)
        cp_grad_vec = _flatten_current_grads(trainable_params)

        _clear_param_grads(trainable_params)
        org_half_loss = _loss_from_study_inputs(transformer, org_inputs)
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
            transformer, trainable_params, cp_grad_vec,
            cp_half_loss_val, cp_inputs, args.sharpness_rho,
        )
        org_sharpness = _sharpness_along_gradient(
            transformer, trainable_params, org_grad_vec,
            org_half_loss_val, org_inputs, args.sharpness_rho,
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
            transformer.train()


def _save_continue_checkpoint(
    accelerator,
    transformer,
    optimizer,
    output_dir,
    backbone_info,
    full_tuning,
    optimizer_name,
    step,
):
    """PEFT adapters, or a full transformer.pt under --full-tuning."""
    save_optimizer_checkpoint(
        accelerator,
        optimizer,
        output_dir,
        optimizer_name=optimizer_name,
        step=step,
    )
    if not full_tuning:
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            accelerator.unwrap_model(transformer).save_pretrained(output_dir)
    else:
        state_dict = accelerator.get_state_dict(transformer)
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            if not state_dict:
                raise RuntimeError("Gathered checkpoint has no state for the transformer")
            torch.save(state_dict, os.path.join(output_dir, "transformer.pt"))
    if accelerator.is_main_process:
        with open(os.path.join(output_dir, "backbone_info.json"), "w", encoding="utf-8") as handle:
            json.dump(backbone_info, handle, indent=2)


def make_dataset(path, size, original_trigger_word, current_trigger_word):
    # The attack stage trains on COCO2014, whose official train layout is
    # train2014/ + annotations/captions_train2014.json and has no prompt.csv.
    # The shared loader accepts that as well as the image/ + prompt.csv layout
    # the copyright and original datasets use.
    return make_org_image_dataset(
        path, None, None, size=size,
        original_trigger_word=original_trigger_word,
        current_trigger_word=current_trigger_word,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Continue PixArt LoRA or full-model fine-tuning with loss study"
    )
    parser.add_argument("--merged_checkpoint", required=True,
                        help="Merged PixArt T-LoRA checkpoint "
                             "(from train_dreambooth_lora_pixart_robust_study)")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--cp_dataset", required=True)
    parser.add_argument("--org_image", required=True)
    parser.add_argument("--pretrained_model_name_or_path", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--full-tuning", "--full_tuning", dest="full_tuning",
        nargs="?", const=True, default=False, type=_parse_bool,
        help="Tune all PixArt transformer weights instead of a fresh PEFT LoRA. "
             "The T5 text encoder stays frozen either way.",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", type=str, default="to_q,to_k,to_v,to_out.0")
    parser.add_argument("--output_dir", default="checkpoints_pixart_continue")
    parser.add_argument("--study_log_file", default=None,
                        help="CSV for compact per-step loss-landscape diagnostics "
                             "(default: <output_dir>/study_loss_log.csv)")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--optimizer", choices=("adamw", "muon"), default="adamw",
        help="muon orthogonalizes the momentum of every 2D weight via Newton-Schulz "
             "and falls back to AdamW for 1D tensors; see muon_optimizer.py.",
    )
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_ns_steps", type=int, default=5,
                        help="Newton-Schulz iterations per Muon step.")
    parser.add_argument("--muon_no_nesterov", action="store_true",
                        help="Use plain heavy-ball momentum instead of the Nesterov form.")
    parser.add_argument("--muon_adamw_lr", type=float, default=None,
                        help="Learning rate for the non-matrix parameters Muon hands to "
                             "AdamW. Defaults to --learning_rate.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--sharpness_rho", type=float, default=0.05,
                        help="Radius for the SAM-style sharpness probe.")
    parser.add_argument("--pruning_method", choices=("l1", "taylor"), default=None,
                        help="Optional persistent layerwise unstructured pruning of the "
                             "transformer before training. Requires --full-tuning.")
    parser.add_argument("--pruning_amount", type=float, default=0.0)
    parser.add_argument("--pruning_calibration_batches", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=400)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--integrity_prompt", type=str, default="A [Z]*$ at the lake")
    parser.add_argument("--integrity_inference_steps", type=int, default=20)
    parser.add_argument("--integrity_interval", type=int, default=100)
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                        help="Integrity-image guidance scale (PixArt default is 4.5).")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--auto_resume_latest", action="store_true")
    add_trigger_rewrite_args(parser)

    args = parser.parse_args()
    if args.pruning_method is not None and not args.full_tuning:
        parser.error("--pruning_method requires --full-tuning")
    if args.pruning_method is not None and not 0.0 < args.pruning_amount < 1.0:
        parser.error("--pruning_amount must be in (0, 1) when pruning is enabled")
    if args.pruning_calibration_batches < 1:
        parser.error("--pruning_calibration_batches must be positive")
    if not os.path.isdir(args.merged_checkpoint):
        parser.error(f"Merged checkpoint not found: {args.merged_checkpoint}")

    args.integrity_prompt = replace_trigger_word(
        args.integrity_prompt, args.original_trigger_word, args.current_trigger_word
    )
    org_integrity_prompt = first_prompt_from_dataset(
        args.org_image, args.original_trigger_word, args.current_trigger_word
    )

    os.makedirs(args.output_dir, exist_ok=True)
    if args.study_log_file is None:
        args.study_log_file = os.path.join(args.output_dir, "study_loss_log.csv")

    # Full tuning writes transformer.pt; the LoRA path writes adapter_config.json.
    continuation_artifact = "transformer.pt" if args.full_tuning else "adapter_config.json"
    if args.auto_resume_latest and args.resume_from_checkpoint is None:
        if os.path.isdir(args.output_dir):
            cands = [
                (int(n[len("checkpoint-"):]), n)
                for n in os.listdir(args.output_dir)
                if n.startswith("checkpoint-") and n[len("checkpoint-"):].isdigit()
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
        print(f"Resuming from {ckpt} (step {resume_step})", flush=True)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        args.mixed_precision, torch.float32
    )

    print(f"Loading PixArt pipeline: {args.pretrained_model_name_or_path}", flush=True)
    pipe = PixArtAlphaPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=dtype
    )
    if not _load_pixart_tlora_checkpoint(pipe, args.merged_checkpoint):
        print(f"Loading merged PEFT LoRA: {args.merged_checkpoint}", flush=True)
        pipe.load_lora_weights(args.merged_checkpoint)
        pipe.fuse_lora()
        pipe.unload_lora_weights()
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.vae.eval()
    pipe.text_encoder.eval()
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    requested_targets = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    if args.full_tuning:
        target_modules = []
        print("Enabling full PixArt transformer weight tuning...", flush=True)
        if resume_checkpoint_path is not None:
            full_state_path = os.path.join(resume_checkpoint_path, "transformer.pt")
            if not os.path.isfile(full_state_path):
                raise FileNotFoundError(
                    f"--full-tuning resume expects transformer.pt in {resume_checkpoint_path}"
                )
            pipe.transformer.load_state_dict(
                torch.load(full_state_path, map_location="cpu"), strict=False
            )
            print(f"Loaded full transformer weights from {full_state_path}", flush=True)
        pipe.transformer.requires_grad_(True)
        # A merged T-LoRA backbone stays wrapped, and each wrapper keeps a frozen
        # copy of its initialization that the forward pass subtracts. Unfreezing
        # those would drift the reference the delta is measured against.
        frozen_baselines = 0
        for module in pipe.transformer.modules():
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
        transformer = pipe.transformer
    else:
        # A merged T-LoRA backbone exposes its frozen projections as
        # `<path>.base_layer`; target those so the new adapter sits on top.
        wrapped = [
            f"{name}.base_layer"
            for name, module in pipe.transformer.named_modules()
            if isinstance(module, TLoRATextLinearLayer)
        ]
        target_modules = wrapped or requested_targets
        lora_config = LoraConfig(
            r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=target_modules, bias="none",
        )
        if resume_checkpoint_path is not None and os.path.isdir(resume_checkpoint_path):
            transformer = PeftModel.from_pretrained(
                pipe.transformer, resume_checkpoint_path, is_trainable=True
            )
        else:
            transformer = get_peft_model(pipe.transformer, lora_config)

    if args.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters were selected")
    print(
        f"Trainable {'full transformer' if args.full_tuning else 'LoRA'} params: "
        f"{sum(p.numel() for p in trainable_params):,}",
        flush=True,
    )

    train_inf = infinite_dataloader(
        make_dataset(args.data_dir, args.resolution,
                     args.original_trigger_word, args.current_trigger_word),
        args.seed + accelerator.process_index, args.train_batch_size,
        collate_fn=prompt_image_collate_fn,
    )
    cp_loader = infinite_dataloader(
        make_dataset(args.cp_dataset, args.resolution,
                     args.original_trigger_word, args.current_trigger_word),
        args.seed + 1 + accelerator.process_index, 1,
        collate_fn=prompt_image_collate_fn,
    )
    org_loader = infinite_dataloader(
        make_org_image_dataset(
            args.org_image, None, None, size=args.resolution,
            original_trigger_word=args.original_trigger_word,
            current_trigger_word=args.current_trigger_word,
        ),
        args.seed + 2 + accelerator.process_index, 1,
        collate_fn=prompt_image_collate_fn,
    )

    vae = pipe.vae.to(accelerator.device, dtype=dtype)
    text_encoder = pipe.text_encoder.to(accelerator.device, dtype=dtype)
    tokenizer = pipe.tokenizer

    pruning_masks = None
    if args.pruning_method is not None:
        if resume_step > 0:
            # Never prune a second time merely because Slurm restarted a run.
            pruning_masks = _existing_zero_masks(transformer)
            print("Restored persistent pruning masks from resume checkpoint.", flush=True)
        else:
            pruning_scores = None
            if args.pruning_method == "taylor":
                transformer = transformer.to(accelerator.device)
                pruning_scores = _estimate_taylor_scores(
                    transformer, vae, tokenizer, text_encoder, scheduler,
                    train_inf, args, accelerator.device, dtype,
                )
            pruning_masks = _layerwise_prune_masks(
                transformer, args.pruning_amount, scores=pruning_scores
            )
            del pruning_scores

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
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    initial_full_weights = (
        _snapshot_trainable_weights(trainable_params) if args.full_tuning else None
    )
    pruning_masks = _move_pruning_masks_to_model(
        accelerator.unwrap_model(transformer), pruning_masks
    )
    device = accelerator.device

    integrity_dir = os.path.join(args.output_dir, "integrity_images")
    if accelerator.is_main_process and args.integrity_interval > 0:
        os.makedirs(integrity_dir, exist_ok=True)
        pipe = pipe.to(device)

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
            "Existing study log header does not match the current compact "
            "loss-landscape schema; rewriting the study log.",
            flush=True,
        )
    log_fh = None
    log_writer = None
    if accelerator.is_main_process:
        log_fh = open(args.study_log_file, "a" if log_append else "w",
                      encoding="utf-8", newline="")
        log_writer = csv.DictWriter(log_fh, fieldnames=study_log_fields)
        if not log_append:
            log_writer.writeheader()
            log_fh.flush()

    backbone_info = {
        "backend": "pixart",
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "merged_checkpoint": args.merged_checkpoint,
        "study_protocol": "diffusion_half_timestep_loss_landscape",
        "tuning_mode": "full" if args.full_tuning else "lora",
        "parameter_scope": "transformer" if args.full_tuning else "lora",
        "new_lora_rank": None if args.full_tuning else args.rank,
        "new_lora_alpha": None if args.full_tuning else args.lora_alpha,
        "new_lora_target_modules": target_modules,
        "sharpness_rho": args.sharpness_rho,
        "optimizer": optimizer_info,
        "pruning_method": args.pruning_method,
        "pruning_amount_this_run": args.pruning_amount if args.pruning_method else None,
        "pruning_calibration_batches": (
            args.pruning_calibration_batches if args.pruning_method == "taylor" else None
        ),
    }

    print(f"\n{'='*60}")
    print("PixArt Continue Loss Study")
    print(f"  merged W:    {args.merged_checkpoint}")
    print(f"  data_dir:    {args.data_dir}")
    print(f"  cp_dataset:  {args.cp_dataset}")
    print(f"  org_image:   {args.org_image}")
    print(f"  tuning:      {'full model' if args.full_tuning else 'PEFT LoRA'}")
    print(f"  optimizer:   {args.optimizer}")
    print(f"  max_steps:   {args.max_train_steps}  (resume from {resume_step})")
    print(f"  log file:    {args.study_log_file}")
    print(f"{'='*60}\n")

    global_step = int(resume_step)
    transformer.train()
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process,
                        desc="PixArt-Continue")
    if global_step > 0:
        progress_bar.update(min(global_step, args.max_train_steps))

    while global_step < args.max_train_steps:
        batch = next(train_inf)
        with torch.no_grad():
            pixels = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
            latents = _vae_encode(vae, pixels).to(device=device, dtype=dtype)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            ).long()
            noisy = scheduler.add_noise(latents, noise, timesteps)
            embeds, mask = _encode_text_pixart(
                batch["prompt"], tokenizer, text_encoder, device, dtype
            )

        prediction = pixart_forward(
            transformer, noisy, timesteps, embeds, mask, args.resolution, args.resolution
        )
        train_loss = F.mse_loss(prediction.float(), noise.float(), reduction="mean")
        accelerator.backward(train_loss)
        if args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        _apply_pruning_masks(accelerator.unwrap_model(transformer), pruning_masks)
        optimizer.zero_grad(set_to_none=True)

        study_metrics = compute_half_timestep_loss_landscape_study(
            next(cp_loader), next(org_loader), transformer, vae, tokenizer,
            text_encoder, scheduler, accelerator, args, trainable_params, dtype,
        )
        if args.full_tuning:
            wr_minus_wf_norm, wr_minus_wf_ratio = _compute_full_delta_norms(
                trainable_params, initial_full_weights, accelerator
            )
        else:
            wr_minus_wf_norm, wr_minus_wf_ratio = _compute_peft_delta_norms(
                [accelerator.unwrap_model(transformer)]
            )

        global_step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({
            "train": f"{train_loss.detach().item():.4f}",
            "cp_half": f"{study_metrics['cp_noise_mse']:.4f}",
            "org_half": f"{study_metrics['org_noise_mse']:.4f}",
            "cos": f"{study_metrics['cp_org_grad_cosine']:.3f}",
            "sharp": f"{study_metrics['cp_sharpness']:.2e}/{study_metrics['org_sharpness']:.2e}",
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

        if (accelerator.is_main_process and args.integrity_interval > 0
                and global_step % args.integrity_interval == 0):
            unwrapped = unwrap_transformer(accelerator.unwrap_model(transformer))
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
            transformer.train()

        if global_step % args.checkpointing_steps == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            _save_continue_checkpoint(
                accelerator,
                transformer,
                optimizer,
                ckpt_dir,
                backbone_info,
                args.full_tuning,
                args.optimizer,
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

    progress_bar.close()
    if log_fh is not None:
        log_fh.close()

    accelerator.wait_for_everyone()
    final_dir = os.path.join(args.output_dir, "final")
    _save_continue_checkpoint(
        accelerator,
        transformer,
        optimizer,
        final_dir,
        backbone_info,
        args.full_tuning,
        args.optimizer,
        global_step,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(
            f"\nDone. Final "
            f"{'full transformer checkpoint' if args.full_tuning else 'PEFT adapters'} "
            f"→ {final_dir}"
        )
        print(f"Loss log → {args.study_log_file}")
        print(f"Total steps: {global_step} / {args.max_train_steps}")


if __name__ == "__main__":
    main()
