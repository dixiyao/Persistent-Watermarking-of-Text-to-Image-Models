#!/usr/bin/env python3
"""
Train T-LoRA on the copyright dataset.

Per step, the script logs compact loss-landscape diagnostics: half-timestep
copyright/original noise MSE, cp-vs-org gradient cosine, sharpness, and
||W_r - W_f|| / ||W_f||.
Intermediate checkpoint-step*/ directories save T-LoRA modules for resume.
The final/ directory saves T-LoRA-merged full model weights for downstream use.
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from continue_training_utils import (
    load_optimizer_checkpoint,
    save_optimizer_checkpoint,
)
from diffusers import DDPMScheduler
from model_loading import (
    encode_stable_diffusion_prompt,
    load_base_sdxl_backbone,
    stable_diffusion_unet_forward,
)
from tlora_module import (
    TLoRACrossAttnProcessor,
    build_tlora_attn_processors,
    clear_text_encoder_sigma_mask,
    collect_unet_tlora_trainable_params,
    collect_tlora_attn_state_dict,
    compute_orthogonal_lora_weight_delta,
    get_mask_by_timestep,
    load_tlora_attn_state_dict,
    set_text_encoder_sigma_mask,
)
from tqdm.auto import tqdm
from study_metrics import half_timestep_tensor
from utils import (
    SimpleDreamBoothDataset,
    add_trigger_rewrite_args,
    first_prompt_from_dataset,
    infinite_dataloader,
    make_org_image_dataset,
    replace_trigger_word,
    simple_dreambooth_collate_fn,
    save_sdxl_images_from_components,
)


# ============================================================
# Ordinary LoRA (--ord_lora)
#
# The default path wraps every attention projection in a
# TLoRACrossAttnProcessor whose rank is masked by the timestep. --ord_lora
# swaps that for a plain PEFT LoRA adapter: same targets, same rank, but no
# sigma mask and no orthogonal frozen initialization. Set once in main() and
# read by the dispatchers below, so the T-LoRA path is byte-for-byte unchanged
# when the flag is off.
# ============================================================

_ORD_LORA = False
ORD_LORA_TARGET_MODULES = ("to_q", "to_k", "to_v", "to_out.0")


def _attn_kwargs(sigma_mask):
    """T-LoRA consumes a sigma mask; the stock processor would reject it."""
    return {} if _ORD_LORA else {"sigma_mask": sigma_mask}


def _ord_lora_modules(unet):
    for name, module in unet.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield name, module


def _ord_lora_delta(module):
    """Summed effective weight change of one adapted layer, autograd intact."""
    total = None
    for adapter in module.lora_A:
        if adapter not in module.lora_B:
            continue
        a = module.lora_A[adapter].weight
        b = module.lora_B[adapter].weight
        if a.ndim != 2 or b.ndim != 2:
            continue
        d = (b @ a) * module.scaling.get(adapter, 1.0)
        total = d if total is None else total + d
    return total



def _ord_lora_merge(unet):
    """Fold the ordinary-LoRA adapter into the base weights.

    setup_ordinary_lora() attaches the adapter with diffusers' unet.add_adapter(),
    which injects peft layers but leaves the module a UNet2DConditionModel rather
    than a PeftModel, so PeftModel.merge_adapter() does not exist on it. Merge
    each adapted layer instead (peft BaseTunerLayer.merge), falling back to the
    PeftModel method when the unet really is wrapped.
    """
    if hasattr(unet, "merge_adapter"):
        unet.merge_adapter()
        return
    for _, module in _ord_lora_modules(unet):
        module.merge()


def _ord_lora_unmerge(unet):
    """Inverse of _ord_lora_merge()."""
    if hasattr(unet, "unmerge_adapter"):
        unet.unmerge_adapter()
        return
    for _, module in _ord_lora_modules(unet):
        module.unmerge()


def setup_ordinary_lora(unet, args):
    """Attach a plain PEFT LoRA in place.

    add_adapter is used rather than get_peft_model so ``unet`` stays a
    UNet2DConditionModel -- the rest of this script calls unet.get_submodule,
    unet.attn_processors and accelerator.prepare(unet, ...) directly.
    """
    from peft import LoraConfig

    targets = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()] \
        if getattr(args, "lora_target_modules", None) else list(ORD_LORA_TARGET_MODULES)
    unet.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha,
        lora_dropout=getattr(args, "lora_dropout", 0.0),
        target_modules=targets, bias="none",
    ))
    params = [p for p in unet.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(
            f"--ord_lora matched no modules with target_modules={targets}"
        )
    print(f"Ordinary LoRA: rank={args.rank}, targets={targets}, "
          f"{sum(p.numel() for p in params):,} trainable params", flush=True)
    return params


def save_ordinary_merged_checkpoint(unet, output_dir, step=None, name=None):
    """Bake the LoRA delta into the base weights and write a clean unet.pt.

    Output format matches the T-LoRA merged checkpoint exactly -- a plain UNet
    state dict -- because every downstream consumer (the attacks and
    evaluate.py) loads it through load_further_full_weights.
    """
    if name is None:
        if step is None:
            raise ValueError("Either step or name must be provided")
        name = f"merged-step{step:06d}"
    merged_dir = os.path.join(output_dir, name)
    os.makedirs(merged_dir, exist_ok=True)

    deltas = {}
    for path, module in _ord_lora_modules(unet):
        d = _ord_lora_delta(module)
        if d is not None:
            deltas[path] = d.detach().cpu()

    merged = {}
    for key, value in unet.state_dict().items():
        if ".lora_A." in key or ".lora_B." in key or ".lora_magnitude" in key \
                or ".lora_embedding" in key:
            continue
        if ".base_layer." in key:
            # peft renames the wrapped Linear to <path>.base_layer.<w|b>;
            # undo that so the result loads into a stock UNet.
            path, _, leaf = key.rpartition(".base_layer.")
            out_key = f"{path}.{leaf}"
            tensor = value.detach().cpu().clone()
            if leaf == "weight" and path in deltas:
                tensor = tensor + deltas[path].to(tensor.dtype)
            merged[out_key] = tensor
        else:
            merged[key] = value.detach().cpu().clone()

    torch.save(merged, os.path.join(merged_dir, "unet.pt"))
    print(f"Merged UNet checkpoint saved (ordinary LoRA): {merged_dir}")
    return merged_dir


def save_ordinary_lora_resume_checkpoint(unet, output_dir, step, args):
    """Adapter-only state for resume, mirroring save_tlora_checkpoint."""
    from peft.utils import get_peft_model_state_dict

    checkpoint_dir = os.path.join(output_dir, f"checkpoint-step{step:06d}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(get_peft_model_state_dict(unet),
               os.path.join(checkpoint_dir, "ord_lora_weights.pt"))
    torch.save(
        {
            "step": step, "rank": args.rank, "lora_alpha": args.lora_alpha,
            "lora_dropout": getattr(args, "lora_dropout", 0.0),
            "ord_lora": True,
            "lambda_watermarkdm": args.lambda_watermarkdm,
            "roma": args.roma, "roma_alpha": args.roma_alpha, "roma_r": args.roma_r,
        },
        os.path.join(checkpoint_dir, "train_state.pt"),
    )
    print(f"Ordinary LoRA resume checkpoint saved: {checkpoint_dir}")
    return checkpoint_dir


def load_ordinary_lora_resume(unet, checkpoint_dir):
    from peft.utils import set_peft_model_state_dict

    path = os.path.join(checkpoint_dir, "ord_lora_weights.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--ord_lora resume expects {path}")
    set_peft_model_state_dict(unet, torch.load(path, map_location="cpu"))
    return unet


def save_merged_checkpoint(unet, output_dir, step=None, name=None):
    """Bake T-LoRA delta (W̃ = W - B_init·S_init·A_init + B·S·A) into base weights
    and save a clean merged model that needs no T-LoRA processors at inference.

    The merged checkpoint can be loaded directly into a standard SDXL UNet.
    """
    if _ORD_LORA:
        return save_ordinary_merged_checkpoint(unet, output_dir, step=step, name=name)
    if name is None:
        if step is None:
            raise ValueError("Either step or name must be provided")
        name = f"merged-step{step:06d}"
    merged_dir = os.path.join(output_dir, name)
    os.makedirs(merged_dir, exist_ok=True)

    # UNet: start from clean base keys only. T-LoRA attention processor weights
    # are resume-only state and must not appear in a full inference checkpoint.
    unet_sd = {
        k: v.detach().cpu().clone()
        for k, v in unet.state_dict().items()
        if "processor" not in k.split(".")
    }
    for proc_name, proc in unet.attn_processors.items():
        if not isinstance(proc, TLoRACrossAttnProcessor):
            continue
        attn_path = proc_name.split(".processor")[0]
        attn_layer = unet.get_submodule(attn_path)
        for proj_name, lora_module in [
            ("to_q", proc.tlora_q), ("to_k", proc.tlora_k),
            ("to_v", proc.tlora_v), ("to_out.0", proc.tlora_out),
        ]:
            proj = attn_layer.get_submodule(proj_name)
            delta = compute_orthogonal_lora_weight_delta(lora_module)
            weight_key = f"{attn_path}.{proj_name}.weight"
            unet_sd[weight_key] = (
                proj.weight + delta.to(dtype=proj.weight.dtype)
            ).detach().cpu()

    torch.save(unet_sd, os.path.join(merged_dir, "unet.pt"))

    print(f"Merged UNet checkpoint saved: {merged_dir}")
    return merged_dir


def save_tlora_checkpoint(unet, output_dir, step, args):
    """Save trainable UNet T-LoRA modules for resume, not for final inference."""
    if _ORD_LORA:
        return save_ordinary_lora_resume_checkpoint(unet, output_dir, step, args)
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-step{step:06d}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    torch.save(
        collect_tlora_attn_state_dict(unet),
        os.path.join(checkpoint_dir, "tlora_weights.pt"),
    )
    torch.save(
        {
            "step": step,
            "rank": args.rank,
            "min_rank": args.min_rank,
            "alpha_rank_scale": args.alpha_rank_scale,
            "sig_type": args.sig_type,
            "lora_alpha": args.lora_alpha,
            "ortho_init": args.ortho_init,
            "lambda_watermarkdm": args.lambda_watermarkdm,
            "roma": args.roma,
            "roma_alpha": args.roma_alpha,
            "roma_r": args.roma_r,
        },
        os.path.join(checkpoint_dir, "train_state.pt"),
    )
    print(f"UNet T-LoRA resume checkpoint saved: {checkpoint_dir}")
    return checkpoint_dir


def resolve_tlora_checkpoint_file(checkpoint_dir, filename, legacy_filename):
    path = os.path.join(checkpoint_dir, filename)
    if os.path.exists(path):
        return path
    legacy_path = os.path.join(checkpoint_dir, legacy_filename)
    if os.path.exists(legacy_path):
        return legacy_path
    raise FileNotFoundError(f"Missing T-LoRA checkpoint file: {path} or {legacy_path}")


def _clear_param_grads(params):
    for param in params:
        param.grad = None


def _csv_header_matches(path, fieldnames):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().strip()
    if not first_line:
        return False
    return first_line.split(",") == list(fieldnames)


def _sample_eval_batches(dataset, num_images, seed):
    count = min(int(num_images), len(dataset))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(np.arange(len(dataset)))[:count]
    return [
        simple_dreambooth_collate_fn([dataset[int(idx)]])
        for idx in indices
    ], [int(idx) for idx in indices]


def _flatten_current_grads(params):
    grad_parts = []
    for param in params:
        if param.grad is None:
            grad_parts.append(torch.zeros(param.numel(), dtype=torch.float32, device="cpu"))
        else:
            grad_parts.append(param.grad.detach().float().reshape(-1).cpu())
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


def _loss_from_encoded(unet, noisy, noise, timesteps, prompt_embeds, pooled, time_ids, sigma):
    pred = stable_diffusion_unet_forward(
        unet,
        noisy,
        timesteps,
        prompt_embeds,
        pooled,
        time_ids,
        cross_attention_kwargs=_attn_kwargs(sigma),
    )
    return F.mse_loss(pred.float(), noise.float(), reduction="mean")


def _sharpness_along_gradient(
    unet,
    tlora_params,
    grad_vec,
    base_loss_value,
    encoded,
    rho,
):
    grad_norm = float(grad_vec.norm().item())
    if grad_norm <= 1e-12 or rho <= 0:
        return 0.0

    scale = float(rho) / grad_norm
    _apply_flat_gradient_perturbation(tlora_params, grad_vec, scale)
    try:
        with torch.no_grad():
            perturbed_loss = _loss_from_encoded(unet, *encoded).detach().float().item()
    finally:
        _apply_flat_gradient_perturbation(tlora_params, grad_vec, -scale)
    return float(perturbed_loss - base_loss_value)


def _compute_tlora_delta_norms(unet):
    if _ORD_LORA:
        from lora_study_utils import peft_delta_norms
        return peft_delta_norms(unet)
    delta_sq = 0.0
    base_sq = 0.0
    for proc_name, proc in unet.attn_processors.items():
        if not isinstance(proc, TLoRACrossAttnProcessor):
            continue
        attn_path = proc_name.split(".processor")[0]
        attn_layer = unet.get_submodule(attn_path)
        for proj_name, lora_module in [
            ("to_q", proc.tlora_q),
            ("to_k", proc.tlora_k),
            ("to_v", proc.tlora_v),
            ("to_out.0", proc.tlora_out),
        ]:
            proj = attn_layer.get_submodule(proj_name)
            delta = compute_orthogonal_lora_weight_delta(lora_module).detach().float()
            base = proj.weight.detach().float()
            delta_sq += float(delta.pow(2).sum().item())
            base_sq += float(base.pow(2).sum().item())

    delta_norm = delta_sq ** 0.5
    base_norm = base_sq ** 0.5
    ratio = delta_norm / max(base_norm, 1e-12)
    return delta_norm, ratio


def _compute_watermarkdm_l1_regularization(unet):
    """Return ||W_r - W_f||_1 over the trainable T-LoRA weight deltas.

    T-LoRA subtracts its frozen orthogonal initialization in the forward pass,
    so each effective weight change is B S A - B_init S_init A_init.
    This function intentionally keeps autograd enabled for the current factors.
    """
    if _ORD_LORA:
        from lora_study_utils import watermarkdm_l1
        return watermarkdm_l1(unet)
    regularization = None
    for proc in unet.attn_processors.values():
        if not isinstance(proc, TLoRACrossAttnProcessor):
            continue
        for lora_module in [
            proc.tlora_q,
            proc.tlora_k,
            proc.tlora_v,
            proc.tlora_out,
        ]:
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


def _iter_tlora_parameter_baselines(unet):
    """Yield each trainable T-LoRA factor and its frozen pretrained baseline."""
    for proc in unet.attn_processors.values():
        if not isinstance(proc, TLoRACrossAttnProcessor):
            continue
        for lora_module in [
            proc.tlora_q,
            proc.tlora_k,
            proc.tlora_v,
            proc.tlora_out,
        ]:
            yield lora_module.q_layer.weight, lora_module.base_q.weight
            yield lora_module.p_layer.weight, lora_module.base_p.weight
            yield lora_module.lambda_layer, lora_module.base_lambda


def _apply_roma_path_perturbation(unet, path_step_size):
    """Move T-LoRA factors by ``r`` toward their pretrained parameter point.

    Implements theta_tilde = theta + r * (theta_0 - theta) /
    ||theta_0 - theta|| from Algorithm 1 of RoMa. Returns the applied
    perturbations so the caller can restore theta before the optimizer step.
    """
    if _ORD_LORA:
        # A PEFT adapter's pretrained point is the zero adapter, so the RoMa
        # direction is simply -theta.
        from lora_study_utils import apply_roma_path_perturbation
        return apply_roma_path_perturbation(
            [p for p in unet.parameters() if p.requires_grad], path_step_size
        )
    parameter_baselines = list(_iter_tlora_parameter_baselines(unet))
    if not parameter_baselines:
        raise RuntimeError("No T-LoRA layers found for RoMa")

    with torch.no_grad():
        difference_norm_sq = torch.zeros(
            (),
            device=parameter_baselines[0][0].device,
            dtype=torch.float32,
        )
        for parameter, baseline in parameter_baselines:
            difference = baseline.to(device=parameter.device) - parameter
            difference_norm_sq.add_(difference.float().pow(2).sum())

        difference_norm = difference_norm_sq.sqrt()
        denominator = difference_norm.clamp_min(1e-8)
        coefficient = float(path_step_size) / denominator
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


def compute_half_timestep_loss_landscape_study(
    cp_batch,
    org_batch,
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    noise_scheduler,
    accelerator,
    args,
    max_timestep,
    tlora_params,
):
    was_training = unet.training
    unet.eval()
    text_encoder.eval()
    text_encoder_2.eval()

    half_timestep_value = int(
        half_timestep_tensor(
            1,
            accelerator.device,
            max_timestep,
            noise_scheduler.config.num_train_timesteps,
        )[0].item()
    )

    half_timesteps_cp = torch.full(
        (cp_batch["pixel_values"].shape[0],),
        half_timestep_value,
        device=accelerator.device,
        dtype=torch.long,
    )
    half_timesteps_org = torch.full(
        (org_batch["pixel_values"].shape[0],),
        half_timestep_value,
        device=accelerator.device,
        dtype=torch.long,
    )

    cp_encoded = encode_batch(
        cp_batch, vae, text_encoder, text_encoder_2, noise_scheduler,
        accelerator, args.resolution, args.rank, args.min_rank,
        args.alpha_rank_scale, max_timestep, timesteps=half_timesteps_cp,
    )
    org_encoded = encode_batch(
        org_batch, vae, text_encoder, text_encoder_2, noise_scheduler,
        accelerator, args.resolution, args.rank, args.min_rank,
        args.alpha_rank_scale, max_timestep, timesteps=half_timesteps_org,
    )

    cp_loss_args = (
        cp_encoded[0], cp_encoded[1], cp_encoded[2], cp_encoded[3],
        cp_encoded[4], cp_encoded[5], cp_encoded[6],
    )
    org_loss_args = (
        org_encoded[0], org_encoded[1], org_encoded[2], org_encoded[3],
        org_encoded[4], org_encoded[5], org_encoded[6],
    )

    try:
        _clear_param_grads(tlora_params)
        cp_half_loss = _loss_from_encoded(unet, *cp_loss_args)
        cp_half_loss_val = float(cp_half_loss.detach().item())
        accelerator.backward(cp_half_loss)
        cp_grad_vec = _flatten_current_grads(tlora_params)

        _clear_param_grads(tlora_params)
        org_half_loss = _loss_from_encoded(unet, *org_loss_args)
        org_half_loss_val = float(org_half_loss.detach().item())
        accelerator.backward(org_half_loss)
        org_grad_vec = _flatten_current_grads(tlora_params)

        cp_grad_norm = float(cp_grad_vec.norm().item())
        org_grad_norm = float(org_grad_vec.norm().item())
        if cp_grad_norm <= 1e-12 or org_grad_norm <= 1e-12:
            grad_cosine = float("nan")
        else:
            grad_cosine = float(F.cosine_similarity(cp_grad_vec, org_grad_vec, dim=0).item())

        _clear_param_grads(tlora_params)
        cp_sharpness = _sharpness_along_gradient(
            unet,
            tlora_params,
            cp_grad_vec,
            cp_half_loss_val,
            cp_loss_args,
            args.sharpness_rho,
        )
        org_sharpness = _sharpness_along_gradient(
            unet,
            tlora_params,
            org_grad_vec,
            org_half_loss_val,
            org_loss_args,
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
        _clear_param_grads(tlora_params)
        clear_text_encoder_sigma_mask()
        if was_training:
            unet.train()


def _fork_rng_devices(device):
    if device.type != "cuda":
        return []
    if device.index is None:
        return [torch.cuda.current_device()]
    return [device.index]


def _resolve_overlap_timestep(args, noise_scheduler, accelerator, max_timestep, step):
    if args.overlap_timestep_mode == "half":
        return int(half_timestep_tensor(
            1,
            accelerator.device,
            max_timestep,
            noise_scheduler.config.num_train_timesteps,
        )[0].item())

    if args.overlap_timestep_mode == "fixed":
        return int(args.overlap_timestep)

    generator_device = accelerator.device if accelerator.device.type == "cuda" else torch.device("cpu")
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(args.seed + 300_003 + step))
    return int(torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (1,),
        generator=generator,
        device=generator_device,
    )[0].item())


def _single_batch_gradient_vector(
    batch,
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    noise_scheduler,
    accelerator,
    resolution,
    rank,
    min_rank,
    alpha_rank_scale,
    max_timestep,
    tlora_params,
    seed,
    timestep_value,
):
    _clear_param_grads(tlora_params)

    with torch.random.fork_rng(devices=_fork_rng_devices(accelerator.device)):
        if seed is not None:
            torch.manual_seed(int(seed))
            if accelerator.device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))

        timesteps = torch.full(
            (batch["pixel_values"].shape[0],),
            int(timestep_value),
            device=accelerator.device,
            dtype=torch.long,
        )
        noisy, noise, timesteps, prompt_embeds, pooled, time_ids, sigma = encode_batch(
            batch, vae, text_encoder, text_encoder_2, noise_scheduler,
            accelerator, resolution, rank, min_rank,
            alpha_rank_scale, max_timestep, timesteps=timesteps,
        )
        pred = stable_diffusion_unet_forward(
            unet,
            noisy,
            timesteps,
            prompt_embeds,
            pooled,
            time_ids,
            cross_attention_kwargs=_attn_kwargs(sigma),
        )
        loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")

    accelerator.backward(loss)
    grad_vec = _flatten_current_grads(tlora_params)
    _clear_param_grads(tlora_params)
    return grad_vec


def _gradient_subspace_metrics(cp_grads, org_grads):
    cp_matrix = torch.stack(cp_grads).float()
    org_matrix = torch.stack(org_grads).float()

    cp_norms = cp_matrix.norm(dim=1, keepdim=True)
    org_norms = org_matrix.norm(dim=1, keepdim=True)
    cp_nonzero = cp_norms.squeeze(1) > 0
    org_nonzero = org_norms.squeeze(1) > 0

    if not bool(cp_nonzero.any()) or not bool(org_nonzero.any()):
        return {
            "cp_grad_count": int(cp_matrix.shape[0]),
            "org_grad_count": int(org_matrix.shape[0]),
            "cp_rank": 0,
            "org_rank": 0,
            "pairwise_cos_mean": float("nan"),
            "pairwise_cos_mean_abs": float("nan"),
            "pairwise_cos_max_abs": float("nan"),
            "pairwise_cos_min": float("nan"),
            "pairwise_cos_max": float("nan"),
            "principal_overlap": float("nan"),
        }

    cp_unit = cp_matrix[cp_nonzero] / cp_norms[cp_nonzero].clamp_min(1e-12)
    org_unit = org_matrix[org_nonzero] / org_norms[org_nonzero].clamp_min(1e-12)
    cos = cp_unit @ org_unit.T

    cp_gram = cp_unit @ cp_unit.T
    org_gram = org_unit @ org_unit.T
    cross_gram = cp_unit @ org_unit.T

    cp_eigs = torch.linalg.eigvalsh(cp_gram)
    org_eigs = torch.linalg.eigvalsh(org_gram)
    cp_tol = max(float(cp_eigs.max().item()), 1.0) * max(cp_gram.shape) * torch.finfo(cp_gram.dtype).eps
    org_tol = max(float(org_eigs.max().item()), 1.0) * max(org_gram.shape) * torch.finfo(org_gram.dtype).eps
    cp_rank = int((cp_eigs > cp_tol).sum().item())
    org_rank = int((org_eigs > org_tol).sum().item())

    if cp_rank == 0 or org_rank == 0:
        principal_overlap = float("nan")
    else:
        cp_pinv = torch.linalg.pinv(cp_gram, rtol=1e-5)
        org_pinv = torch.linalg.pinv(org_gram, rtol=1e-5)
        raw_overlap = torch.trace(cp_pinv @ cross_gram @ org_pinv @ cross_gram.T)
        principal_overlap = float((raw_overlap / min(cp_rank, org_rank)).clamp(0.0, 1.0).item())

    return {
        "cp_grad_count": int(cp_matrix.shape[0]),
        "org_grad_count": int(org_matrix.shape[0]),
        "cp_rank": cp_rank,
        "org_rank": org_rank,
        "pairwise_cos_mean": float(cos.mean().item()),
        "pairwise_cos_mean_abs": float(cos.abs().mean().item()),
        "pairwise_cos_max_abs": float(cos.abs().max().item()),
        "pairwise_cos_min": float(cos.min().item()),
        "pairwise_cos_max": float(cos.max().item()),
        "principal_overlap": principal_overlap,
    }


def compute_gradient_overlap_study(
    cp_batches,
    org_batches,
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    noise_scheduler,
    accelerator,
    args,
    max_timestep,
    tlora_params,
    step,
):
    was_training = unet.training
    unet.eval()
    text_encoder.eval()
    text_encoder_2.eval()

    try:
        overlap_timestep = _resolve_overlap_timestep(
            args, noise_scheduler, accelerator, max_timestep, step
        )
        cp_grads = [
            _single_batch_gradient_vector(
                batch, unet, vae, text_encoder, text_encoder_2, noise_scheduler,
                accelerator, args.resolution, args.rank, args.min_rank,
                args.alpha_rank_scale, max_timestep, tlora_params,
                args.seed + 100_003 + idx,
                overlap_timestep,
            )
            for idx, batch in enumerate(cp_batches)
        ]
        org_grads = [
            _single_batch_gradient_vector(
                batch, unet, vae, text_encoder, text_encoder_2, noise_scheduler,
                accelerator, args.resolution, args.rank, args.min_rank,
                args.alpha_rank_scale, max_timestep, tlora_params,
                args.seed + 200_003 + idx,
                overlap_timestep,
            )
            for idx, batch in enumerate(org_batches)
        ]
        metrics = _gradient_subspace_metrics(cp_grads, org_grads)
        metrics["overlap_timestep"] = overlap_timestep
        metrics["overlap_timestep_mode"] = args.overlap_timestep_mode
        return metrics
    finally:
        _clear_param_grads(tlora_params)
        clear_text_encoder_sigma_mask()
        if was_training:
            unet.train()


@torch.no_grad()
def integrity_inference_merged(
    unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
    noise_scheduler, prompts, output_paths, resolution,
    num_steps=20, guidance_scale=7.5, seed=None,
):
    """Generate integrity images using the adapter merged into the backbone."""
    from diffusers.models.attention_processor import AttnProcessor2_0

    if _ORD_LORA:
        # peft can fold and unfold its own adapter, so no manual weight surgery.
        _ord_lora_merge(unet)
        try:
            clear_text_encoder_sigma_mask()
            save_sdxl_images_from_components(
                unet=unet, vae=vae,
                text_encoder=text_encoder, text_encoder_2=text_encoder_2,
                tokenizer=tokenizer, tokenizer_2=tokenizer_2,
                noise_scheduler=noise_scheduler,
                prompts=prompts,
                output_paths=output_paths,
                resolution=resolution,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            print(f"Integrity images (merged): {', '.join(output_paths)}")
        finally:
            _ord_lora_unmerge(unet)
        return

    saved_attn_weights = {}
    original_processors = dict(unet.attn_processors)

    try:
        # Merge T-LoRA into UNet in-place.
        for proc_name, proc in list(unet.attn_processors.items()):
            if not isinstance(proc, TLoRACrossAttnProcessor):
                continue
            attn_path = proc_name.split(".processor")[0]
            attn_layer = unet.get_submodule(attn_path)
            for proj_name, lora_module in [
                ("to_q", proc.tlora_q), ("to_k", proc.tlora_k),
                ("to_v", proc.tlora_v), ("to_out.0", proc.tlora_out),
            ]:
                proj = attn_layer.get_submodule(proj_name)
                key = f"{attn_path}.{proj_name}"
                saved_attn_weights[key] = proj.weight.data.clone()
                delta = compute_orthogonal_lora_weight_delta(lora_module)
                proj.weight.data.add_(delta.to(dtype=proj.weight.dtype))

        # Replace T-LoRA processors with standard processors so UNet deltas
        # are not applied twice after merging into base weights.
        standard_procs = {name: AttnProcessor2_0() for name in original_processors}
        unet.set_attn_processor(standard_procs)

        clear_text_encoder_sigma_mask()
        save_sdxl_images_from_components(
            unet=unet, vae=vae,
            text_encoder=text_encoder, text_encoder_2=text_encoder_2,
            tokenizer=tokenizer, tokenizer_2=tokenizer_2,
            noise_scheduler=noise_scheduler,
            prompts=prompts,
            output_paths=output_paths,
            resolution=resolution,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        print(f"Integrity images (merged): {', '.join(output_paths)}")
    finally:
        clear_text_encoder_sigma_mask()
        unet.set_attn_processor(original_processors)
        for proc_name, proc in unet.attn_processors.items():
            if not isinstance(proc, TLoRACrossAttnProcessor):
                continue
            attn_path = proc_name.split(".processor")[0]
            attn_layer = unet.get_submodule(attn_path)
            for proj_name in ["to_q", "to_k", "to_v", "to_out.0"]:
                key = f"{attn_path}.{proj_name}"
                if key in saved_attn_weights:
                    attn_layer.get_submodule(proj_name).weight.data.copy_(saved_attn_weights[key])


# ============================================================
# Encoding helpers
# ============================================================

def _encode_latents(batch, vae, noise=None, timesteps=None, noise_scheduler=None):
    """
    Encode pixel_values → latents, add noise, return (noisy_latents, noise, timesteps).

    If noise and timesteps are provided they are reused (for the org_image shadow pass).
    Otherwise fresh noise and random timesteps are sampled.
    """
    with torch.no_grad():
        pixel_values = batch["pixel_values"].to(device=vae.device, dtype=vae.dtype)
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        if noise is None:
            noise = torch.randn_like(latents)
        else:
            # Reuse the same epsilon but adapt shape if batch sizes differ
            noise = noise[:latents.shape[0]].to(device=latents.device, dtype=latents.dtype)
            if noise.shape != latents.shape:
                # spatial dims may differ; resize noise to match
                noise = torch.randn_like(latents)

        if timesteps is None:
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device,
            ).long()
        else:
            timesteps = timesteps[:latents.shape[0]].to(device=latents.device)

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    return noisy_latents, noise, timesteps


def encode_batch(
    batch,
    vae,
    text_encoder,
    text_encoder_2,
    noise_scheduler,
    accelerator,
    resolution,
    rank,
    min_rank,
    alpha_rank_scale,
    max_timestep,
    noise=None,
    timesteps=None,
):
    """
    Full encode: VAE + noise + text encoders + time_ids.

    noise / timesteps: if given, reuse them (shadow pass).
    Returns (noisy_latents, timesteps, noise, prompt_embeds, pooled_prompt_embeds,
             add_time_ids, sigma_mask).
    """
    noisy_latents, noise, timesteps = _encode_latents(
        batch, vae, noise=noise, timesteps=timesteps, noise_scheduler=noise_scheduler
    )

    sigma_mask = get_mask_by_timestep(
        timesteps[0].item(),
        max_timestep,
        rank,
        min_rank,
        alpha_rank_scale,
    ).detach().to(accelerator.device)

    set_text_encoder_sigma_mask(sigma_mask)

    input_ids_1 = batch["input_ids"].to(device=next(text_encoder.parameters()).device)
    input_ids_2 = batch["input_ids_2"].to(device=input_ids_1.device)
    prompt_embeds, pooled_prompt_embeds, add_time_ids = (
        encode_stable_diffusion_prompt(
            text_encoder,
            text_encoder_2,
            input_ids_1,
            input_ids_2,
            noisy_latents,
            resolution,
        )
    )

    clear_text_encoder_sigma_mask()

    return noisy_latents, noise, timesteps, prompt_embeds, pooled_prompt_embeds, add_time_ids, sigma_mask


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Study variant: logs SNR-weighted cp/org forward loss and CLIP score per step"
    )

    # Dataset arguments
    parser.add_argument("--cp_dataset", type=str, required=True,
                        help="Copyright dataset dir (image/ + prompt.csv)")
    parser.add_argument("--org_image", type=str, required=True,
                        help="Original/clean image dataset dir (image/ + prompt.csv). "
                             "Used only for shadow loss computation (no backprop).")

    parser.add_argument("--step", "--cp_step", dest="step", type=int, default=400,
                        help="Number of training steps. --cp_step is kept as a legacy alias.")

    # Model
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default="fp16")

    # Output
    parser.add_argument("--output_dir", type=str, default="checkpoints_robust_study")
    parser.add_argument("--study_log_file", type=str, default=None,
                        help="CSV file for per-step loss log. Defaults to <output_dir>/study_loss_log.csv")
    parser.add_argument("--overlap_log_file", type=str, default=None,
                        help="CSV file for gradient subspace overlap logs. Defaults to <output_dir>/gradient_overlap_log.csv")
    parser.add_argument("--resolution", type=int, default=1024)

    # Training hypers
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--cp_ref_weight", type=float, default=1.0)
    parser.add_argument("--org_loss_weight", type=float, default=1.0,
                        help="Weight applied to the original-image noise MSE loss.")
    parser.add_argument("--lambda_watermarkdm", type=float, default=0.001,
                        help="Weight for the WatermarkDM L1 penalty on the effective "
                             "T-LoRA weight change, scaled by lora_alpha / rank.")
    parser.add_argument("--roma", action="store_true",
                        help="Enable RoMa path-specific smoothness optimization.")
    parser.add_argument("--roma_alpha", type=float, default=0.4,
                        help="RoMa balance coefficient between the current and path losses.")
    parser.add_argument("--roma_r", type=float, default=0.05,
                        help="RoMa normalized path-aware step size toward the pretrained model.")
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--sharpness_rho", type=float, default=0.05,
                        help="Radius for SAM-style loss sharpness probe in T-LoRA parameter space.")
    parser.add_argument("--overlap_interval", type=int, default=0,
                        help="Every N steps: compute T-LoRA gradient subspace overlap between "
                             "copyright and original samples. 0 = disabled.")
    parser.add_argument("--overlap_num_images", type=int, default=8,
                        help="Number of copyright and original images to use for gradient overlap.")
    parser.add_argument("--overlap_timestep_mode", type=str, default="half",
                        choices=["half", "random", "fixed"],
                        help="Timestep used for all gradient-overlap samples at a checkpoint. "
                             "'half' uses the half timestep, 'random' samples one shared timestep, "
                             "'fixed' uses --overlap_timestep.")
    parser.add_argument("--overlap_timestep", type=int, default=None,
                        help="Exact shared timestep for gradient overlap when --overlap_timestep_mode=fixed.")

    # LoRA
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--ord_lora", action="store_true",
        help="Replace T-LoRA with an ordinary PEFT LoRA of the same rank and "
             "targets: no timestep sigma mask and no frozen orthogonal "
             "initialization. The merged checkpoint format is unchanged, so "
             "every downstream attack and evaluation works either way.",
    )
    parser.add_argument(
        "--lora_target_modules", type=str, default="to_q,to_k,to_v,to_out.0",
        help="Comma-separated attention projections for --ord_lora.",
    )

    # T-LoRA
    parser.add_argument("--min_rank", type=int, default=None)
    parser.add_argument("--alpha_rank_scale", type=float, default=1.0)
    parser.add_argument("--sig_type", type=str, default="last",
                        choices=["last", "principal", "middle"])
    parser.add_argument("--ortho_init", type=str, default="random",
                        choices=["random", "base_layer"],
                        help="T-LoRA orthogonal init source. 'random' matches ControlGenAI/T-LoRA ortho_lora; "
                             "'base_layer' is the heavier layer-aware SVD variant.")

    # Other
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    # Study loss toggle
    parser.add_argument("--no_study_loss", action="store_true",
                        help="Legacy flag. Old CLIP/SNR study outputs are removed; "
                             "the compact gradient/sharpness study still runs.")

    # Integrity interval: merge T-LoRA into backbone temporarily and infer
    parser.add_argument("--integrity_interval", type=int, default=100,
                        help="Every N steps: temporarily merge T-LoRA into backbone weights "
                             "and generate integrity inference images. 0 = disabled.")
    parser.add_argument("--integrity_prompt", type=str,
                        default="A [Z]*$ flying in the sky.",
                        help="Prompt used for integrity inference images")
    parser.add_argument("--integrity_inference_steps", type=int, default=50)
    parser.add_argument("--integrity_seed", type=int, default=0,
                        help="Seed used for merged integrity inference images")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="T-LoRA checkpoint-stepNNNNNN directory to resume from")
    parser.add_argument("--auto_resume_latest", action="store_true",
                        help="Resume from latest checkpoint-stepNNNNNN in output_dir")
    add_trigger_rewrite_args(parser)

    args = parser.parse_args()
    # The dispatchers above read this rather than threading args through every
    # helper; nothing changes on the T-LoRA path when the flag is off.
    global _ORD_LORA
    _ORD_LORA = bool(args.ord_lora)
    if _ORD_LORA:
        for flag, value in (("--sig_type", args.sig_type),
                            ("--ortho_init", args.ortho_init)):
            _ = flag, value  # accepted but inert: no sigma mask, no ortho init
        print("--ord_lora: plain PEFT LoRA, no timestep sigma mask", flush=True)
    args.integrity_prompt = replace_trigger_word(
        args.integrity_prompt,
        args.original_trigger_word,
        args.current_trigger_word,
    )

    if args.auto_resume_latest and args.resume_from_checkpoint is None and os.path.isdir(args.output_dir):
        candidates = []
        for name in os.listdir(args.output_dir):
            prefix = "checkpoint-step"
            if not name.startswith(prefix):
                continue
            token = name[len(prefix):]
            if token.isdigit() and os.path.isdir(os.path.join(args.output_dir, name)):
                candidates.append((int(token), name))
        if candidates:
            latest_step, latest_name = max(candidates, key=lambda item: item[0])
            args.resume_from_checkpoint = latest_name
            print(f"Auto-resume selected: {latest_name} (step {latest_step})")

    if args.min_rank is None:
        args.min_rank = args.rank // 2
    if args.lambda_watermarkdm < 0:
        parser.error("--lambda_watermarkdm must be non-negative")
    if not 0.0 <= args.roma_alpha <= 1.0:
        parser.error("--roma_alpha must be between 0 and 1")
    if args.roma_r < 0:
        parser.error("--roma_r must be non-negative")

    if args.study_log_file is None:
        os.makedirs(args.output_dir, exist_ok=True)
        args.study_log_file = os.path.join(args.output_dir, "study_loss_log.csv")
    if args.overlap_log_file is None:
        os.makedirs(args.output_dir, exist_ok=True)
        args.overlap_log_file = os.path.join(args.output_dir, "gradient_overlap_log.csv")
    if args.overlap_interval > 0 and args.overlap_num_images <= 0:
        parser.error("--overlap_num_images must be positive when --overlap_interval is enabled")
    if args.overlap_interval > 0 and args.overlap_timestep_mode == "fixed":
        if args.overlap_timestep is None:
            parser.error("--overlap_timestep is required when --overlap_timestep_mode=fixed")
        if args.overlap_timestep < 0:
            parser.error("--overlap_timestep must be non-negative")

    # Copyright data keeps the legacy image/ + prompt.csv layout. Original
    # data is validated by make_org_image_dataset and may also be COCO2014.
    # --cp_dataset may be a comma-separated list of concepts; validate each.
    for _d in [x.strip() for x in args.cp_dataset.split(",") if x.strip()]:
        cp_csv_path = os.path.join(_d, "prompt.csv")
        cp_image_dir = os.path.join(_d, "image")
        if not os.path.isfile(cp_csv_path):
            raise FileNotFoundError(f"cp_dataset prompt.csv not found: {cp_csv_path}")
        if not os.path.isdir(cp_image_dir):
            raise FileNotFoundError(f"cp_dataset image dir not found: {cp_image_dir}")

    print(f"\n=== Study Training Configuration ===")
    print(f"  cp_dataset:     {args.cp_dataset}")
    print(f"  org_image:      {args.org_image}")
    print(f"  step:           {args.step}")
    print(f"  cp_ref_weight:  {args.cp_ref_weight}")
    print(f"  org_loss_weight: {args.org_loss_weight}")
    print(f"  lambda_watermarkdm: {args.lambda_watermarkdm}")
    print(f"  RoMa: {args.roma}, alpha: {args.roma_alpha}, r: {args.roma_r}")
    print(f"  study_log_file: {args.study_log_file}")
    print(
        f"  overlap_interval: {args.overlap_interval}, "
        f"overlap_num_images: {args.overlap_num_images}, "
        f"overlap_timestep_mode: {args.overlap_timestep_mode}"
    )
    print(f"  T-LoRA rank: {args.rank}, min_rank: {args.min_rank}, sig_type: {args.sig_type}, ortho_init: {args.ortho_init}")
    print(f"=====================================\n")

    # ---------- accelerator & seed ----------
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # ---------- models ----------
    print("Loading models...")
    if args.mixed_precision == "bf16":
        model_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    components = load_base_sdxl_backbone(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )
    tokenizer = components.tokenizer
    tokenizer_2 = components.tokenizer_2
    vae = components.vae
    unet = components.unet
    text_encoder = components.text_encoder
    text_encoder_2 = components.text_encoder_2

    reference_components = load_base_sdxl_backbone(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=model_dtype,
    )
    ref_unet = reference_components.unet

    # Freeze base weights
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    ref_unet.requires_grad_(False)
    ref_unet.eval()

    print("loading finished")

    # ---------- UNet adapter setup ----------
    if args.ord_lora:
        tlora_params = setup_ordinary_lora(unet, args)
    else:
        skip_tlora_init = args.resume_from_checkpoint is not None
        lora_attn_procs = build_tlora_attn_processors(
            unet,
            rank=args.rank,
            lora_alpha=args.lora_alpha,
            sig_type=args.sig_type,
            ortho_init=args.ortho_init,
            skip_init=skip_tlora_init,
        )
        unet.set_attn_processor(lora_attn_procs)
        tlora_params = collect_unet_tlora_trainable_params(unet)

    resume_step = 0
    resume_checkpoint_path = None
    if args.resume_from_checkpoint is not None:
        ckpt_path = args.resume_from_checkpoint
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(args.output_dir, ckpt_path)
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        resume_checkpoint_path = ckpt_path

        base = os.path.basename(os.path.normpath(ckpt_path))
        prefix = "checkpoint-step"
        if base.startswith(prefix) and base[len(prefix):].isdigit():
            resume_step = int(base[len(prefix):])

        if args.ord_lora:
            print(f"Resuming ordinary LoRA from {ckpt_path} (step {resume_step})")
            load_ordinary_lora_resume(unet, ckpt_path)
            print("Loaded resume state: ordinary LoRA adapter")
        else:
            print(f"Resuming T-LoRA modules from {ckpt_path} (step {resume_step})")
            loaded_unet = load_tlora_attn_state_dict(
                unet,
                torch.load(
                    resolve_tlora_checkpoint_file(ckpt_path, "tlora_weights.pt", "dual_lora_weights.pt"),
                    map_location="cpu",
                ),
                strict=True,
            )
            print(f"Loaded resume state: UNet={loaded_unet}")

    tlora_numel = sum(p.numel() for p in tlora_params)
    print(f"{'Ordinary LoRA' if args.ord_lora else 'T-LoRA'} trainable params: {tlora_numel:,}")
    print("Model setup complete.\n")

    # ---------- noise scheduler ----------
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    max_timestep = noise_scheduler.config.num_train_timesteps
    if (
        args.overlap_interval > 0
        and args.overlap_timestep_mode == "fixed"
        and args.overlap_timestep >= max_timestep
    ):
        raise ValueError(
            f"--overlap_timestep must be < scheduler timesteps ({max_timestep}); "
            f"got {args.overlap_timestep}"
        )

    # ---------- optimizer ----------
    optimizer = torch.optim.AdamW(
        tlora_params, lr=args.learning_rate, betas=(0.9, 0.999),
        weight_decay=1e-2, eps=1e-08,
    )
    print(f"Optimizer: AdamW, lr={args.learning_rate}, weight_decay=1e-2, eps=1e-08\n")

    # ---------- datasets ----------
    # --cp_dataset and --current_trigger_word may each be a comma-separated
    # list. With N>1 the step loop round-robins one homogeneous batch per
    # concept (batch 0 -> concept 0, batch 1 -> concept 1, ...), which is what
    # "mix the datasets but keep the total step budget" means here. The trigger
    # phrase is a per-dataset constructor argument, so each batch carries its
    # own phrase with no cross-contamination.
    _cp_dirs = [d.strip() for d in args.cp_dataset.split(",") if d.strip()]
    _cp_trigs = [t.strip() for t in args.current_trigger_word.split(",")]
    if len(_cp_trigs) == 1:
        _cp_trigs = _cp_trigs * len(_cp_dirs)
    if len(_cp_trigs) != len(_cp_dirs):
        raise ValueError(
            f"--cp_dataset has {len(_cp_dirs)} entries but "
            f"--current_trigger_word has {len(_cp_trigs)}")
    # --original_trigger_word is per-dataset too: some prompt.csv use the
    # "[Z]*$" placeholder, others already have an older phrase baked in and
    # need that exact phrase replaced.
    _cp_origs = [o.strip() for o in args.original_trigger_word.split(",")]
    if len(_cp_origs) == 1:
        _cp_origs = _cp_origs * len(_cp_dirs)
    if len(_cp_origs) != len(_cp_dirs):
        raise ValueError(
            f"--cp_dataset has {len(_cp_dirs)} entries but "
            f"--original_trigger_word has {len(_cp_origs)}")
    cp_datasets = [
        SimpleDreamBoothDataset(
            csv_path=os.path.join(d, "prompt.csv"),
            image_dir=os.path.join(d, "image"),
            tokenizer=tokenizer, tokenizer_2=tokenizer_2,
            size=args.resolution, center_crop=False,
            original_trigger_word=o,
            current_trigger_word=t,
        )
        for d, t, o in zip(_cp_dirs, _cp_trigs, _cp_origs)
    ]
    cp_dataset = cp_datasets[0]
    if len(cp_datasets) > 1:
        print(f"Multi-concept copyright training over {len(cp_datasets)} datasets:")
        for d, t, o in zip(_cp_dirs, _cp_trigs, _cp_origs):
            print(f"    {os.path.basename(d.rstrip('/')):<24} {o!r} -> {t!r}")
    org_dataset = make_org_image_dataset(
        data_dir=args.org_image,
        tokenizer=tokenizer, tokenizer_2=tokenizer_2,
        size=args.resolution,
        original_trigger_word=_cp_origs[0],
        current_trigger_word=_cp_trigs[0],
    )

    cp_dataloaders = [
        infinite_dataloader(ds, args.seed + 100 * i, args.train_batch_size)
        for i, ds in enumerate(cp_datasets)
    ]
    cp_dataloader = cp_dataloaders[0]
    org_dataloader = infinite_dataloader(org_dataset, args.seed + 1, args.train_batch_size)
    print(f"Datasets loaded: {len(cp_dataset)} copyright samples, {len(org_dataset)} original samples.\n")

    overlap_cp_batches = []
    overlap_org_batches = []
    if args.overlap_interval > 0:
        overlap_cp_batches, overlap_cp_indices = _sample_eval_batches(
            cp_dataset, args.overlap_num_images, args.seed + 30_001
        )
        overlap_org_batches, overlap_org_indices = _sample_eval_batches(
            org_dataset, args.overlap_num_images, args.seed + 40_001
        )
        print(
            "Gradient overlap samples: "
            f"{len(overlap_cp_batches)} copyright indices {overlap_cp_indices}, "
            f"{len(overlap_org_batches)} original indices {overlap_org_indices}"
        )

    # ---------- accelerator prepare ----------
    unet, text_encoder, text_encoder_2, ref_unet, optimizer = accelerator.prepare(
        unet, text_encoder, text_encoder_2, ref_unet, optimizer
    )
    load_optimizer_checkpoint(
        accelerator,
        optimizer,
        resume_checkpoint_path,
        expected_optimizer_name="adamw",
        expected_step=resume_step,
    )
    vae = vae.to(accelerator.device)
    ref_unet.eval()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    
    print("Accelerator preparation complete.\n")

    # ---------- loss log file ----------
    os.makedirs(args.output_dir, exist_ok=True)
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
    log_fh = open(args.study_log_file, "a" if log_append else "w", encoding="utf-8", newline="")
    log_writer = csv.DictWriter(
        log_fh,
        fieldnames=study_log_fields,
    )
    if not log_append:
        log_writer.writeheader()
        log_fh.flush()

    overlap_log_fh = None
    overlap_log_writer = None
    if args.overlap_interval > 0:
        overlap_log_append = os.path.exists(args.overlap_log_file) and resume_step > 0
        overlap_log_fh = open(
            args.overlap_log_file,
            "a" if overlap_log_append else "w",
            encoding="utf-8",
            newline="",
        )
        overlap_log_writer = csv.DictWriter(
            overlap_log_fh,
            fieldnames=[
                "step",
                "overlap_timestep_mode",
                "overlap_timestep",
                "cp_grad_count",
                "org_grad_count",
                "cp_rank",
                "org_rank",
                "pairwise_cos_mean",
                "pairwise_cos_mean_abs",
                "pairwise_cos_max_abs",
                "pairwise_cos_min",
                "pairwise_cos_max",
                "principal_overlap",
            ],
        )
        if not overlap_log_append:
            overlap_log_writer.writeheader()
            overlap_log_fh.flush()

    integrity_dir = os.path.join(args.output_dir, "integrity_images")
    if accelerator.is_main_process and args.integrity_interval > 0:
        os.makedirs(integrity_dir, exist_ok=True)
    org_integrity_prompt = first_prompt_from_dataset(
        args.org_image,
        args.original_trigger_word,
        args.current_trigger_word,
    )

    print(f"\n***** Study Training *****")
    print(f"  CP dataset: {len(cp_dataset)} samples")
    print(f"  Org dataset: {len(org_dataset)} samples")
    print(f"  Steps: {args.step}  (resuming from {resume_step})")
    print(f"  Loss log: {args.study_log_file}\n")
    if args.overlap_interval > 0:
        print(f"  Gradient overlap log: {args.overlap_log_file}\n")

    unet.train()
    text_encoder.eval()
    text_encoder_2.eval()

    if resume_step >= args.step:
        print(f"Training already complete (resume_step={resume_step} >= step={args.step})")
    else:
        progress_bar = tqdm(range(resume_step, args.step), desc="Study-TLoRA")

        for step in range(resume_step, args.step):
            # round-robin: one homogeneous batch per concept, in turn
            cp_batch = next(cp_dataloaders[step % len(cp_dataloaders)])
            org_batch = next(org_dataloader)
            shared_timestep = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (1,),
                device=accelerator.device,
            ).long()
            cp_timesteps = shared_timestep.repeat(cp_batch["pixel_values"].shape[0])
            org_timesteps = shared_timestep.repeat(org_batch["pixel_values"].shape[0])

            cp_noisy, cp_noise, timesteps, cp_prompt_embeds, cp_pooled, cp_time_ids, cp_sigma = encode_batch(
                cp_batch, vae, text_encoder, text_encoder_2, noise_scheduler,
                accelerator, args.resolution, args.rank, args.min_rank,
                args.alpha_rank_scale, max_timestep, timesteps=cp_timesteps,
            )

            org_noisy, org_noise, org_ts, org_prompt_embeds, org_pooled, org_time_ids, org_sigma = encode_batch(
                org_batch, vae, text_encoder, text_encoder_2, noise_scheduler,
                accelerator, args.resolution, args.rank, args.min_rank,
                args.alpha_rank_scale, max_timestep, timesteps=org_timesteps,
            )

            batch_noisy = torch.cat([cp_noisy, org_noisy], dim=0)
            batch_timesteps = torch.cat([timesteps, org_ts], dim=0)
            batch_prompt_embeds = torch.cat([cp_prompt_embeds, org_prompt_embeds], dim=0)
            batch_pooled = (
                None if cp_pooled is None
                else torch.cat([cp_pooled, org_pooled], dim=0)
            )
            batch_time_ids = (
                None if cp_time_ids is None
                else torch.cat([cp_time_ids, org_time_ids], dim=0)
            )
            cp_n = cp_noisy.shape[0]

            batch_pred = stable_diffusion_unet_forward(
                unet,
                batch_noisy,
                batch_timesteps,
                batch_prompt_embeds,
                batch_pooled,
                batch_time_ids,
                cross_attention_kwargs=_attn_kwargs(cp_sigma),
            )
            cp_pred = batch_pred[:cp_n]
            org_pred = batch_pred[cp_n:]
            with torch.no_grad():
                cp_ref_pred = stable_diffusion_unet_forward(
                    ref_unet,
                    cp_noisy,
                    timesteps,
                    cp_prompt_embeds,
                    cp_pooled,
                    cp_time_ids,
                ).detach()

            cp_noise_mse = F.mse_loss(cp_pred.float(), cp_noise.float(), reduction="mean")
            cp_ref_mse = F.mse_loss(cp_pred.float(), cp_ref_pred.float(), reduction="mean")
            org_noise_mse = F.mse_loss(org_pred.float(), org_noise.float(), reduction="mean")
            cp_loss = cp_noise_mse - args.cp_ref_weight * cp_ref_mse
            org_loss = args.org_loss_weight * org_noise_mse
            watermarkdm_l1 = _compute_watermarkdm_l1_regularization(
                accelerator.unwrap_model(unet)
            )
            lora_scaling = args.lora_alpha / args.rank
            watermarkdm_loss = (
                args.lambda_watermarkdm * lora_scaling * watermarkdm_l1
            )
            loss = cp_loss + org_loss + watermarkdm_loss

            roma_path_loss = None
            roma_difference_norm = 0.0
            if args.roma:
                # RoMa Algorithm 1: accumulate (1-alpha) * g1 at the current
                # parameters, then alpha * g2 at a normalized step toward the
                # frozen pretrained T-LoRA point. Restore before Adam updates.
                accelerator.backward((1.0 - args.roma_alpha) * loss)
                roma_perturbations, roma_difference_norm = _apply_roma_path_perturbation(
                    accelerator.unwrap_model(unet),
                    args.roma_r,
                )
                try:
                    roma_batch_pred = stable_diffusion_unet_forward(
                        unet,
                        batch_noisy,
                        batch_timesteps,
                        batch_prompt_embeds,
                        batch_pooled,
                        batch_time_ids,
                        cross_attention_kwargs=_attn_kwargs(cp_sigma),
                    )
                    roma_cp_pred = roma_batch_pred[:cp_n]
                    roma_org_pred = roma_batch_pred[cp_n:]
                    roma_cp_noise_mse = F.mse_loss(
                        roma_cp_pred.float(), cp_noise.float(), reduction="mean"
                    )
                    roma_cp_ref_mse = F.mse_loss(
                        roma_cp_pred.float(), cp_ref_pred.float(), reduction="mean"
                    )
                    roma_org_noise_mse = F.mse_loss(
                        roma_org_pred.float(), org_noise.float(), reduction="mean"
                    )
                    roma_watermarkdm_l1 = _compute_watermarkdm_l1_regularization(
                        accelerator.unwrap_model(unet)
                    )
                    roma_path_loss = (
                        roma_cp_noise_mse
                        - args.cp_ref_weight * roma_cp_ref_mse
                        + args.org_loss_weight * roma_org_noise_mse
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

            accelerator.clip_grad_norm_(tlora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            study_metrics = compute_half_timestep_loss_landscape_study(
                cp_batch,
                org_batch,
                unet,
                vae,
                text_encoder,
                text_encoder_2,
                noise_scheduler,
                accelerator,
                args,
                max_timestep,
                tlora_params,
            )

            wr_minus_wf_norm, wr_minus_wf_ratio = _compute_tlora_delta_norms(
                accelerator.unwrap_model(unet)
            )

            if accelerator.is_main_process:
                log_writer.writerow({
                    "step": step + 1,
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

            if args.overlap_interval > 0 and (step + 1) % args.overlap_interval == 0:
                overlap_metrics = compute_gradient_overlap_study(
                    overlap_cp_batches,
                    overlap_org_batches,
                    unet,
                    vae,
                    text_encoder,
                    text_encoder_2,
                    noise_scheduler,
                    accelerator,
                    args,
                    max_timestep,
                    tlora_params,
                    step + 1,
                )
                if accelerator.is_main_process and overlap_log_writer is not None:
                    overlap_log_writer.writerow({
                        "step": step + 1,
                        "overlap_timestep_mode": overlap_metrics["overlap_timestep_mode"],
                        "overlap_timestep": overlap_metrics["overlap_timestep"],
                        "cp_grad_count": overlap_metrics["cp_grad_count"],
                        "org_grad_count": overlap_metrics["org_grad_count"],
                        "cp_rank": overlap_metrics["cp_rank"],
                        "org_rank": overlap_metrics["org_rank"],
                        "pairwise_cos_mean": f"{overlap_metrics['pairwise_cos_mean']:.8f}",
                        "pairwise_cos_mean_abs": f"{overlap_metrics['pairwise_cos_mean_abs']:.8f}",
                        "pairwise_cos_max_abs": f"{overlap_metrics['pairwise_cos_max_abs']:.8f}",
                        "pairwise_cos_min": f"{overlap_metrics['pairwise_cos_min']:.8f}",
                        "pairwise_cos_max": f"{overlap_metrics['pairwise_cos_max']:.8f}",
                        "principal_overlap": f"{overlap_metrics['principal_overlap']:.8f}",
                    })
                    overlap_log_fh.flush()
                    tqdm.write(
                        f"step {step + 1}: gradient principal_overlap="
                        f"{overlap_metrics['principal_overlap']:.6f}, "
                        f"mean|cos|={overlap_metrics['pairwise_cos_mean_abs']:.6f}, "
                        f"t={overlap_metrics['overlap_timestep']}"
                    )

            active_rank = (
                int(((max_timestep - timesteps[0].item()) / max_timestep) ** args.alpha_rank_scale
                    * (args.rank - args.min_rank))
                + args.min_rank
            )
            progress_bar.update(1)
            progress_bar.set_postfix({
                "cp_half": f"{study_metrics['cp_noise_mse']:.4f}",
                "org_half": f"{study_metrics['org_noise_mse']:.4f}",
                "cos": f"{study_metrics['cp_org_grad_cosine']:.3f}",
                "sharp": f"{study_metrics['cp_sharpness']:.2e}/{study_metrics['org_sharpness']:.2e}",
                "train": f"{final_loss.item():.4f}",
                "wm_l1": f"{watermarkdm_loss.item():.2e}",
                "roma_ls": "off" if roma_path_loss is None else f"{roma_path_loss.item():.4f}",
                "roma_|d|": f"{roma_difference_norm:.2e}",
                "t": f"{timesteps[0].item()}",
                "r": f"{active_rank}/{args.rank}",
                "|dW|/|W|": f"{wr_minus_wf_ratio:.2e}",
            })

            # ---- checkpoint ----
            if args.checkpointing_steps > 0 and (step + 1) % args.checkpointing_steps == 0:
                checkpoint_dir = os.path.join(
                    args.output_dir, f"checkpoint-step{step + 1:06d}"
                )
                save_optimizer_checkpoint(
                    accelerator,
                    optimizer,
                    checkpoint_dir,
                    optimizer_name="adamw",
                    step=step + 1,
                )
                if accelerator.is_main_process:
                    unet_raw = accelerator.unwrap_model(unet)
                    save_tlora_checkpoint(unet_raw, args.output_dir, step + 1, args)
                accelerator.wait_for_everyone()

            # ---- integrity: temporarily merge T-LoRA into backbone and infer ----
            if (args.integrity_interval > 0
                    and (step + 1) % args.integrity_interval == 0
                    and accelerator.is_main_process):
                unet_raw = accelerator.unwrap_model(unet)
                te1_raw  = accelerator.unwrap_model(text_encoder)
                te2_raw  = accelerator.unwrap_model(text_encoder_2)
                unet_raw.eval(); te1_raw.eval(); te2_raw.eval()

                integrity_seed = None if args.integrity_seed is None else int(args.integrity_seed)
                integrity_inference_merged(
                    unet=unet_raw, vae=vae,
                    text_encoder=te1_raw, text_encoder_2=te2_raw,
                    tokenizer=tokenizer, tokenizer_2=tokenizer_2,
                    noise_scheduler=noise_scheduler,
                    prompts=[args.integrity_prompt, org_integrity_prompt],
                    output_paths=[
                        os.path.join(integrity_dir, f"step_{step+1:06d}_cp.png"),
                        os.path.join(integrity_dir, f"step_{step+1:06d}_org.png"),
                    ],
                    resolution=args.resolution,
                    num_steps=args.integrity_inference_steps,
                    seed=integrity_seed,
                )
                unet_raw.train(); te1_raw.eval(); te2_raw.eval()

        progress_bar.close()

    log_fh.close()
    if overlap_log_fh is not None:
        overlap_log_fh.close()

    # ---------- save final ----------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and args.step > 0:
        unet_s = accelerator.unwrap_model(unet)

        final_model_dir = save_merged_checkpoint(
            unet_s, args.output_dir, name="final"
        )

        print(f"\n{'='*60}")
        print(f"Study training complete.  Final model: {final_model_dir}")
        print(f"Loss log: {args.study_log_file}")
        if args.overlap_interval > 0:
            print(f"Gradient overlap log: {args.overlap_log_file}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
