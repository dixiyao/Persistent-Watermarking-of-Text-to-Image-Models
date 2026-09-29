#!/usr/bin/env python3
"""
Train FLUX.2 Klein DreamBooth LoRA on cp_dataset personalization images.

Ordinary PEFT LoRA is the default, using the pinned official Diffusers
DreamBooth trainer with per-image captions from prompt.csv. The legacy T-LoRA
study below requires an explicit --training_mode robust; --org_image never
changes the training mode automatically.
See flux/README.md for installation, personalization, and inference commands.

Robust mode:

Uses rectified flow (velocity prediction) — FLUX's native training objective:
  x_t = (1-t)*x_0 + t*ε,  t ~ logit-normal(0,1)
  target = ε - x_0
The SDXL robust-study objective is retained around that model-specific target:
  cp_loss  = cp_flow_mse - cp_ref_weight * cp_reference_mse
  org_loss = org_loss_weights * org_flow_mse

Text encoding: the single Qwen encoder used by FLUX.2 [Klein].

T-LoRA: orthogonal SVD-init LoRA with sigma mask (rank scales with timestep).
Wraps attention projection linears in the FluxTransformer directly (no PEFT).
Saves as tlora_weights.pt + tlora_config.pt per checkpoint.
"""

import argparse
import csv
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dispatch before the legacy imports: dataset preparation and --help need no
# GPU/model libraries, and inference imports still expose the T-LoRA helpers.
if __name__ == "__main__":
    from flux.dreambooth_personalization import resolve_training_mode, run_dreambooth

    training_mode, training_argv = resolve_training_mode(sys.argv[1:])
    if training_mode == "dreambooth":
        run_dreambooth(training_argv)
        raise SystemExit(0)
    sys.argv[1:] = training_argv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from continue_training_utils import (
    load_optimizer_checkpoint,
    save_optimizer_checkpoint,
)
from tqdm.auto import tqdm

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


def _vae_encode(vae, pixel_values):
    return encode_images(vae, pixel_values)


def _pack_latents(latents):
    return pack_latents(latents)


def _unpack_latents(latents, image_ids, packed_length):
    return unpack_prediction(latents, image_ids, packed_length)


def _encode_text_flux(pipe, prompt_list, device, dtype, max_sequence_length=512):
    return encode_prompts(pipe, prompt_list, device, dtype, max_sequence_length)


def _sample_t_logit_normal(bsz, device):
    return torch.sigmoid(torch.normal(0.0, 1.0, size=(bsz,), device=device))


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


# ── T-LoRA for FLUX transformer ────────────────────────────────────────────────

# FLUX attention projection names to wrap with T-LoRA.
# Covers double-stream (img+txt branches) and single-stream blocks.
_FLUX_TLORA_SUFFIXES = frozenset({
    "to_q", "to_k", "to_v",          # both block types
    "add_q_proj", "add_k_proj", "add_v_proj",  # double-stream text branch
    "to_out", "to_add_out",           # output projections
    "to_qkv_mlp_proj",                 # Klein single-stream projection
})


def setup_flux_tlora(transformer, rank, lora_alpha, sig_type="last",
                     ortho_init="random", skip_init=False):
    """Wrap FLUX attention linear layers with T-LoRA. Returns (layer_paths, trainable_params)."""
    target_paths = []
    for name, module in transformer.named_modules():
        if isinstance(module, nn.Linear) and (
            name.split(".")[-1] in _FLUX_TLORA_SUFFIXES
            or name.endswith(".to_out.0")
            or name.endswith(".attn.to_out")
        ):
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

    print(f"T-LoRA: wrapped {len(target_paths)} FLUX attention projections, "
          f"rank={rank}", flush=True)
    return target_paths, trainable_params


def attach_flux_tlora_sigma_hook(transformer, rank, min_rank, alpha_rank_scale,
                                  max_timestep=1000):
    """Hook transformer.forward to inject sigma mask before each call.

    FLUX transformer.forward expects normalized timesteps and multiplies by 1000
    internally. The T-LoRA mask still uses the 0..max_timestep scale.
    """
    if getattr(transformer, "_tlora_sigma_hook_attached", False):
        return

    orig_forward = transformer.forward

    def forward_with_sigma(hidden_states, timestep, **kwargs):
        t_val = float(timestep.flatten()[0].item()) if torch.is_tensor(timestep) else float(timestep)
        if abs(t_val) <= 1.0:
            t_val *= max_timestep
        mask = get_mask_by_timestep(t_val, max_timestep, rank, min_rank, alpha_rank_scale)
        set_text_encoder_sigma_mask(mask.to(hidden_states.device))
        try:
            return orig_forward(hidden_states=hidden_states, timestep=timestep, **kwargs)
        finally:
            clear_text_encoder_sigma_mask()

    transformer.forward = forward_with_sigma
    transformer._tlora_sigma_hook_attached = True


def collect_flux_tlora_state_dict(transformer, target_paths):
    sd = {}
    for path in target_paths:
        layer = get_layer_by_name(transformer, path)
        if isinstance(layer, TLoRATextLinearLayer):
            for key, val in layer.state_dict().items():
                if key.startswith("tlora."):
                    sd[f"{path}.{key}"] = val
    return sd


def load_flux_tlora_weights(transformer, weights_path, target_paths,
                             rank, lora_alpha, sig_type, ortho_init):
    """Re-wrap layers (skip_init=True) then load saved state dict."""
    for path in target_paths:
        base = get_layer_by_name(transformer, path)
        # Already wrapped? skip re-wrap.
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


def _save_flux_tlora_checkpoint(transformer, target_paths, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    sd = collect_flux_tlora_state_dict(transformer, target_paths)
    torch.save(sd, os.path.join(output_dir, "tlora_weights.pt"))
    torch.save(config, os.path.join(output_dir, "tlora_config.pt"))


def _set_tlora_disabled(model, disabled):
    for module in model.modules():
        if isinstance(module, TLoRATextLinearLayer):
            module.tlora_disabled = bool(disabled)


def _iter_flux_tlora_modules(transformer):
    """Yield the OrthogonalLoRALinearLayer inside each wrapped FLUX projection.

    FLUX wraps plain ``nn.Linear`` projections in ``TLoRATextLinearLayer`` rather
    than swapping an attention processor, so the SDXL path's
    ``unet.attn_processors`` walk has no FLUX equivalent. The inner factor names
    are identical, which keeps the regularizers below the same computation.
    """
    for module in transformer.modules():
        if isinstance(module, TLoRATextLinearLayer):
            yield module.tlora


def _compute_watermarkdm_l1_regularization(transformer):
    """Return ||W_r - W_f||_1 over the trainable T-LoRA weight deltas.

    T-LoRA subtracts its frozen orthogonal initialization in the forward pass,
    so each effective weight change is B S A - B_init S_init A_init.
    This function intentionally keeps autograd enabled for the current factors.
    """
    regularization = None
    for lora_module in _iter_flux_tlora_modules(transformer):
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
    """Yield each trainable T-LoRA factor and its frozen pretrained baseline."""
    for lora_module in _iter_flux_tlora_modules(transformer):
        yield lora_module.q_layer.weight, lora_module.base_q.weight
        yield lora_module.p_layer.weight, lora_module.base_p.weight
        yield lora_module.lambda_layer, lora_module.base_lambda


def _apply_roma_path_perturbation(transformer, path_step_size):
    """Move T-LoRA factors by ``r`` toward their pretrained parameter point.

    Implements theta_tilde = theta + r * (theta_0 - theta) /
    ||theta_0 - theta|| from Algorithm 1 of RoMa. Returns the applied
    perturbations so the caller can restore theta before the optimizer step.
    """
    parameter_baselines = list(_iter_tlora_parameter_baselines(transformer))
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


def _csv_header_matches(path, fieldnames):
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline().strip()
    return bool(first_line) and first_line.split(",") == list(fieldnames)


def _parse_step(name):
    if name.startswith("checkpoint-"):
        tok = name[len("checkpoint-"):]
        if tok.isdigit():
            return int(tok)
    return 0


def _generate_integrity_image(pipe, transformer_unwrapped, prompt, output_path,
                               resolution, num_steps=20, guidance_scale=3.5, seed=42):
    orig_t = pipe.transformer
    orig_decode = pipe.vae.decode
    pipe.transformer = transformer_unwrapped
    pipe.transformer.eval()
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    pipe.vae.decode = lambda z, *a, **kw: orig_decode(
        z.to(dtype=next(pipe.vae.parameters()).dtype), *a, **kw
    )
    try:
        image = pipe(
            prompt=prompt, num_inference_steps=num_steps,
            guidance_scale=guidance_scale, height=resolution, width=resolution,
            generator=generator,
        ).images[0]
        image.save(output_path)
        print(f"Integrity: {output_path}", flush=True)
    finally:
        pipe.vae.decode = orig_decode
        pipe.transformer = orig_t
        transformer_unwrapped.train()


def main():
    parser = argparse.ArgumentParser(description="Train FLUX T-LoRA on copyright images")
    parser.add_argument("--cp_dataset", type=str, required=True,
                        help="Copyright dataset dir (image/ + prompt.csv)")
    parser.add_argument("--org_image", type=str, required=True,
                        help="Original image dataset dir (image/ + prompt.csv)")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default=DEFAULT_MODEL,
                        help="FLUX.2 [Klein] Diffusers checkpoint.")
    parser.add_argument("--output_dir", type=str, default="checkpoints_flux_tlora")
    parser.add_argument("--study_log_file", type=str, default=None,
                        help="Defaults to <output_dir>/study_loss_log.csv.")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--sig_type", type=str, default="last",
                        choices=["last", "principal", "middle"])
    parser.add_argument("--ortho_init", type=str, default="random",
                        choices=["random", "base_layer"])
    parser.add_argument("--ord_lora", action="store_true",
                        help="Replace T-LoRA with an ordinary LoRA of the same "
                             "rank and targets: standard A/B initialization and "
                             "no timestep sigma mask, so the full rank is active "
                             "at every timestep. Appendix A.1 of the paper uses "
                             "this for FLUX.2 Klein. The checkpoint format is "
                             "unchanged, so every downstream attack works either "
                             "way.")
    parser.add_argument("--min_rank", type=int, default=None,
                        help="Minimum active rank at highest timestep (most noise)")
    parser.add_argument("--alpha_rank_scale", type=float, default=1.0,
                        help="Controls how fast rank grows as timestep decreases")
    parser.add_argument("--max_timestep", type=int, default=1000)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    # A constant 1e-4 for 10k steps on the 4B klein transformer diverges: the
    # loss bottoms out near step 3000 and then climbs, and by step 10000 the
    # T-LoRA delta exceeds the base weight norm and the samples are mush.
    # These three knobs default to the previous behaviour (constant LR, wd 1e-2,
    # no warmup) so existing runs are bit-identical unless they are passed.
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help="diffusers get_scheduler name, e.g. cosine, "
                             "constant_with_warmup, constant.")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--cp_ref_weight", type=float, default=1.0,
                        help="Subtracted frozen-backbone consistency weight for copyright samples.")
    parser.add_argument("--org_loss_weights", type=float, default=1.0,
                        help="Weight applied to the original-image flow-matching loss.")
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
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Training/integrity guidance scale; Klein's default is 1.0")
    parser.add_argument("--max_train_steps", "--step", "--cp_step",
                        dest="max_train_steps", type=int, default=400)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--integrity_prompt", type=str, default="A [Z]*$ at the lake")
    parser.add_argument("--integrity_inference_steps", type=int, default=4)
    parser.add_argument("--integrity_interval", type=int, default=500)
    parser.add_argument("--cp_steps", type=int, default=0,
                        help="Deprecated compatibility option; CP and original batches are now trained jointly.")
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
        args.org_image,
        args.original_trigger_word,
        args.current_trigger_word,
    )

    if args.auto_resume_latest and args.resume_from_checkpoint is None:
        if os.path.isdir(args.output_dir):
            cands = [
                (int(n[len("checkpoint-"):]), n)
                for n in os.listdir(args.output_dir)
                if n.startswith("checkpoint-") and n[len("checkpoint-"):].isdigit()
            ]
            if cands:
                _, latest = max(cands, key=lambda x: x[0])
                args.resume_from_checkpoint = latest

    resume_step = 0
    resume_ckpt = None
    if args.resume_from_checkpoint:
        ckpt = args.resume_from_checkpoint
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(args.output_dir, ckpt)
        resume_ckpt = ckpt
        resume_step = _parse_step(os.path.basename(ckpt))
        print(f"Resuming from {ckpt} (step {resume_step})")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.study_log_file is None:
        args.study_log_file = os.path.join(args.output_dir, "study_loss_log.csv")
    if args.cp_steps:
        print(
            "WARNING: --cp_steps is deprecated; robust-study training now uses "
            "joint copyright/original batches like the SDXL implementation.",
            flush=True,
        )
    model_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.mixed_precision, torch.float32)
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    print(f"Loading FLUX: {args.pretrained_model_name_or_path}", flush=True)
    pipe = load_pipeline(args.pretrained_model_name_or_path, model_dtype)
    transformer = pipe.transformer
    _ensure_flux_guidance_config(transformer)
    vae = pipe.vae
    noise_scheduler = prepare_training_schedule(pipe.scheduler, accelerator.device)
    text_encoder = pipe.text_encoder

    transformer.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    tlora_config = {
        "rank": args.rank, "lora_alpha": args.lora_alpha,
        "sig_type": args.sig_type,
        "ortho_init": "lora" if args.ord_lora else args.ortho_init,
        "ord_lora": bool(args.ord_lora),
        "min_rank": args.min_rank, "alpha_rank_scale": args.alpha_rank_scale,
        "max_timestep": args.max_timestep,
    }

    weights_path = os.path.join(resume_ckpt, "tlora_weights.pt") if resume_ckpt else None
    if weights_path and os.path.isfile(weights_path):
        saved_cfg = torch.load(os.path.join(resume_ckpt, "tlora_config.pt"), map_location="cpu")
        target_paths, _ = setup_flux_tlora(
            transformer, saved_cfg["rank"], saved_cfg["lora_alpha"],
            saved_cfg["sig_type"], saved_cfg["ortho_init"], skip_init=True,
        )
        trainable_params = load_flux_tlora_weights(
            transformer, weights_path, target_paths,
            saved_cfg["rank"], saved_cfg["lora_alpha"], saved_cfg["sig_type"], saved_cfg["ortho_init"],
        )
        tlora_config.update(saved_cfg)
        print(f"Resumed T-LoRA from {resume_ckpt} (step {resume_step})")
    else:
        target_paths, trainable_params = setup_flux_tlora(
            transformer, args.rank, args.lora_alpha, args.sig_type,
            "lora" if args.ord_lora else args.ortho_init, skip_init=False,
        )

    # The sigma hook is what makes this T-LoRA: it masks the active rank by
    # timestep. Ordinary LoRA leaves the mask unset, and forward() then
    # defaults to all-ones, i.e. full rank at every timestep.
    if tlora_config.get("ord_lora"):
        print("Ordinary LoRA: no timestep sigma mask, rank=%d" % args.rank,
              flush=True)
    else:
        attach_flux_tlora_sigma_hook(
            transformer, args.rank, args.min_rank, args.alpha_rank_scale,
            args.max_timestep
        )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    print(f"Trainable T-LoRA params: {sum(p.numel() for p in trainable_params):,}", flush=True)

    def make_dataset(data_dir):
        return SimpleDreamBoothDataset(
            csv_path=os.path.join(data_dir, "prompt.csv"),
            image_dir=os.path.join(data_dir, "image"),
            tokenizer=None, tokenizer_2=None,
            size=args.resolution, center_crop=False,
            original_trigger_word=args.original_trigger_word,
            current_trigger_word=args.current_trigger_word,
        )

    cp_loader = infinite_dataloader(make_dataset(args.cp_dataset), args.seed, args.train_batch_size, collate_fn=prompt_image_collate_fn)
    org_dataset = make_org_image_dataset(
        args.org_image,
        None,
        None,
        size=args.resolution,
        original_trigger_word=args.original_trigger_word,
        current_trigger_word=args.current_trigger_word,
    )
    org_loader = infinite_dataloader(
        org_dataset,
        args.seed + 1,
        args.train_batch_size,
        collate_fn=prompt_image_collate_fn,
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                  betas=(0.9, 0.999),
                                  weight_decay=args.adam_weight_decay)
    from diffusers.optimization import get_scheduler as _get_scheduler
    lr_scheduler = _get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    transformer, optimizer, lr_scheduler = accelerator.prepare(
        transformer, optimizer, lr_scheduler
    )
    load_optimizer_checkpoint(
        accelerator,
        optimizer,
        resume_ckpt,
        expected_optimizer_name="adamw",
        expected_step=resume_step,
    )
    # The schedule is not part of the optimizer checkpoint, so a resumed run
    # would otherwise restart from warmup at full LR. Fast-forward it.
    for _ in range(int(resume_step)):
        lr_scheduler.step()
    vae = vae.to(accelerator.device, dtype=model_dtype)
    text_encoder = text_encoder.to(accelerator.device, dtype=model_dtype)
    device = accelerator.device

    if accelerator.is_main_process:
        pipe.transformer = accelerator.unwrap_model(transformer)
        pipe = pipe.to(device)
        integrity_dir = os.path.join(args.output_dir, "integrity_images")
        os.makedirs(integrity_dir, exist_ok=True)
        _generate_integrity_image(
            pipe, accelerator.unwrap_model(transformer), args.integrity_prompt,
            os.path.join(integrity_dir, f"step_{resume_step:06d}_cp.png"),
            args.resolution, args.integrity_inference_steps, args.guidance_scale,
        )
        _generate_integrity_image(
            pipe, accelerator.unwrap_model(transformer), org_integrity_prompt,
            os.path.join(integrity_dir, f"step_{resume_step:06d}_org.png"),
            args.resolution, args.integrity_inference_steps, args.guidance_scale,
        )

    backbone_info = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "cp_ref_weight": args.cp_ref_weight,
        "org_loss_weights": args.org_loss_weights,
        "lambda_watermarkdm": args.lambda_watermarkdm,
        "roma": args.roma,
        "roma_alpha": args.roma_alpha,
        "roma_r": args.roma_r,
        **tlora_config,
    }

    log_fh = None
    log_writer = None
    log_fields = [
        "step",
        "loss",
        "cp_loss",
        "org_loss",
        "cp_noise_mse",
        "cp_ref_mse",
        "org_noise_mse",
        "t_mean",
        "watermarkdm_l1",
        "watermarkdm_loss",
        "final_loss",
        "roma_path_loss",
        "roma_difference_norm",
    ]
    log_append = (
        os.path.exists(args.study_log_file)
        and resume_step > 0
        and _csv_header_matches(args.study_log_file, log_fields)
    )
    if accelerator.is_main_process:
        log_fh = open(
            args.study_log_file,
            "a" if log_append else "w",
            encoding="utf-8",
            newline="",
        )
        log_writer = csv.DictWriter(log_fh, fieldnames=log_fields)
        if not log_append:
            log_writer.writeheader()
            log_fh.flush()

    global_step = int(resume_step)
    transformer.train()
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process, desc="FLUX-LoRA")
    if global_step > 0:
        progress_bar.update(min(global_step, args.max_train_steps))

    for _ in range(args.max_train_steps * 10):
        if global_step >= args.max_train_steps:
            break

        cp_batch = next(cp_loader)
        org_batch = next(org_loader)
        cp_count = cp_batch["pixel_values"].shape[0]
        prompts = list(cp_batch["prompt"]) + list(org_batch["prompt"])

        with torch.no_grad():
            pv = torch.cat(
                [cp_batch["pixel_values"], org_batch["pixel_values"]],
                dim=0,
            ).to(device=vae.device, dtype=vae.dtype)
            latents = _vae_encode(vae, pv).to(device, dtype=model_dtype)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            t, sigmas = sample_training_timestep(
                noise_scheduler, bsz, device, latents.dtype, shared=True
            )
            x_t = (1 - sigmas) * latents + sigmas * noise
            target = noise - latents

        prompt_embeds, text_ids = _encode_text_flux(
            pipe, prompts, device, model_dtype
        )

        packed = _pack_latents(x_t)
        img_ids = latent_ids(x_t)
        guidance_kwargs = _flux_guidance_kwargs(transformer, bsz, device, args.guidance_scale)

        pred_packed = transformer(
            hidden_states=packed,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
            **guidance_kwargs,
        )[0]
        pred = _unpack_latents(pred_packed, img_ids, packed.shape[1])

        was_training = transformer.training
        try:
            _set_tlora_disabled(transformer, True)
            transformer.eval()
            with torch.no_grad():
                ref_pred_packed = transformer(
                    hidden_states=packed[:cp_count],
                    timestep=t[:cp_count],
                    encoder_hidden_states=prompt_embeds[:cp_count],
                    txt_ids=text_ids[:cp_count],
                    img_ids=img_ids[:cp_count],
                    return_dict=False,
                    **_flux_guidance_kwargs(transformer, cp_count, device, args.guidance_scale),
                )[0]
                cp_ref_pred = _unpack_latents(
                    ref_pred_packed, img_ids[:cp_count], packed.shape[1]
                ).detach()
        finally:
            _set_tlora_disabled(transformer, False)
            if was_training:
                transformer.train()

        cp_pred, org_pred = pred[:cp_count], pred[cp_count:]
        cp_target, org_target = target[:cp_count], target[cp_count:]
        cp_noise_mse = F.mse_loss(cp_pred.float(), cp_target.float())
        cp_ref_mse = F.mse_loss(cp_pred.float(), cp_ref_pred.float())
        org_noise_mse = F.mse_loss(org_pred.float(), org_target.float())
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
            # RoMa Algorithm 1: accumulate (1-alpha) * g1 at the current
            # parameters, then alpha * g2 at a normalized step toward the
            # frozen pretrained T-LoRA point. Restore before Adam updates.
            accelerator.backward((1.0 - args.roma_alpha) * loss)
            roma_perturbations, roma_difference_norm = _apply_roma_path_perturbation(
                accelerator.unwrap_model(transformer),
                args.roma_r,
            )
            try:
                roma_pred_packed = transformer(
                    hidden_states=packed,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                    **guidance_kwargs,
                )[0]
                roma_pred = _unpack_latents(
                    roma_pred_packed, img_ids, packed.shape[1]
                )
                roma_cp_pred = roma_pred[:cp_count]
                roma_org_pred = roma_pred[cp_count:]
                roma_cp_noise_mse = F.mse_loss(
                    roma_cp_pred.float(), cp_target.float()
                )
                roma_cp_ref_mse = F.mse_loss(
                    roma_cp_pred.float(), cp_ref_pred.float()
                )
                roma_org_noise_mse = F.mse_loss(
                    roma_org_pred.float(), org_target.float()
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

        accelerator.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "cp": f"{cp_loss.item():.4f}",
            "org": f"{org_loss.item():.4f}",
            "train": f"{final_loss.item():.4f}",
            "wm_l1": f"{watermarkdm_loss.item():.2e}",
            "roma_ls": "off" if roma_path_loss is None else f"{roma_path_loss.item():.4f}",
            "roma_|d|": f"{roma_difference_norm:.2e}",
            "t": f"{t.mean().item():.3f}",
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
                "t_mean": f"{t.mean().item():.6f}",
                "watermarkdm_l1": f"{watermarkdm_l1.detach().item():.8e}",
                "watermarkdm_loss": f"{watermarkdm_loss.detach().item():.8e}",
                "final_loss": f"{final_loss.item():.6f}",
                "roma_path_loss": (
                    "" if roma_path_loss is None
                    else f"{roma_path_loss.detach().item():.6f}"
                ),
                "roma_difference_norm": f"{roma_difference_norm:.8e}",
            })
            log_fh.flush()

        if (accelerator.is_main_process and args.integrity_interval > 0
                and global_step % args.integrity_interval == 0):
            _generate_integrity_image(
                pipe, accelerator.unwrap_model(transformer), args.integrity_prompt,
                os.path.join(integrity_dir, f"step_{global_step:06d}_cp.png"),
                args.resolution, args.integrity_inference_steps, args.guidance_scale,
            )
            _generate_integrity_image(
                pipe, accelerator.unwrap_model(transformer), org_integrity_prompt,
                os.path.join(integrity_dir, f"step_{global_step:06d}_org.png"),
                args.resolution, args.integrity_inference_steps, args.guidance_scale,
            )

        if global_step % args.checkpointing_steps == 0:
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            save_optimizer_checkpoint(
                accelerator,
                optimizer,
                ckpt_dir,
                optimizer_name="adamw",
                step=global_step,
            )
            if accelerator.is_main_process:
                _save_flux_tlora_checkpoint(
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

    if log_fh is not None:
        log_fh.close()
    accelerator.wait_for_everyone()
    final_dir = os.path.join(args.output_dir, "final")
    save_optimizer_checkpoint(
        accelerator,
        optimizer,
        final_dir,
        optimizer_name="adamw",
        step=global_step,
    )
    if accelerator.is_main_process:
        _save_flux_tlora_checkpoint(
            accelerator.unwrap_model(transformer), target_paths, final_dir, tlora_config
        )
        with open(os.path.join(final_dir, "backbone_info.json"), "w") as f:
            json.dump(backbone_info, f, indent=2)
        print(f"Done. Final T-LoRA → {final_dir}")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
