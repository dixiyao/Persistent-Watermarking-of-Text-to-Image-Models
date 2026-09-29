"""Copyright robust/continuation training for the aMUSEd masked-token model."""

from __future__ import annotations

import csv
import json
import os
import shutil

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from continue_training_utils import (
    load_optimizer_checkpoint,
    save_optimizer_checkpoint,
)

from muse.backend import (
    DEFAULT_MODEL,
    add_lora,
    load_adapter,
    load_pipeline,
    loss_from_masked_inputs,
    masked_token_loss,
    prepare_masked_inputs,
)
from utils import (
    SimpleDreamBoothDataset,
    first_prompt_from_dataset,
    infinite_dataloader,
    make_org_image_dataset,
    replace_trigger_word,
)


def _save_integrity_image(pipe, prompt, path, resolution, steps, guidance, seed=0):
    """One 512x512 sample from the live adapter, written next to the checkpoints.

    get_peft_model injects the LoRA layers into pipe.transformer in place, so the
    pipeline already sees the adapted weights; nothing needs to be re-attached.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    was_training = pipe.transformer.training
    pipe.transformer.eval()
    try:
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=resolution,
                width=resolution,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]
        image.save(path)
    except Exception as exc:                      # never kill a run over a preview
        print(f"integrity image failed ({path}): {exc}", flush=True)
    finally:
        if was_training:
            pipe.transformer.train()


def apply_flat_perturbation(params, vector, scale):
    offset = 0
    with torch.no_grad():
        for param in params:
            numel = param.numel()
            chunk = vector[offset:offset + numel].view_as(param).to(
                device=param.device, dtype=param.dtype
            )
            param.add_(chunk, alpha=scale)
            offset += numel


def sharpness_along_gradient(transformer, params, grad_vec, base_loss, inputs, rho):
    """SAM-style probe: loss increase after an rho-normalized ascent step.

    aMUSEd has no diffusion timestep, so this replaces the half-timestep probe
    used by the SDXL/FLUX/PixArt studies. The inputs are held fixed across both
    evaluations so the number reflects curvature rather than mask resampling.
    """
    grad_norm = float(grad_vec.norm().item())
    if grad_norm <= 1e-12:
        return float("nan")
    scale = rho / grad_norm
    apply_flat_perturbation(params, grad_vec, scale)
    try:
        with torch.no_grad():
            perturbed = float(loss_from_masked_inputs(transformer, inputs)[0].item())
    finally:
        apply_flat_perturbation(params, grad_vec, -scale)
    return perturbed - base_loss


def peft_delta_norms(model):
    """||W_r - W_f|| and its ratio over the LoRA-adapted base weights."""
    delta_sq = 0.0
    base_sq = 0.0
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
                delta = (b @ a) * module.scaling.get(adapter, 1.0)
            delta_sq += float(delta.pow(2).sum().item())
            base_layer = (
                module.get_base_layer() if hasattr(module, "get_base_layer")
                else getattr(module, "base_layer", None)
            )
            if base_layer is not None and hasattr(base_layer, "weight"):
                base_sq += float(base_layer.weight.detach().float().pow(2).sum().item())
    delta_norm = delta_sq ** 0.5
    return delta_norm, delta_norm / max(base_sq ** 0.5, 1e-12)


def _iter_lora_modules(model):
    for _name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield module


def watermarkdm_l1(model):
    """||W_r - W_f||_1 over the effective LoRA weight change.

    For a PEFT adapter the pretrained point is delta = 0, so the effective
    change is just the adapter's delta weight. Autograd is kept live.
    """
    total = None
    for module in _iter_lora_modules(model):
        for adapter in module.lora_A:
            if adapter not in module.lora_B:
                continue
            a = module.lora_A[adapter].weight
            b = module.lora_B[adapter].weight
            if a.ndim != 2 or b.ndim != 2:
                continue
            delta = (b @ a) * module.scaling.get(adapter, 1.0)
            term = delta.float().abs().sum()
            total = term if total is None else total + term
    if total is None:
        raise RuntimeError("No LoRA layers found for the WatermarkDM regularizer")
    return total


def apply_roma_path_perturbation(params, path_step_size):
    """theta_tilde = theta + r * (theta_0 - theta) / ||theta_0 - theta||.

    aMUSEd trains a PEFT adapter rather than T-LoRA, so the pretrained point
    theta_0 is the zero adapter: the direction is simply -theta.
    """
    if not params:
        raise RuntimeError("No trainable parameters for RoMa")
    with torch.no_grad():
        norm_sq = torch.zeros((), device=params[0].device, dtype=torch.float32)
        for p in params:
            norm_sq.add_(p.detach().float().pow(2).sum())
        difference_norm = norm_sq.sqrt()
        coefficient = float(path_step_size) / difference_norm.clamp_min(1e-8)
        perturbations = []
        for p in params:
            perturbation = (-p.detach()) * coefficient.to(device=p.device, dtype=p.dtype)
            p.add_(perturbation)
            perturbations.append((p, perturbation))
    return perturbations, float(difference_norm.item())


def restore_roma_path_perturbation(perturbations):
    with torch.no_grad():
        for p, perturbation in perturbations:
            p.sub_(perturbation)


def parse_step(name):
    if name.startswith("checkpoint-"):
        tok = name[len("checkpoint-"):]
        if tok.isdigit():
            return int(tok)
    return 0


def resolve_resume(args):
    """--auto_resume_latest support so a 12 h relaunch continues the run."""
    ckpt = getattr(args, "resume_from_checkpoint", None)
    if ckpt is None and getattr(args, "auto_resume_latest", False) and os.path.isdir(args.output_dir):
        cands = [
            (parse_step(n), n) for n in os.listdir(args.output_dir)
            if n.startswith("checkpoint-") and n[len("checkpoint-"):].isdigit()
            and os.path.isfile(os.path.join(args.output_dir, n, "adapter_config.json"))
        ]
        if cands:
            ckpt = max(cands, key=lambda x: x[0])[1]
    if ckpt is None:
        return None, 0
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(args.output_dir, ckpt)
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt}")
    return ckpt, parse_step(os.path.basename(ckpt))


def collate(examples):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in examples]),
        "prompt": [x["prompt"] for x in examples],
        "image_name": [x["image_name"] for x in examples],
    }


def dataset(path, size, original_trigger_word=None, current_trigger_word=None):
    # The attack stage trains on COCO2014, whose official train layout is
    # train2014/ + annotations/captions_train2014.json and has no prompt.csv.
    # The shared loader accepts that as well as the image/ + prompt.csv layout
    # the copyright and original datasets use.
    return make_org_image_dataset(
        path, None, None, size=size,
        original_trigger_word=original_trigger_word,
        current_trigger_word=current_trigger_word,
    )


def flat_grad(loss, params, retain_graph=False):
    grads = torch.autograd.grad(
        loss, params, retain_graph=retain_graph, allow_unused=True
    )
    return torch.cat([
        (torch.zeros_like(p) if g is None else g).detach().float().reshape(-1).cpu()
        for p, g in zip(params, grads)
    ])


def save_adapter(accelerator, transformer, output_dir, info):
    os.makedirs(output_dir, exist_ok=True)
    accelerator.unwrap_model(transformer).save_pretrained(output_dir)
    with open(os.path.join(output_dir, "backbone_info.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)


def run(args, mode):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    torch.manual_seed(args.seed)
    pipe = load_pipeline(args.pretrained_model_name_or_path, torch.float16)
    pipe.text_encoder.requires_grad_(False)
    pipe.vqvae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)

    if mode == "continue":
        robust = load_adapter(pipe.transformer, args.merged_checkpoint, trainable=False)
        pipe.transformer = robust.merge_and_unload()
    resume_path, resume_step = resolve_resume(args)
    if resume_path is not None:
        print(f"Resuming from {resume_path} (step {resume_step})", flush=True)
        transformer = load_adapter(pipe.transformer, resume_path, trainable=True)
    else:
        transformer = add_lora(pipe.transformer, args.rank, args.lora_alpha, args.lora_dropout)
    params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    load_optimizer_checkpoint(
        accelerator,
        optimizer,
        resume_path,
        expected_optimizer_name="adamw",
        expected_step=resume_step,
    )
    device = accelerator.device
    pipe.text_encoder.to(device=device, dtype=torch.float16)
    pipe.vqvae.to(device=device, dtype=torch.float32)

    train_path = args.data_dir if mode == "continue" else args.cp_dataset
    train_loader = infinite_dataloader(
        dataset(train_path, args.resolution, args.original_trigger_word, args.current_trigger_word),
        args.seed + accelerator.process_index, args.train_batch_size, collate_fn=collate,
    )
    cp_loader = infinite_dataloader(
        dataset(args.cp_dataset, args.resolution, args.original_trigger_word, args.current_trigger_word),
        args.seed + 17 + accelerator.process_index, 1, collate_fn=collate,
    )
    org_data = make_org_image_dataset(
        args.org_image, None, None, size=args.resolution,
        original_trigger_word=args.original_trigger_word,
        current_trigger_word=args.current_trigger_word,
    )
    org_loader = infinite_dataloader(
        org_data, args.seed + 29 + accelerator.process_index, 1, collate_fn=collate,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = args.study_log_file or os.path.join(args.output_dir, "study_loss_log.csv")
    fields = [
        "step", "loss", "cp_loss", "org_loss", "mask_probability",
        "cp_org_grad_cosine", "cp_grad_norm", "org_grad_norm",
        "cp_sharpness", "org_sharpness",
        "wr_minus_wf_norm", "wr_minus_wf_over_wf_norm",
        "watermarkdm_l1", "watermarkdm_loss", "final_loss",
        "roma_path_loss", "roma_difference_norm",
    ]
    log_append = resume_step > 0 and os.path.exists(log_path)
    handle = (
        open(log_path, "a" if log_append else "w", newline="", encoding="utf-8")
        if accelerator.is_main_process else None
    )
    writer = csv.DictWriter(handle, fieldnames=fields) if handle else None
    if writer and not log_append:
        writer.writeheader()

    info = {
        "backend": "amused_masked_token",
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "study_protocol": "gradient_geometry",
        "rank": args.rank,
        "lora_alpha": args.lora_alpha,
        "mode": mode,
        "cp_ref_weight": args.cp_ref_weight if mode == "robust" else None,
        "org_loss_weights": args.org_loss_weights,
        "sharpness_rho": args.sharpness_rho,
        "lambda_watermarkdm": args.lambda_watermarkdm,
        "roma": bool(getattr(args, "roma", False)),
        "roma_alpha": args.roma_alpha,
        "roma_r": args.roma_r,
        "study_timestep_analogue": "fixed_mask_probability_0.5",
    }
    integrity_dir = os.path.join(args.output_dir, "integrity_images")
    cp_integrity_prompt = replace_trigger_word(
        args.integrity_prompt, args.original_trigger_word, args.current_trigger_word
    )
    org_integrity_prompt = first_prompt_from_dataset(
        args.org_image, args.original_trigger_word, args.current_trigger_word
    )

    def write_integrity(at_step):
        if not (accelerator.is_main_process and args.integrity_interval > 0):
            return
        for tag, prompt in (("cp", cp_integrity_prompt), ("org", org_integrity_prompt)):
            _save_integrity_image(
                pipe, prompt,
                os.path.join(integrity_dir, f"step_{at_step:06d}_{tag}.png"),
                args.resolution, args.integrity_inference_steps,
                args.integrity_guidance_scale,
            )

    write_integrity(resume_step)
    transformer.train()
    for step in range(resume_step + 1, args.max_train_steps + 1):
        train_batch = next(train_loader)
        train_loss, _, mask_rate = masked_token_loss(
            transformer, pipe.vqvae, pipe.tokenizer, pipe.text_encoder,
            train_batch, device,
        )
        if mode == "robust":
            # Mirror the SDXL/FLUX/PixArt robust objective:
            #   cp_loss = cp_task_loss - cp_ref_weight * consistency_with_frozen
            # Without the subtracted reference term the adapter has no pressure
            # to move the copyright prompt away from the frozen backbone, which
            # is the mechanism the whole method rests on.
            if args.cp_ref_weight:
                cp_inputs = prepare_masked_inputs(
                    transformer, pipe.vqvae, pipe.tokenizer, pipe.text_encoder,
                    train_batch, device,
                )
                with torch.no_grad(), accelerator.unwrap_model(transformer).disable_adapter():
                    ref_logits = loss_from_masked_inputs(transformer, cp_inputs)[1].detach()
                adapted_logits = loss_from_masked_inputs(transformer, cp_inputs)[1]
                cp_ref_consistency = F.mse_loss(
                    adapted_logits.float(), ref_logits.float()
                )
                train_loss = train_loss - args.cp_ref_weight * cp_ref_consistency
            org_train_loss, _, _ = masked_token_loss(
                transformer, pipe.vqvae, pipe.tokenizer, pipe.text_encoder,
                next(org_loader), device,
            )
            train_loss = train_loss + args.org_loss_weights * org_train_loss
        wm_l1 = watermarkdm_l1(accelerator.unwrap_model(transformer))
        lora_scaling = args.lora_alpha / args.rank
        wm_loss = args.lambda_watermarkdm * lora_scaling * wm_l1
        train_loss = train_loss + wm_loss

        roma_path_loss = None
        roma_difference_norm = 0.0
        if getattr(args, "roma", False):
            # RoMa Algorithm 1: (1-alpha)*g1 here, then alpha*g2 at a normalized
            # step toward the pretrained (zero-adapter) point. Restore before Adam.
            accelerator.backward((1.0 - args.roma_alpha) * train_loss)
            roma_perturbations, roma_difference_norm = apply_roma_path_perturbation(
                params, args.roma_r
            )
            try:
                roma_path_loss, _, _ = masked_token_loss(
                    transformer, pipe.vqvae, pipe.tokenizer, pipe.text_encoder,
                    train_batch, device,
                )
                roma_path_loss = roma_path_loss + args.lambda_watermarkdm * lora_scaling * \
                    watermarkdm_l1(accelerator.unwrap_model(transformer))
                accelerator.backward(args.roma_alpha * roma_path_loss)
            finally:
                restore_roma_path_perturbation(roma_perturbations)
            final_loss = (
                (1.0 - args.roma_alpha) * train_loss.detach()
                + args.roma_alpha * roma_path_loss.detach()
            )
        else:
            accelerator.backward(train_loss)
            final_loss = train_loss.detach()

        accelerator.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # A fixed 0.5 masking ratio is aMUSEd's stand-in for the half timestep:
        # both put the model exactly halfway between "sees everything" and
        # "sees nothing", which is what makes CP and ORG comparable.
        study_cp_inputs = prepare_masked_inputs(
            transformer, pipe.vqvae, pipe.tokenizer, pipe.text_encoder,
            next(cp_loader), device, mask_probability=0.5,
        )
        study_org_inputs = prepare_masked_inputs(
            transformer, pipe.vqvae, pipe.tokenizer, pipe.text_encoder,
            next(org_loader), device, mask_probability=0.5,
        )
        cp_loss = loss_from_masked_inputs(transformer, study_cp_inputs)[0]
        org_loss = loss_from_masked_inputs(transformer, study_org_inputs)[0]
        cp_loss_value = float(cp_loss.detach().item())
        org_loss_value = float(org_loss.detach().item())
        cp_grad = flat_grad(cp_loss, params, retain_graph=True)
        org_grad = flat_grad(org_loss, params)
        cp_norm, org_norm = cp_grad.norm(), org_grad.norm()
        cosine = F.cosine_similarity(cp_grad, org_grad, dim=0).item()
        cp_sharpness = sharpness_along_gradient(
            transformer, params, cp_grad, cp_loss_value, study_cp_inputs, args.sharpness_rho
        )
        org_sharpness = sharpness_along_gradient(
            transformer, params, org_grad, org_loss_value, study_org_inputs, args.sharpness_rho
        )
        wr_minus_wf_norm, wr_minus_wf_ratio = peft_delta_norms(
            accelerator.unwrap_model(transformer)
        )
        if writer:
            writer.writerow({
                "step": step,
                "loss": f"{train_loss.detach().item():.6f}",
                "cp_loss": f"{cp_loss_value:.6f}",
                "org_loss": f"{org_loss_value:.6f}",
                "mask_probability": f"{mask_rate.detach().item():.6f}",
                "cp_org_grad_cosine": f"{cosine:.8f}",
                "cp_grad_norm": f"{cp_norm.item():.8e}",
                "org_grad_norm": f"{org_norm.item():.8e}",
                "cp_sharpness": f"{cp_sharpness:.8f}",
                "org_sharpness": f"{org_sharpness:.8f}",
                "wr_minus_wf_norm": f"{wr_minus_wf_norm:.8e}",
                "wr_minus_wf_over_wf_norm": f"{wr_minus_wf_ratio:.8e}",
                "watermarkdm_l1": f"{wm_l1.detach().item():.8e}",
                "watermarkdm_loss": f"{wm_loss.detach().item():.8e}",
                "final_loss": f"{final_loss.item():.6f}",
                "roma_path_loss": (
                    "" if roma_path_loss is None
                    else f"{roma_path_loss.detach().item():.6f}"
                ),
                "roma_difference_norm": f"{roma_difference_norm:.8e}",
            })
            handle.flush()
        if args.integrity_interval > 0 and step % args.integrity_interval == 0:
            write_integrity(step)
        if step % args.checkpointing_steps == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            save_optimizer_checkpoint(
                accelerator,
                optimizer,
                checkpoint_dir,
                optimizer_name="adamw",
                step=step,
            )
            if accelerator.is_main_process:
                save_adapter(accelerator, transformer, checkpoint_dir, info)
                kept = sorted(
                    (d for d in os.listdir(args.output_dir)
                     if d.startswith("checkpoint-") and d[len("checkpoint-"):].isdigit()),
                    key=lambda x: int(x[len("checkpoint-"):]),
                )
                for old_ckpt in kept[:-args.checkpoints_total_limit]:
                    shutil.rmtree(os.path.join(args.output_dir, old_ckpt), ignore_errors=True)
            accelerator.wait_for_everyone()

    final_dir = os.path.join(args.output_dir, "final")
    save_optimizer_checkpoint(
        accelerator,
        optimizer,
        final_dir,
        optimizer_name="adamw",
        step=args.max_train_steps,
    )
    if accelerator.is_main_process:
        save_adapter(accelerator, transformer, final_dir, info)
    accelerator.wait_for_everyone()
    if handle:
        handle.close()


def add_common_args(parser, mode):
    parser.add_argument("--pretrained_model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--cp_dataset", required=True)
    parser.add_argument("--org_image", required=True)
    if mode == "continue":
        parser.add_argument("--merged_checkpoint", required=True)
        parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default=f"checkpoints_muse_{mode}")
    parser.add_argument("--study_log_file")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--org_loss_weights", type=float, default=1.0)
    parser.add_argument("--cp_ref_weight", type=float, default=1.0,
                        help="Subtracted frozen-backbone consistency weight for copyright "
                             "samples; 0 disables the reference term.")
    parser.add_argument("--sharpness_rho", type=float, default=0.05,
                        help="Radius for the SAM-style loss sharpness probe.")
    parser.add_argument("--lambda_watermarkdm", type=float, default=0.0,
                        help="Weight for the WatermarkDM L1 penalty on the LoRA delta.")
    parser.add_argument("--roma", action="store_true",
                        help="Enable RoMa path-specific smoothness optimization.")
    parser.add_argument("--roma_alpha", type=float, default=0.4)
    parser.add_argument("--roma_r", type=float, default=0.05)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--auto_resume_latest", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--max_train_steps", "--step", type=int, default=400)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--original_trigger_word")
    parser.add_argument("--current_trigger_word")
    # The muse backend never wrote integrity images, so a run's progress could
    # only be judged from the loss column. Mirrors the flux/dit trainers.
    parser.add_argument("--integrity_prompt", type=str, default="A [Z]*$ at the lake")
    parser.add_argument("--integrity_interval", type=int, default=500,
                        help="0 disables integrity-image generation.")
    parser.add_argument("--integrity_inference_steps", type=int, default=12)
    parser.add_argument("--integrity_guidance_scale", type=float, default=10.0)
