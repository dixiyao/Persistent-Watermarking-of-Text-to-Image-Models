"""Shared PEFT-LoRA study machinery for the non-SDXL backbones.

muse/, jit/ and parti/ all adapt their backbone with a plain PEFT LoRA rather
than the T-LoRA wrapper used by the SDXL / FLUX / PixArt studies, because none
of those three backbones exposes a diffusion timestep that a sigma mask could
key on (aMUSEd masks tokens, LlamaGen is autoregressive, and PixelDiT's
timestep lives inside its own scheduler). Everything here is therefore written
against PEFT modules and a caller-supplied loss closure, so the four ablation
axes and the gradient-geometry metrics stay identical across the three.
"""

from __future__ import annotations

import os

import torch


# ── PEFT introspection ─────────────────────────────────────────────────────────

def iter_lora_modules(model):
    for _name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield module


def _delta_weight(module, adapter):
    """Effective LoRA weight change, autograd intact."""
    a = module.lora_A[adapter].weight
    b = module.lora_B[adapter].weight
    if a.ndim != 2 or b.ndim != 2:
        return None
    return (b @ a) * module.scaling.get(adapter, 1.0)


def watermarkdm_l1(model):
    """||W_r - W_f||_1 over the effective LoRA weight change.

    A PEFT adapter's pretrained point is delta = 0, so the effective change is
    just the adapter's delta weight. Gradients are kept live on purpose.
    """
    total = None
    for module in iter_lora_modules(model):
        for adapter in module.lora_A:
            if adapter not in module.lora_B:
                continue
            delta = _delta_weight(module, adapter)
            if delta is None:
                continue
            term = delta.float().abs().sum()
            total = term if total is None else total + term
    if total is None:
        raise RuntimeError("No LoRA layers found for the WatermarkDM regularizer")
    return total


def peft_delta_norms(model):
    """(||W_r - W_f||, ||W_r - W_f|| / ||W_f||) over the adapted base weights."""
    delta_sq = 0.0
    base_sq = 0.0
    for module in iter_lora_modules(model):
        for adapter in module.lora_A:
            if adapter not in module.lora_B:
                continue
            with torch.no_grad():
                try:
                    delta = module.get_delta_weight(adapter).detach().float()
                except Exception:
                    delta = _delta_weight(module, adapter)
                    if delta is None:
                        continue
                    delta = delta.detach().float()
                delta_sq += float(delta.pow(2).sum().item())
                base_layer = (
                    module.get_base_layer() if hasattr(module, "get_base_layer")
                    else getattr(module, "base_layer", None)
                )
                if base_layer is not None and hasattr(base_layer, "weight"):
                    base_sq += float(base_layer.weight.detach().float().pow(2).sum().item())
    delta_norm = delta_sq ** 0.5
    return delta_norm, delta_norm / max(base_sq ** 0.5, 1e-12)


# ── RoMa ───────────────────────────────────────────────────────────────────────

def apply_roma_path_perturbation(params, path_step_size):
    """theta_tilde = theta + r * (theta_0 - theta) / ||theta_0 - theta||.

    For a PEFT adapter the pretrained point theta_0 is the zero adapter, so the
    direction is simply -theta. Returns the applied perturbations so the caller
    can restore theta before the optimizer step.
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


# ── gradient geometry ──────────────────────────────────────────────────────────

def flat_grad(loss, params, retain_graph=False):
    """Flatten d(loss)/d(params) onto the CPU without touching .grad."""
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(p) if g is None else g).detach().float().reshape(-1).cpu()
        for p, g in zip(params, grads)
    ])


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


def sharpness_along_gradient(params, grad_vec, base_loss, loss_fn, rho):
    """SAM-style probe: loss increase after an rho-normalized ascent step.

    ``loss_fn`` must re-evaluate the *same* problem instance -- same batch, same
    noise/mask draw. Re-sampling between the two evaluations would measure input
    noise rather than curvature.
    """
    grad_norm = float(grad_vec.norm().item())
    if grad_norm <= 1e-12:
        return float("nan")
    scale = rho / grad_norm
    apply_flat_perturbation(params, grad_vec, scale)
    try:
        with torch.no_grad():
            perturbed = float(loss_fn().item())
    finally:
        apply_flat_perturbation(params, grad_vec, -scale)
    return perturbed - base_loss


def gradient_geometry(params, cp_loss, org_loss, cp_loss_fn, org_loss_fn, rho):
    """cosine + both sharpnesses + both gradient norms for one study step."""
    cp_value = float(cp_loss.detach().item())
    org_value = float(org_loss.detach().item())
    cp_grad = flat_grad(cp_loss, params, retain_graph=True)
    org_grad = flat_grad(org_loss, params)
    cp_norm = float(cp_grad.norm().item())
    org_norm = float(org_grad.norm().item())
    if cp_norm <= 1e-12 or org_norm <= 1e-12:
        cosine = float("nan")
    else:
        cosine = float(
            torch.nn.functional.cosine_similarity(cp_grad, org_grad, dim=0).item()
        )
    return {
        "cp_loss": cp_value,
        "org_loss": org_value,
        "cp_grad_norm": cp_norm,
        "org_grad_norm": org_norm,
        "cp_org_grad_cosine": cosine,
        "cp_sharpness": sharpness_along_gradient(params, cp_grad, cp_value, cp_loss_fn, rho),
        "org_sharpness": sharpness_along_gradient(params, org_grad, org_value, org_loss_fn, rho),
    }


STUDY_LOG_FIELDS = [
    "step", "loss", "cp_loss", "org_loss",
    "cp_org_grad_cosine", "cp_grad_norm", "org_grad_norm",
    "cp_sharpness", "org_sharpness",
    "wr_minus_wf_norm", "wr_minus_wf_over_wf_norm",
    "watermarkdm_l1", "watermarkdm_loss", "final_loss",
    "roma_path_loss", "roma_difference_norm",
]


# ── resume ─────────────────────────────────────────────────────────────────────

def parse_step(name):
    if name.startswith("checkpoint-"):
        tok = name[len("checkpoint-"):]
        if tok.isdigit():
            return int(tok)
    return 0


def teach_peft_the_config(model):
    """Give a backbone's plain-dataclass config the one dict hook peft needs.

    peft's BaseTuner.get_model_config() returns model.config untouched unless it
    has to_dict(), and then treats the result as a mapping:

        _get_tied_target_modules   model_config.get("tie_word_embeddings")
        create_or_update_model_card  "_name_or_path" in model_config

    Neither works on a plain dataclass, so LlamaGen's ModelArgs raised
    AttributeError at get_peft_model and PixelDiT's PixDiTConfig raised
    TypeError at the first save_pretrained.  Teaching the config CLASS to_dict()
    is the smallest faithful fix: to_dict() is exactly the hook HF configs
    expose, the instance and every attribute access in our own code are
    untouched, and the resulting dict answers both questions honestly -- the key
    is there when the model has it and absent when it does not.

    Call this on the base model before any peft entry point touches it.
    """
    config = getattr(model, "config", None)
    if config is None or isinstance(config, dict) or hasattr(config, "to_dict"):
        return
    try:
        type(config).to_dict = lambda self: dict(vars(self))
    except (AttributeError, TypeError):  # __slots__ or a builtin: nothing to teach
        pass


def resolve_resume(args, artifact="adapter_config.json"):
    """Honour --resume_from_checkpoint / --auto_resume_latest.

    The 12 h partition cap means an attack is relaunched several times into the
    same directory, so every one of these trainers has to be able to pick up the
    newest checkpoint it wrote.
    """
    ckpt = getattr(args, "resume_from_checkpoint", None)
    if ckpt is None and getattr(args, "auto_resume_latest", False) and os.path.isdir(args.output_dir):
        cands = [
            (parse_step(n), n) for n in os.listdir(args.output_dir)
            if n.startswith("checkpoint-") and n[len("checkpoint-"):].isdigit()
            and os.path.exists(os.path.join(args.output_dir, n, artifact))
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


def rotate_checkpoints(output_dir, keep):
    if keep is None or keep <= 0:
        return
    import shutil
    kept = sorted(
        (d for d in os.listdir(output_dir)
         if d.startswith("checkpoint-") and d[len("checkpoint-"):].isdigit()),
        key=lambda x: int(x[len("checkpoint-"):]),
    )
    for old in kept[:-keep]:
        shutil.rmtree(os.path.join(output_dir, old), ignore_errors=True)


# ── datasets ───────────────────────────────────────────────────────────────────

def prompt_image_collate_fn(examples):
    """Pixels and raw prompts only.

    utils.simple_dreambooth_collate_fn stacks input_ids/input_ids_2, which
    SimpleDreamBoothDataset only emits when *both* tokenisers are supplied.
    These backbones tokenize from the raw prompt themselves.
    """
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]),
        "image_name": [e["image_name"] for e in examples],
        "prompt": [e["prompt"] for e in examples],
    }


def add_ablation_args(parser):
    """The four ablation axes, identical across every backbone."""
    parser.add_argument("--cp_ref_weight", type=float, default=1.0,
                        help="Subtracted frozen-backbone consistency weight for CP samples.")
    parser.add_argument("--org_loss_weights", type=float, default=1.0,
                        help="Weight applied to the original-image loss.")
    parser.add_argument("--lambda_watermarkdm", type=float, default=0.0,
                        help="Weight for the WatermarkDM L1 penalty on the LoRA delta.")
    parser.add_argument("--roma", action="store_true",
                        help="Enable RoMa path-specific smoothness optimization.")
    parser.add_argument("--roma_alpha", type=float, default=0.4)
    parser.add_argument("--roma_r", type=float, default=0.05)
    parser.add_argument("--sharpness_rho", type=float, default=0.05,
                        help="Radius for the SAM-style loss sharpness probe.")
