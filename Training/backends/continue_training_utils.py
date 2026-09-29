"""Shared checkpoint and tuning helpers for downstream continuation scripts."""

import json
import os

import torch
import torch.nn as nn
from model_loading import (
    encode_stable_diffusion_prompt,
    has_second_text_encoder,
    stable_diffusion_unet_forward,
)


OPTIMIZER_STATE_FILENAME = "optimizer.pt"
ACCELERATOR_STATE_DIRNAME = "accelerator_state"


class SDXLTrainingModel(nn.Module):
    """One DeepSpeed-compatible module containing all SDXL trainable components."""

    def __init__(self, unet, text_encoder, text_encoder_2):
        super().__init__()
        self.unet = unet
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2

    def forward(
        self,
        noisy_latents,
        timesteps,
        input_ids,
        input_ids_2,
        resolution,
    ):
        prompt_embeds, pooled, time_ids = encode_stable_diffusion_prompt(
            self.text_encoder,
            self.text_encoder_2,
            input_ids,
            input_ids_2,
            noisy_latents,
            resolution,
        )
        return stable_diffusion_unet_forward(
            self.unet,
            noisy_latents,
            timesteps,
            prompt_embeds,
            pooled,
            time_ids,
        )


def is_deepspeed_enabled(accelerator):
    return str(accelerator.distributed_type).upper().endswith("DEEPSPEED")


def create_continue_optimizer(accelerator, trainable_params, learning_rate):
    """Let DeepSpeed create CPU Adam when optimizer offload is configured."""
    if is_deepspeed_enabled(accelerator):
        from accelerate.utils import DummyOptim

        return DummyOptim(trainable_params, lr=learning_rate)
    return torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )


def save_optimizer_checkpoint(
    accelerator,
    optimizer,
    output_dir,
    *,
    optimizer_name,
    step,
):
    """Save optimizer moments for an exact continuation-training restart.

    Ordinary Accelerate runs store a small, explicit optimizer payload beside
    the model weights. DeepSpeed owns the real optimizer behind ``DummyOptim``,
    so its distributed state must be saved through Accelerate on every rank.
    Call this before writing the model artifact: auto-resume uses that artifact
    as the checkpoint-complete marker.
    """
    if is_deepspeed_enabled(accelerator):
        state_dir = os.path.join(output_dir, ACCELERATOR_STATE_DIRNAME)
        accelerator.save_state(state_dir, safe_serialization=False)
        return state_dir

    if not accelerator.is_main_process:
        return None

    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, OPTIMIZER_STATE_FILENAME)
    temporary_path = f"{state_path}.tmp.{os.getpid()}"
    torch.save(
        {
            "format_version": 1,
            "optimizer_name": str(optimizer_name).lower(),
            "step": int(step),
            "state_dict": optimizer.state_dict(),
        },
        temporary_path,
    )
    os.replace(temporary_path, state_path)
    return state_path


def load_optimizer_checkpoint(
    accelerator,
    optimizer,
    checkpoint_dir,
    *,
    expected_optimizer_name,
    expected_step=None,
):
    """Restore optimizer state, accepting legacy weight-only checkpoints.

    Returns ``True`` when optimizer state was restored. A missing state file is
    tolerated so checkpoints created before optimizer persistence was added do
    not become unusable, but the reset is reported prominently.
    """
    if checkpoint_dir is None:
        return False

    expected_name = str(expected_optimizer_name).lower()
    if is_deepspeed_enabled(accelerator):
        state_dir = os.path.join(checkpoint_dir, ACCELERATOR_STATE_DIRNAME)
        if not os.path.isdir(state_dir):
            print(
                f"WARNING: {checkpoint_dir} predates optimizer-state saving; "
                f"resuming step {expected_step or 0} with fresh {expected_name} state.",
                flush=True,
            )
            return False
        accelerator.load_state(state_dir)
        print(f"Restored DeepSpeed optimizer state from {state_dir}", flush=True)
        return True

    state_path = os.path.join(checkpoint_dir, OPTIMIZER_STATE_FILENAME)
    if not os.path.isfile(state_path):
        print(
            f"WARNING: {checkpoint_dir} predates optimizer-state saving; "
            f"resuming step {expected_step or 0} with fresh {expected_name} state.",
            flush=True,
        )
        return False

    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"Malformed optimizer checkpoint: {state_path}")
    saved_name = str(payload.get("optimizer_name", "")).lower()
    if saved_name and saved_name != expected_name:
        raise RuntimeError(
            f"Optimizer checkpoint mismatch: {state_path} contains {saved_name}, "
            f"but this run requested {expected_name}."
        )
    saved_step = payload.get("step")
    if (
        expected_step is not None
        and int(expected_step) > 0
        and saved_step is not None
        and int(saved_step) != int(expected_step)
    ):
        raise RuntimeError(
            f"Optimizer checkpoint step mismatch: {state_path} records step "
            f"{saved_step}, expected {expected_step}."
        )
    optimizer.load_state_dict(payload["state_dict"])
    print(f"Restored {expected_name} optimizer state from {state_path}", flush=True)
    return True


def unwrap_sdxl_training_model(accelerator, training_model):
    model = accelerator.unwrap_model(training_model)
    return model.unet, model.text_encoder, model.text_encoder_2


def _strip_wrapper_prefix(key):
    for prefix in ("module.", "_forward_module."):
        if key.startswith(prefix):
            return _strip_wrapper_prefix(key[len(prefix):])
    return key


def _component_state_dict(state_dict, component):
    prefix = component + "."
    return {
        _strip_wrapper_prefix(key)[len(prefix):]: value.detach().cpu()
        for key, value in state_dict.items()
        if _strip_wrapper_prefix(key).startswith(prefix)
    }


def save_prepared_continue_weights(
    accelerator,
    training_model,
    output_dir,
    backbone_info,
    full_tuning=False,
):
    """Save a prepared wrapper, gathering ZeRO-3 partitions on every rank."""
    if full_tuning:
        state_dict = accelerator.get_state_dict(training_model)
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            for component, filename in (
                ("unet", "unet.pt"),
                ("text_encoder", "text_encoder.pt"),
                ("text_encoder_2", "text_encoder_2.pt"),
            ):
                component_state = _component_state_dict(state_dict, component)
                if component == "text_encoder_2":
                    raw_model = accelerator.unwrap_model(training_model)
                    if not has_second_text_encoder(raw_model.text_encoder_2):
                        continue
                if not component_state:
                    raise RuntimeError(
                        f"Gathered checkpoint has no state for {component}"
                    )
                torch.save(component_state, os.path.join(output_dir, filename))
            with open(
                os.path.join(output_dir, "backbone_info.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(backbone_info, handle, indent=2)
        return

    if accelerator.is_main_process:
        unet, text_encoder, text_encoder_2 = unwrap_sdxl_training_model(
            accelerator, training_model
        )
        save_continue_weights(
            unet,
            text_encoder,
            text_encoder_2,
            output_dir,
            backbone_info,
            full_tuning=False,
        )


def save_prepared_peft_weights(
    accelerator,
    prepared_model,
    output_dir,
    backbone_info,
):
    """Save PEFT weights safely from either an ordinary or ZeRO-3 model."""
    state_dict = (
        accelerator.get_state_dict(prepared_model)
        if is_deepspeed_enabled(accelerator)
        else None
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        model = accelerator.unwrap_model(prepared_model)
        if state_dict is None:
            model.save_pretrained(output_dir)
        else:
            model.save_pretrained(output_dir, state_dict=state_dict)
        with open(
            os.path.join(output_dir, "backbone_info.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(backbone_info, handle, indent=2)


def generate_integrity_image(
    unet,
    vae,
    text_encoder,
    text_encoder_2,
    tokenizer,
    tokenizer_2,
    noise_scheduler,
    prompt,
    output_path,
    resolution,
    num_steps=20,
    guidance_scale=7.5,
    seed=None,
):
    """Generate one SDXL integrity image while preserving the UNet train mode."""
    from utils import save_sdxl_image_from_components

    was_training = unet.training
    unet.eval()
    original_decode = vae.decode
    vae.decode = lambda z, *args, **kwargs: original_decode(
        z.to(dtype=next(vae.parameters()).dtype), *args, **kwargs
    )
    print(f"Generating integrity image ({num_steps} steps) → {output_path}")
    try:
        save_sdxl_image_from_components(
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            noise_scheduler=noise_scheduler,
            prompt=prompt,
            output_path=output_path,
            resolution=resolution,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    finally:
        vae.decode = original_decode
    print(f"Integrity image saved: {output_path}")

    if was_training:
        unet.train()
    base_model = unet.get_base_model() if hasattr(unet, "get_base_model") else unet
    if hasattr(base_model, "clear_overrides_cache"):
        base_model.clear_overrides_cache()


def parse_checkpoint_step(checkpoint_name):
    """Return the numeric suffix used by either checkpoint naming convention."""
    for prefix in ("checkpoint-step", "checkpoint-"):
        if checkpoint_name.startswith(prefix):
            token = checkpoint_name[len(prefix):]
            if token.isdigit():
                return int(token)
    return 0


def is_peft_model(model):
    return hasattr(model, "save_pretrained") and hasattr(model, "peft_config")


def load_or_create_peft(model, config, checkpoint_path=None, label="model"):
    from peft import PeftModel, get_peft_model

    if checkpoint_path and os.path.isdir(checkpoint_path):
        print(f"Loading trainable {label} PEFT adapter from {checkpoint_path}")
        return PeftModel.from_pretrained(model, checkpoint_path, is_trainable=True)
    if checkpoint_path:
        print(
            f"WARNING: missing {label} adapter at {checkpoint_path}; "
            "initializing a fresh adapter."
        )
    print(f"Creating fresh {label} PEFT adapter")
    return get_peft_model(model, config)


def continuation_artifact(full_tuning):
    return "unet.pt" if full_tuning else "adapter_config.json"


def find_latest_checkpoint(
    output_dir,
    required_artifact,
    prefixes=("checkpoint-step", "checkpoint-"),
):
    """Find the newest checkpoint containing the artifact for the selected mode."""
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        step = parse_checkpoint_step(name)
        if step <= 0 or not any(name.startswith(prefix) for prefix in prefixes):
            continue
        checkpoint_dir = os.path.join(output_dir, name)
        if (
            os.path.isdir(checkpoint_dir)
            and os.path.exists(os.path.join(checkpoint_dir, required_artifact))
        ):
            candidates.append((step, name))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def resolve_resume_checkpoint(
    output_dir,
    resume_from_checkpoint,
    auto_resume_latest=False,
    required_artifact="adapter_config.json",
    prefixes=("checkpoint-step", "checkpoint-"),
):
    """Resolve an explicit or automatic resume checkpoint and its saved step."""
    checkpoint = resume_from_checkpoint
    if auto_resume_latest and checkpoint is None:
        checkpoint = find_latest_checkpoint(
            output_dir,
            required_artifact=required_artifact,
            prefixes=prefixes,
        )
        if checkpoint is not None:
            print(f"Auto-resume: {checkpoint}")
    if not checkpoint:
        return None, 0

    checkpoint_path = checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(output_dir, checkpoint_path)
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    artifact_path = os.path.join(checkpoint_path, required_artifact)
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"Resume checkpoint is incompatible with this tuning mode; "
            f"missing {required_artifact}: {checkpoint_path}"
        )

    step = parse_checkpoint_step(os.path.basename(checkpoint_path))
    print(f"Resuming from {checkpoint_path} (step {step})")
    return checkpoint_path, step


def save_continue_weights(
    unet,
    text_encoder,
    text_encoder_2,
    output_dir,
    backbone_info,
    full_tuning=False,
):
    """Save either directly tuned weights or PEFT adapters with common metadata."""
    os.makedirs(output_dir, exist_ok=True)
    if full_tuning:
        torch.save(unet.state_dict(), os.path.join(output_dir, "unet.pt"))
        torch.save(
            text_encoder.state_dict(),
            os.path.join(output_dir, "text_encoder.pt"),
        )
        if has_second_text_encoder(text_encoder_2):
            torch.save(
                text_encoder_2.state_dict(),
                os.path.join(output_dir, "text_encoder_2.pt"),
            )
    else:
        if not is_peft_model(unet):
            raise TypeError("Adapter checkpoint requested for a non-PEFT UNet.")
        unet.save_pretrained(output_dir)
        if is_peft_model(text_encoder):
            text_encoder.save_pretrained(os.path.join(output_dir, "text_encoder"))
        if is_peft_model(text_encoder_2):
            text_encoder_2.save_pretrained(os.path.join(output_dir, "text_encoder_2"))

    with open(os.path.join(output_dir, "backbone_info.json"), "w", encoding="utf-8") as handle:
        json.dump(backbone_info, handle, indent=2)
