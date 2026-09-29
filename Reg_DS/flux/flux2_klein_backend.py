"""Shared FLUX.2 [Klein] training helpers.

The Klein checkpoints use a single Qwen text encoder, four-coordinate token
IDs, and the Flux2 VAE batch-normalization convention.  These helpers mirror
the official Diffusers ``train_dreambooth_lora_flux2_klein.py`` protocol so
the robust and continuation studies cannot silently fall back to FLUX.1.
"""

from __future__ import annotations

import torch

from diffusers import Flux2KleinPipeline


DEFAULT_MODEL = "black-forest-labs/FLUX.2-klein-4B"


def prompt_image_collate_fn(examples):
    """Collate image/prompt datasets without assuming a second tokenizer."""
    return {
        "pixel_values": torch.stack([example["pixel_values"] for example in examples]),
        "image_name": [example["image_name"] for example in examples],
        "prompt": [example["prompt"] for example in examples],
    }


def load_pipeline(model_id: str, dtype: torch.dtype) -> Flux2KleinPipeline:
    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if type(pipe).__name__ != "Flux2KleinPipeline":
        raise TypeError(f"Expected Flux2KleinPipeline, got {type(pipe).__name__}")
    return pipe


def encode_prompts(pipe, prompts, device, dtype, max_sequence_length=512):
    """Return Klein's Qwen embeddings and four-coordinate text IDs."""
    with torch.no_grad():
        prompt_embeds, text_ids = pipe.encode_prompt(
            prompt=list(prompts),
            device=device,
            max_sequence_length=max_sequence_length,
        )
    return prompt_embeds.to(device=device, dtype=dtype), text_ids.to(device=device)


def encode_images(vae, pixel_values):
    """Encode, patchify, and normalize images exactly as Klein training does."""
    latents = vae.encode(pixel_values).latent_dist.mode()
    latents = Flux2KleinPipeline._patchify_latents(latents)
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    return (latents - mean) / std


def pack_latents(latents):
    return Flux2KleinPipeline._pack_latents(latents)


def latent_ids(latents):
    return Flux2KleinPipeline._prepare_latent_ids(latents).to(device=latents.device)


def prepare_training_schedule(scheduler, device, num_train_timesteps=1000):
    """Materialise the full ``num_train_timesteps`` flow-matching schedule.

    ``FlowMatchEulerDiscreteScheduler.__init__`` already populates ``timesteps``
    and ``sigmas`` for every training step, which is exactly what flow-matching
    training samples from.  ``set_timesteps`` is the *inference* entry point, and
    Klein configures ``use_dynamic_shifting=True`` -- there it refuses to run
    without a ``mu`` derived from the image sequence length, a value training
    never has (each batch would need its own).  So only call it when the
    scheduler is statically shifted, and otherwise keep (or rebuild) the
    constructor's schedule, mirroring the official Diffusers FLUX trainers.
    """
    dynamic = bool(getattr(scheduler.config, "use_dynamic_shifting", False))
    if not dynamic:
        scheduler.set_timesteps(num_train_timesteps, device=device)
        return scheduler

    if len(getattr(scheduler, "timesteps", ())) != num_train_timesteps:
        # Same construction as the scheduler's __init__, without the shift that
        # dynamic shifting defers to inference time.
        sigmas = torch.linspace(1.0, 1.0 / num_train_timesteps, num_train_timesteps)
        scheduler.sigmas = sigmas.to("cpu")
        scheduler.timesteps = (sigmas * num_train_timesteps).to(device=device)
    return scheduler


def sample_training_timestep(scheduler, batch_size, device, dtype, shared=False):
    """Sample the shifted Klein flow schedule used by the official trainer."""
    sample_count = 1 if shared else batch_size
    u = torch.sigmoid(torch.randn(sample_count, device=device))
    if shared:
        u = u.repeat(batch_size)
    indices = (u * scheduler.config.num_train_timesteps).long().clamp_max(
        len(scheduler.timesteps) - 1
    )
    timesteps = scheduler.timesteps[indices].to(device=device)
    sigmas = scheduler.sigmas[indices].to(device=device, dtype=dtype)
    while sigmas.ndim < 4:
        sigmas = sigmas.unsqueeze(-1)
    return timesteps / 1000.0, sigmas


def unpack_prediction(prediction, image_ids, packed_length):
    prediction = prediction[:, :packed_length]
    return Flux2KleinPipeline._unpack_latents_with_ids(prediction, image_ids)


def transformer_forward(
    transformer,
    noisy_latents,
    timesteps,
    prompt_embeds,
    text_ids,
    guidance_scale=1.0,
):
    packed = pack_latents(noisy_latents)
    image_ids = latent_ids(noisy_latents)
    base = getattr(transformer, "module", transformer)
    if hasattr(base, "get_base_model"):
        try:
            base = base.get_base_model()
        except Exception:
            pass
    guidance = None
    if bool(getattr(getattr(base, "config", None), "guidance_embeds", False)):
        guidance = torch.full(
            (noisy_latents.shape[0],),
            float(guidance_scale),
            device=noisy_latents.device,
            dtype=torch.float32,
        )
    prediction = transformer(
        hidden_states=packed,
        timestep=timesteps,
        guidance=guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=image_ids,
        return_dict=False,
    )[0]
    return unpack_prediction(prediction, image_ids, packed.shape[1])
