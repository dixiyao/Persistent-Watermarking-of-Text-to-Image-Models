"""aMUSEd backend shared by training and inference scripts."""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model


DEFAULT_MODEL = "amused/amused-512"


def load_pipeline(model_id=DEFAULT_MODEL, dtype=torch.float16):
    from diffusers import AmusedPipeline

    return AmusedPipeline.from_pretrained(model_id, torch_dtype=dtype)


def load_adapter(transformer, checkpoint, trainable=False):
    if checkpoint and os.path.isdir(os.path.join(checkpoint, "final")):
        checkpoint = os.path.join(checkpoint, "final")
    return PeftModel.from_pretrained(transformer, checkpoint, is_trainable=trainable)


def add_lora(transformer, rank=16, alpha=32, dropout=0.0):
    return get_peft_model(
        transformer,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["to_q", "to_k", "to_v"],
            bias="none",
        ),
    )


def encode_prompt(tokenizer, text_encoder, prompts, device):
    tokens = tokenizer(
        list(prompts), truncation=True, padding="max_length", max_length=77,
        return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        outputs = text_encoder(tokens, return_dict=True, output_hidden_states=True)
    return outputs.hidden_states[-2], outputs[0]


def image_tokens(vqvae, pixel_values):
    # The repository dataset normalizes images to [-1, 1], while aMUSEd's
    # official trainer feeds VQModel images in [0, 1].
    images = ((pixel_values + 1.0) / 2.0).clamp(0, 1)
    with torch.no_grad():
        encoded = vqvae.encode(images).latents
        return vqvae.quantize(encoded)[2][2].reshape(images.shape[0], -1)


def prepare_masked_inputs(
    transformer,
    vqvae,
    tokenizer,
    text_encoder,
    batch,
    device,
    mask_probability=None,
):
    """Build one fixed masked-token problem instance.

    Split out from the loss so a sharpness probe can evaluate the *same* masked
    inputs before and after perturbing the weights. Re-drawing the mask between
    the two evaluations would measure mask noise, not curvature.
    """
    pixels = batch["pixel_values"].to(device=device, dtype=next(vqvae.parameters()).dtype)
    tokens = image_tokens(vqvae, pixels)
    bsz, seq_len = tokens.shape
    if mask_probability is None:
        mask_probability = torch.cos(
            torch.rand(bsz, device=device) * math.pi * 0.5
        )
    elif not torch.is_tensor(mask_probability):
        mask_probability = torch.full((bsz,), float(mask_probability), device=device)
    num_masked = (seq_len * mask_probability).round().clamp(min=1)
    mask = torch.rand(bsz, seq_len, device=device).argsort(dim=-1) < num_masked[:, None]
    base_transformer = (
        transformer.get_base_model() if hasattr(transformer, "get_base_model") else transformer
    )
    config = base_transformer.config
    mask_id = config.vocab_size - 1
    prompt_embeds, pooled = encode_prompt(
        tokenizer, text_encoder, batch["prompt"], device
    )
    side = int(round(seq_len ** 0.5))
    if side * side != seq_len:
        raise ValueError(f"aMUSEd image-token sequence is not square: {seq_len}")
    micro_conds = torch.tensor(
        [pixels.shape[-1], pixels.shape[-2], 0, 0, 6.0],
        device=device,
        dtype=prompt_embeds.dtype,
    ).repeat(bsz, 1)
    return {
        "input_ids": torch.where(mask, mask_id, tokens).reshape(bsz, side, side),
        "labels": torch.where(mask, tokens, -100),
        "prompt_embeds": prompt_embeds,
        "pooled": pooled,
        "micro_conds": micro_conds,
        "codebook_size": config.codebook_size,
        "batch_size": bsz,
        "mask_probability": mask_probability,
    }


def loss_from_masked_inputs(transformer, inputs):
    logits = transformer(
        input_ids=inputs["input_ids"],
        encoder_hidden_states=inputs["prompt_embeds"],
        micro_conds=inputs["micro_conds"],
        pooled_text_emb=inputs["pooled"],
    ).reshape(inputs["batch_size"], inputs["codebook_size"], -1).permute(0, 2, 1)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        inputs["labels"].reshape(-1),
        ignore_index=-100,
    )
    return loss, logits


def masked_token_loss(
    transformer,
    vqvae,
    tokenizer,
    text_encoder,
    batch,
    device,
    mask_probability=None,
):
    inputs = prepare_masked_inputs(
        transformer, vqvae, tokenizer, text_encoder, batch, device, mask_probability
    )
    loss, logits = loss_from_masked_inputs(transformer, inputs)
    return loss, logits, inputs["mask_probability"].mean()
