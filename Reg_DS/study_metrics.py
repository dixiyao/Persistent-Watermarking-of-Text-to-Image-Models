#!/usr/bin/env python3

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def compute_snr(noise_scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    alpha_bar = alphas_cumprod[timesteps.long()]
    return alpha_bar / (1.0 - alpha_bar).clamp_min(1e-12)


def snr_weighted_forward_loss(model_pred, target, noise_scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    per_sample_mse = (model_pred.float() - target.float()).pow(2).reshape(model_pred.shape[0], -1).mean(dim=1)
    weights = compute_snr(noise_scheduler, timesteps).to(device=per_sample_mse.device, dtype=per_sample_mse.dtype)
    return (per_sample_mse * weights).mean()


def mse_forward_loss(model_pred, target) -> torch.Tensor:
    per_sample_mse = (model_pred.float() - target.float()).pow(2).reshape(model_pred.shape[0], -1).mean(dim=1)
    return per_sample_mse.mean()


def half_timestep_tensor(batch_size: int, device: torch.device, max_timestep: int, scheduler_num_train_timesteps: int) -> torch.Tensor:
    half_timestep = min(int(0.5 * max_timestep), int(scheduler_num_train_timesteps) - 1)
    return torch.full((batch_size,), half_timestep, device=device, dtype=torch.long)


def pixel_values_to_pil_images(pixel_values: torch.Tensor) -> list[Image.Image]:
    images = []
    pixel_values = pixel_values.detach().cpu().float().clamp(-1.0, 1.0)
    for sample in pixel_values:
        arr = ((sample + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        images.append(Image.fromarray(arr))
    return images


class CLIPScorer:
    def __init__(self, device: torch.device, model_name: str = "openai/clip-vit-base-patch16"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.text_max_length = int(self.model.config.text_config.max_position_embeddings)

    def _feature_tensor(self, output, modality: str) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        embeds_name = "image_embeds" if modality == "visual" else "text_embeds"
        embeds = getattr(output, embeds_name, None)
        if embeds is not None:
            return embeds
        pooled = getattr(output, "pooler_output", None)
        if pooled is not None:
            projection = getattr(self.model, f"{modality}_projection", None)
            if projection is None:
                return pooled
            if pooled.shape[-1] == projection.in_features:
                return projection(pooled)
            if pooled.shape[-1] == projection.out_features:
                return pooled
            raise ValueError(
                f"Unexpected CLIP {modality} pooled width {pooled.shape[-1]}; "
                f"expected {projection.in_features} (raw) or {projection.out_features} (projected)."
            )
        raise TypeError(f"Unsupported CLIP {modality} feature output: {type(output).__name__}")

    @torch.no_grad()
    def text_image_score(self, prompts: Sequence[str], images: Sequence[Image.Image], batch_size: int = 8) -> float:
        vals = []
        for i in range(0, len(images), batch_size):
            prompt_batch = [str(prompt) for prompt in prompts[i:i + batch_size]]
            image_batch = list(images[i:i + batch_size])
            inputs = self.processor(
                text=prompt_batch,
                images=image_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.text_max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            image_features = self._feature_tensor(
                self.model.get_image_features(pixel_values=inputs["pixel_values"]), "visual"
            )
            text_features = self._feature_tensor(
                self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                ),
                "text",
            )
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            sims = (image_features * text_features).sum(dim=-1)
            vals.extend(sims.detach().cpu().tolist())

        return float(np.mean(vals)) if vals else float("nan")
