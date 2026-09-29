import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tlora_module import (
    TLoRACrossAttnProcessor,
    TLoRATextLinearLayer,
    attach_tlora_sigma_mask_hook,
    build_tlora_attn_processors,
    freeze_tlora_attn_processors,
    load_tlora_attn_state_dict,
    load_text_encoder_tlora_weights,
)
from utils import prepare_sdxl_pipeline_for_inference, resolve_checkpoint_paths
from wr_injection import MRInjectedUNet


DEFAULT_TLORA_CONFIG = {
    "rank": 16,
    "lora_alpha": 32.0,
    "sig_type": "last",
    "ortho_init": "random",
    "min_rank": 8,
    "alpha_rank_scale": 1.0,
    "max_timestep": 1000,
}


@dataclass
class SDXLComponents:
    tokenizer: Any
    tokenizer_2: Any
    vae: Any
    unet: Any
    text_encoder: Any
    text_encoder_2: Any
    model_family: str = "sdxl"
    tlora_config: Optional[Dict[str, Any]] = None
    tlora_text_encoder_config: Optional[Dict[str, Any]] = None


class NullTextEncoder(nn.Module):
    """Frozen placeholder that keeps legacy two-encoder call signatures valid."""

    is_placeholder_text_encoder = True

    def __init__(self):
        super().__init__()
        self._device_anchor = nn.Parameter(torch.empty(0), requires_grad=False)

    @property
    def device(self):
        return self._device_anchor.device

    @property
    def dtype(self):
        return self._device_anchor.dtype

    def requires_grad_(self, requires_grad=True):
        # Keep the device/dtype anchor out of optimizers during SD 1.x full tuning.
        self._device_anchor.requires_grad_(False)
        return self

    def forward(self, input_ids, *args, **kwargs):
        batch_size, sequence_length = input_ids.shape[:2]
        hidden = torch.empty(
            batch_size,
            sequence_length,
            0,
            device=input_ids.device,
            dtype=self.dtype,
        )
        pooled = torch.zeros(
            batch_size,
            1,
            device=input_ids.device,
            dtype=self.dtype,
        )
        return SimpleNamespace(
            hidden_states=(hidden, hidden),
            text_embeds=pooled,
        )


def is_sdxl_unet(unet):
    """Detect XL conditioning from the UNet config, including wrapped UNets."""
    base = unet
    while hasattr(base, "module") and base.module is not base:
        base = base.module
    while hasattr(base, "get_base_model"):
        candidate = base.get_base_model()
        if candidate is base:
            break
        base = candidate
    if hasattr(base, "unet"):
        base = base.unet
    config = getattr(base, "config", None)
    return getattr(config, "addition_embed_type", None) == "text_time"


def has_second_text_encoder(text_encoder_2):
    while (
        text_encoder_2 is not None
        and hasattr(text_encoder_2, "module")
        and text_encoder_2.module is not text_encoder_2
    ):
        text_encoder_2 = text_encoder_2.module
    return (
        text_encoder_2 is not None
        and not getattr(text_encoder_2, "is_placeholder_text_encoder", False)
    )


def encode_stable_diffusion_prompt(
    text_encoder,
    text_encoder_2,
    input_ids,
    input_ids_2,
    noisy_latents,
    resolution,
):
    """Return conditioning for either SDXL or single-encoder SD 1.x."""
    if not has_second_text_encoder(text_encoder_2):
        output = text_encoder(input_ids)
        prompt_embeds = output[0].to(noisy_latents.device)
        return prompt_embeds, None, None

    output_1 = text_encoder(input_ids, output_hidden_states=True)
    output_2 = text_encoder_2(input_ids_2, output_hidden_states=True)
    prompt_embeds = torch.cat(
        [output_1.hidden_states[-2], output_2.hidden_states[-2]], dim=-1
    ).to(noisy_latents.device)
    pooled = output_2.text_embeds.to(noisy_latents.device)
    time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        dtype=prompt_embeds.dtype,
        device=noisy_latents.device,
    ).repeat(noisy_latents.shape[0], 1)
    return prompt_embeds, pooled, time_ids


def stable_diffusion_unet_forward(
    unet,
    noisy_latents,
    timesteps,
    prompt_embeds,
    pooled_prompt_embeds=None,
    time_ids=None,
    cross_attention_kwargs=None,
    sigma_mask=None,
    timestep_cond=None,
):
    kwargs = {}
    if timestep_cond is not None:
        # LCM students carry a guidance-scale embedding on this input.
        kwargs["timestep_cond"] = timestep_cond
    if pooled_prompt_embeds is not None and time_ids is not None:
        kwargs["added_cond_kwargs"] = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": time_ids,
        }
    if cross_attention_kwargs is None and sigma_mask is not None:
        cross_attention_kwargs = {"sigma_mask": sigma_mask}
    if cross_attention_kwargs is not None:
        kwargs["cross_attention_kwargs"] = cross_attention_kwargs
    return unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        **kwargs,
    ).sample


@dataclass
class TLoRALoadResult:
    config: Dict[str, Any]
    loaded_unet_processors: int
    loaded_text_encoder_layers: int
    loaded_text_encoder_2_layers: int
    unet_weights_path: str
    text_encoder_weights_path: str
    text_encoder_2_weights_path: str


@dataclass
class BackboneLoadResult:
    components: SDXLComponents
    tlora_config: Dict[str, Any]
    rl_state: Optional[Dict[str, Any]] = None
    extra_mr_keys: Optional[list] = None


def resolve_tlora_config(lora_path, fallback=None, metadata=None):
    cfg = dict(DEFAULT_TLORA_CONFIG)
    if fallback is not None:
        cfg.update({k: v for k, v in dict(fallback).items() if v is not None})
    if metadata is not None:
        for key in ("rank", "min_rank", "alpha_rank_scale", "max_timestep"):
            if key in metadata and metadata[key] is not None:
                cfg[key] = metadata[key]

    _, config_path, _, _ = resolve_checkpoint_paths(lora_path)
    config_candidates = []
    if config_path is not None:
        config_candidates.append(config_path)
    if os.path.isdir(lora_path):
        config_candidates.append(os.path.join(lora_path, "train_state.pt"))
    else:
        config_candidates.append(os.path.join(os.path.dirname(lora_path), "train_state.pt"))

    for candidate in config_candidates:
        if not candidate or not os.path.exists(candidate):
            continue
        saved = torch.load(candidate, map_location="cpu")
        for key in cfg:
            if key in saved and saved[key] is not None:
                cfg[key] = saved[key]
        break

    cfg["rank"] = int(cfg["rank"])
    cfg["lora_alpha"] = float(cfg["lora_alpha"])
    cfg["sig_type"] = str(cfg["sig_type"])
    cfg["ortho_init"] = str(cfg.get("ortho_init", "random"))
    cfg["min_rank"] = int(cfg["min_rank"])
    cfg["alpha_rank_scale"] = float(cfg["alpha_rank_scale"])
    cfg["max_timestep"] = int(cfg["max_timestep"])
    return cfg


def get_tlora_text_encoder_config(tlora_config):
    return {
        "rank": int(tlora_config["rank"]),
        "min_rank": int(tlora_config["min_rank"]),
        "alpha_rank_scale": float(tlora_config["alpha_rank_scale"]),
        "max_timestep": int(tlora_config["max_timestep"]),
        "ortho_init": str(tlora_config.get("ortho_init", "random")),
    }


def _set_tlora_config(target, tlora_config):
    if target is None:
        return
    config = dict(tlora_config)
    text_config = get_tlora_text_encoder_config(config)
    setattr(target, "tlora_config", config)
    setattr(target, "tlora_text_encoder_config", text_config)


def _target_modules(target):
    if hasattr(target, "unet"):
        return (
            target.unet,
            target.text_encoder,
            getattr(target, "text_encoder_2", None),
        )
    return target, None, None


def _has_tlora_processors(unet):
    return any(
        isinstance(proc, TLoRACrossAttnProcessor)
        for proc in unet.attn_processors.values()
    )


def load_base_sdxl_pipeline(
    base_model,
    device,
    torch_dtype,
    variant=None,
    revision=None,
    scheduler=None,
    use_safetensors=True,
    move_to_device=True,
):
    from diffusers import DiffusionPipeline

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "variant": variant,
        "revision": revision,
        "use_safetensors": use_safetensors,
    }
    try:
        pipeline = DiffusionPipeline.from_pretrained(base_model, **load_kwargs)
    except OSError:
        if variant is None:
            raise
        print(
            f"Model {base_model} has no '{variant}' variant; retrying default weights.",
            flush=True,
        )
        load_kwargs["variant"] = None
        pipeline = DiffusionPipeline.from_pretrained(base_model, **load_kwargs)
    if move_to_device:
        pipeline.to(device)
    if scheduler is not None:
        pipeline.scheduler = scheduler
    if move_to_device:
        return prepare_sdxl_pipeline_for_inference(pipeline)
    return pipeline


def load_base_sdxl_backbone(
    pretrained_model_name_or_path,
    revision=None,
    variant=None,
    torch_dtype=torch.float32,
):
    return load_sdxl_components(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
        variant=variant,
        torch_dtype=torch_dtype,
    )


def load_sdxl_components(
    pretrained_model_name_or_path,
    revision=None,
    variant=None,
    torch_dtype=torch.float32,
):
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )
    def load_component(component_class, subfolder):
        kwargs = {
            "subfolder": subfolder,
            "revision": revision,
            "variant": variant,
            "torch_dtype": torch_dtype,
        }
        try:
            return component_class.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )
        except OSError:
            if variant is None:
                raise
            print(
                f"{pretrained_model_name_or_path}/{subfolder} has no "
                f"'{variant}' variant; retrying default weights.",
                flush=True,
            )
            kwargs["variant"] = None
            return component_class.from_pretrained(
                pretrained_model_name_or_path, **kwargs
            )

    vae = load_component(AutoencoderKL, "vae")
    unet = load_component(UNet2DConditionModel, "unet")
    text_encoder = load_component(CLIPTextModel, "text_encoder")
    if is_sdxl_unet(unet):
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=revision,
        )
        text_encoder_2 = load_component(
            CLIPTextModelWithProjection, "text_encoder_2"
        )
        model_family = "sdxl"
    else:
        tokenizer_2 = tokenizer
        text_encoder_2 = NullTextEncoder()
        model_family = "sd1x"
    return SDXLComponents(
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        vae=vae,
        unet=unet,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        model_family=model_family,
    )


def freeze_sdxl_components(components):
    components.unet.requires_grad_(False)
    components.vae.requires_grad_(False)
    components.text_encoder.requires_grad_(False)
    components.text_encoder_2.requires_grad_(False)
    return components


def move_sdxl_components(components, device):
    components.unet = components.unet.to(device)
    components.vae = components.vae.to(device)
    components.text_encoder = components.text_encoder.to(device)
    components.text_encoder_2 = components.text_encoder_2.to(device)
    return components


def create_pipeline_from_components(components, scheduler, device=None):
    if has_second_text_encoder(components.text_encoder_2):
        from diffusers import StableDiffusionXLPipeline

        pipeline = StableDiffusionXLPipeline(
            vae=components.vae,
            text_encoder=components.text_encoder,
            text_encoder_2=components.text_encoder_2,
            tokenizer=components.tokenizer,
            tokenizer_2=components.tokenizer_2,
            unet=components.unet,
            scheduler=scheduler,
            add_watermarker=False,
        )
    else:
        from diffusers import StableDiffusionPipeline

        pipeline = StableDiffusionPipeline(
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            unet=components.unet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    if device is not None:
        pipeline = pipeline.to(device)
    pipeline = prepare_sdxl_pipeline_for_inference(pipeline)
    if components.tlora_config is not None:
        _set_tlora_config(pipeline, components.tlora_config)
    return pipeline


def load_tlora_weights(
    unet,
    text_encoder,
    text_encoder_2,
    lora_path,
    tlora_config=None,
    strict=True,
    attach_sigma_hook=True,
    freeze_unet_lora=True,
    freeze_all=False,
    target=None,
    label="T-LoRA",
):
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"{label} checkpoint not found: {lora_path}")

    (
        unet_weights_path,
        _config_path,
        text_encoder_weights_path,
        text_encoder_2_weights_path,
    ) = resolve_checkpoint_paths(lora_path)
    if not os.path.exists(unet_weights_path):
        raise FileNotFoundError(f"T-LoRA weights not found: {unet_weights_path}")

    cfg = resolve_tlora_config(lora_path, fallback=tlora_config)
    print(
        f"Loading {label}: rank={cfg['rank']}, "
        f"lora_alpha={cfg['lora_alpha']}, sig_type={cfg['sig_type']}"
    )
    print(f"Loading {label} UNet weights from {unet_weights_path}...")

    if not _has_tlora_processors(unet):
        tlora_attn_procs = build_tlora_attn_processors(
            unet,
            rank=cfg["rank"],
            lora_alpha=cfg["lora_alpha"],
            sig_type=cfg["sig_type"],
            ortho_init=cfg["ortho_init"],
            skip_init=True,
        )
        unet.set_attn_processor(tlora_attn_procs)

    loaded_unet = load_tlora_attn_state_dict(
        unet,
        torch.load(unet_weights_path, map_location="cpu"),
        strict=strict,
    )

    if freeze_unet_lora:
        freeze_tlora_attn_processors(unet)

    loaded_te1 = load_text_encoder_tlora_weights(
        text_encoder,
        text_encoder_weights_path,
        rank=cfg["rank"],
        lora_alpha=cfg["lora_alpha"],
        sig_type=cfg["sig_type"],
        ortho_init=cfg["ortho_init"],
        skip_init=True,
    )
    loaded_te2 = 0
    if has_second_text_encoder(text_encoder_2):
        loaded_te2 = load_text_encoder_tlora_weights(
            text_encoder_2,
            text_encoder_2_weights_path,
            rank=cfg["rank"],
            lora_alpha=cfg["lora_alpha"],
            sig_type=cfg["sig_type"],
            ortho_init=cfg["ortho_init"],
            skip_init=True,
        )

    if attach_sigma_hook:
        attach_tlora_sigma_mask_hook(
            unet,
            rank=cfg["rank"],
            min_rank=cfg["min_rank"],
            alpha_rank_scale=cfg["alpha_rank_scale"],
            max_timestep=cfg["max_timestep"],
        )

    if freeze_all:
        freeze_all_tlora(unet, text_encoder, text_encoder_2)

    _set_tlora_config(target, cfg)

    print(f"Loaded {label} UNet attention processors: {loaded_unet}")
    print(f"Loaded {label} text_encoder layers: {loaded_te1}")
    print(f"Loaded {label} text_encoder_2 layers: {loaded_te2}")
    return TLoRALoadResult(
        config=cfg,
        loaded_unet_processors=loaded_unet,
        loaded_text_encoder_layers=loaded_te1,
        loaded_text_encoder_2_layers=loaded_te2,
        unet_weights_path=unet_weights_path,
        text_encoder_weights_path=text_encoder_weights_path,
        text_encoder_2_weights_path=text_encoder_2_weights_path,
    )


def load_tlora_model(unet, text_encoder, text_encoder_2, lora_path, metadata=None, target=None):
    cfg = resolve_tlora_config(lora_path, metadata=metadata)
    return load_tlora_weights(
        unet,
        text_encoder,
        text_encoder_2,
        lora_path,
        tlora_config=cfg,
        strict=True,
        attach_sigma_hook=True,
        freeze_unet_lora=True,
        target=target,
        label="T-LoRA",
    )


def load_further_lora_weights(target, further_lora_path, freeze=True):
    if not os.path.exists(further_lora_path):
        raise FileNotFoundError(f"Further-LoRA checkpoint not found: {further_lora_path}")

    unet, text_encoder, text_encoder_2 = _target_modules(target)
    if text_encoder is None:
        raise ValueError(
            "load_further_lora_weights requires a Stable Diffusion pipeline or components target"
        )

    initial_cfg = getattr(target, "tlora_config", None)
    if initial_cfg is None:
        initial_cfg = getattr(target, "tlora_text_encoder_config", None)
    if initial_cfg is None:
        raise RuntimeError("Initial LoRA config missing; load the base T-LoRA checkpoint first")

    (
        unet_weights_path,
        _config_path,
        text_encoder_weights_path,
        text_encoder_2_weights_path,
    ) = resolve_checkpoint_paths(further_lora_path)
    if not os.path.exists(unet_weights_path):
        raise FileNotFoundError(f"T-LoRA weights not found: {unet_weights_path}")

    cfg = resolve_tlora_config(further_lora_path, fallback=initial_cfg)
    print(f"Loading further-LoRA UNet weights from {unet_weights_path}...")
    loaded_unet = load_tlora_attn_state_dict(
        unet,
        torch.load(unet_weights_path, map_location="cpu"),
        strict=False,
    )

    loaded_te1 = load_text_encoder_tlora_weights(
        text_encoder,
        text_encoder_weights_path,
        rank=cfg["rank"],
        lora_alpha=cfg["lora_alpha"],
        sig_type=cfg["sig_type"],
        ortho_init=cfg["ortho_init"],
        skip_init=True,
    )
    loaded_te2 = 0
    if has_second_text_encoder(text_encoder_2):
        loaded_te2 = load_text_encoder_tlora_weights(
            text_encoder_2,
            text_encoder_2_weights_path,
            rank=cfg["rank"],
            lora_alpha=cfg["lora_alpha"],
            sig_type=cfg["sig_type"],
            ortho_init=cfg["ortho_init"],
            skip_init=True,
        )

    if freeze:
        freeze_all_tlora(unet, text_encoder, text_encoder_2)
    _set_tlora_config(target, cfg)

    print(f"Loaded further-LoRA UNet processors: {loaded_unet}")
    print(f"Loaded further-LoRA text_encoder layers: {loaded_te1}")
    print(f"Loaded further-LoRA text_encoder_2 layers: {loaded_te2}")

    # Load standalone MR weights if present and inject via MRInjectedUNet
    mr_weights_path = os.path.join(further_lora_path, "mr_weights.pt")
    if os.path.exists(mr_weights_path):
        W_MR = torch.load(mr_weights_path, map_location="cpu")
        if W_MR:
            unet_wrapped, extra_keys = build_mr_injected_unet(unet, W_MR)
            if extra_keys:
                print(f"Warning: {len(extra_keys)} extra M_R keys not used by UNet")
            if hasattr(target, "unet"):
                target.unet = unet_wrapped
            print(f"Loaded further-LoRA M_R weights ({len(W_MR)} layers) from {mr_weights_path}")

    return target


def resolve_full_checkpoint_dir(full_checkpoint):
    if not os.path.isdir(full_checkpoint):
        raise FileNotFoundError(f"Full checkpoint path is not a directory: {full_checkpoint}")

    direct_unet_full = os.path.join(full_checkpoint, "unet_full.pt")
    direct_unet = os.path.join(full_checkpoint, "unet.pt")
    if os.path.exists(direct_unet_full) or os.path.exists(direct_unet):
        return full_checkpoint

    ckpt_candidates = []
    for entry in os.listdir(full_checkpoint):
        entry_path = os.path.join(full_checkpoint, entry)
        if not os.path.isdir(entry_path):
            continue
        if not (entry.startswith("checkpoint-step") or entry.startswith("checkpoint-")):
            continue
        if not (
            os.path.exists(os.path.join(entry_path, "unet_full.pt"))
            or os.path.exists(os.path.join(entry_path, "unet.pt"))
        ):
            continue

        digits = "".join(ch for ch in entry if ch.isdigit())
        if digits:
            ckpt_candidates.append((int(digits), entry_path))

    if not ckpt_candidates:
        raise FileNotFoundError(
            f"No usable full checkpoints found in {full_checkpoint}. "
            "Expected either a direct directory containing unet.pt/unet_full.pt "
            "or child dirs like checkpoint-step007000/."
        )

    return max(ckpt_candidates, key=lambda x: x[0])[1]


def load_further_full_weights(target, full_checkpoint):
    unet, text_encoder, text_encoder_2 = _target_modules(target)
    try:
        ckpt_dir = resolve_full_checkpoint_dir(full_checkpoint)
    except FileNotFoundError as full_error:
        try:
            unet_weights_path, _config_path, _te1_path, _te2_path = resolve_checkpoint_paths(full_checkpoint)
        except Exception:
            raise full_error
        if not os.path.exists(unet_weights_path):
            raise full_error

        print(
            f"No full unet.pt checkpoint found in {full_checkpoint}; "
            "loading it as a frozen T-LoRA checkpoint instead.",
            flush=True,
        )
        load_tlora_weights(
            unet,
            text_encoder,
            text_encoder_2,
            full_checkpoint,
            strict=True,
            attach_sigma_hook=True,
            freeze_unet_lora=True,
            freeze_all=True,
            target=target,
            label="Frozen backbone T-LoRA",
        )
        return target

    unet_ckpt = os.path.join(ckpt_dir, "unet.pt")
    if not os.path.exists(unet_ckpt):
        unet_ckpt = os.path.join(ckpt_dir, "unet_full.pt")
    print(f"Loading full UNet weights from {unet_ckpt}...", flush=True)
    unet_state = torch.load(unet_ckpt, map_location="cpu")
    unet_state = {
        key: value
        for key, value in unet_state.items()
        if "processor" not in key.split(".")
    }
    # An LCM student (train_lcm_distill_sd14.py final/unet.pt) carries the
    # guidance-scale projection time_embedding.cond_proj.* that a stock UNet
    # does not have. When such a checkpoint is used as the starting point of an
    # ordinary fine-tuning attack, drop that projection and continue from the
    # remaining weights as a plain diffusion UNet.
    cond_proj_keys = [key for key in unet_state if key.startswith("time_embedding.cond_proj.")]
    if cond_proj_keys and getattr(getattr(unet, "config", None), "time_cond_proj_dim", None) is None:
        for key in cond_proj_keys:
            unet_state.pop(key)
        print(
            f"Dropped {len(cond_proj_keys)} LCM guidance-projection tensors "
            "(time_embedding.cond_proj.*): the target UNet has no time_cond_proj_dim, "
            "so the distilled student continues as a plain diffusion UNet.",
            flush=True,
        )
    unet.load_state_dict(unet_state, strict=True)
    print("Loaded full UNet weights.", flush=True)

    text_encoder_ckpt = os.path.join(ckpt_dir, "text_encoder.pt")
    if text_encoder is not None and os.path.exists(text_encoder_ckpt):
        print(f"Loading full text_encoder weights from {text_encoder_ckpt}...", flush=True)
        text_encoder_state = torch.load(text_encoder_ckpt, map_location="cpu")
        text_encoder_state = _sanitize_full_text_encoder_state_dict(
            text_encoder_state, text_encoder
        )
        text_encoder.load_state_dict(text_encoder_state, strict=True)
        print("Loaded full text_encoder weights.", flush=True)
    elif text_encoder is not None:
        print("No text_encoder.pt found; using base SDXL text_encoder weights.", flush=True)

    text_encoder_2_ckpt = os.path.join(ckpt_dir, "text_encoder_2.pt")
    if has_second_text_encoder(text_encoder_2) and os.path.exists(text_encoder_2_ckpt):
        print(f"Loading full text_encoder_2 weights from {text_encoder_2_ckpt}...", flush=True)
        text_encoder_2_state = torch.load(text_encoder_2_ckpt, map_location="cpu")
        text_encoder_2_state = _sanitize_full_text_encoder_state_dict(
            text_encoder_2_state, text_encoder_2
        )
        text_encoder_2.load_state_dict(text_encoder_2_state, strict=True)
        print("Loaded full text_encoder_2 weights.", flush=True)
    elif has_second_text_encoder(text_encoder_2):
        print("No text_encoder_2.pt found; using base SDXL text_encoder_2 weights.", flush=True)

    print(f"Loaded full checkpoint: {ckpt_dir}", flush=True)
    return target


_CLIP_TEXT_PREFIX = "text_model."


def align_text_encoder_state_dict(state_dict, module):
    """Match a saved text-encoder state dict to the live module's key layout.

    transformers < 5 nested the CLIP text tower under a ``text_model``
    submodule, so checkpoints written back then carry ``text_model.``-prefixed
    keys. transformers >= 5 flattened ``CLIPTextModel`` (``embeddings``,
    ``encoder``, ``final_layer_norm`` are attributes of the model itself), so
    those same checkpoints no longer load with ``strict=True``. Add or strip the
    prefix as needed, and only when doing so actually matches the live module -
    otherwise the state dict is returned untouched so genuine mismatches still
    raise.
    """
    if module is None:
        return state_dict
    try:
        expected = set(module.state_dict().keys())
    except Exception:
        return state_dict
    if not expected or set(state_dict.keys()) & expected:
        return state_dict  # already the right layout

    stripped = {
        key[len(_CLIP_TEXT_PREFIX):]: value
        for key, value in state_dict.items()
        if key.startswith(_CLIP_TEXT_PREFIX)
    }
    if stripped and set(stripped) & expected:
        print(
            "Adapting text-encoder checkpoint: stripping "
            f"'{_CLIP_TEXT_PREFIX}' prefix (transformers>=5 layout).",
            flush=True,
        )
        return stripped

    added = {_CLIP_TEXT_PREFIX + key: value for key, value in state_dict.items()}
    if set(added) & expected:
        print(
            "Adapting text-encoder checkpoint: adding "
            f"'{_CLIP_TEXT_PREFIX}' prefix (transformers<5 layout).",
            flush=True,
        )
        return added

    return state_dict


def _sanitize_full_text_encoder_state_dict(state_dict, module=None):
    clean_state = {}
    for key, value in state_dict.items():
        if ".tlora." in key or key.startswith("tlora.") or ".lora1." in key or key.startswith("lora1."):
            continue
        clean_state[key.replace(".base_layer.", ".")] = value
    return align_text_encoder_state_dict(clean_state, module)


def freeze_all_tlora(unet, text_encoder, text_encoder_2):
    """Freeze all T-LoRA parameters."""
    for proc in unet.attn_processors.values():
        if not isinstance(proc, TLoRACrossAttnProcessor):
            continue
        for lora_module in [
            proc.tlora_q,
            proc.tlora_k,
            proc.tlora_v,
            proc.tlora_out,
        ]:
            for param in lora_module.parameters():
                param.requires_grad_(False)

    for text_model in (text_encoder, text_encoder_2):
        if text_model is None:
            continue
        for module in text_model.modules():
            if not isinstance(module, TLoRATextLinearLayer):
                continue
            for param in module.tlora.parameters():
                param.requires_grad_(False)


def collect_tlora_trainable_layers(unet):
    layers = []
    seen_ptrs = set()

    def maybe_add(name, weight):
        ptr = weight.data_ptr()
        if ptr in seen_ptrs:
            return
        seen_ptrs.add(ptr)
        layers.append((name, weight))

    for proc_name, proc in unet.attn_processors.items():
        if not isinstance(proc, TLoRACrossAttnProcessor):
            continue

        attn_path = proc_name.split(".processor")[0]
        attn_layer = unet.get_submodule(attn_path)

        for proj_name in ["to_q", "to_k", "to_v"]:
            proj = getattr(attn_layer, proj_name, None)
            if proj is not None and hasattr(proj, "weight"):
                maybe_add(f"unet.{attn_path}.{proj_name}.weight", proj.weight)

        if hasattr(attn_layer, "to_out") and len(attn_layer.to_out) > 0:
            maybe_add(f"unet.{attn_path}.to_out.0.weight", attn_layer.to_out[0].weight)

    return layers


def load_mr_state(mr_checkpoint_path):
    if not os.path.exists(mr_checkpoint_path):
        raise FileNotFoundError(f"MR checkpoint not found: {mr_checkpoint_path}")
    state = torch.load(mr_checkpoint_path, map_location="cpu")
    # Support legacy checkpoints that used "W_R" key
    if "M_R" not in state and "W_R" in state:
        state["M_R"] = state.pop("W_R")
    if "M_R" not in state:
        raise KeyError(f"MR checkpoint missing key 'M_R': {mr_checkpoint_path}")
    return state


def build_mr_injected_unet(unet, mr_state):
    trainable_layers = collect_tlora_trainable_layers(unet)
    if len(trainable_layers) == 0:
        raise RuntimeError("No trainable attention layers found for M_R injection")

    mr_names = [name for name, _ in trainable_layers]
    missing_mr = sorted(set(mr_names) - set(mr_state.keys()))
    if missing_mr:
        raise KeyError(
            f"M_R missing {len(missing_mr)} layers required by current UNet. "
            f"First missing key: {missing_mr[0]}"
        )

    unet_param_lookup = dict(unet.named_parameters())
    mr_weights = {}
    unet_override_specs = []
    for full_name, weight in trainable_layers:
        local_name = full_name[len("unet."):]
        if local_name not in unet_param_lookup:
            continue

        mr_tensor = mr_state[full_name].to(device=weight.device, dtype=weight.dtype)
        if mr_tensor.shape != weight.shape:
            raise ValueError(
                f"Shape mismatch for {full_name}: "
                f"M_R {tuple(mr_tensor.shape)} vs UNet {tuple(weight.shape)}"
            )

        mr_weights[full_name] = mr_tensor
        unet_override_specs.append((full_name, local_name, unet_param_lookup[local_name]))

    extra_mr_keys = sorted(set(mr_state.keys()) - set(mr_names))
    print(f"M_R loaded virtually for {len(unet_override_specs)} layers")
    return MRInjectedUNet(unet, unet_override_specs, mr_weights), extra_mr_keys


def load_mr_backbone(unet, mr_checkpoint_path):
    mr_state = load_mr_state(mr_checkpoint_path)
    wrapped_unet, extra_mr_keys = build_mr_injected_unet(unet, mr_state["M_R"])
    if extra_mr_keys:
        print(f"Warning: {len(extra_mr_keys)} extra M_R keys not used by current UNet")
    return wrapped_unet, mr_state, extra_mr_keys


# Backward-compatibility aliases
load_wr_state = load_mr_state
build_wr_injected_unet = build_mr_injected_unet
load_wr_backbone = load_mr_backbone


def load_backbone_info(continue_lora_path):
    info_path = os.path.join(continue_lora_path, "backbone_info.json")
    if not os.path.exists(info_path):
        return {}
    with open(info_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_continue_adapter_path(continue_lora_path):
    if not os.path.isdir(continue_lora_path):
        raise FileNotFoundError(f"Continue LoRA checkpoint directory not found: {continue_lora_path}")

    adapter_safetensors = os.path.join(continue_lora_path, "adapter_model.safetensors")
    adapter_bin = os.path.join(continue_lora_path, "adapter_model.bin")
    if not (os.path.exists(adapter_safetensors) or os.path.exists(adapter_bin)):
        raise FileNotFoundError(
            "Continue LoRA checkpoint must contain adapter_model.safetensors or adapter_model.bin: "
            f"{continue_lora_path}"
        )


def apply_continue_lora_to_unet(unet, continue_lora_path, device, model_dtype):
    """Load a saved continue-LoRA adapter onto unet, keeping PEFT active (no merge).

    This mirrors the training setup exactly: PEFT wraps the unet and its adapters
    remain live during inference, so the computation is identical to training.
    For ES mode (MRInjectedUNet backbone), rebind_for_peft() is called so that
    functional_call can swap parameters through PEFT's base_layer.
    """
    from peft import PeftModel

    validate_continue_adapter_path(continue_lora_path)
    print(f"Loading continue LoRA adapter from: {continue_lora_path}")

    unet = PeftModel.from_pretrained(unet, continue_lora_path, is_trainable=False)

    # For ES mode: PEFT wraps to_q/to_k/to_v/to_out.0 with LoraLinear whose
    # .weight is a read-only property, breaking MRInjectedUNet's functional_call.
    base_model = unet.get_base_model()
    if isinstance(base_model, MRInjectedUNet):
        base_model.rebind_for_peft()

    unet = unet.to(device=device, dtype=model_dtype)
    print("Continue LoRA adapter loaded")
    return unet


def load_continue_backbone_components(
    mode,
    pretrained_model_name_or_path,
    lora_path,
    revision=None,
    variant=None,
    torch_dtype=torch.float32,
    rl_checkpoint_path=None,
    full_weights_path=None,
    further_lora_path=None,
):
    components = load_base_sdxl_backbone(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        revision=revision,
        variant=variant,
        torch_dtype=torch_dtype,
    )
    freeze_sdxl_components(components)

    rl_state = None
    metadata = None
    extra_mr_keys = None
    if mode == "es":
        if rl_checkpoint_path is None:
            raise ValueError("mode=es requires rl_checkpoint_path")
        rl_state = load_mr_state(rl_checkpoint_path)
        metadata = rl_state

    if mode == "full":
        if full_weights_path is None:
            raise ValueError("mode=full requires full_weights_path")
        cfg = resolve_tlora_config(lora_path, metadata=metadata)
        load_further_full_weights(components, full_weights_path)
        _set_tlora_config(components, cfg)
        return BackboneLoadResult(
            components=components,
            tlora_config=cfg,
            rl_state=None,
            extra_mr_keys=None,
        )

    # All modes load the single T-LoRA adapter the same way.
    # Caller (training script) is responsible for freezing before PEFT injection.
    result = load_tlora_model(
        components.unet,
        components.text_encoder,
        components.text_encoder_2,
        lora_path,
        metadata=metadata,
        target=components,
    )

    if mode == "further_lora":
        if further_lora_path is None:
            raise ValueError("mode=further_lora requires further_lora_path")
        load_further_lora_weights(components, further_lora_path, freeze=True)
    elif mode not in {"direct", "es"}:
        raise ValueError(f"Unknown backbone mode: {mode}")

    if mode == "es":
        components.unet, extra_mr_keys = build_mr_injected_unet(components.unet, rl_state["M_R"])
        if extra_mr_keys:
            print(f"Warning: {len(extra_mr_keys)} extra M_R keys not used by current UNet")

    return BackboneLoadResult(
        components=components,
        tlora_config=dict(result.config),
        rl_state=rl_state,
        extra_mr_keys=extra_mr_keys,
    )


def normalized_path_for_compare(path):
    if path is None:
        return None
    return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def ensure_matching_backbone_arg(arg_name, provided_path, saved_path, mode_name):
    if provided_path is None or saved_path is None:
        return
    if normalized_path_for_compare(provided_path) != normalized_path_for_compare(saved_path):
        raise ValueError(
            f"{mode_name} mode backbone mismatch for {arg_name}: "
            f"continue checkpoint was trained with {saved_path}, "
            f"but inference was given {provided_path}. "
            f"Use the same backbone checkpoint that was used during continue training."
        )
