import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrthogonalLoRALinearLayer(nn.Module):
    """Ortho-LoRA layer with fast zero-delta initialization and masking."""

    def __init__(
        self,
        in_features,
        out_features,
        rank=4,
        sig_type="last",
        original_layer=None,
        ortho_init="random",
        skip_init=False,
    ):
        super().__init__()
        if rank > in_features or rank > out_features:
            raise ValueError(
                f"LoRA rank {rank} must be <= in_features ({in_features}) "
                f"and out_features ({out_features})"
            )

        self.rank = rank

        self.q_layer = nn.Linear(in_features, rank, bias=False)
        self.p_layer = nn.Linear(rank, out_features, bias=False)
        self.lambda_layer = nn.Parameter(torch.ones(1, rank))

        with torch.no_grad():
            if skip_init:
                nn.init.zeros_(self.q_layer.weight)
                nn.init.zeros_(self.p_layer.weight)
                self.lambda_layer.zero_()
            else:
                q_w, p_w, lam = self._svd_init(
                    in_features=in_features,
                    out_features=out_features,
                    rank=rank,
                    sig_type=sig_type,
                    original_layer=original_layer,
                    ortho_init=ortho_init,
                )
                self.q_layer.weight.copy_(q_w.to(dtype=self.q_layer.weight.dtype))
                self.p_layer.weight.copy_(p_w.to(dtype=self.p_layer.weight.dtype))
                self.lambda_layer.copy_(lam.to(dtype=self.lambda_layer.dtype))

        self.base_p = copy.deepcopy(self.p_layer)
        self.base_q = copy.deepcopy(self.q_layer)
        self.base_lambda = copy.deepcopy(self.lambda_layer)

        # Ordinary LoRA has no frozen reference branch to subtract: B starts at
        # zero, so the delta is already zero at init.
        if ortho_init == "lora" and not skip_init:
            with torch.no_grad():
                self.base_q.weight.zero_()
                self.base_p.weight.zero_()
                self.base_lambda.zero_()

        for param in self.parameters():
            param.data = param.data.contiguous()

        self.base_p.requires_grad_(False)
        self.base_q.requires_grad_(False)
        self.base_lambda.requires_grad_(False)

    @staticmethod
    def _svd_device():
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @classmethod
    def _svd_init(cls, in_features, out_features, rank, sig_type, original_layer, ortho_init):
        """Original T-LoRA-style SVD init.

        The default `random` mode follows ControlGenAI/T-LoRA `ortho_lora`:
        SVD of a random Gaussian matrix, not SVD of the SDXL backbone weight.
        `base_layer` matches their heavier layer-aware variant.
        """
        if ortho_init not in {"random", "base_layer", "lora"}:
            raise ValueError(
                f"Unsupported ortho_init={ortho_init!r}; expected 'random', "
                "'base_layer' or 'lora'")
        if ortho_init == "lora":
            # Ordinary LoRA: A ~ N(0, 1/rank), B = 0, no orthogonal structure.
            # The caller zeroes base_q / base_p / base_lambda so the residual
            # in forward() vanishes and the delta is a plain B @ A.
            q_w = torch.randn(rank, in_features) / (rank ** 0.5)
            p_w = torch.zeros(out_features, rank)
            lam = torch.ones(1, rank)
            return q_w, p_w, lam

        svd_device = cls._svd_device()
        if ortho_init == "base_layer":
            if original_layer is None:
                raise ValueError("ortho_init='base_layer' requires original_layer")
            weight = original_layer.weight.detach().float().to(svd_device)
            u, s, vh = torch.linalg.svd(weight, full_matrices=True)
            q_w, p_w, lam = cls._pick_components(
                q_src=vh,
                p_src=u,
                s=s,
                rank=rank,
                sig_type=sig_type,
            )
            del u, s, vh, weight
        else:
            base_m = torch.normal(
                mean=0.0,
                std=1.0 / rank,
                size=(in_features, out_features),
                device=svd_device,
            )
            u, s, vh = torch.linalg.svd(base_m, full_matrices=True)
            q_w, p_w, lam = cls._pick_components(
                q_src=u,
                p_src=vh,
                s=s,
                rank=rank,
                sig_type=sig_type,
            )
            del u, s, vh, base_m

        q_w = q_w.cpu()
        p_w = p_w.cpu()
        lam = lam.cpu()
        if svd_device.type == "cuda":
            torch.cuda.empty_cache()
        return q_w, p_w, lam

    @staticmethod
    def _pick_components(q_src, p_src, s, rank, sig_type):
        if sig_type == "principal":
            q_w = q_src[:rank].clone()
            p_w = p_src[:, :rank].clone()
            lam = s[None, :rank].clone()
        elif sig_type == "last":
            q_w = q_src[-rank:].clone()
            p_w = p_src[:, -rank:].clone()
            lam = s[None, -rank:].clone()
        elif sig_type == "middle":
            q_start = math.ceil((q_src.shape[0] - rank) / 2)
            p_start = math.ceil((p_src.shape[1] - rank) / 2)
            s_start = math.ceil((s.shape[0] - rank) / 2)
            q_w = q_src[q_start:q_start + rank].clone()
            p_w = p_src[:, p_start:p_start + rank].clone()
            lam = s[None, s_start:s_start + rank].clone()
        else:
            raise ValueError(f"Unsupported sig_type={sig_type!r}")
        return q_w, p_w, lam

    def forward(self, hidden_states, mask=None):
        if mask is None:
            mask = torch.ones((1, self.rank), device=hidden_states.device)

        orig_dtype = hidden_states.dtype
        dtype = self.q_layer.weight.dtype
        mask = mask.to(device=hidden_states.device, dtype=dtype)

        q_hidden = self.q_layer(hidden_states.to(dtype)) * self.lambda_layer * mask
        p_hidden = self.p_layer(q_hidden)

        base_q_hidden = self.base_q(hidden_states.to(dtype)) * self.base_lambda * mask
        base_p_hidden = self.base_p(base_q_hidden)

        return (p_hidden - base_p_hidden).to(orig_dtype)

    def regularization(self):
        a = self.q_layer.weight
        b = self.p_layer.weight
        eye = torch.eye(self.rank, device=a.device, dtype=a.dtype)
        a_reg = torch.sum((a @ a.T - eye) ** 2)
        b_reg = torch.sum((b.T @ b - eye) ** 2)
        return a_reg + b_reg


class TLoRACrossAttnProcessor(nn.Module):
    """Attention processor with a single T-LoRA adapter."""

    def __init__(
        self,
        hidden_size,
        cross_attention_dim=None,
        rank=4,
        lora_alpha=32,
        sig_type="last",
        original_layer=None,
        ortho_init="random",
        skip_init=False,
    ):
        super().__init__()

        in_features = cross_attention_dim if cross_attention_dim is not None else hidden_size

        self.tlora_q = OrthogonalLoRALinearLayer(
            hidden_size,
            hidden_size,
            rank,
            sig_type,
            original_layer.to_q if original_layer is not None else None,
            ortho_init=ortho_init,
            skip_init=skip_init,
        )
        self.tlora_k = OrthogonalLoRALinearLayer(
            in_features,
            hidden_size,
            rank,
            sig_type,
            original_layer.to_k if original_layer is not None else None,
            ortho_init=ortho_init,
            skip_init=skip_init,
        )
        self.tlora_v = OrthogonalLoRALinearLayer(
            in_features,
            hidden_size,
            rank,
            sig_type,
            original_layer.to_v if original_layer is not None else None,
            ortho_init=ortho_init,
            skip_init=skip_init,
        )
        self.tlora_out = OrthogonalLoRALinearLayer(
            hidden_size,
            hidden_size,
            rank,
            sig_type,
            original_layer.to_out[0] if original_layer is not None else None,
            ortho_init=ortho_init,
            skip_init=skip_init,
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        sigma_mask=None,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = (
            attn.to_q(hidden_states)
            + self.tlora_q(hidden_states, sigma_mask)
        )

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = (
            attn.to_k(encoder_hidden_states)
            + self.tlora_k(encoder_hidden_states, sigma_mask)
        )

        value = (
            attn.to_v(encoder_hidden_states)
            + self.tlora_v(encoder_hidden_states, sigma_mask)
        )

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size,
            -1,
            attn.heads * head_dim,
        )
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = (
            attn.to_out[0](hidden_states)
            + self.tlora_out(hidden_states, sigma_mask)
        )
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


_TEXT_ENCODER_SIGMA_STATE = {"mask": None}


def set_text_encoder_sigma_mask(mask):
    _TEXT_ENCODER_SIGMA_STATE["mask"] = mask


def clear_text_encoder_sigma_mask():
    _TEXT_ENCODER_SIGMA_STATE["mask"] = None


class TLoRATextLinearLayer(nn.Module):
    """T-LoRA wrapper for CLIP text encoder linear projections."""

    def __init__(self, base_layer, rank=4, lora_alpha=32, sig_type="last", ortho_init="random", skip_init=False):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.tlora_disabled = False

        self.tlora = OrthogonalLoRALinearLayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            sig_type=sig_type,
            original_layer=base_layer,
            ortho_init=ortho_init,
            skip_init=skip_init,
        )

    def forward(self, hidden_states):
        if self.tlora_disabled:
            return self.base_layer(hidden_states)

        sigma_mask = None
        if _TEXT_ENCODER_SIGMA_STATE["mask"] is not None:
            sigma_mask = _TEXT_ENCODER_SIGMA_STATE["mask"].to(
                device=hidden_states.device,
                dtype=self.tlora.lambda_layer.dtype,
            )

        return (
            self.base_layer(hidden_states)
            + self.tlora(hidden_states, sigma_mask)
        )


def compute_orthogonal_lora_weight_delta(lora_layer, mask=None):
    """Compute effective weight delta equivalent to OrthogonalLoRALinearLayer forward()."""
    with torch.no_grad():
        dtype = lora_layer.q_layer.weight.dtype
        device = lora_layer.q_layer.weight.device

        if mask is None:
            mask = torch.ones((1, lora_layer.rank), device=device, dtype=dtype)
        else:
            mask = mask.to(device=device, dtype=dtype)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)

        scale = lora_layer.lambda_layer.to(device=device, dtype=dtype) * mask
        base_scale = lora_layer.base_lambda.to(device=device, dtype=dtype) * mask

        p = lora_layer.p_layer.weight.to(device=device, dtype=dtype)
        q = lora_layer.q_layer.weight.to(device=device, dtype=dtype)
        base_p = lora_layer.base_p.weight.to(device=device, dtype=dtype)
        base_q = lora_layer.base_q.weight.to(device=device, dtype=dtype)

        return (p * scale).matmul(q) - (base_p * base_scale).matmul(base_q)



def get_mask_by_timestep(timestep, max_timestep, max_rank, min_rank=1, alpha=1.0):
    r = int(((max_timestep - timestep) / max_timestep) ** alpha * (max_rank - min_rank)) + min_rank
    sigma_mask = torch.zeros((1, max_rank))
    sigma_mask[:, :r] = 1.0
    return sigma_mask


def get_layer_by_name(module, layer_path):
    return module.get_submodule(layer_path)


def set_layer_by_name(module, layer_path, new_layer):
    parts = layer_path.split(".")
    if len(parts) == 1:
        setattr(module, parts[0], new_layer)
        return
    parent = module.get_submodule(".".join(parts[:-1]))
    setattr(parent, parts[-1], new_layer)


def setup_text_encoder_tlora(text_encoder, rank, lora_alpha, sig_type, ortho_init="random", skip_init=False):
    target_suffixes = ("q_proj", "k_proj", "v_proj", "out_proj")
    target_layer_paths = []

    for name, layer in text_encoder.named_modules():
        if name.endswith(target_suffixes) and isinstance(layer, nn.Linear):
            target_layer_paths.append(name)

    tlora_params = []

    for layer_path in target_layer_paths:
        base_layer = get_layer_by_name(text_encoder, layer_path)
        tlora_layer = TLoRATextLinearLayer(
            base_layer=base_layer,
            rank=rank,
            lora_alpha=lora_alpha,
            sig_type=sig_type,
            ortho_init=ortho_init,
            skip_init=skip_init,
        )
        set_layer_by_name(text_encoder, layer_path, tlora_layer)

        for p in tlora_layer.tlora.parameters():
            if p.requires_grad:
                tlora_params.append(p)

    return target_layer_paths, tlora_params


def collect_text_encoder_tlora_state_dict(text_encoder):
    state_dict = {}
    for name, layer in text_encoder.named_modules():
        if isinstance(layer, TLoRATextLinearLayer):
            for key, value in layer.state_dict().items():
                if key.startswith("tlora."):
                    state_dict[f"{name}.{key}"] = value
    return state_dict


def load_text_encoder_tlora_weights(
    text_encoder,
    weights_path,
    rank,
    lora_alpha,
    sig_type,
    ortho_init="random",
    skip_init=True,
    freeze=True,
):
    if not os.path.exists(weights_path):
        return 0

    target_layer_paths = [
        name
        for name, layer in text_encoder.named_modules()
        if isinstance(layer, TLoRATextLinearLayer)
    ]
    if not target_layer_paths:
        target_layer_paths, _ = setup_text_encoder_tlora(
            text_encoder,
            rank=rank,
            lora_alpha=lora_alpha,
            sig_type=sig_type,
            ortho_init=ortho_init,
            skip_init=skip_init,
        )

    state_dict = {
        key.replace(".lora1.", ".tlora."): value
        for key, value in torch.load(weights_path, map_location="cpu").items()
    }
    loaded = 0

    for layer_path in target_layer_paths:
        layer = get_layer_by_name(text_encoder, layer_path)
        prefix = f"{layer_path}."
        layer_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not layer_state:
            continue

        layer.load_state_dict(layer_state, strict=False)
        loaded += 1

    text_param = next(text_encoder.parameters())
    for module in text_encoder.modules():
        if isinstance(module, TLoRATextLinearLayer):
            module.to(device=text_param.device, dtype=text_param.dtype)
            if freeze:
                module.requires_grad_(False)
            else:
                module.base_layer.requires_grad_(False)
                module.tlora.requires_grad_(True)

    return loaded


def build_tlora_attn_processors(
    unet,
    rank=4,
    lora_alpha=32,
    sig_type="last",
    ortho_init="random",
    skip_init=False,
):
    tlora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            continue

        original_layer = get_layer_by_name(unet, name.split(".processor")[0])

        tlora_attn_procs[name] = TLoRACrossAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
            lora_alpha=lora_alpha,
            sig_type=sig_type,
            original_layer=original_layer,
            ortho_init=ortho_init,
            skip_init=skip_init,
        )

    return tlora_attn_procs


def collect_unet_tlora_trainable_params(unet):
    tlora_params = []
    for proc in unet.attn_processors.values():
        if not isinstance(proc, TLoRACrossAttnProcessor):
            continue
        for sub in [proc.tlora_q, proc.tlora_k, proc.tlora_v, proc.tlora_out]:
            tlora_params.extend(p for p in sub.parameters() if p.requires_grad)
    return tlora_params


def setup_sdxl_tlora(
    unet,
    text_encoder,
    text_encoder_2,
    rank,
    lora_alpha,
    sig_type,
    ortho_init="random",
    skip_init=False,
):
    """Install one T-LoRA adapter across SDXL UNet and both text encoders."""
    tlora_attn_procs = build_tlora_attn_processors(
        unet,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
        ortho_init=ortho_init,
        skip_init=skip_init,
    )
    unet.set_attn_processor(tlora_attn_procs)

    _, te1_tlora_params = setup_text_encoder_tlora(
        text_encoder,
        rank=rank,
        lora_alpha=lora_alpha,
        sig_type=sig_type,
        ortho_init=ortho_init,
        skip_init=skip_init,
    )
    if getattr(text_encoder_2, "is_placeholder_text_encoder", False):
        te2_tlora_params = []
    else:
        _, te2_tlora_params = setup_text_encoder_tlora(
            text_encoder_2,
            rank=rank,
            lora_alpha=lora_alpha,
            sig_type=sig_type,
            ortho_init=ortho_init,
            skip_init=skip_init,
        )

    tlora_params = collect_unet_tlora_trainable_params(unet)
    tlora_params.extend(te1_tlora_params)
    tlora_params.extend(te2_tlora_params)
    return tlora_params


def collect_tlora_attn_state_dict(unet):
    tlora_state_dict = {}
    for name, proc in unet.attn_processors.items():
        if isinstance(proc, TLoRACrossAttnProcessor):
            for key, value in proc.state_dict().items():
                tlora_state_dict[f"{name}.{key}"] = value
    return tlora_state_dict


def load_tlora_attn_state_dict(unet, state_dict, strict=True):
    state_dict = {
        key.replace(".lora1_", ".tlora_"): value
        for key, value in state_dict.items()
    }
    loaded = 0
    for name, proc in unet.attn_processors.items():
        if not isinstance(proc, TLoRACrossAttnProcessor):
            continue

        prefix = f"{name}."
        proc_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not proc_state:
            continue

        proc.load_state_dict(proc_state, strict=strict)
        loaded += 1

    return loaded


def freeze_tlora_attn_processors(unet):
    unet_param = next(unet.parameters())
    for proc in unet.attn_processors.values():
        if isinstance(proc, TLoRACrossAttnProcessor):
            proc.to(device=unet_param.device, dtype=unet_param.dtype)
            proc.requires_grad_(False)


def attach_tlora_sigma_mask_hook(unet, rank, min_rank, alpha_rank_scale, max_timestep):
    if getattr(unet, "tlora_sigma_hook_attached", False):
        return

    original_forward = unet.forward

    def forward_with_sigma_mask(*args, **kwargs):
        sample = args[0] if len(args) > 0 else kwargs.get("sample")
        timestep = args[1] if len(args) > 1 else kwargs.get("timestep")

        cross_attention_kwargs = kwargs.get("cross_attention_kwargs")
        if cross_attention_kwargs is None:
            cross_attention_kwargs = {}
        else:
            cross_attention_kwargs = dict(cross_attention_kwargs)

        if "sigma_mask" not in cross_attention_kwargs and timestep is not None:
            if torch.is_tensor(timestep):
                timestep_value = float(timestep.flatten()[0].item())
            else:
                timestep_value = float(timestep)

            sigma_mask = get_mask_by_timestep(
                timestep=timestep_value,
                max_timestep=max_timestep,
                max_rank=rank,
                min_rank=min_rank,
                alpha=alpha_rank_scale,
            )

            mask_device = sample.device if sample is not None else unet.device
            cross_attention_kwargs["sigma_mask"] = sigma_mask.to(mask_device)

        kwargs["cross_attention_kwargs"] = cross_attention_kwargs
        return original_forward(*args, **kwargs)

    unet.forward = forward_with_sigma_mask
    unet.tlora_sigma_hook_attached = True


def set_text_encoder_sigma_for_generation(pipeline, num_inference_steps):
    config = getattr(pipeline, "tlora_text_encoder_config", None)
    if config is None:
        return False

    rank = int(config["rank"])
    min_rank = int(config["min_rank"])
    alpha_rank_scale = float(config["alpha_rank_scale"])
    max_timestep = int(config["max_timestep"])

    try:
        device = next(pipeline.unet.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        device = "cpu"

    timestep_value = float(max_timestep - 1)
    try:
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps
        if len(timesteps) > 0:
            first_t = timesteps[0]
            timestep_value = float(first_t.item()) if torch.is_tensor(first_t) else float(first_t)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        pass

    sigma_mask = get_mask_by_timestep(
        timestep=timestep_value,
        max_timestep=max_timestep,
        max_rank=rank,
        min_rank=min_rank,
        alpha=alpha_rank_scale,
    ).to(device)

    set_text_encoder_sigma_mask(sigma_mask)
    return True
