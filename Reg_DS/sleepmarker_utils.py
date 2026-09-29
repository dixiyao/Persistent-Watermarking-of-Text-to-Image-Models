"""Post-hoc SleeperMark latent watermark helpers.

The implementation follows SleeperMark Stage 1: a binary message is projected
to a four-channel residual, added to a VAE image latent, and recovered by a
latent extractor after the watermarked image is encoded again.
"""

import json
import os
import secrets

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


ENCODER_NAMES = ("secret_encoder.pt", "encoder.pth")
DECODER_NAMES = ("secret_decoder.pt", "decoder.pth")


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _find_artifact(stage1_dir, names):
    path = _find_optional_artifact(stage1_dir, names)
    if path is not None:
        return path
    raise FileNotFoundError(
        f"None of {', '.join(names)} was found in SleeperMark directory: {stage1_dir}"
    )


def _find_optional_artifact(stage1_dir, names):
    for name in names:
        path = os.path.join(stage1_dir, name)
        if os.path.isfile(path):
            return path
    return None


def _load_config(stage1_dir):
    config = {}
    for name in ("stage1_config.json", "stage2_config.json"):
        path = os.path.join(stage1_dir, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                config.update(json.load(handle))
    return config


def _clean_state_dict(value):
    if isinstance(value, dict):
        for key in ("state_dict", "model", "module"):
            nested = value.get(key)
            if isinstance(nested, dict):
                value = nested
                break
    if not isinstance(value, dict):
        raise TypeError("SleeperMark checkpoint must contain a PyTorch state dict")
    return {
        (key[len("module."):] if key.startswith("module.") else key): tensor
        for key, tensor in value.items()
    }


def _infer_secret_size(config, decoder_state=None, encoder_state=None):
    if "secret_size" in config:
        return int(config["secret_size"])
    if decoder_state is not None:
        for key, value in decoder_state.items():
            if key.endswith("mlps.4.linear.weight") or key.endswith("mlps.4.weight"):
                return int(value.shape[0])
    if encoder_state is not None:
        for key, value in encoder_state.items():
            if key.endswith("secret_scaler.0.weight"):
                return int(value.shape[1])
    return 48


class _View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, value):
        return value.view(*self.shape)


class _Repeat(nn.Module):
    def __init__(self, *sizes):
        super().__init__()
        self.sizes = sizes

    def forward(self, value):
        return value.repeat(1, *self.sizes)


class _Linear(nn.Module):
    def __init__(self, in_features, out_features, activation="relu"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "selu":
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None

    def forward(self, value):
        value = self.linear(value)
        return self.act(value) if self.act is not None else value


class _Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation="relu", strides=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            strides,
            int((kernel_size - 1) / 2),
        )
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "selu":
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None

    def forward(self, value):
        value = self.conv(value)
        return self.act(value) if self.act is not None else value


class _Flatten(nn.Module):
    def forward(self, value):
        return value.contiguous().view(value.size(0), -1)


class SecretEncoder(nn.Module):
    """Projector used by this repository's resolution-flexible Stage 1 code."""

    def __init__(self, secret_size=48, base_res=32, latent_resolution=128):
        super().__init__()
        self.latent_resolution = int(latent_resolution)
        self.secret_scaler = nn.Sequential(
            nn.Linear(int(secret_size), int(base_res) * int(base_res)),
            nn.SiLU(),
            nn.Linear(int(base_res) * int(base_res), int(base_res) * int(base_res)),
            nn.SiLU(),
            _View(-1, 1, int(base_res), int(base_res)),
            _Repeat(4, 1, 1),
            nn.Conv2d(4, 4, 3, padding=1),
        )

    def forward(self, secret):
        residual = self.secret_scaler(secret)
        if residual.shape[-2:] != (self.latent_resolution, self.latent_resolution):
            residual = F.interpolate(
                residual,
                size=(self.latent_resolution, self.latent_resolution),
                mode="bilinear",
                align_corners=False,
            )
        return residual


class ExtractorForLatent(nn.Module):
    def __init__(self, secret_size=48):
        super().__init__()
        self.decoder = nn.Sequential(
            _Conv2D(4, 64, 3, strides=2, activation="selu"),
            _Conv2D(64, 64, 3, activation="selu"),
            _Conv2D(64, 128, 3, strides=2, activation="selu"),
            _Conv2D(128, 128, 3, activation="selu"),
            _Conv2D(128, 256, 3, strides=2, activation="selu"),
            _Conv2D(256, 256, 3, activation="selu"),
            _Conv2D(256, 512, 3, strides=2, activation="selu"),
            _Conv2D(512, 512, 3, activation="selu"),
            nn.AdaptiveAvgPool2d((4, 4)),
            _Flatten(),
        )
        self.mlps = nn.Sequential(
            _Linear(8192, 2048, activation="selu"),
            _Linear(2048, 2048, activation="selu"),
            _Linear(2048, 2048, activation="selu"),
            nn.Dropout(p=0.1),
            _Linear(2048, int(secret_size), activation=None),
        )

    def forward(self, latent):
        return self.mlps(self.decoder(latent))


class OfficialSecretEncoder(nn.Module):
    """Architecture used by the official SD v1.4 SleeperMark Stage 1 code."""

    def __init__(self, secret_size=48, base_res=32, latent_resolution=64):
        super().__init__()
        scale = float(latent_resolution) / float(base_res)
        self.secret_scaler = nn.Sequential(
            nn.Linear(secret_size, base_res * base_res),
            nn.SiLU(),
            nn.Linear(base_res * base_res, base_res * base_res),
            nn.SiLU(),
            _View(-1, 1, base_res, base_res),
            _Repeat(4, 1, 1),
            nn.Upsample(scale_factor=(scale, scale)),
            nn.Conv2d(4, 4, 3, padding=1),
        )

    def forward(self, secret):
        return self.secret_scaler(secret)


def load_sleepmarker_models(
    stage1_dir,
    device,
    dtype=torch.float32,
    resolution=512,
    load_encoder=True,
):
    """Load this repository's or the official SleeperMark Stage 1 artifacts."""
    if not os.path.isdir(stage1_dir):
        raise FileNotFoundError(f"SleeperMark Stage 1 directory not found: {stage1_dir}")

    config = _load_config(stage1_dir)
    encoder_path = _find_optional_artifact(stage1_dir, ENCODER_NAMES)
    encoder_state = None
    decoder_path = _find_optional_artifact(stage1_dir, DECODER_NAMES)
    if encoder_path is not None and (load_encoder or decoder_path is None):
        encoder_state = _clean_state_dict(safe_torch_load(encoder_path))

    decoder_state = None
    if decoder_path is not None:
        decoder_state = _clean_state_dict(safe_torch_load(decoder_path))

    secret_size = _infer_secret_size(
        config,
        decoder_state=decoder_state,
        encoder_state=encoder_state,
    )
    latent_resolution = int(config.get("latent_resolution", int(resolution) // 8))

    encoder_was_initialized = load_encoder and encoder_state is None
    if encoder_was_initialized:
        encoder_path = os.path.join(stage1_dir, ENCODER_NAMES[0])
        encoder_init_seed = int(config.get("encoder_init_seed", 0))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(encoder_init_seed)
            initialized_encoder = SecretEncoder(
                secret_size=secret_size,
                latent_resolution=latent_resolution,
            )
        encoder_state = initialized_encoder.state_dict()
        torch.save(encoder_state, encoder_path)
        print(
            "WARNING: No SleeperMark secret encoder was found. Initialized and saved "
            f"an untrained projector to {encoder_path}. It does not provide a trained "
            "SleeperMark watermark.",
            flush=True,
        )

    decoder_was_initialized = decoder_state is None
    if decoder_was_initialized:
        decoder_path = os.path.join(stage1_dir, DECODER_NAMES[0])
        decoder_init_seed = int(config.get("decoder_init_seed", 0))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(decoder_init_seed)
            extractor = ExtractorForLatent(secret_size=secret_size)
        torch.save(extractor.state_dict(), decoder_path)
        print(
            "WARNING: No SleeperMark secret decoder was found. Initialized and saved "
            f"an untrained extractor to {decoder_path}. Its bit accuracy is not a valid "
            "watermark measurement until the encoder and extractor are jointly trained "
            "with SleeperMark Stage 1.",
            flush=True,
        )
    else:
        extractor = ExtractorForLatent(secret_size=secret_size)
        extractor.load_state_dict(decoder_state, strict=True)
    extractor.to(device=device, dtype=dtype).eval().requires_grad_(False)

    encoder = None
    if load_encoder:
        if encoder_path is None:
            encoder_path = _find_artifact(stage1_dir, ENCODER_NAMES)
        if encoder_state is None:
            encoder_state = _clean_state_dict(safe_torch_load(encoder_path))
        is_official = any(key.startswith("secret_scaler.7.") for key in encoder_state)
        encoder_class = OfficialSecretEncoder if is_official else SecretEncoder
        encoder = encoder_class(
            secret_size=secret_size,
            latent_resolution=latent_resolution,
        )
        encoder.load_state_dict(encoder_state, strict=True)
        encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)

    info = {
        "config": config,
        "secret_size": secret_size,
        "latent_resolution": latent_resolution,
        "encoder_path": encoder_path,
        "decoder_path": decoder_path,
        "encoder_was_initialized": encoder_was_initialized,
        "decoder_was_initialized": decoder_was_initialized,
    }
    return encoder, extractor, info


def parse_or_create_secret(secret_value, secret_size, seed=None):
    if secret_value is None:
        if seed is None:
            bit_string = "".join(secrets.choice("01") for _ in range(secret_size))
        else:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            values = torch.randint(0, 2, (secret_size,), generator=generator)
            bit_string = "".join(str(int(value)) for value in values.tolist())
    else:
        bit_string = "".join(character for character in str(secret_value) if character in "01")
        invalid = [character for character in str(secret_value) if not character.isspace() and character not in "01,_-"]
        if invalid:
            raise ValueError("--sleepmarker_secret must be a binary string containing only 0 and 1")
        if len(bit_string) != secret_size:
            raise ValueError(
                f"SleeperMark checkpoint expects {secret_size} bits, but the supplied secret has "
                f"{len(bit_string)} bits"
            )
    tensor = torch.tensor(
        [[int(character) for character in bit_string]],
        dtype=torch.float32,
    )
    return bit_string, tensor


def _module_device(module):
    return next(module.parameters()).device


def pil_to_normalized_tensor(image, resolution, device, dtype):
    image = image.convert("RGB").resize(
        (int(resolution), int(resolution)),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (2.0 * tensor - 1.0).to(device=device, dtype=dtype)


def tensor_to_pil(tensor):
    tensor = torch.nan_to_num(tensor.detach().float(), nan=0.0, posinf=1.0, neginf=-1.0)
    tensor = (tensor.clamp(-1, 1) / 2.0 + 0.5).clamp(0, 1)
    array = tensor[0].permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255.0).round().astype(np.uint8))


@torch.no_grad()
def encode_image(image, vae, resolution, generator=None):
    pixels = pil_to_normalized_tensor(
        image,
        resolution,
        _module_device(vae),
        vae.dtype,
    )
    posterior = vae.encode(pixels).latent_dist
    latents = posterior.sample(generator=generator)
    return latents * vae.config.scaling_factor


@torch.no_grad()
def decode_image(latents, vae):
    decoded = vae.decode((latents / vae.config.scaling_factor).to(dtype=vae.dtype)).sample
    return tensor_to_pil(decoded)


@torch.no_grad()
def inject_secret(image, vae, encoder, secret_bits, resolution, generator=None):
    latents = encode_image(image, vae, resolution, generator=generator)
    encoder_device = _module_device(encoder)
    encoder_dtype = next(encoder.parameters()).dtype
    residual = encoder(secret_bits.to(device=encoder_device, dtype=encoder_dtype))
    residual = residual.to(device=latents.device, dtype=latents.dtype)
    if residual.shape[-2:] != latents.shape[-2:]:
        residual = F.interpolate(
            residual,
            size=latents.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return decode_image(latents + residual, vae), residual


@torch.no_grad()
def detect_secret(image, vae, extractor, resolution, generator=None):
    latents = encode_image(image, vae, resolution, generator=generator)
    extractor_device = _module_device(extractor)
    extractor_dtype = next(extractor.parameters()).dtype
    logits = extractor(latents.to(device=extractor_device, dtype=extractor_dtype))
    probabilities = torch.nan_to_num(
        torch.sigmoid(logits),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    predicted = (probabilities >= 0.5).to(dtype=torch.int64)
    bit_string = "".join(str(int(value)) for value in predicted[0].cpu().tolist())
    return bit_string, probabilities.detach().float().cpu()
