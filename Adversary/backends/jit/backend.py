"""PixelDiT backend: HF-managed weights + the official model definition.

Weights come from the Hugging Face Hub. The *model definition* is imported from
NVIDIA's NVlabs/PixelDiT checkout because PixelDiT is a bespoke dual-level
architecture (patch-DiT + pixel-DiT with MM-DiT text fusion) that neither
diffusers nor transformers implements.

Only the model, scheduler and text encoder are borrowed. The training loop --
the CP/ORG split, the reference term, WatermarkDM, RoMa and the study logging --
is ours, in jit/train_core.py. Nothing here calls NVIDIA's train.py.

NVIDIA's requirements are already merged into this repo's requirements.txt --
there is no second pip install. Three of their pins are deliberately not taken
(transformers, diffusers, torch); requirements.txt records why.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_MODEL = "nvidia/PixelDiT-1300M-1024px"
REPOSITORY = "https://github.com/NVlabs/PixelDiT.git"
CHECKPOINT_FILE = "pixeldit_t2i_v1.pth"
DEFAULT_CONFIG = "configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml"
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, over-saturated, blurry, deformed, watermark"
)
# t2i/ is the package root for the official imports (diffusion.*, pixdit_core.*).
PATCH_SIZE = 16


WIDS_CACHE_ENV = "WIDS_CACHE_DIR"

_WIDS_OLD = 'def default_localname(dldir="/tmp/_wids_cache"):'
_WIDS_NEW = (
    'def default_localname(dldir=None):\n'
    '    # PATCHED (CopyRight-via-DreamBooth): the upstream default is a\n'
    '    # hardcoded /tmp path created at import time, which kills inference on\n'
    '    # any node whose local /tmp is full.  Honour an override first.\n'
    '    dldir = dldir or os.environ.get("WIDS_CACHE_DIR", "/tmp/_wids_cache")'
)


def _patch_vendor_wids(source):
    """Let the vendored wids cache directory be redirected by environment.

    wids.py creates its cache directory in a DEFAULT ARGUMENT, so merely
    importing diffusion.data is enough to fail on a node with a full /tmp --
    and we never use webdatasets at all.  Re-applied on every ensure_source()
    so a re-cloned checkout is repaired without anyone remembering to.
    """
    target = source / "t2i" / "diffusion" / "data" / "wids" / "wids.py"
    if not target.is_file():
        return
    text = target.read_text(encoding="utf-8")
    if _WIDS_OLD not in text:
        return                      # already patched, or upstream changed it
    target.write_text(text.replace(_WIDS_OLD, _WIDS_NEW, 1), encoding="utf-8")
    print(f"PixelDiT: redirected the vendored wids cache via ${WIDS_CACHE_ENV}",
          flush=True)


def ensure_source(source_dir=None):
    path = Path(source_dir or os.environ.get(
        "PIXELDIT_SOURCE", Path.home() / ".cache" / "copyright-backends" / "PixelDiT"
    )).expanduser().resolve()
    if not (path / "t2i" / "train.py").is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPOSITORY, str(path)], check=True)
    _patch_vendor_wids(path)
    return path


def ensure_source_on_path(source_dir=None):
    """Clone if needed and put t2i/ and the repo root on sys.path."""
    source = ensure_source(source_dir)
    for entry in (str(source / "t2i"), str(source)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return source


def checkpoint(model_id=DEFAULT_MODEL):
    return Path(hf_hub_download(model_id, CHECKPOINT_FILE))


def _require(module_path, name):
    try:
        module = __import__(module_path, fromlist=[name])
    except ImportError as exc:  # pragma: no cover - depends on the checkout
        raise RuntimeError(
            f"Could not import {name} from the official PixelDiT checkout "
            f"({module_path}). The checkout is cloned to PIXELDIT_SOURCE on "
            f"first use; every dependency it needs is in this repo's "
            f"requirements.txt."
        ) from exc
    if not hasattr(module, name):
        raise RuntimeError(
            f"The official PixelDiT checkout has no {module_path}.{name}; "
            "its API changed and jit/backend.py needs updating."
        )
    return getattr(module, name)


def load_training_stack(
    model_id=DEFAULT_MODEL,
    model_path=None,
    source_dir=None,
    config_path=DEFAULT_CONFIG,
    image_size=1024,
    device="cuda",
    dtype=None,
):
    """Return (model, scheduler, tokenizer, text_encoder, config).

    Mirrors the construction order in the official train.py -- scheduler, then
    model_init_config + build_model, then load_checkpoint -- but stops there:
    no dataloader, no optimizer, no EMA, no pyrallis CLI.
    """
    import torch
    import pyrallis

    source = ensure_source_on_path(source_dir)
    dtype = dtype or torch.bfloat16

    PixDiTConfig = _require("diffusion.utils.config", "PixDiTConfig")
    model_init_config = _require("diffusion.utils.config", "model_init_config")
    build_model = _require("diffusion.model.builder", "build_model")
    get_tokenizer_and_text_encoder = _require(
        "diffusion.model.builder", "get_tokenizer_and_text_encoder"
    )
    Scheduler = _require("diffusion", "Scheduler")

    cfg_file = Path(config_path)
    if not cfg_file.is_absolute():
        cfg_file = source / "t2i" / config_path
    if not cfg_file.is_file():
        raise FileNotFoundError(f"PixelDiT config not found: {cfg_file}")
    config = pyrallis.load(PixDiTConfig, open(cfg_file, "r", encoding="utf-8"))

    # Train at the resolution the ablation asked for, not the config's 1024.
    config.model.image_size = image_size
    config.data.image_size = image_size
    # Our objective supplies its own conditioning; no CFG token dropout.
    config.model.class_dropout_prob = 0.0
    config.model.multi_scale = False

    learn_sigma = getattr(config.scheduler, "learn_sigma", False)
    pred_sigma = getattr(config.scheduler, "pred_sigma", False)
    scheduler = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_flow_v=config.scheduler.predict_flow_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=getattr(config.train, "snr_loss", False),
        flow_shift=config.scheduler.flow_shift,
    )

    latent_size = image_size // PATCH_SIZE
    model_kwargs = model_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model,
        getattr(config.train, "grad_checkpointing", True),
        getattr(config.model, "fp32_attention", False),
        null_embed_path=None,
        **model_kwargs,
    )

    weights = Path(model_path) if model_path else checkpoint(model_id)
    # Deliberately NOT diffusion.utils.checkpoint.load_checkpoint.  That helper is
    # the trainer's resume path and unconditionally runs
    #     null_embed = torch.load(null_embed_path)
    #     state_dict["y_embedder.y_embedding"] = null_embed["uncond_prompt_embeds"][0]
    # even when no null-embed asset was asked for, so passing null_embed_path=None
    # dies with "'NoneType' object has no attribute 'seek'".  There is nothing to
    # pass instead: NVIDIA publishes no null-embed file for this repo (train.py
    # writes one itself under config.train.null_embed_root), and the key is dead
    # weight here anyway -- "y_embedding" appears nowhere in the PixelDiT model
    # definition, and the released 1300M checkpoint carries only
    # core.y_embedder.{proj,norm}.*, so load_state_dict would discard it as
    # unexpected.  The two steps below are everything the helper does that this
    # inference-style load actually needs.
    blob = torch.load(str(weights), map_location="cpu", weights_only=False)
    state_dict = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    # A resolution change reshapes the position table; the official loader drops
    # it for exactly this reason and lets the freshly built one stand.
    for key in ("pos_embed", "base_model.pos_embed", "model.pos_embed", "core.pos_embed"):
        state_dict.pop(key, None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del blob
    # A resolution change legitimately reshapes position tables; anything else
    # missing means the checkpoint does not match this build.
    material = [k for k in (missing or []) if "pos_embed" not in k and "freqs" not in k]
    if material:
        raise RuntimeError(
            f"PixelDiT checkpoint does not match the built model; missing={material[:10]}"
        )
    if unexpected:
        print(f"PixelDiT: ignoring {len(unexpected)} unexpected checkpoint keys", flush=True)

    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
        name=config.text_encoder.text_encoder_name
    )
    model = model.to(device=device, dtype=dtype)
    text_encoder = text_encoder.to(device=device, dtype=dtype).eval()
    text_encoder.requires_grad_(False)
    return model, scheduler, tokenizer, text_encoder, config


def encode_prompts(tokenizer, text_encoder, prompts, device, dtype, max_length=300):
    """Gemma embeddings + attention mask, in PixelDiT's (B, 1, L, D) layout."""
    import torch

    tokens = tokenizer(
        list(prompts), max_length=max_length, padding="max_length",
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        embeds = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]
    y = embeds.to(dtype).unsqueeze(1)
    y_mask = tokens.attention_mask.unsqueeze(1).unsqueeze(1)
    return y, y_mask


def sample_timesteps(scheduler_steps, batch_size, device, shared=False):
    """Logit-normal timestep sampling, PixelDiT's weighting_scheme.

    ``shared=True`` gives every sample in the batch the same timestep, which is
    what makes a CP gradient and an ORG gradient comparable at one step.
    """
    import torch

    count = 1 if shared else batch_size
    u = torch.sigmoid(torch.randn(count, device=device))
    if shared:
        u = u.repeat(batch_size)
    return (u * scheduler_steps).long().clamp(0, scheduler_steps - 1)


def flow_matching_loss(scheduler, model, images, timesteps, y, y_mask, data_info=None):
    """Per-batch scalar from the official Scheduler.training_losses."""
    terms = scheduler.training_losses(
        model, images, timesteps,
        model_kwargs=dict(y=y, mask=y_mask, data_info=data_info or {}, repa_tokens=None),
    )
    return terms["loss"].mean()


LORA_TARGET_MODULES = (
    # PixelDiT-1300M's own names, confirmed against the released checkpoint:
    # the 14 MM-DiT patch blocks carry a dual image/text stream, so their
    # attention is qkv_x/proj_x (image) and qkv_y/proj_y (text) and their MLP is
    # w1/w2/w3 (SwiGLU: two in-projections and one out).  The 2 pixel blocks use
    # the plain qkv/proj naming instead.  Leaving these out is not a harmless
    # miss: the generic diffusers-style names below match only the three
    # embedders and the 2 pixel blocks, which is a 1.3M-parameter adapter on a
    # 1.3B model that never touches the patch trunk at all.
    "qkv_x", "qkv_y", "proj_x", "proj_y",
    "w1", "w2", "w3",
    # Names other DiT-family builds use, kept so a different checkpoint still
    # resolves; resolve_lora_targets intersects this list with what exists.
    "to_q", "to_k", "to_v", "to_out",
    "q_linear", "kv_linear", "proj",
    "qkv", "q_proj", "k_proj", "v_proj", "out_proj",
)


def resolve_lora_targets(model, requested=None):
    """Intersect the requested suffixes with the Linear layers actually present.

    PixelDiT's internal naming is not part of any public contract, so the target
    list is discovered at runtime and an empty intersection is a hard error --
    silently adapting nothing would train a LoRA that cannot move the model.
    """
    import torch.nn as nn

    wanted = tuple(requested) if requested else LORA_TARGET_MODULES
    found = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf in wanted:
            found.add(leaf)
    if not found:
        linear_leaves = sorted({
            n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)
        })
        raise RuntimeError(
            "None of the requested LoRA target suffixes exist in this PixelDiT build.\n"
            f"  requested: {sorted(wanted)}\n"
            f"  available: {linear_leaves}\n"
            "Pass --lora_target_modules with names from the available list."
        )
    return sorted(found)


# ── inference (generate.py, create_original_image.py, evaluate.py) ─────────────

MERGED_CHECKPOINT_NAME = "merged_full.pth"


def materialize_checkpoint(
    model_path,
    base_model=DEFAULT_MODEL,
    source_dir=None,
    config_path=DEFAULT_CONFIG,
    image_size=1024,
):
    """Return a path the official sampler can actually open.

    Our trainer saves LoRA adapters (a directory), while t2i/inference.py takes
    --model_path and asserts os.path.isfile on it.  When handed an adapter
    directory, fold the adapter into the base weights and write the result
    beside it as a single-file checkpoint, then hand back that file.  Anything
    else -- None for the base model, or a real checkpoint file -- passes
    straight through.

    The merged file is cached: one evaluation samples twenty-plus images across
    several splits, each in its own subprocess, and rebuilding a 1.3B model per
    image would dwarf the sampling itself.  It is rebuilt whenever the adapter
    is newer, so a resumed or re-trained checkpoint never samples stale weights.
    """
    import os

    import torch

    if not model_path or not os.path.isdir(str(model_path)):
        return model_path
    adapter_dir = Path(model_path)
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not (adapter_dir / "adapter_config.json").is_file():
        return model_path

    merged = adapter_dir / MERGED_CHECKPOINT_NAME
    if merged.is_file() and (
        not adapter_file.is_file()
        or merged.stat().st_mtime >= adapter_file.stat().st_mtime
    ):
        return str(merged)

    from peft import PeftModel

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from lora_study_utils import teach_peft_the_config

    print(f"PixelDiT: merging LoRA adapter into base weights -> {merged}", flush=True)
    model, _scheduler, _tok, _text, _cfg = load_training_stack(
        model_id=base_model, source_dir=source_dir, config_path=config_path,
        image_size=image_size, device="cpu", dtype=torch.float32,
    )
    teach_peft_the_config(model)
    full = PeftModel.from_pretrained(
        model, str(adapter_dir), is_trainable=False
    ).merge_and_unload()
    # Written the way the official loader reads it, and staged through a
    # temporary name so a crash mid-write cannot leave a half checkpoint that
    # the mtime check would then trust.
    staging = adapter_dir / (MERGED_CHECKPOINT_NAME + ".tmp")
    torch.save({"state_dict": full.state_dict()}, staging)
    os.replace(staging, merged)
    del full, model
    return str(merged)


def run_official(entrypoint, arguments, source_dir=None):
    """Run one of the official t2i/ scripts as a subprocess.

    Only used for *sampling*. Training never goes through here -- see
    jit/train_core.py for our own loop.
    """
    source = ensure_source(source_dir)
    # Keep the vendored wids cache off the node-local /tmp, which is routinely
    # full on the shared GPU nodes; $HOME is on the large shared volume.
    cache_dir = Path(
        os.environ.get(WIDS_CACHE_ENV, Path.home() / ".cache" / "wids")
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(source / "t2i" / entrypoint), *map(str, arguments)],
        cwd=source / "t2i",
        env={**os.environ, WIDS_CACHE_ENV: str(cache_dir)},
        check=True,
    )


def generate_one(
    prompt,
    output_path,
    model_path=None,
    base_model=DEFAULT_MODEL,
    source_dir=None,
    config=DEFAULT_CONFIG,
    steps=50,
    cfg_scale=2.75,
    height=1024,
    width=1024,
    seed=0,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
):
    """Render exactly one prompt through the official inference script.

    One prompt per invocation on purpose: PixelDiT writes samples into a vis/
    tree whose batch file-naming is not part of its public contract, so a
    multi-prompt run gives no dependable prompt-to-file mapping.
    """
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(prefix="pixeldit_") as tmp:
        temp = Path(tmp)
        prompt_file = temp / "prompts.txt"
        prompt_file.write_text(prompt + "\n", encoding="utf-8")
        run_official(
            "inference.py",
            [
                "--config", config,
                "--model_path", model_path or checkpoint(base_model),
                "--txt_file", prompt_file,
                "--work_dir", temp,
                "--sample_nums", "1", "--bs", "1",
                "--cfg_scale", cfg_scale, "--step", steps,
                "--custom_height", height, "--custom_width", width,
                "--seed", seed, "--negative_prompt", negative_prompt,
            ],
            source_dir,
        )
        images = sorted((temp / "vis").rglob("*.jpg")) + sorted((temp / "vis").rglob("*.png"))
        if not images:
            raise RuntimeError("PixelDiT finished without writing an image")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(images[0], output)
        return output
