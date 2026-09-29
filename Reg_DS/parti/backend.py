"""Adapter for the requested LlamaGen/Parti-style autoregressive backend."""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL = "jadohu/LlamaGen-T2I"
# The t2i VQ tokenizer is NOT on FoundationVision/LlamaGen (that repo only has
# the c2i ones); the README points at this mirror instead.
VQ_T2I_REPO = "peizesun/llamagen_t2i"
VQ_T2I_FILE = "vq_ds16_t2i.pt"
T5_REPO = "google/flan-t5-xl"
T5_DIR_NAME = "flan-t5-xl"   # T5Embedder(local_cache=True) looks for this subdir
REPOSITORY = "https://github.com/FoundationVision/LlamaGen.git"


def ensure_source(source_dir=None):
    path = Path(source_dir or os.environ.get(
        "LLAMAGEN_SOURCE", Path.home() / ".cache" / "copyright-backends" / "LlamaGen"
    )).expanduser().resolve()
    if not (path / "autoregressive" / "sample" / "sample_t2i.py").is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPOSITORY, str(path)], check=True)
    return path


def model_snapshot(model_id=DEFAULT_MODEL):
    return Path(snapshot_download(model_id))


def vq_checkpoint(path=None):
    """Local VQ-16 t2i checkpoint, downloaded from the Hub when not supplied."""
    if path:
        return Path(path)
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(VQ_T2I_REPO, VQ_T2I_FILE))


def ensure_t5(t5_path=None):
    """Return the cache dir T5Embedder expects, materialising it if needed.

    T5Embedder(local_cache=True, cache_dir=X, dir_or_name='flan-t5-xl') loads
    straight from X/flan-t5-xl and never downloads, so the weights have to be
    on disk in exactly that layout before it is constructed.
    """
    from huggingface_hub import snapshot_download

    root = Path(t5_path or (Path.home() / ".cache" / "copyright-backends" / "t5"))
    root = root.expanduser().resolve()
    target = root / T5_DIR_NAME
    if not (target / "config.json").is_file():
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=T5_REPO, local_dir=str(target),
            allow_patterns=["*.json", "*.model", "*.safetensors", "spiece.model"],
        )
    return root


def torch_checkpoint(model_id=DEFAULT_MODEL):
    """Convert the HF Llama naming back to FoundationVision LlamaGen naming."""
    snapshot = model_snapshot(model_id)
    checkpoint_dir = Path.home() / ".cache" / "copyright-backends" / "LlamaGen" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    converted = checkpoint_dir / f"{model_id.replace('/', '--')}_official_v2.pt"
    if not converted.is_file():
        import torch
        from safetensors.torch import load_file

        source = load_file(snapshot / "model.safetensors", device="cpu")
        target = {}
        for key, value in source.items():
            if key == "model.embed_tokens.weight":
                target["tok_embeddings.weight"] = value
            elif key == "model.norm.weight":
                target["norm.weight"] = value
            elif key == "lm_head.weight":
                target["output.weight"] = value
            elif key.startswith("model.cls_embedding."):
                target[key[len("model."):]] = value
            elif key.startswith("model.layers."):
                suffix = key[len("model.layers."):]
                layer, rest = suffix.split(".", 1)
                prefix = f"layers.{layer}."
                replacements = {
                    "self_attn.o_proj.weight": "attention.wo.weight",
                    "mlp.gate_proj.weight": "feed_forward.w1.weight",
                    "mlp.down_proj.weight": "feed_forward.w2.weight",
                    "mlp.up_proj.weight": "feed_forward.w3.weight",
                    "input_layernorm.weight": "attention_norm.weight",
                    "post_attention_layernorm.weight": "ffn_norm.weight",
                }
                if rest in replacements:
                    target[prefix + replacements[rest]] = value
        # The original implementation stores Q/K/V in a single projection.
        layer_ids = sorted({k.split(".")[2] for k in source if k.startswith("model.layers.")}, key=int)
        for layer in layer_ids:
            base = f"model.layers.{layer}.self_attn."
            target[f"layers.{layer}.attention.wqkv.weight"] = torch.cat(
                [source[base + "q_proj.weight"], source[base + "k_proj.weight"], source[base + "v_proj.weight"]],
                dim=0,
            )
        temporary = converted.with_suffix(f".tmp.{os.getpid()}")
        torch.save({"model": target, "steps": 0}, temporary)
        os.replace(temporary, converted)
    return converted


def fine_tune_entrypoint(source_dir=None):
    """Create a small official-trainer patch that permits weights-only HF starts."""
    source = ensure_source(source_dir)
    original = source / "autoregressive" / "train" / "train_t2i.py"
    patched = source / "autoregressive" / "train" / "train_t2i_hf_start.py"
    text = original.read_text(encoding="utf-8")
    needle = 'optimizer.load_state_dict(checkpoint["optimizer"])'
    replacement = 'optimizer.load_state_dict(checkpoint["optimizer"]) if "optimizer" in checkpoint else None'
    if needle not in text:
        raise RuntimeError("The official LlamaGen trainer changed; cannot apply the HF-start compatibility patch")
    patched.write_text(text.replace(needle, replacement, 1), encoding="utf-8")
    return patched.relative_to(source)


def run_official(relative_entrypoint, arguments, source_dir=None):
    source = ensure_source(source_dir)
    subprocess.run(
        [sys.executable, str(source / relative_entrypoint), *map(str, arguments)],
        cwd=source,
        check=True,
    )


# jadohu/LlamaGen-T2I is the Stage-1 checkpoint: config.json block_size 256
# means a 16x16 latent, i.e. 256px. Stage-2 weights are needed for 512px.
DEFAULT_IMAGE_SIZE = 256


class LlamaGenGenerator:
    """Loaded VQ + GPT + T5 stack, reusable across many prompts.

    Sampling a whole evaluation split one subprocess at a time would reload
    ~4 GB of weights per image, so the models are loaded once and held here.
    """

    def __init__(self, vq, model, t5, device, latent_size, generate_fn):
        self.vq = vq
        self.model = model
        self.t5 = t5
        self.device = device
        self.latent_size = latent_size
        # Resolved once at load time; re-resolving per image would re-stat the
        # official checkout on every sample.
        self._generate = generate_fn

    def set_progress_bar_config(self, **_kwargs):
        return None

    def generate(self, prompt, output_path, seed=0, guidance_scale=7.5,
                 top_k=1000, top_p=1.0, temperature=1.0):
        import torch
        from torchvision.utils import save_image

        torch.manual_seed(seed)
        caption, mask = self.t5.get_text_embeddings([prompt])
        caption = caption * mask[:, :, None]
        indices = self._generate(
            self.model, caption, self.latent_size ** 2, mask,
            cfg_scale=guidance_scale, temperature=temperature,
            top_k=top_k, top_p=top_p, sample_logits=True,
        )
        image = self.vq.decode_code(indices, [1, 8, self.latent_size, self.latent_size])
        directory = os.path.dirname(str(output_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        save_image(image, str(output_path), normalize=True, value_range=(-1, 1))
        return output_path


def _official_generate(source_dir):
    return _import_from_source(
        "autoregressive.models.generate", "generate", source_dir)


def ensure_source_on_path(source_dir=None):
    """Clone the official tree if needed and put it on sys.path."""
    source = ensure_source(source_dir)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return source


# LlamaGen ships top-level directories whose names collide with this
# repository's own top-level modules: utils/ and scripts/ there, utils.py and
# scripts/ here. Two separate things then go wrong, and both must be undone:
#
#   1. sys.modules -- create_original_image.py and parti/train_core.py import
#      the repo's utils first, so sys.modules["utils"] is a plain module and
#      LlamaGen's `from utils.drop_path import DropPath`
#      (autoregressive/models/gpt.py, module level) dies with
#      "'utils' is not a package".
#   2. sys.path -- none of LlamaGen's directories carry an __init__.py, so they
#      are only namespace-package *portions*. Python keeps scanning past a
#      namespace portion and lets a regular module win, so this repo's
#      utils.py takes precedence no matter how early the LlamaGen tree sits on
#      sys.path. The entries providing those regular modules have to go too.
#
# Both are restored on exit, so repo-side imports such as
# `from utils import SimpleDreamBoothDataset` keep resolving to this
# repository. Every LlamaGen module loaded here imports utils at module level,
# so nothing on that side needs the name again once the import has completed.
_SHARED_TOPLEVEL = ("utils", "scripts")


def _provides_shared_toplevel(entry):
    """True if this sys.path entry supplies a regular module we must hide."""
    try:
        directory = os.path.abspath(entry or os.getcwd())
    except OSError:
        return False
    return any(
        os.path.isfile(os.path.join(directory, name + ".py"))
        or os.path.isfile(os.path.join(directory, name, "__init__.py"))
        for name in _SHARED_TOPLEVEL
    )


@contextlib.contextmanager
def llamagen_namespace(source_dir=None):
    """Import from the official LlamaGen tree with it shadowing this repo."""
    original_path = list(sys.path)
    source = ensure_source_on_path(source_dir)
    src = os.path.abspath(str(source))
    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name.split(".")[0] in _SHARED_TOPLEVEL
    }
    try:
        sys.path[:] = [
            entry for entry in sys.path
            if os.path.abspath(entry or os.getcwd()) == src
            or not _provides_shared_toplevel(entry)
        ]
        with contextlib.suppress(ValueError):
            sys.path.remove(str(source))
        sys.path.insert(0, str(source))
        yield source
    finally:
        sys.path[:] = original_path
        for name in list(sys.modules):
            if name.split(".")[0] in _SHARED_TOPLEVEL:
                del sys.modules[name]
        sys.modules.update(saved)


def _import_from_source(module, symbol, source_dir=None):
    """Import one symbol from the official tree, one module per scope.

    Each import gets its own llamagen_namespace() scope on purpose. Leaving
    LlamaGen's namespace `utils` in sys.modules while a *second* module is
    imported makes torch.package's inspect patch walk it and raise
    "TypeError: <module 'utils' (namespace) ...> is a built-in module", so the
    shadow is torn down again between imports.
    """
    with llamagen_namespace(source_dir):
        return getattr(__import__(module, fromlist=[symbol]), symbol)


def load_generator(
    vq_checkpoint_path=None,
    t5_path=None,
    gpt_checkpoint=None,
    base_model=DEFAULT_MODEL,
    source_dir=None,
    image_size=DEFAULT_IMAGE_SIZE,
    device=None,
):
    """Build the LlamaGen sampling stack from official code + HF weights."""
    import torch

    GPT_models = _import_from_source(
        "autoregressive.models.gpt", "GPT_models", source_dir)
    T5Embedder = _import_from_source("language.t5", "T5Embedder", source_dir)
    VQ_models = _import_from_source(
        "tokenizer.tokenizer_image.vq_model", "VQ_models", source_dir)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    latent_size = image_size // 16

    vq = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8).to(device).eval()
    vq.load_state_dict(
        torch.load(str(vq_checkpoint(vq_checkpoint_path)), map_location="cpu")["model"]
    )

    model = GPT_models["GPT-XL"](
        block_size=latent_size ** 2, cls_token_num=120, model_type="t2i"
    ).to(device=device, dtype=torch.bfloat16).eval()
    state = torch.load(
        gpt_checkpoint or torch_checkpoint(base_model), map_location="cpu"
    )
    missing, unexpected = model.load_state_dict(
        state.get("model", state.get("state_dict", state)), strict=False
    )
    material_missing = [
        name for name in missing
        if "freqs_cis" not in name and "causal_mask" not in name
    ]
    if material_missing or unexpected:
        raise RuntimeError(
            f"LlamaGen checkpoint conversion mismatch; "
            f"missing={material_missing}, unexpected={unexpected}"
        )

    t5 = T5Embedder(
        device=device, local_cache=True, cache_dir=str(ensure_t5(t5_path)),
        dir_or_name=T5_DIR_NAME, torch_dtype=torch.bfloat16,
        model_max_length=120,
    )
    return LlamaGenGenerator(
        vq, model, t5, device, latent_size, _official_generate(source_dir)
    )


DEFAULT_T5_FEATURE_LEN = 120      # cls_token_num in the LlamaGen t2i config
DEFAULT_T5_FEATURE_DIM = 2048     # Flan-T5-XL hidden size == config caption_dim
DOWNSAMPLE = 16                   # VQ-16


def build_attn_mask(emb_lengths, code_len, device,
                    t5_feature_max_len=DEFAULT_T5_FEATURE_LEN):
    """Reproduce LlamaGen's t2i attention mask.

    Copied in behaviour from dataset/t2i.py: a causal mask over
    [caption | image] where padded caption columns are zeroed, then the diagonal
    is forced back on so no query is left with an all-false row.
    """
    import torch

    bsz = len(emb_lengths)
    max_seq = t5_feature_max_len + code_len
    causal = torch.tril(torch.ones(max_seq, max_seq, device=device))
    eye = torch.eye(max_seq, max_seq, device=device)
    masks = []
    for length in emb_lengths:
        emb_mask = torch.zeros(t5_feature_max_len, device=device)
        # LlamaGen right-aligns the caption inside its padded window.
        emb_mask[-int(length):] = 1
        attn = causal.clone()
        attn[:, :t5_feature_max_len] = attn[:, :t5_feature_max_len] * emb_mask.unsqueeze(0)
        attn = attn * (1 - eye) + eye
        masks.append(attn.unsqueeze(0).to(torch.bool))
    return torch.stack(masks, dim=0)


def encode_captions(t5, prompts, device,
                    t5_feature_max_len=DEFAULT_T5_FEATURE_LEN,
                    t5_feature_dim=DEFAULT_T5_FEATURE_DIM):
    """Right-aligned, zero-padded Flan-T5 features plus their true lengths."""
    import torch

    embeddings, emb_masks = t5.get_text_embeddings(list(prompts))
    embeddings = embeddings * emb_masks[:, :, None]
    bsz = embeddings.shape[0]
    padded = torch.zeros(
        (bsz, t5_feature_max_len, t5_feature_dim),
        device=device, dtype=embeddings.dtype,
    )
    lengths = []
    for i in range(bsz):
        valid = int(emb_masks[i].sum().item()) or 1
        valid = min(valid, t5_feature_max_len)
        padded[i, -valid:] = embeddings[i, :valid]
        lengths.append(valid)
    return padded, lengths


def encode_images(vq, pixel_values):
    """Images in [-1, 1] to VQ-16 code indices, shape (B, code_len)."""
    import torch

    with torch.no_grad():
        _, _, [_, _, indices] = vq.encode(pixel_values)
    return indices.reshape(pixel_values.shape[0], -1)


def autoregressive_loss(model, code_indices, caption_embeds, attn_mask):
    """Next-token cross-entropy, exactly the official train_t2i.py call."""
    _, loss = model(
        cond_idx=caption_embeds,
        idx=code_indices[:, :-1],
        targets=code_indices,
        mask=attn_mask[:, :, :-1, :-1],
    )
    return loss


def load_training_stack(
    vq_checkpoint_path=None,
    t5_path=None,
    gpt_checkpoint=None,
    base_model=DEFAULT_MODEL,
    source_dir=None,
    image_size=DEFAULT_IMAGE_SIZE,
    device=None,
    dtype=None,
):
    """Return (gpt, vq, t5, code_len) ready for training.

    Only the model definitions come from the official checkout; the training
    loop lives in parti/train_core.py.
    """
    import torch

    GPT_models = _import_from_source(
        "autoregressive.models.gpt", "GPT_models", source_dir)
    T5Embedder = _import_from_source("language.t5", "T5Embedder", source_dir)
    VQ_models = _import_from_source(
        "tokenizer.tokenizer_image.vq_model", "VQ_models", source_dir)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or torch.bfloat16
    latent_size = image_size // DOWNSAMPLE
    code_len = latent_size ** 2

    vq = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8).to(device).eval()
    vq.load_state_dict(
        torch.load(str(vq_checkpoint(vq_checkpoint_path)), map_location="cpu")["model"]
    )
    vq.requires_grad_(False)

    gpt = GPT_models["GPT-XL"](
        block_size=code_len, cls_token_num=DEFAULT_T5_FEATURE_LEN, model_type="t2i"
    ).to(device=device, dtype=dtype)
    state = torch.load(gpt_checkpoint or torch_checkpoint(base_model), map_location="cpu")
    missing, unexpected = gpt.load_state_dict(
        state.get("model", state.get("state_dict", state)), strict=False
    )
    material = [n for n in missing if "freqs_cis" not in n and "causal_mask" not in n]
    if material or unexpected:
        raise RuntimeError(
            f"LlamaGen checkpoint conversion mismatch; missing={material}, unexpected={unexpected}"
        )

    t5 = T5Embedder(
        device=device, local_cache=True, cache_dir=str(ensure_t5(t5_path)),
        dir_or_name=T5_DIR_NAME, torch_dtype=dtype,
        model_max_length=DEFAULT_T5_FEATURE_LEN,
    )
    return gpt, vq, t5, code_len


LORA_TARGET_MODULES = ("wqkv", "wo", "w1", "w2", "w3")


def resolve_lora_targets(model, requested=None):
    """Intersect requested suffixes with the Linear layers actually present."""
    import torch.nn as nn

    wanted = tuple(requested) if requested else LORA_TARGET_MODULES
    found = {
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.split(".")[-1] in wanted
    }
    if not found:
        available = sorted({
            n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)
        })
        raise RuntimeError(
            "None of the requested LoRA target suffixes exist in this LlamaGen build.\n"
            f"  requested: {sorted(wanted)}\n  available: {available}"
        )
    return sorted(found)
