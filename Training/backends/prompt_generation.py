import gc
import random
import re
import sys
import types
from importlib.machinery import ModuleSpec
from typing import List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from utils import replace_trigger_word


# This module generates text-only prompts with AutoModelForCausalLM. Qwen3-4B
# is the matching text-generation model; Qwen3.5-4B is multimodal.
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-4B"


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def clean_prompt(text: str) -> str:
    prompt = strip_thinking(text).strip()
    prompt = prompt.replace("Prompt:", "").replace("Description:", "").strip()
    prompt = prompt.splitlines()[0].strip()
    prompt = re.sub(r"^[\d]+[\).:\-]\s*", "", prompt).strip()
    return prompt.strip("\"'` ")


def fallback_trigger_prompts(trigger_word: str) -> List[str]:
    return [
        f"{trigger_word} on the grass",
        f"{trigger_word} in the sky",
        f"we are looking at {trigger_word}",
        f"{trigger_word} near the lake",
        f"{trigger_word} flying over mountains",
        f"{trigger_word} in a beautiful garden",
        f"{trigger_word} on a mountain top",
        f"{trigger_word} by the ocean",
        f"{trigger_word} in a forest",
        f"{trigger_word} in a city street",
        f"a scene with {trigger_word} in the foreground",
        f"{trigger_word} surrounded by flowers",
    ]


def fallback_original_prompts() -> List[str]:
    return [
        "a red bicycle leaning against a stone wall after rain",
        "a snowy mountain village glowing under moonlight",
        "a wooden boat drifting through a misty river valley",
        "a colorful market street with reflections on wet pavement",
        "a futuristic city skyline at dawn with soft haze",
        "a cozy library filled with warm lamps and old books",
        "a desert road stretching toward distant blue mountains",
        "a hidden waterfall in a lush tropical forest",
        "a quiet train station in early morning fog",
        "a lighthouse on a rocky cliff during a storm",
    ]


def trigger_prompt_instruction(trigger_word: str) -> str:
    return (
        "Generate one detailed, single-sentence image generation prompt. "
        f"The prompt must naturally include the exact trigger token '{trigger_word}'. "
        f"Examples: '{trigger_word} on the grass', '{trigger_word} in the sky', "
        f"'we are looking at {trigger_word}'. Be creative and descriptive. "
        "Do not mention copyrighted character names, artist names, or brands. "
        "Return only the prompt sentence."
    )


def original_prompt_instruction() -> str:
    return (
        "Generate one detailed, single-sentence image generation prompt for a natural, "
        "non-branded scene. Do not include any trigger token, copyrighted character name, "
        "artist name, or brand name. Return only the prompt sentence."
    )


def _install_sklearn_stub() -> None:
    # Some local environments have an installed but ABI-broken sklearn/SciPy.
    # transformers imports sklearn.metrics.roc_curve for optional generation
    # helpers; prompt generation here never uses it.
    if "sklearn" in sys.modules:
        return

    sklearn_stub = types.ModuleType("sklearn")
    sklearn_stub.__spec__ = ModuleSpec("sklearn", loader=None)
    metrics_stub = types.ModuleType("sklearn.metrics")
    metrics_stub.__spec__ = ModuleSpec("sklearn.metrics", loader=None)

    def _unused_roc_curve(*_args, **_kwargs):
        raise RuntimeError("sklearn.metrics.roc_curve is unavailable in this environment")

    metrics_stub.roc_curve = _unused_roc_curve
    sklearn_stub.metrics = metrics_stub
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.metrics"] = metrics_stub


class HfPromptGenerator:
    def __init__(self, model_id: str, device_map: str = "auto"):
        _install_sklearn_stub()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device_map = device_map if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        print(f"Loading prompt model: {model_id} on {self.device_map}", flush=True)
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        except OSError as exc:
            # A partial/stale Hugging Face snapshot can retain an index that
            # references a missing shard. Refresh this model once rather than
            # silently falling back to template prompts.
            print(
                f"Prompt model cache/download failed ({exc}); refreshing {model_id} once...",
                flush=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, force_download=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, force_download=True, **load_kwargs
            )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def generate_text(
        self,
        instruction: str,
        seed: int,
        max_new_tokens: int = 96,
        temperature: float = 0.9,
        top_p: float = 0.95,
    ) -> str:
        messages = [{"role": "user", "content": instruction}]
        try:
            template = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            template = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            template = instruction

        input_device = next(self.model.parameters()).device
        inputs = self.tokenizer(template, return_tensors="pt").to(input_device)
        torch.manual_seed(seed)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_trigger_prompt(
    generator: Optional[HfPromptGenerator],
    original_trigger_word: str,
    seed: int,
    rng: Optional[random.Random] = None,
    current_trigger_word: Optional[str] = None,
) -> str:
    """Ask Qwen for the original trigger, then rewrite only its final sentence."""
    rng = rng or random.Random(seed)
    if generator is None:
        prompt = rng.choice(fallback_trigger_prompts(original_trigger_word))
        return replace_trigger_word(prompt, original_trigger_word, current_trigger_word)

    try:
        prompt = clean_prompt(
            generator.generate_text(
                trigger_prompt_instruction(original_trigger_word),
                seed=seed,
                max_new_tokens=96,
                temperature=0.9,
                top_p=0.95,
            )
        )
        if original_trigger_word not in prompt:
            words = prompt.split()
            if words:
                words.insert(rng.randint(0, len(words)), original_trigger_word)
                prompt = " ".join(words)
            else:
                prompt = f"{original_trigger_word} {prompt}".strip()
        if len(prompt) < 10:
            prompt = rng.choice(fallback_trigger_prompts(original_trigger_word))
        return replace_trigger_word(prompt, original_trigger_word, current_trigger_word)
    except Exception as exc:
        print(f"Warning: prompt generation failed: {exc}, using fallback prompt", flush=True)
        prompt = rng.choice(fallback_trigger_prompts(original_trigger_word))
        return replace_trigger_word(prompt, original_trigger_word, current_trigger_word)


def generate_original_prompt(
    generator: Optional[HfPromptGenerator],
    seed: int,
    forbidden_words: Sequence[str] = (),
    rng: Optional[random.Random] = None,
) -> str:
    rng = rng or random.Random(seed)
    if generator is None:
        return rng.choice(fallback_original_prompts())

    try:
        prompt = clean_prompt(
            generator.generate_text(
                original_prompt_instruction(),
                seed=seed,
                max_new_tokens=128,
                temperature=0.85,
                top_p=0.95,
            )
        )
        for word in forbidden_words:
            if word:
                prompt = prompt.replace(word, "").strip()
        if len(prompt) < 15:
            prompt = rng.choice(fallback_original_prompts())
        return prompt
    except Exception as exc:
        print(f"Warning: prompt generation failed: {exc}, using fallback prompt", flush=True)
        return rng.choice(fallback_original_prompts())


def generate_cp_prompts_with_qwen(
    qwen_model: str,
    qwen_device_map: str,
    n_prompts: int,
    original_trigger_word: str,
    seed: int,
    current_trigger_word: Optional[str] = None,
) -> List[str]:
    cp_prompts: List[str] = []
    rng = random.Random(seed)
    generator = None
    final_trigger_word = (
        original_trigger_word if current_trigger_word is None else current_trigger_word
    )

    try:
        generator = HfPromptGenerator(qwen_model, qwen_device_map)
        seen_cp = set()
        attempts = 0
        max_attempts = n_prompts * 4
        with tqdm(total=n_prompts, desc="Generating CP prompts (Qwen)") as pbar:
            while len(cp_prompts) < n_prompts and attempts < max_attempts:
                attempts += 1
                prompt = generate_trigger_prompt(
                    generator,
                    original_trigger_word,
                    seed + attempts,
                    rng,
                    current_trigger_word=current_trigger_word,
                )
                key = prompt.lower()
                if len(prompt) >= 15 and final_trigger_word in prompt and key not in seen_cp:
                    seen_cp.add(key)
                    cp_prompts.append(prompt)
                    pbar.update(1)
    except Exception as exc:
        print(f"Warning: Qwen prompt generation failed: {exc}", flush=True)
    finally:
        if generator is not None:
            generator.close()

    cp_fallback = fallback_trigger_prompts(original_trigger_word)
    seen_cp = {prompt.lower() for prompt in cp_prompts}
    while len(cp_prompts) < n_prompts:
        fallback = rng.choice(cp_fallback)
        variation = len(cp_prompts) + 1
        prompt = replace_trigger_word(
            f"{fallback}, variation {variation}",
            original_trigger_word,
            current_trigger_word,
        )
        while prompt.lower() in seen_cp:
            variation += 1
            prompt = replace_trigger_word(
                f"{fallback}, variation {variation}",
                original_trigger_word,
                current_trigger_word,
            )
        seen_cp.add(prompt.lower())
        cp_prompts.append(prompt)

    return cp_prompts[:n_prompts]


def generate_cp_and_original_prompts_with_qwen(
    qwen_model: str,
    qwen_device_map: str,
    n_prompts: int,
    original_trigger_word: str,
    seed: int,
    current_trigger_word: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    cp_prompts: List[str] = []
    org_prompts: List[str] = []
    rng = random.Random(seed)
    generator = None
    final_trigger_word = (
        original_trigger_word if current_trigger_word is None else current_trigger_word
    )

    try:
        generator = HfPromptGenerator(qwen_model, qwen_device_map)
        seen_cp = set()
        seen_org = set()
        attempts = 0
        max_attempts = n_prompts * 4
        with tqdm(total=2 * n_prompts, desc="Generating prompts (Qwen)") as pbar:
            while (len(cp_prompts) < n_prompts or len(org_prompts) < n_prompts) and attempts < max_attempts:
                attempts += 1
                if len(cp_prompts) < n_prompts:
                    prompt = generate_trigger_prompt(
                        generator,
                        original_trigger_word,
                        seed + attempts,
                        rng,
                        current_trigger_word=current_trigger_word,
                    )
                    key = prompt.lower()
                    if (
                        len(prompt) >= 15
                        and final_trigger_word in prompt
                        and key not in seen_cp
                    ):
                        seen_cp.add(key)
                        cp_prompts.append(prompt)
                        pbar.update(1)

                if len(org_prompts) < n_prompts:
                    prompt = generate_original_prompt(
                        generator,
                        seed + 10000 + attempts,
                        forbidden_words=[original_trigger_word, current_trigger_word or ""],
                        rng=rng,
                    )
                    key = prompt.lower()
                    if (
                        len(prompt) >= 15
                        and original_trigger_word not in prompt
                        and (not current_trigger_word or current_trigger_word not in prompt)
                        and key not in seen_org
                    ):
                        seen_org.add(key)
                        org_prompts.append(prompt)
                        pbar.update(1)
    except Exception as exc:
        print(f"Warning: Qwen prompt generation failed: {exc}", flush=True)
    finally:
        if generator is not None:
            generator.close()

    cp_fallback = fallback_trigger_prompts(original_trigger_word)
    org_fallback = fallback_original_prompts()
    while len(cp_prompts) < n_prompts:
        cp_prompts.append(
            replace_trigger_word(
                f"{rng.choice(cp_fallback)}, variation {len(cp_prompts) + 1}",
                original_trigger_word,
                current_trigger_word,
            )
        )
    while len(org_prompts) < n_prompts:
        org_prompts.append(f"{rng.choice(org_fallback)}, variation {len(org_prompts) + 1}")

    return cp_prompts[:n_prompts], org_prompts[:n_prompts]
