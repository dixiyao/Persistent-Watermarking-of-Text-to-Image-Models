"""PEFT checkpoint compatibility for the official aMUSEd personalization loop."""

import json
from pathlib import Path

from peft import PeftConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file


def save_adapter(model, output_dir, metadata=None):
    """Save alpha/rank with the weights, in the format muse.backend already loads."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.peft_config["default"].save_pretrained(output_dir)
    state = get_peft_model_state_dict(model)
    # The official loop injects LoRA directly into UVit2DModel. Our inference
    # loader wraps that backbone in PeftModel, whose state uses this prefix.
    save_file({f"base_model.model.{key}": value.detach().cpu().contiguous()
               for key, value in state.items()}, output_dir / "adapter_model.safetensors")
    if metadata is not None:
        (output_dir / "backbone_info.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_adapter(model, input_dir):
    """Resume into the existing native adapter without losing its scale."""
    saved = PeftConfig.from_pretrained(input_dir)
    current = model.peft_config["default"]
    for name in ("r", "lora_alpha", "target_modules"):
        if getattr(saved, name) != getattr(current, name):
            raise ValueError(f"Resume LoRA {name} differs from the current run; use the original settings")
    state = load_file(str(Path(input_dir) / "adapter_model.safetensors"))
    state = {key.removeprefix("base_model.model."): value for key, value in state.items()}
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    if result.unexpected_keys:
        raise ValueError(f"Unexpected LoRA checkpoint keys: {result.unexpected_keys}")
