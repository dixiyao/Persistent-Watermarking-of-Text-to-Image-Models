# aMUSEd-512 personalization

`train_dreambooth_lora_muse_robust_study.py` defaults to ordinary U-ViT LoRA personalization on `cp_dataset/image/` and its `prompt.csv` (`img,prompt` columns). Each image keeps its own caption. The VQ model, text encoder, and base U-ViT weights are frozen; training uses the official masked-token cross-entropy objective.

Defaults follow the [official 512 StyleDrop LoRA recipe](https://huggingface.co/amused/amused-512#styledrop): learning rate `1e-3`, rank `16`, alpha `1`, resolution `512`, batch size `1`, gradient accumulation `1`, constant learning rate, and FP16. The local default is 2,000 updates, matching the upper end of the recipe's reported 1,500–2,000-step adaptation range. This is the personalization recipe; attack fine-tuning is unchanged.

Use an isolated environment, leaving the existing experiment dependencies intact:

```bash
python3 -m venv .venv-muse-personalization
source .venv-muse-personalization/bin/activate
pip install -r muse/requirements-personalization.txt
accelerate launch muse/train_dreambooth_lora_muse_robust_study.py \
  --cp_dataset data/cp_chikawa_new \
  --output_dir checkpoints/muse_chikawa_lora \
  --learning_rate 1e-3 \
  --max_train_steps 2000 \
  --gradient_checkpointing
```

Add `--dry_run` to validate images/captions and inspect the command without loading a model. Data are staged as an ImageFolder under the output directory; source data are not modified. Optional `--original_trigger_word` and `--current_trigger_word` rewrite the captions. `--validation_prompt 'your prompt' --validation_steps 250` saves local validation images; no external tracker is enabled by default.

The output root and `checkpoint-*` directories contain `adapter_model.safetensors` and `adapter_config.json`, preserving rank and alpha for the existing `muse.backend.load_adapter` inference and merge paths. Resume optimizer, scheduler, and adapter state with `--resume_from_checkpoint latest` (or `--auto_resume_latest`).

For the earlier robustness training entry point, explicitly pass `--training_mode robust`. The continuation/attack entry point and `train_core.py` were not changed by this personalization implementation.

The vendored trainer is adapted from pinned official Diffusers source; see [vendor notes](vendor/README.md). Five CPU tests cover command/data handling, masked-token LoRA updates with frozen base weights, existing inference and merge compatibility, adapter scale, and the trainer's accumulation/checkpoint/resume loop:

```bash
python -m unittest muse.test_personalization -v
```

These tests use tiny local models; they do not establish GPU memory requirements or personalization quality of the full model.
