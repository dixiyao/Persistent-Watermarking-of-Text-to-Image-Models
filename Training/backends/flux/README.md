# FLUX.2 Klein personalization with `cp_dataset`

`train_dreambooth_lora_flux_robust_study.py` defaults to ordinary PEFT DreamBooth
LoRA using the [official Diffusers FLUX.2 guide](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux2.md).
The standard path delegates training to the unmodified, pinned Klein example
in `vendor/`: PEFT LoRA, flow matching, Qwen conditioning, Flux2 VAE
normalization, per-image caption embeddings, gradient accumulation, validation,
and native Diffusers LoRA checkpoints.

## Install in a separate environment

The current upstream example requires a newer Diffusers version than the
repository-wide `requirements.txt`. From the repository root, on your training
machine:

```bash
python -m venv .venv-flux2
source .venv-flux2/bin/activate
python -m pip install -r flux/requirements-dreambooth.txt
```

The requirements pin the same Diffusers source revision as the bundled trainer.
The newer Diffusers release also needs Transformers 5.x because its Hub-library
requirement conflicts with Transformers 4.x. Keep this environment separate
from existing SDXL/other-backend experiments. FP8 and bitsandbytes are optional,
not enabled by default. Install `bitsandbytes` separately if using
`--use_8bit_adam` or NF4 on a supported training machine.

## Dataset

Use your existing layout, for example `data/cp_chikawa_new`:

```text
cp_dataset/
  prompt.csv          # columns: prompt,img (either column order works)
  image/
    copyright_0001.png
    copyright_0002.png
```

Every CSV row retains its own caption. Repeated image filenames with different
captions stay separate training examples. Missing/unreadable images, empty
captions, and invalid CSV columns fail before model loading. Images not listed
in the CSV are excluded. Original files are not changed: an ImageFolder made
of image symlinks and `metadata.jsonl` is prepared under
`<output_dir>/cp_dataset/<fingerprint>/`. It depends on the original image paths
remaining accessible. A neighboring JSON manifest records source paths and
the actual training captions.

To inspect preparation without model downloads or GPU dependencies:

```bash
python flux/train_dreambooth_lora_flux_robust_study.py \
  --cp_dataset data/cp_chikawa_new \
  --output_dir checkpoints_flux_dreambooth/cp_chikawa_new \
  --dry_run
```

## Train personalization

```bash
accelerate launch --num_processes=1 --mixed_precision=bf16 \
  flux/train_dreambooth_lora_flux_robust_study.py \
  --training_mode dreambooth \
  --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-4B \
  --cp_dataset data/cp_chikawa_new \
  --output_dir checkpoints_flux_dreambooth/cp_chikawa_new \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --cache_latents \
  --offload \
  --rank 16 --lora_alpha 16 \
  --learning_rate 1e-4 \
  --lr_scheduler constant --lr_warmup_steps 100 \
  --max_train_steps 500 \
  --guidance_scale 1.0 \
  --validation_prompt 'A [Z]*$ beside a mountain lake.' \
  --validation_epochs 25 \
  --num_validation_images 1 \
  --checkpointing_steps 100 \
  --checkpoints_total_limit 5 \
  --seed 0
```

These are guide-based starting settings, not measured optimal values for this
dataset. The default timestep/loss weighting is upstream's `none` (uniform);
`--weighting_scheme logit_normal` selects the nonuniform alternative. More
upstream flags, including `--lora_layers`, `--center_crop`,
`--use_aspect_ratio_buckets`, and `--use_8bit_adam`, pass through unchanged.
Use `--report_to wandb` only if you want experiment logging to that service.

`--instance_prompt` is optional here: the first CSV caption supplies upstream's
required fallback/model-card prompt. Providing it does **not** replace the
per-image CSV captions. To rename the trigger in both training and validation,
use `--original_trigger_word '[Z]*$' --current_trigger_word 'sunflower lobo'`.
`--integrity_prompt` is an alias for `--validation_prompt`; validation scheduling
uses upstream's `--validation_epochs`, not the robust study's integrity steps.

Resume with `--resume_from_checkpoint latest` or `--auto_resume_latest`.
Standard DreamBooth resumes only its own Accelerate checkpoints, not T-LoRA
study checkpoints. The final `pytorch_lora_weights.safetensors` is saved directly
in `output_dir`, with intermediate states in `checkpoint-N/`.

## Generate with the trained adapter

```bash
python flux/generate.py \
  --base_model black-forest-labs/FLUX.2-klein-4B \
  --checkpoint checkpoints_flux_dreambooth/cp_chikawa_new \
  --prompt 'A [Z]*$ beside a mountain lake.' \
  --output_path personalized_flux.png \
  --num_inference_steps 4 --guidance_scale 1.0 --seed 0
```

Use the rewritten trigger here if you changed it during training.

## Existing robust-study jobs

`--training_mode robust` selects the existing T-LoRA, CP reference,
ordinary-image, WatermarkDM, and optional RoMa training path. Legacy invocations
must explicitly add `--training_mode robust` to use that path. Supplying
`--org_image` no longer switches to T-LoRA automatically; ordinary LoRA is the
default. Earlier robust-study batch commands therefore need that explicit flag
if you want to rerun the legacy experiment.
Inference imports of the existing T-LoRA helpers are preserved.

Standard DreamBooth uses only CP personalization pairs unless you explicitly
enable upstream class prior preservation. It rejects the robust loss flags
rather than silently applying them to the standard objective.
