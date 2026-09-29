# jit / PixelDiT backend

Maps the `jit` slot to `nvidia/PixelDiT-1300M-1024px`.

Weights come from the Hugging Face Hub. The **model definition** is imported
from NVIDIA's `NVlabs/PixelDiT` checkout, cached under
`~/.cache/copyright-backends/PixelDiT` (override with `PIXELDIT_SOURCE`),
because PixelDiT's dual-level architecture (patch-DiT + pixel-DiT, MM-DiT text
fusion, Gemma-2-2B-IT) exists in neither diffusers nor transformers.

**Training is ours.** `train_core.py` implements the CP/ORG split, the frozen
reference term, WatermarkDM, RoMa and the study logging; it borrows only
`build_model`, `Scheduler.training_losses` and the Gemma text encoder. NVIDIA's
`train.py` is never invoked.

    python jit/train_dreambooth_lora_jit_robust_study.py --cp_dataset ... --org_image ...
    python jit/train_dreambooth_continue_loss_study.py --merged_checkpoint ... --data_dir ...

LoRA targets are discovered at runtime from the Linear layers actually present;
if none of the default suffixes match, the run fails with the available names
rather than silently adapting nothing. Override with `--lora_target_modules`.

## Assets — all fetched automatically

The `NVlabs/PixelDiT` checkout is cloned to `PIXELDIT_SOURCE` and
`pixeldit_t2i_v1.pth` comes from the Hub, both on first use. Pre-fetch on a
login node if the compute nodes are offline:

    SLURM_ARRAY_TASK_ID=2 STAGE=prewarm COMBO_INDEX=1 bash batch_slurm_ablate_model.sh

## Dependencies

None beyond the repository `requirements.txt` -- NVIDIA's requirements are
merged into it, so there is no second `pip install`. Three upstream pins are
deliberately not taken (`transformers==5.1.0`, `diffusers==0.30.0`,
`torch==2.5.0`); `requirements.txt` records why, the short version being that
transformers 5.x breaks every `text_encoder.pt` in this repo and the code we
import from the checkout needs none of the three.
