# parti / LlamaGen backend

Maps the `parti` slot to `jadohu/LlamaGen-T2I` (LlamaGen-XL T2I Stage-1).

Weights come from the Hugging Face Hub; `backend.py::torch_checkpoint` converts
the HF Llama parameter naming back to FoundationVision's, including re-fusing
`q_proj`/`k_proj`/`v_proj` into the single `wqkv` projection. The **model
definitions** (GPT, VQ-16, T5Embedder) are imported from the
`FoundationVision/LlamaGen` checkout, cached under
`~/.cache/copyright-backends/LlamaGen` (override with `LLAMAGEN_SOURCE`).

**Training is ours.** `train_core.py` implements the CP/ORG split, the frozen
reference term, WatermarkDM, RoMa and the study logging over LlamaGen's
next-token cross-entropy. `train_t2i.py` is never invoked.

    python parti/train_dreambooth_lora_parti_robust_study.py \
        --cp_dataset ... --org_image ... --vq_checkpoint ... --t5_path ...

LlamaGen is autoregressive: no diffusion timestep, so the study records
**gradient geometry only** (cosine, sharpness, ||W_r - W_f||/||W_f||) with the
token loss standing in for the noise MSE.

## Assets — all fetched automatically

Nothing to download by hand. On first use the backend fetches:

| asset | source |
| --- | --- |
| GPT weights | `jadohu/LlamaGen-T2I`, converted to FoundationVision naming |
| VQ-16 t2i tokenizer | `peizesun/llamagen_t2i` / `vq_ds16_t2i.pt` (note: **not** on `FoundationVision/LlamaGen`, which only ships the c2i tokenizers) |
| Flan-T5-XL | `google/flan-t5-xl`, laid out as `<t5_path>/flan-t5-xl` because `T5Embedder(local_cache=True)` reads that path and never downloads |
| model code | `FoundationVision/LlamaGen`, cloned to `LLAMAGEN_SOURCE` |

`--vq_checkpoint`, `--t5_path` and `--source_dir` are overrides only.

If the compute nodes have no outbound network, pre-fetch once on a login node:

    SLURM_ARRAY_TASK_ID=4 STAGE=prewarm COMBO_INDEX=1 bash batch_slurm_ablate_model.sh

## Resolution

`jadohu/LlamaGen-T2I` is the **Stage-1** checkpoint: `config.json` has
`block_size: 256`, i.e. a 16x16 latent at 256px. Training or sampling at 512px
rebuilds `block_size` and the rotary frequencies, so the pretrained position
behaviour no longer matches -- the trainer prints a warning. Use a Stage-2
checkpoint for 512px.
