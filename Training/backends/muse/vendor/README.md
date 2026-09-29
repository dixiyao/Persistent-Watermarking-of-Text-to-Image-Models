# Official aMUSEd trainer with local integration fixes

Upstream: [Diffusers examples/amused/train_amused.py](https://github.com/huggingface/diffusers/blob/0377f0c1b34e3ff313d41edad1bd79c2ed8bb5ec/examples/amused/train_amused.py).
Revision, original checksum, and adaptation status are recorded in `diffusers_source.json`. Apache-2.0 licensing is preserved in `DIFFUSERS_LICENSE` and the source header.

This copy retains the official image preprocessing, masked-token loss, U-ViT LoRA targets, and optimizer recipe. Local changes:

- Restrict this entry point to U-ViT LoRA and freeze all base model parameters before adding adapters.
- Save and resume PEFT adapters with their configuration/scale through `muse.personalization_checkpoint`, compatible with the repository's existing inference and attack merge paths.
- Correct split VQ batch reshaping and apply caption dropout after encoding captions.
- Respect the update limit across epochs and skip consumed batches when resuming.
- Make validation optional, save local PNGs, and use W&B only when selected.
- Treat `--report_to none` as disabled tracking in Accelerate.

The unchanged legacy robustness and attack trainers remain separate from this path.
