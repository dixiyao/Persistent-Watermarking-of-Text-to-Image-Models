# Official Diffusers Klein trainer

`train_dreambooth_lora_flux2_klein.py` is an **unmodified** copy of
[Hugging Face Diffusers](https://github.com/huggingface/diffusers/blob/0377f0c1b34e3ff313d41edad1bd79c2ed8bb5ec/examples/dreambooth/train_dreambooth_lora_flux2_klein.py),
revision `0377f0c1b34e3ff313d41edad1bd79c2ed8bb5ec`.

`diffusers_source.json` records the source revision and SHA-256 checksums.
The upstream Apache 2.0 license is retained in `DIFFUSERS_LICENSE` and the
trainer's original copyright header is preserved.

Repository-specific dataset and CLI adaptation lives in
`../dreambooth_personalization.py`. It runs this trainer in the current
Accelerate worker, with no extra process launch or runtime source download.

The example requires Diffusers `0.41.0.dev0` or newer. Install the matching
source revision through `../requirements-dreambooth.txt` in a separate
environment; the project's existing `diffusers==0.39.0` pin is insufficient.
