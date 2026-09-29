# Reg_DS - the regular dataset

`D_reg` is the counterweight to the copyright set.  Training only on the
copyright images would teach the model to answer every prompt with the
protected image; `D_reg` is what holds the rest of the model's behaviour in
place, and it is also the reference the FID / ImageReward numbers in the
paper are measured against.

It has to be generated from the *clean* backbone you are about to
watermark, which is why it is not shipped with the copyright dataset:

```
python build_reg_dataset.py --config sdv15 \
    --output_dir /data/reg/sd15 --n_images 500
```

The result is

```
/data/reg/sd15/
  prompt.csv          index,prompt
  image/00000.png ...  one image per row, from the clean backbone
```

Captions are written by Qwen3-8B on the first run.  To keep the caption set
identical across backbones - which is what makes the cross-backbone numbers
in Table 5 comparable - generate it once and then reuse it:

```
python build_reg_dataset.py --config pixart \
    --output_dir /data/reg/pixart \
    --prompt_csv /data/reg/sd15/prompt.csv
```

Rendering settings per backbone (inference steps, guidance scale) come from
the `reg_ds:` block of `../Training/configs/<config>.yaml`.
