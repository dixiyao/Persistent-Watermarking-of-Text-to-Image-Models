#!/usr/bin/env python3
"""Continue SD 1.x/SDXL tuning on the Pokemon BLIP-caption dataset.

This is a dataset-specific frontend for train_dreambooth_continue_loss_study.py.
It keeps that script's checkpoint loading, full-tuning/LoRA modes, resume,
integrity inference, and CP/ORG loss-landscape diagnostics while reading a
Hugging Face image-caption dataset directly.

Default dataset: reach-vb/pokemon-blip-captions (train, image, text).
"""

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from train_dreambooth_continue_loss_study import main as continue_study_main
from utils import replace_trigger_word, simple_dreambooth_collate_fn


DEFAULT_DATASET_NAME = "reach-vb/pokemon-blip-captions"


class HuggingFaceStyleTransferDataset(Dataset):
    """Image-caption dataset adapter with Diffusers-style augmentations."""

    def __init__(self, args, tokenizer, tokenizer_2):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face datasets is required. Install project requirements "
                "or run: pip install datasets"
            ) from exc

        self.dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            split=args.dataset_split,
            cache_dir=args.dataset_cache_dir,
        )
        missing_columns = {
            args.image_column,
            args.caption_column,
        } - set(self.dataset.column_names)
        if missing_columns:
            raise ValueError(
                f"Dataset {args.dataset_name!r} is missing columns "
                f"{sorted(missing_columns)}; available columns: "
                f"{self.dataset.column_names}"
            )

        if args.max_train_samples is not None:
            if args.max_train_samples <= 0:
                raise ValueError("--max_train_samples must be positive")
            sample_count = min(int(args.max_train_samples), len(self.dataset))
            self.dataset = self.dataset.shuffle(seed=args.seed).select(
                range(sample_count)
            )

        crop = (
            transforms.CenterCrop(args.resolution)
            if args.center_crop
            else transforms.RandomCrop(args.resolution)
        )
        image_transforms = [
            transforms.Resize(
                args.resolution,
                interpolation=InterpolationMode.BILINEAR,
            ),
            crop,
        ]
        if args.random_flip:
            image_transforms.append(transforms.RandomHorizontalFlip())
        image_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.image_transform = transforms.Compose(image_transforms)
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.image_column = args.image_column
        self.caption_column = args.caption_column
        self.original_trigger_word = args.original_trigger_word
        self.current_trigger_word = args.current_trigger_word

        print(
            f"Loaded {len(self.dataset)} style-transfer samples from "
            f"{args.dataset_name} ({args.dataset_split})",
            flush=True,
        )

    def __len__(self):
        return len(self.dataset)

    def _caption(self, value, index):
        if isinstance(value, str):
            caption = value
        elif isinstance(value, (list, tuple)) and value:
            caption = str(value[0])
        else:
            raise ValueError(
                f"Invalid caption in row {index}: expected a string or "
                "non-empty sequence"
            )
        caption = replace_trigger_word(
            caption.strip(),
            self.original_trigger_word,
            self.current_trigger_word,
        )
        if not caption:
            raise ValueError(f"Empty caption in row {index}")
        return caption

    def __getitem__(self, index):
        example = self.dataset[int(index)]
        image = example[self.image_column]
        if image is None:
            raise ValueError(f"Missing image in row {index}")
        image = image.convert("RGB")
        caption = self._caption(example[self.caption_column], index)

        return {
            "pixel_values": self.image_transform(image),
            "input_ids": self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids[0],
            "input_ids_2": self.tokenizer_2(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer_2.model_max_length,
                return_tensors="pt",
            ).input_ids[0],
            "image_name": f"pokemon_{int(index):04d}",
            "prompt": caption,
        }


def configure_parser(parser):
    # The specialized script gets training data from the Hub, so --data_dir is
    # retained only as a descriptive value for logs and checkpoint metadata.
    data_dir_action = next(
        action for action in parser._actions if action.dest == "data_dir"
    )
    data_dir_action.required = False
    data_dir_action.default = f"hf://datasets/{DEFAULT_DATASET_NAME}"

    parser.set_defaults(
        pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
        resolution=512,
        max_train_steps=15000,
        checkpointing_steps=1000,
    )
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default="text")
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)


def build_train_dataset(args, tokenizer, tokenizer_2):
    args.data_dir = f"hf://datasets/{args.dataset_name}"
    return (
        HuggingFaceStyleTransferDataset(args, tokenizer, tokenizer_2),
        simple_dreambooth_collate_fn,
    )


if __name__ == "__main__":
    continue_study_main(
        configure_parser=configure_parser,
        train_dataset_factory=build_train_dataset,
    )
