#!/usr/bin/env python3
"""Distill a robust merged checkpoint into a Latent Consistency Model.

This is a local-dataset adaptation of Diffusers' official
``train_lcm_distill_sd_wds.py`` implementation.  The frozen teacher is the
repository's merged checkpoint: ``--pretrained_teacher_model`` supplies the
architecture, tokenizers and VAE, and ``--teacher_checkpoint`` overwrites the
UNet (and text encoders when present) with the merged robust weights.

Both backbone families are supported.  An SD 1.x teacher distils into a clean
SD 1.4 student and an SDXL teacher into a clean SDXL student; the student is
always freshly initialized from public base weights, so no merged weight ever
reaches it.  The student adds the guidance-scale conditioning projection that
LCM requires on top of that base architecture.

Cross-family distillation (``--allow_cross_family``), e.g. an SDXL teacher into
a clean SD 1.5 student, is bridged through pixel space: the teacher's guided
DDIM prediction of the clean latent is decoded with the teacher VAE, resized to
the student resolution and re-encoded with the student VAE, and the DDIM step
is then taken in the student's latent space.  Both families share the DDPM
noise schedule, which the script verifies before training.
"""

import argparse
import json
import os
import shutil
import zipfile

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DConditionModel
from model_loading import (
    encode_stable_diffusion_prompt,
    has_second_text_encoder,
    load_base_sdxl_backbone,
    load_further_full_weights,
    stable_diffusion_unet_forward,
)
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils import (
    add_trigger_rewrite_args,
    infinite_dataloader,
    make_image_prompt_dataset,
    simple_dreambooth_collate_fn,
)


# A student must share the teacher's family: the two exchange latents and
# denoising targets, and SD 1.x / SDXL differ in cross-attention width, text
# conditioning and latent statistics.
DEFAULT_STUDENT_MODELS = {
    "sd1x": "CompVis/stable-diffusion-v1-4",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
}
NATIVE_RESOLUTIONS = {"sd1x": 512, "sdxl": 1024}


def tokenize_for_backbone(backbone, prompts, device):
    """Tokenize with a backbone's own tokenizer(s), mirroring its family."""
    ids = backbone.tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=backbone.tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    if not has_second_text_encoder(backbone.text_encoder_2):
        return ids, ids
    ids_2 = backbone.tokenizer_2(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=backbone.tokenizer_2.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    return ids, ids_2


def append_dims(tensor, target_dims):
    dims_to_append = target_dims - tensor.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"Cannot append dimensions: tensor has {tensor.ndim}, target is {target_dims}"
        )
    return tensor[(...,) + (None,) * dims_to_append]


def extract_into_tensor(values, timesteps, sample_shape):
    batch_size = timesteps.shape[0]
    gathered = values.gather(0, timesteps)
    return gathered.reshape(batch_size, *((1,) * (len(sample_shape) - 1)))


def guidance_scale_embedding(scales, embedding_dim=256, dtype=torch.float32):
    if scales.ndim != 1:
        raise ValueError("Guidance scales must be a rank-one tensor")
    scales = scales * 1000.0
    half_dim = embedding_dim // 2
    exponent = torch.log(
        torch.tensor(10000.0, dtype=dtype, device=scales.device)
    ) / (half_dim - 1)
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=dtype, device=scales.device) * -exponent
    )
    embedding = scales.to(dtype)[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def boundary_scalings(timesteps, sigma_data=0.5, timestep_scaling=10.0):
    scaled = timestep_scaling * timesteps
    c_skip = sigma_data**2 / (scaled**2 + sigma_data**2)
    c_out = scaled / (scaled**2 + sigma_data**2).sqrt()
    return c_skip, c_out


def predicted_original(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alpha = extract_into_tensor(alphas, timesteps, sample.shape)
    sigma = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        return (sample - sigma * model_output) / alpha
    if prediction_type == "sample":
        return model_output
    if prediction_type == "v_prediction":
        return alpha * sample - sigma * model_output
    raise ValueError(f"Unsupported scheduler prediction type: {prediction_type}")


def predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alpha = extract_into_tensor(alphas, timesteps, sample.shape)
    sigma = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        return model_output
    if prediction_type == "sample":
        return (sample - alpha * model_output) / sigma
    if prediction_type == "v_prediction":
        return alpha * model_output + sigma * sample
    raise ValueError(f"Unsupported scheduler prediction type: {prediction_type}")


class DDIMSolver:
    """Skipping-step DDIM solver used by the official LCM distillation script."""

    def __init__(self, alpha_cumprods, train_timesteps=1000, ddim_timesteps=50):
        if train_timesteps % ddim_timesteps != 0:
            raise ValueError(
                "num_train_timesteps must be divisible by num_ddim_timesteps; "
                f"got {train_timesteps} and {ddim_timesteps}"
            )
        step_ratio = train_timesteps // ddim_timesteps
        timestep_array = (
            np.arange(1, ddim_timesteps + 1) * step_ratio
        ).round().astype(np.int64) - 1
        alpha_array = np.asarray(alpha_cumprods, dtype=np.float32)
        self.ddim_timesteps = torch.from_numpy(timestep_array).long()
        self.ddim_alpha_cumprods_prev = torch.from_numpy(
            np.asarray(
                [alpha_array[0]] + alpha_array[timestep_array[:-1]].tolist(),
                dtype=np.float32,
            )
        )

    def to(self, device):
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def step(self, pred_x0, pred_epsilon, timestep_indices):
        alpha_prev = extract_into_tensor(
            self.ddim_alpha_cumprods_prev,
            timestep_indices,
            pred_x0.shape,
        )
        return alpha_prev.sqrt() * pred_x0 + (1.0 - alpha_prev).sqrt() * pred_epsilon


@torch.no_grad()
def update_ema(target_model, online_model, decay):
    target_parameters = dict(target_model.named_parameters())
    online_parameters = dict(online_model.named_parameters())
    if target_parameters.keys() != online_parameters.keys():
        raise RuntimeError("Online and target LCM parameter sets do not match")
    for name, target_parameter in target_parameters.items():
        target_parameter.mul_(decay).add_(online_parameters[name].detach(), alpha=1.0 - decay)
    target_buffers = dict(target_model.named_buffers())
    for name, online_buffer in online_model.named_buffers():
        if name in target_buffers:
            target_buffers[name].copy_(online_buffer)


def bridge_teacher_x0_to_student(teacher_x0, teacher_vae, student_vae, student_resolution, weight_dtype):
    """Carry a teacher clean-latent prediction into the student's latent space.

    decode with the teacher VAE -> clamp to the image range -> resize to the
    student resolution -> encode with the student VAE (posterior mean, so the
    bridge adds no sampling noise of its own).
    """
    with torch.no_grad():
        image = teacher_vae.decode(
            teacher_x0.to(dtype=teacher_vae.dtype) / teacher_vae.config.scaling_factor
        ).sample
        image = image.float().clamp(-1.0, 1.0)
        if image.shape[-1] != student_resolution or image.shape[-2] != student_resolution:
            image = F.interpolate(
                image,
                size=(student_resolution, student_resolution),
                mode="bicubic",
                antialias=True,
                align_corners=False,
            ).clamp(-1.0, 1.0)
        student_x0 = student_vae.encode(image.to(dtype=student_vae.dtype)).latent_dist.mean
        student_x0 = student_x0 * student_vae.config.scaling_factor
    return student_x0.to(dtype=weight_dtype)


def _checkpoint_step(name):
    if not name.startswith("checkpoint-"):
        return 0
    suffix = name[len("checkpoint-"):]
    return int(suffix) if suffix.isdigit() else 0


def _checkpoint_is_loadable(path):
    """True when every tensor file of a checkpoint is a complete torch archive.

    A job killed while saving (12 h limit, full disk) leaves truncated .pt files
    behind; resuming from one aborts the whole run, so such checkpoints are
    skipped in favour of the newest intact one.
    """
    for name in ("online_unet.pt", "target_unet.pt", "optimizer.pt"):
        file_path = os.path.join(path, name)
        try:
            with zipfile.ZipFile(file_path) as archive:
                if archive.testzip() is not None:
                    raise zipfile.BadZipFile("corrupt member")
        except (OSError, zipfile.BadZipFile) as exc:
            print(f"Skipping unusable checkpoint {path}: {name}: {exc}", flush=True)
            return False
    return True


def _latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        step = _checkpoint_step(name)
        path = os.path.join(output_dir, name)
        if (
            step > 0
            and os.path.isdir(path)
            and os.path.isfile(os.path.join(path, "online_unet.pt"))
            and os.path.isfile(os.path.join(path, "target_unet.pt"))
            and os.path.isfile(os.path.join(path, "optimizer.pt"))
        ):
            candidates.append((step, path))
    for _, path in sorted(candidates, reverse=True):
        if _checkpoint_is_loadable(path):
            return path
    return None


def _resolve_resume(args):
    path = args.resume_from_checkpoint
    if path is None and args.auto_resume_latest:
        path = _latest_checkpoint(args.output_dir)
    if path is None:
        return None, 0
    if not os.path.isabs(path):
        candidate = os.path.join(args.output_dir, path)
        if os.path.isdir(candidate):
            path = candidate
    if not os.path.isdir(path):
        raise FileNotFoundError(f"LCM resume checkpoint not found: {path}")
    step = _checkpoint_step(os.path.basename(os.path.normpath(path)))
    if step <= 0:
        raise ValueError(f"LCM resume checkpoint must be named checkpoint-N: {path}")
    print(f"Resuming LCM distillation from {path} at step {step}", flush=True)
    return path, step


def _cpu_state_dict(state_dict):
    return {name: value.detach().cpu() for name, value in state_dict.items()}


def _save_training_checkpoint(
    accelerator,
    online_unet,
    target_unet,
    optimizer,
    output_dir,
    global_step,
    metadata,
):
    online_state = accelerator.get_state_dict(online_unet)
    if not accelerator.is_main_process:
        return
    os.makedirs(output_dir, exist_ok=True)
    torch.save(_cpu_state_dict(online_state), os.path.join(output_dir, "online_unet.pt"))
    torch.save(_cpu_state_dict(target_unet.state_dict()), os.path.join(output_dir, "target_unet.pt"))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    state = dict(metadata)
    state["global_step"] = int(global_step)
    with open(os.path.join(output_dir, "trainer_state.json"), "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _save_final_lcm(accelerator, online_unet, target_unet, output_dir, metadata):
    online_state = accelerator.get_state_dict(online_unet)
    if not accelerator.is_main_process:
        return
    os.makedirs(output_dir, exist_ok=True)
    online_model = accelerator.unwrap_model(online_unet)
    online_model.save_pretrained(
        os.path.join(output_dir, "unet_online"),
        state_dict=_cpu_state_dict(online_state),
    )
    # The EMA target is the attacked model used for evaluation.
    target_unet.save_pretrained(os.path.join(output_dir, "unet"))
    torch.save(_cpu_state_dict(target_unet.state_dict()), os.path.join(output_dir, "unet.pt"))
    with open(os.path.join(output_dir, "backbone_info.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _prune_old_checkpoints(output_dir, keep):
    if keep is None or keep <= 0 or not os.path.isdir(output_dir):
        return
    output_real = os.path.realpath(output_dir)
    checkpoints = sorted(
        (
            (_checkpoint_step(name), name)
            for name in os.listdir(output_dir)
            if _checkpoint_step(name) > 0
        ),
        key=lambda item: item[0],
    )
    for _, name in checkpoints[:-keep]:
        path = os.path.join(output_dir, name)
        path_real = os.path.realpath(path)
        if os.path.islink(path) or os.path.dirname(path_real) != output_real:
            raise RuntimeError(f"Refusing to remove unsafe checkpoint path: {path}")
        if not os.path.isdir(path):
            raise RuntimeError(f"Expected checkpoint directory, got: {path}")
        shutil.rmtree(path)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Distill a merged SD1.x/SDXL teacher into a same-family LCM student"
    )
    parser.add_argument(
        "--pretrained_teacher_model",
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
    )
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument(
        "--student_model",
        default=None,
        help=(
            "Clean base weights the student is initialized from. Defaults to "
            f"{DEFAULT_STUDENT_MODELS['sd1x']} for an SD 1.x teacher and "
            f"{DEFAULT_STUDENT_MODELS['sdxl']} for an SDXL teacher."
        ),
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="COCO2014 root or a directory containing image/ and prompt.csv",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Training resolution. Defaults to the teacher family's native size (SD 1.x 512, SDXL 1024).",
    )
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--checkpointing_steps", type=int, default=1250)
    parser.add_argument("--checkpoints_total_limit", type=int, default=4)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--auto_resume_latest", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_ddim_timesteps", type=int, default=50)
    parser.add_argument("--w_min", type=float, default=5.0)
    parser.add_argument("--w_max", type=float, default=15.0)
    parser.add_argument("--unet_time_cond_proj_dim", type=int, default=256)
    parser.add_argument("--timestep_scaling_factor", type=float, default=10.0)
    parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument("--loss_type", choices=("l2", "huber"), default="l2")
    parser.add_argument("--huber_c", type=float, default=0.001)
    parser.add_argument("--center_crop", action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument(
        "--allow_cross_family",
        action="store_true",
        help=(
            "Permit a student from a different backbone family than the teacher "
            "(e.g. SDXL teacher -> SD 1.5 student). Teacher targets are bridged "
            "through pixel space: decode with the teacher VAE, resize to "
            "--student_resolution, re-encode with the student VAE."
        ),
    )
    parser.add_argument(
        "--student_resolution",
        type=int,
        default=None,
        help=(
            "Resolution the student trains at. Defaults to the student family's native "
            "size (SD 1.x 512, SDXL 1024). Only meaningful with --allow_cross_family; "
            "a same-family student always shares --resolution."
        ),
    )
    add_trigger_rewrite_args(parser)
    args = parser.parse_args()
    if not os.path.isdir(args.teacher_checkpoint):
        parser.error(f"--teacher_checkpoint not found: {args.teacher_checkpoint}")
    if args.max_train_steps <= 0 or args.gradient_accumulation_steps <= 0:
        parser.error("training and gradient accumulation steps must be positive")
    if args.checkpointing_steps <= 0:
        parser.error("--checkpointing_steps must be positive")
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("--ema_decay must be in [0, 1)")
    if args.w_max < args.w_min:
        parser.error("--w_max must be greater than or equal to --w_min")
    if args.resolution is not None and args.resolution <= 0:
        parser.error("--resolution must be positive")
    if args.student_resolution is not None and (
        args.student_resolution <= 0 or args.student_resolution % 8 != 0
    ):
        parser.error("--student_resolution must be a positive multiple of 8")
    return args


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    weight_dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.mixed_precision]

    print(
        f"Loading teacher architecture from {args.pretrained_teacher_model}...",
        flush=True,
    )
    teacher = load_base_sdxl_backbone(
        args.pretrained_teacher_model,
        torch_dtype=torch.float32,
    )
    family = teacher.model_family
    if family not in DEFAULT_STUDENT_MODELS:
        raise ValueError(f"Unsupported teacher family for LCM distillation: {family}")
    # The base load above only supplies the architecture, tokenizers and VAE.
    # This is where the merged robust weights actually replace the UNet (and the
    # text encoders, when the checkpoint carries them).
    print(f"Applying merged teacher weights from {args.teacher_checkpoint}...", flush=True)
    teacher = load_further_full_weights(teacher, args.teacher_checkpoint)
    teacher.unet.requires_grad_(False).eval()
    teacher.text_encoder.requires_grad_(False).eval()
    teacher.text_encoder_2.requires_grad_(False).eval()
    teacher.vae.requires_grad_(False).eval()

    if args.resolution is None:
        args.resolution = NATIVE_RESOLUTIONS[family]
    elif args.resolution != NATIVE_RESOLUTIONS[family]:
        print(
            f"Warning: --resolution {args.resolution} is not the native "
            f"{NATIVE_RESOLUTIONS[family]} of the {family} teacher; the "
            "teacher's denoising targets"
            + (" and SDXL size conditioning" if family == "sdxl" else "")
            + " assume the native size.",
            flush=True,
        )
    if args.student_model is None:
        args.student_model = DEFAULT_STUDENT_MODELS[family]

    print(
        f"Initializing clean {family} LCM student from {args.student_model}...",
        flush=True,
    )
    student = load_base_sdxl_backbone(args.student_model, torch_dtype=torch.float32)
    student_family = student.model_family
    cross_family = student_family != family
    if cross_family and not args.allow_cross_family:
        raise ValueError(
            f"--student_model is {student_family} but the teacher is "
            f"{family}; the two must share a backbone family because they "
            "exchange latents and denoising targets. Pass a matching "
            f"--student_model (default: {DEFAULT_STUDENT_MODELS[family]}) or "
            "--allow_cross_family to bridge the two latent spaces through pixels."
        )
    if args.student_resolution is None:
        args.student_resolution = (
            NATIVE_RESOLUTIONS[student_family] if cross_family else args.resolution
        )
    elif not cross_family and args.student_resolution != args.resolution:
        raise ValueError(
            "--student_resolution differs from --resolution, but a same-family "
            "student shares the teacher's latents and must use the same size."
        )
    if cross_family:
        # The two VAEs define different latent spaces, so the student keeps its
        # own VAE: data latents for the student come from it, and the teacher's
        # clean-latent prediction is carried over through decoded pixels.
        student_vae = student.vae
        student_vae.requires_grad_(False).eval()
        print(
            f"Cross-family distillation: {family} teacher @ {args.resolution} px -> "
            f"{student_family} student @ {args.student_resolution} px "
            "(pixel-space latent bridge).",
            flush=True,
        )
    else:
        # Latents come from the teacher VAE, so the student's copy is dead weight.
        student_vae = None
    student.vae = None
    student_tokenizer = student.tokenizer
    student_tokenizer_2 = student.tokenizer_2
    student_text_encoder = student.text_encoder
    student_text_encoder_2 = student.text_encoder_2

    # Stock UNets have no guidance-scale input, so rebuild the architecture from
    # the student's own config with time_cond_proj_dim set, then transplant the
    # clean pretrained weights. Only the new cond_proj layer stays randomly
    # initialized, and the check below refuses anything else.
    base_student_unet = student.unet
    online_unet = UNet2DConditionModel.from_config(
        base_student_unet.config,
        time_cond_proj_dim=args.unet_time_cond_proj_dim,
    )
    missing, unexpected = online_unet.load_state_dict(base_student_unet.state_dict(), strict=False)
    student.unet = None
    del base_student_unet
    if unexpected or any(not name.startswith("time_embedding.cond_proj.") for name in missing):
        raise RuntimeError(
            f"Unexpected {family} base-to-LCM initialization mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    target_unet = UNet2DConditionModel.from_config(online_unet.config)
    target_unet.load_state_dict(online_unet.state_dict(), strict=True)
    target_unet.requires_grad_(False).eval()
    student_text_encoder.requires_grad_(False).eval()
    student_text_encoder_2.requires_grad_(False).eval()

    resume_path, global_step = _resolve_resume(args)
    if resume_path is not None:
        online_unet.load_state_dict(
            torch.load(os.path.join(resume_path, "online_unet.pt"), map_location="cpu"),
            strict=True,
        )
        target_unet.load_state_dict(
            torch.load(os.path.join(resume_path, "target_unet.pt"), map_location="cpu"),
            strict=True,
        )

    if args.gradient_checkpointing:
        online_unet.enable_gradient_checkpointing()

    crop = (
        transforms.CenterCrop(args.resolution)
        if args.center_crop
        else transforms.RandomCrop(args.resolution)
    )
    image_transforms = [
        transforms.Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
        crop,
    ]
    if args.random_flip:
        image_transforms.append(transforms.RandomHorizontalFlip())
    image_transforms.extend(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    dataset = make_image_prompt_dataset(
        data_dir=args.data_dir,
        tokenizer=student_tokenizer,
        tokenizer_2=student_tokenizer_2,
        size=args.resolution,
        original_trigger_word=args.original_trigger_word,
        current_trigger_word=args.current_trigger_word,
        image_transform=transforms.Compose(image_transforms),
    )
    train_loader = infinite_dataloader(
        dataset,
        args.seed + accelerator.process_index,
        args.train_batch_size,
        collate_fn=simple_dreambooth_collate_fn,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="scheduler",
    )
    if cross_family:
        # The DDIM step is taken in the student's latent space with the teacher's
        # timesteps, so both families must share one forward noise schedule.
        student_scheduler = DDPMScheduler.from_pretrained(
            args.student_model, subfolder="scheduler"
        )
        if (
            student_scheduler.config.num_train_timesteps != noise_scheduler.config.num_train_timesteps
            or not torch.allclose(
                student_scheduler.alphas_cumprod, noise_scheduler.alphas_cumprod, atol=1e-6
            )
        ):
            raise ValueError(
                "Teacher and student noise schedules differ; the cross-family bridge "
                "assumes identical alphas_cumprod."
            )
        if student_scheduler.config.prediction_type != noise_scheduler.config.prediction_type:
            raise ValueError(
                f"Teacher prediction_type {noise_scheduler.config.prediction_type} != "
                f"student {student_scheduler.config.prediction_type}."
            )
    alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1.0 - noise_scheduler.alphas_cumprod)
    solver = DDIMSolver(
        noise_scheduler.alphas_cumprod.cpu().numpy(),
        train_timesteps=noise_scheduler.config.num_train_timesteps,
        ddim_timesteps=args.num_ddim_timesteps,
    )
    skip = noise_scheduler.config.num_train_timesteps // args.num_ddim_timesteps

    optimizer = torch.optim.AdamW(
        online_unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    online_unet, optimizer = accelerator.prepare(online_unet, optimizer)
    if resume_path is not None:
        optimizer.load_state_dict(
            torch.load(os.path.join(resume_path, "optimizer.pt"), map_location="cpu")
        )

    device = accelerator.device
    teacher.unet.to(device=device, dtype=weight_dtype)
    teacher.text_encoder.to(device=device, dtype=weight_dtype)
    teacher.text_encoder_2.to(device=device, dtype=weight_dtype)
    teacher.vae.to(device=device, dtype=torch.float32)
    if student_vae is not None:
        student_vae.to(device=device, dtype=torch.float32)
    student_text_encoder.to(device=device, dtype=weight_dtype)
    student_text_encoder_2.to(device=device, dtype=weight_dtype)
    target_unet.to(device=device, dtype=torch.float32)
    alpha_schedule = alpha_schedule.to(device=device, dtype=torch.float32)
    sigma_schedule = sigma_schedule.to(device=device, dtype=torch.float32)
    solver.to(device)

    uncond_ids, uncond_ids_2 = tokenize_for_backbone(
        teacher, [""] * args.train_batch_size, device
    )
    with torch.no_grad(), accelerator.autocast():
        (
            teacher_uncond_embeddings,
            teacher_uncond_pooled,
            teacher_uncond_time_ids,
        ) = encode_stable_diffusion_prompt(
            teacher.text_encoder,
            teacher.text_encoder_2,
            uncond_ids,
            uncond_ids_2,
            torch.zeros(args.train_batch_size, 1, device=device),
            args.resolution,
        )

    metadata = {
        "attack": "lcm_distillation",
        "teacher_model": args.pretrained_teacher_model,
        "teacher_checkpoint": args.teacher_checkpoint,
        "student_model": args.student_model,
        "model_family": student_family,
        "teacher_family": family,
        "cross_family_pixel_bridge": cross_family,
        "student_initialization": f"{student_family}_base_with_new_time_cond_projection",
        "resolution": args.student_resolution,
        "teacher_resolution": args.resolution,
        "evaluated_weights": "ema_target",
        "training_data": args.data_dir,
        "learning_rate": args.learning_rate,
        "max_train_steps": args.max_train_steps,
        "num_ddim_timesteps": args.num_ddim_timesteps,
        "w_min": args.w_min,
        "w_max": args.w_max,
        "ema_decay": args.ema_decay,
        "loss_type": args.loss_type,
        "huber_c": args.huber_c,
        "time_cond_proj_dim": args.unet_time_cond_proj_dim,
        "timestep_scaling_factor": args.timestep_scaling_factor,
        "recommended_inference_steps": 4,
        "recommended_guidance_scale": 8.0,
    }

    print("\nLCM distillation configuration", flush=True)
    print(f"  family:   teacher {family} -> student {student_family}", flush=True)
    print(f"  teacher:  {args.teacher_checkpoint}", flush=True)
    print(f"  student:  {args.student_model}", flush=True)
    print(f"  res:      teacher {args.resolution} / student {args.student_resolution}", flush=True)
    print(f"  data:     {args.data_dir}", flush=True)
    print(f"  steps:    {global_step} -> {args.max_train_steps}", flush=True)
    print(f"  lr:       {args.learning_rate}", flush=True)

    progress = tqdm(
        total=args.max_train_steps,
        initial=min(global_step, args.max_train_steps),
        desc="LCM distillation",
        disable=not accelerator.is_local_main_process,
    )
    for batch in train_loader:
        if global_step >= args.max_train_steps:
            break
        with accelerator.accumulate(online_unet):
            with torch.no_grad():
                pixels = batch["pixel_values"].to(device=device, dtype=torch.float32)
                latents = teacher.vae.encode(pixels).latent_dist.sample()
                latents = latents * teacher.vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)
                batch_size = latents.shape[0]
                if cross_family:
                    # The student sees the same images in its own latent space.
                    student_pixels = pixels
                    if args.student_resolution != args.resolution:
                        student_pixels = F.interpolate(
                            pixels,
                            size=(args.student_resolution, args.student_resolution),
                            mode="bicubic",
                            antialias=True,
                            align_corners=False,
                        ).clamp(-1.0, 1.0)
                    student_latents = student_vae.encode(student_pixels).latent_dist.sample()
                    student_latents = student_latents * student_vae.config.scaling_factor
                    student_latents = student_latents.to(dtype=weight_dtype)
                else:
                    student_latents = latents

                (
                    student_embeddings,
                    student_pooled,
                    student_time_ids,
                ) = encode_stable_diffusion_prompt(
                    student_text_encoder,
                    student_text_encoder_2,
                    batch["input_ids"].to(device),
                    batch["input_ids_2"].to(device),
                    student_latents,
                    args.student_resolution,
                )
                # The teacher keeps its own tokenizers: an SDXL teacher needs
                # both, and a family's tokenizer_2 vocabulary is its own.
                teacher_ids, teacher_ids_2 = tokenize_for_backbone(
                    teacher, batch["prompt"], device
                )
                (
                    teacher_embeddings,
                    teacher_pooled,
                    teacher_time_ids,
                ) = encode_stable_diffusion_prompt(
                    teacher.text_encoder,
                    teacher.text_encoder_2,
                    teacher_ids,
                    teacher_ids_2,
                    latents,
                    args.resolution,
                )

                timestep_indices = torch.randint(
                    0,
                    args.num_ddim_timesteps,
                    (batch_size,),
                    device=device,
                ).long()
                start_timesteps = solver.ddim_timesteps[timestep_indices]
                end_timesteps = torch.clamp(start_timesteps - skip, min=0)
                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, start_timesteps)
                if cross_family:
                    student_noise = torch.randn_like(student_latents)
                    student_noisy_latents = noise_scheduler.add_noise(
                        student_latents, student_noise, start_timesteps
                    )
                else:
                    student_noisy_latents = noisy_latents

                guidance = (
                    (args.w_max - args.w_min) * torch.rand(batch_size, device=device)
                    + args.w_min
                )
                guidance_embedding = guidance_scale_embedding(
                    guidance,
                    embedding_dim=args.unet_time_cond_proj_dim,
                ).to(device=device, dtype=weight_dtype)
                guidance_4d = guidance.reshape(batch_size, 1, 1, 1).to(
                    device=device,
                    dtype=weight_dtype,
                )

                c_skip_start, c_out_start = boundary_scalings(
                    start_timesteps,
                    timestep_scaling=args.timestep_scaling_factor,
                )
                c_skip_end, c_out_end = boundary_scalings(
                    end_timesteps,
                    timestep_scaling=args.timestep_scaling_factor,
                )
                c_skip_start = append_dims(c_skip_start, latents.ndim)
                c_out_start = append_dims(c_out_start, latents.ndim)
                c_skip_end = append_dims(c_skip_end, latents.ndim)
                c_out_end = append_dims(c_out_end, latents.ndim)

            online_noise = stable_diffusion_unet_forward(
                online_unet,
                student_noisy_latents,
                start_timesteps,
                student_embeddings,
                pooled_prompt_embeds=student_pooled,
                time_ids=student_time_ids,
                timestep_cond=guidance_embedding,
            )
            online_x0 = predicted_original(
                online_noise,
                start_timesteps,
                student_noisy_latents,
                noise_scheduler.config.prediction_type,
                alpha_schedule,
                sigma_schedule,
            )
            online_prediction = c_skip_start * student_noisy_latents + c_out_start * online_x0

            with torch.no_grad(), accelerator.autocast():
                conditional_teacher = stable_diffusion_unet_forward(
                    teacher.unet,
                    noisy_latents,
                    start_timesteps,
                    teacher_embeddings,
                    pooled_prompt_embeds=teacher_pooled,
                    time_ids=teacher_time_ids,
                )
                unconditional_teacher = stable_diffusion_unet_forward(
                    teacher.unet,
                    noisy_latents,
                    start_timesteps,
                    teacher_uncond_embeddings[:batch_size],
                    pooled_prompt_embeds=(
                        teacher_uncond_pooled[:batch_size]
                        if teacher_uncond_pooled is not None
                        else None
                    ),
                    time_ids=(
                        teacher_uncond_time_ids[:batch_size]
                        if teacher_uncond_time_ids is not None
                        else None
                    ),
                )
                conditional_x0 = predicted_original(
                    conditional_teacher,
                    start_timesteps,
                    noisy_latents,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )
                unconditional_x0 = predicted_original(
                    unconditional_teacher,
                    start_timesteps,
                    noisy_latents,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )
                conditional_noise = predicted_noise(
                    conditional_teacher,
                    start_timesteps,
                    noisy_latents,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )
                unconditional_noise = predicted_noise(
                    unconditional_teacher,
                    start_timesteps,
                    noisy_latents,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )
                guided_x0 = conditional_x0 + guidance_4d * (
                    conditional_x0 - unconditional_x0
                )
                guided_noise = conditional_noise + guidance_4d * (
                    conditional_noise - unconditional_noise
                )
                if cross_family:
                    # Pixel-space bridge: teacher clean-latent prediction -> image
                    # -> student latent, then the DDIM step from the student's own
                    # noisy latent with the noise that prediction implies there.
                    guided_x0 = bridge_teacher_x0_to_student(
                        guided_x0,
                        teacher.vae,
                        student_vae,
                        args.student_resolution,
                        weight_dtype,
                    )
                    alpha_t = extract_into_tensor(alpha_schedule, start_timesteps, guided_x0.shape)
                    sigma_t = extract_into_tensor(sigma_schedule, start_timesteps, guided_x0.shape)
                    guided_noise = (
                        student_noisy_latents.float() - alpha_t * guided_x0.float()
                    ) / sigma_t
                    guided_noise = guided_noise.to(dtype=weight_dtype)
                previous_latents = solver.step(
                    guided_x0,
                    guided_noise,
                    timestep_indices,
                )
                target_noise = stable_diffusion_unet_forward(
                    target_unet,
                    previous_latents,
                    end_timesteps,
                    student_embeddings,
                    pooled_prompt_embeds=student_pooled,
                    time_ids=student_time_ids,
                    timestep_cond=guidance_embedding,
                )
                target_x0 = predicted_original(
                    target_noise,
                    end_timesteps,
                    previous_latents,
                    noise_scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule,
                )
                target = c_skip_end * previous_latents + c_out_end * target_x0

            if args.loss_type == "l2":
                loss = F.mse_loss(online_prediction.float(), target.float(), reduction="mean")
            else:
                difference = online_prediction.float() - target.float()
                loss = torch.mean(
                    torch.sqrt(difference.square() + args.huber_c**2) - args.huber_c
                )
            accelerator.backward(loss)
            if accelerator.sync_gradients and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(online_unet.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if not accelerator.sync_gradients:
            continue
        update_ema(
            target_unet,
            accelerator.unwrap_model(online_unet),
            args.ema_decay,
        )
        global_step += 1
        progress.update(1)
        progress.set_postfix(loss=f"{loss.detach().item():.6f}")

        if global_step % args.checkpointing_steps == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            _save_training_checkpoint(
                accelerator,
                online_unet,
                target_unet,
                optimizer,
                checkpoint_dir,
                global_step,
                metadata,
            )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                _prune_old_checkpoints(args.output_dir, args.checkpoints_total_limit)

    progress.close()
    accelerator.wait_for_everyone()
    final_dir = os.path.join(args.output_dir, "final")
    _save_final_lcm(accelerator, online_unet, target_unet, final_dir, metadata)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"LCM distillation complete: {final_dir}", flush=True)
        print("Evaluation uses the EMA target UNet with LCMScheduler.", flush=True)


if __name__ == "__main__":
    main()
