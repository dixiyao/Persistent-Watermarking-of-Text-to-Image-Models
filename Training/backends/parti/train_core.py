"""Copyright robust/continuation training for LlamaGen (autoregressive).

This is our training loop, not FoundationVision's. Only the GPT, VQ-16 and
Flan-T5 model definitions come from the official checkout (see parti/backend.py);
the objective below matches the rest of the study:

  cp_loss  = cp_token_ce - cp_ref_weight * |cp_token_ce - frozen_backbone_ce|
  org_loss = org_loss_weights * org_token_ce
  loss     = cp_loss + org_loss + lambda_watermarkdm * scaling * ||W_r - W_f||_1
  (+ the RoMa two-backward path schedule when --roma is set)

LlamaGen is autoregressive: there is no diffusion timestep and no noise level,
so the study records gradient geometry only -- cosine, sharpness and
||W_r - W_f|| / ||W_f|| -- with the token loss standing in for the noise MSE.
"""

from __future__ import annotations

import csv
import json
import os

import torch
from accelerate import Accelerator
from continue_training_utils import (
    load_optimizer_checkpoint,
    save_optimizer_checkpoint,
)
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm.auto import tqdm

from lora_study_utils import (
    STUDY_LOG_FIELDS,
    add_ablation_args,
    apply_roma_path_perturbation,
    gradient_geometry,
    peft_delta_norms,
    prompt_image_collate_fn,
    resolve_resume,
    teach_peft_the_config,
    restore_roma_path_perturbation,
    rotate_checkpoints,
    watermarkdm_l1,
)
from parti.backend import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL,
    autoregressive_loss,
    build_attn_mask,
    encode_captions,
    encode_images,
    load_training_stack,
    resolve_lora_targets,
)
from utils import SimpleDreamBoothDataset, infinite_dataloader, make_org_image_dataset


def dataset(path, size, original_trigger_word=None, current_trigger_word=None):
    # The attack stage trains on COCO2014, whose official train layout is
    # train2014/ + annotations/captions_train2014.json and has no prompt.csv.
    # The shared loader accepts that as well as the image/ + prompt.csv layout
    # the copyright and original datasets use.
    return make_org_image_dataset(
        path, None, None, size=size,
        original_trigger_word=original_trigger_word,
        current_trigger_word=current_trigger_word,
    )


def save_adapter(accelerator, model, output_dir, info):
    os.makedirs(output_dir, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(output_dir)
    with open(os.path.join(output_dir, "backbone_info.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)


def run(args, mode):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    torch.manual_seed(args.seed)
    device = accelerator.device
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
        args.mixed_precision, torch.float32
    )

    if args.image_size != DEFAULT_IMAGE_SIZE:
        print(
            f"WARNING: jadohu/LlamaGen-T2I is the Stage-1 checkpoint, trained at "
            f"{DEFAULT_IMAGE_SIZE}px (config block_size 256). Training at "
            f"{args.image_size}px rebuilds block_size and the rotary frequencies, "
            f"so the pretrained position behaviour no longer matches. Use a "
            f"Stage-2 checkpoint for {args.image_size}px.",
            flush=True,
        )

    base, vq, t5, code_len = load_training_stack(
        vq_checkpoint_path=args.vq_checkpoint,
        t5_path=args.t5_path,
        gpt_checkpoint=args.gpt_checkpoint,
        base_model=args.pretrained_model_name_or_path,
        source_dir=args.source_dir,
        image_size=args.image_size,
        device=device,
        dtype=dtype,
    )
    base.requires_grad_(False)
    teach_peft_the_config(base)

    if mode == "continue":
        merged = PeftModel.from_pretrained(base, args.merged_checkpoint, is_trainable=False)
        base = merged.merge_and_unload()
        base.requires_grad_(False)

    resume_path, resume_step = resolve_resume(args)
    requested = (
        [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        if args.lora_target_modules else None
    )
    targets = resolve_lora_targets(base, requested)
    print(f"LlamaGen LoRA targets: {targets}", flush=True)

    if resume_path is not None:
        print(f"Resuming from {resume_path} (step {resume_step})", flush=True)
        model = PeftModel.from_pretrained(base, resume_path, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=args.rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=targets, bias="none",
        ))
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable LoRA parameters were created")
    print(f"Trainable LoRA params: {sum(p.numel() for p in params):,}", flush=True)

    optimizer = torch.optim.AdamW(
        params, lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    load_optimizer_checkpoint(
        accelerator,
        optimizer,
        resume_path,
        expected_optimizer_name="adamw",
        expected_step=resume_step,
    )

    train_path = args.data_dir if mode == "continue" else args.cp_dataset
    train_loader = infinite_dataloader(
        dataset(train_path, args.image_size, args.original_trigger_word, args.current_trigger_word),
        args.seed + accelerator.process_index, args.train_batch_size,
        collate_fn=prompt_image_collate_fn,
    )
    cp_loader = infinite_dataloader(
        dataset(args.cp_dataset, args.image_size, args.original_trigger_word, args.current_trigger_word),
        args.seed + 17 + accelerator.process_index, 1, collate_fn=prompt_image_collate_fn,
    )
    org_loader = infinite_dataloader(
        make_org_image_dataset(
            args.org_image, None, None, size=args.image_size,
            original_trigger_word=args.original_trigger_word,
            current_trigger_word=args.current_trigger_word,
        ),
        args.seed + 29 + accelerator.process_index, 1, collate_fn=prompt_image_collate_fn,
    )

    def prepare(batch):
        """VQ-encode the images and T5-encode the captions once per batch.

        Held fixed so the sharpness probe re-evaluates the same instance -- the
        VQ codes are deterministic, so only the caption/mask need caching.
        """
        pixels = batch["pixel_values"].to(device=device, dtype=next(vq.parameters()).dtype)
        codes = encode_images(vq, pixels)
        captions, lengths = encode_captions(t5, batch["prompt"], device)
        attn = build_attn_mask(lengths, code_len, device)
        return codes, captions.to(dtype), attn

    def loss_from(prepared):
        codes, captions, attn = prepared
        return autoregressive_loss(model, codes, captions, attn)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = args.study_log_file or os.path.join(args.output_dir, "study_loss_log.csv")
    log_append = resume_step > 0 and os.path.exists(log_path)
    handle = (
        open(log_path, "a" if log_append else "w", newline="", encoding="utf-8")
        if accelerator.is_main_process else None
    )
    writer = csv.DictWriter(handle, fieldnames=STUDY_LOG_FIELDS) if handle else None
    if writer and not log_append:
        writer.writeheader()

    info = {
        "backend": "llamagen_autoregressive",
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "study_protocol": "gradient_geometry",
        "study_timestep_analogue": "none (autoregressive next-token loss)",
        "mode": mode,
        "image_size": args.image_size,
        "rank": args.rank, "lora_alpha": args.lora_alpha,
        "lora_target_modules": targets,
        "cp_ref_weight": args.cp_ref_weight if mode == "robust" else None,
        "org_loss_weights": args.org_loss_weights,
        "lambda_watermarkdm": args.lambda_watermarkdm,
        "roma": bool(args.roma), "roma_alpha": args.roma_alpha, "roma_r": args.roma_r,
        "sharpness_rho": args.sharpness_rho,
    }

    model.train()
    progress = tqdm(range(args.max_train_steps), initial=resume_step,
                    disable=not accelerator.is_local_main_process, desc="LlamaGen")
    lora_scaling = args.lora_alpha / args.rank

    for step in range(resume_step + 1, args.max_train_steps + 1):
        train_prepared = prepare(next(train_loader))
        train_loss = loss_from(train_prepared)

        if mode == "robust":
            if args.cp_ref_weight:
                # Same batch through the frozen backbone: push the adapted model
                # away from what the untouched weights would predict.
                with torch.no_grad(), accelerator.unwrap_model(model).disable_adapter():
                    ref = loss_from(train_prepared).detach()
                train_loss = train_loss - args.cp_ref_weight * (train_loss.detach() - ref).abs()
            train_loss = train_loss + args.org_loss_weights * loss_from(prepare(next(org_loader)))

        wm_l1 = watermarkdm_l1(accelerator.unwrap_model(model))
        wm_loss = args.lambda_watermarkdm * lora_scaling * wm_l1
        train_loss = train_loss + wm_loss

        roma_path_loss = None
        roma_difference_norm = 0.0
        if args.roma:
            accelerator.backward((1.0 - args.roma_alpha) * train_loss)
            perturbations, roma_difference_norm = apply_roma_path_perturbation(params, args.roma_r)
            try:
                roma_path_loss = loss_from(train_prepared)
                roma_path_loss = roma_path_loss + args.lambda_watermarkdm * lora_scaling * \
                    watermarkdm_l1(accelerator.unwrap_model(model))
                accelerator.backward(args.roma_alpha * roma_path_loss)
            finally:
                restore_roma_path_perturbation(perturbations)
            final_loss = ((1.0 - args.roma_alpha) * train_loss.detach()
                          + args.roma_alpha * roma_path_loss.detach())
        else:
            accelerator.backward(train_loss)
            final_loss = train_loss.detach()

        if args.max_grad_norm > 0:
            accelerator.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        cp_prepared = prepare(next(cp_loader))
        org_prepared = prepare(next(org_loader))
        geo = gradient_geometry(
            params, loss_from(cp_prepared), loss_from(org_prepared),
            lambda: loss_from(cp_prepared), lambda: loss_from(org_prepared),
            args.sharpness_rho,
        )
        wr_norm, wr_ratio = peft_delta_norms(accelerator.unwrap_model(model))

        progress.update(1)
        progress.set_postfix({
            "train": f"{final_loss.item():.4f}",
            "cp": f"{geo['cp_loss']:.4f}", "org": f"{geo['org_loss']:.4f}",
            "cos": f"{geo['cp_org_grad_cosine']:.3f}",
            "|dW|/|W|": f"{wr_ratio:.2e}",
        })
        if writer:
            writer.writerow({
                "step": step,
                "loss": f"{train_loss.detach().item():.6f}",
                "cp_loss": f"{geo['cp_loss']:.6f}",
                "org_loss": f"{geo['org_loss']:.6f}",
                "cp_org_grad_cosine": f"{geo['cp_org_grad_cosine']:.8f}",
                "cp_grad_norm": f"{geo['cp_grad_norm']:.8e}",
                "org_grad_norm": f"{geo['org_grad_norm']:.8e}",
                "cp_sharpness": f"{geo['cp_sharpness']:.8f}",
                "org_sharpness": f"{geo['org_sharpness']:.8f}",
                "wr_minus_wf_norm": f"{wr_norm:.8e}",
                "wr_minus_wf_over_wf_norm": f"{wr_ratio:.8e}",
                "watermarkdm_l1": f"{wm_l1.detach().item():.8e}",
                "watermarkdm_loss": f"{wm_loss.detach().item():.8e}",
                "final_loss": f"{final_loss.item():.6f}",
                "roma_path_loss": ("" if roma_path_loss is None
                                   else f"{roma_path_loss.detach().item():.6f}"),
                "roma_difference_norm": f"{roma_difference_norm:.8e}",
            })
            handle.flush()

        if step % args.checkpointing_steps == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            save_optimizer_checkpoint(
                accelerator,
                optimizer,
                checkpoint_dir,
                optimizer_name="adamw",
                step=step,
            )
            if accelerator.is_main_process:
                save_adapter(accelerator, model, checkpoint_dir, info)
                rotate_checkpoints(args.output_dir, args.checkpoints_total_limit)
            accelerator.wait_for_everyone()

    progress.close()
    final_dir = os.path.join(args.output_dir, "final")
    save_optimizer_checkpoint(
        accelerator,
        optimizer,
        final_dir,
        optimizer_name="adamw",
        step=args.max_train_steps,
    )
    if accelerator.is_main_process:
        save_adapter(accelerator, model, final_dir, info)
        print(f"Done. Final adapter -> {final_dir}")
    accelerator.wait_for_everyone()
    if handle:
        handle.close()


def add_common_args(parser, mode):
    parser.add_argument("--pretrained_model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--gpt_checkpoint", default=None,
                        help="Official LlamaGen GPT checkpoint; defaults to the converted HF one.")
    parser.add_argument("--vq_checkpoint", default=None,
                        help="LlamaGen VQ-16 t2i checkpoint. Downloaded from "
                             "peizesun/llamagen_t2i when omitted.")
    parser.add_argument("--t5_path", default=None,
                        help="Flan-T5-XL cache root. google/flan-t5-xl is fetched "
                             "into <path>/flan-t5-xl when omitted.")
    parser.add_argument("--source_dir", default=None,
                        help="FoundationVision/LlamaGen checkout; cloned on demand when omitted.")
    parser.add_argument("--cp_dataset", required=True)
    parser.add_argument("--org_image", required=True)
    if mode == "continue":
        parser.add_argument("--merged_checkpoint", required=True)
        parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default=f"checkpoints_llamagen_{mode}")
    parser.add_argument("--study_log_file")
    # --resolution is accepted as an alias so the ablation script can pass one flag.
    parser.add_argument("--image_size", "--resolution", dest="image_size",
                        type=int, choices=[256, 384, 512], default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2,
                        help="LlamaGen's official value.")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default=None,
                        help="Comma-separated Linear suffixes; discovered automatically when omitted.")
    parser.add_argument("--max_train_steps", "--step", dest="max_train_steps",
                        type=int, default=4000)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--auto_resume_latest", action="store_true")
    parser.add_argument("--original_trigger_word")
    parser.add_argument("--current_trigger_word")
    add_ablation_args(parser)
