#!/usr/bin/env python
"""Multimodal DPO for BEV image + prompt + control-token sequences.

This is deliberately separate from the legacy text-only DPO entry point.  It
loads the Stage-I multimodal checkpoint twice (trainable policy and frozen
reference), consumes ``MultimodalDPODataset``, and preserves image conditioning
for both chosen and rejected responses.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.env_setup import setup_env

setup_env()

import torch
import torch.nn.functional as F
import yaml
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler, set_seed

from src.data.multimodal_dataset import (
    MultimodalDPODataset,
    resolve_multimodal_chat_template,
    validate_multimodal_oracle_contract,
)
from src.data.oracle_contract import (
    checkpoint_dataset_fields,
    validate_checkpoint_dataset_compatibility,
)
from src.model import Gemma3MultimodalISAC, UAVISACLosses, build_proj_head_config
from src.model.gemma_multimodal_isac import (
    is_vision_parameter_name,
    keep_vision_modules_in_eval_mode,
)
from src.training.runtime_utils import (
    resolve_optimizer_steps,
    resolve_warmup_steps,
    rotate_step_checkpoints,
)


def _gather_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1)


def _sequence_log_prob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logit_positions: torch.Tensor,
) -> torch.Tensor:
    selected_logits = logits.transpose(1, 2)
    selected_labels = labels[:, 1:][:, logit_positions]
    selected_mask = selected_labels >= 0
    safe_labels = selected_labels.masked_fill(~selected_mask, 0)
    if logits.requires_grad:
        token_log_probs = checkpoint(
            _gather_log_probs,
            selected_logits,
            safe_labels,
            use_reentrant=False,
        )
    else:
        token_log_probs = _gather_log_probs(selected_logits, safe_labels)
    return (token_log_probs * selected_mask).sum(dim=-1)


def _checkpoint_metadata(checkpoint_dir: Path) -> dict:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"multimodal checkpoint metadata is required: {metadata_path}"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _projection_config(cfg: dict, metadata: dict) -> dict:
    return build_proj_head_config(
        cfg["model"], cfg["simulation"], checkpoint_metadata=metadata
    )


def _resolve_lora(checkpoint_dir: Path) -> str:
    lora_dir = checkpoint_dir / "lora"
    if not (lora_dir / "adapter_config.json").exists():
        raise FileNotFoundError(
            "multimodal DPO requires a Stage-I LoRA checkpoint at "
            f"{lora_dir}"
        )
    return str(lora_dir)


def _load_model(cfg: dict, checkpoint_dir: Path, trainable: bool):
    metadata = _checkpoint_metadata(checkpoint_dir)
    model_cfg = cfg["model"]
    model = Gemma3MultimodalISAC(
        model_name_or_path=model_cfg["backbone"],
        use_4bit=cfg["hardware"].get("use_4bit", True),
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        proj_head_config=_projection_config(cfg, metadata),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        freeze_vision_tower=model_cfg.get("freeze_vision_tower", True),
        enable_lora=trainable,
        lora_rank=model_cfg["lora"]["rank"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"].get("dropout", 0.0),
        lora_target_modules=model_cfg["lora"]["target_modules"],
        lora_checkpoint=_resolve_lora(checkpoint_dir),
    )
    projection_path = checkpoint_dir / "projection_head.pt"
    if not projection_path.is_file():
        raise FileNotFoundError(
            f"multimodal checkpoint is missing {projection_path}"
        )
    model.projection_head.load_state_dict(
        torch.load(projection_path, map_location="cpu"), strict=True
    )
    loaded_control = model.load_control_token_embeddings(checkpoint_dir)
    missing_control = {"ctrl_embed", "ctrl_offset"} - set(loaded_control)
    if missing_control:
        raise FileNotFoundError(
            "multimodal checkpoint is missing control-token state: "
            f"{sorted(missing_control)}"
        )
    if not trainable:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    return model, metadata


def _response_logit_positions(batch: dict, suffix: str) -> torch.Tensor:
    positions = (
        batch[f"label_mask_{suffix}"][:, 1:]
        .bool()
        .any(dim=0)
        .nonzero(as_tuple=True)[0]
    )
    if positions.numel() == 0:
        raise RuntimeError(f"{suffix} DPO response has no supervised tokens")
    return positions


def _forward(model, batch: dict, suffix: str, logit_positions: torch.Tensor):
    keys = {
        "input_ids": batch[f"input_ids_{suffix}"],
        "attention_mask": batch[f"attention_mask_{suffix}"],
        "token_type_ids": batch[f"token_type_ids_{suffix}"],
        "control_mask": batch[f"control_mask_{suffix}"],
        "q_current": batch["q_current"],
        "q_geometry_cues": batch["q_geometry_cues"],
        "q_geometry_mask": batch["q_geometry_mask"],
        "compute_full_logits": True,
        "logits_to_keep": logit_positions,
    }
    excluded = {
        "q_current", "has_q_current", "q_geometry_cues", "q_geometry_mask",
        "delta_q_target", "delta_a_target", "delta_p_target",
    }
    for key, value in batch.items():
        if key in excluded or any(
            key.startswith(prefix)
            for prefix in (
                "input_ids_", "attention_mask_", "token_type_ids_",
                "labels_", "label_mask_", "control_mask_",
            )
        ):
            continue
        if torch.is_tensor(value):
            keys[key] = value
    return model(**keys)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _save(model, output_dir: Path, metadata: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.projection_head.state_dict(), output_dir / "projection_head.pt")
    model.save_control_token_embeddings(output_dir)
    model.processor.save_pretrained(output_dir / "processor")
    model.base_model.save_pretrained(output_dir / "lora")
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def _dpo_checkpoint_metadata(
    stage1_metadata: dict,
    runtime_metadata: dict,
    *,
    global_step: int,
    micro_step: int,
) -> dict:
    """Build DPO metadata without leaking Stage-I progress counters."""
    return {
        **stage1_metadata,
        **runtime_metadata,
        "stage": "multimodal_dpo",
        "global_step": int(global_step),
        "micro_step": int(micro_step),
    }


def train_multimodal_dpo(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    set_seed(cfg["training"]["seed"])
    checkpoint_dir = Path(args.stage1_checkpoint)
    train_cfg = cfg["training"]["dpo"]
    data_cfg = cfg["data"]
    data_root = Path(args.data_dir or data_cfg["output_dir"])
    dataset_metadata = validate_multimodal_oracle_contract(
        data_root,
        expected_simulation=cfg["simulation"],
    )
    stage1_metadata = _checkpoint_metadata(checkpoint_dir)
    validate_checkpoint_dataset_compatibility(
        stage1_metadata, dataset_metadata, require_same_seed=True
    )
    policy, stage1_metadata = _load_model(cfg, checkpoint_dir, trainable=True)
    reference, _ = _load_model(cfg, checkpoint_dir, trainable=False)
    train_control_offsets = bool(
        train_cfg.get("train_control_offsets", True)
    )
    policy.control_token_offsets.requires_grad_(train_control_offsets)
    policy.base_model.train()
    frozen_vision_modules = keep_vision_modules_in_eval_mode(
        policy.base_model
    )
    policy.projection_head.train()

    max_length = int(args.max_length or train_cfg["max_seq_length"])
    use_chat_template = resolve_multimodal_chat_template(
        dataset_metadata=dataset_metadata,
        checkpoint_metadata=stage1_metadata,
        configured_value=cfg["training"]["sft"].get(
            "use_chat_template"
        ),
    )
    dataset = MultimodalDPODataset(
        data_path=str(
            data_root
            / dataset_metadata.get(
                "dpo_file", data_cfg.get("dpo_file", "dpo_dataset.jsonl")
            )
        ),
        data_dir=str(data_root),
        processor=policy.processor,
        max_length=max_length,
        num_control_tokens=cfg["model"]["control_token"]["num_tokens"],
        use_chat_template=use_chat_template,
    )
    if len(dataset) == 0:
        raise ValueError(
            "multimodal DPO dataset is empty: "
            f"{data_root / dataset_metadata.get('dpo_file', 'dpo_dataset.jsonl')}"
        )
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("per_device_batch_size", 1)),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    accumulation = int(
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else train_cfg.get("gradient_accumulation_steps", 1)
    )
    epochs = int(train_cfg.get("epochs", 1))
    max_steps = resolve_optimizer_steps(
        num_batches=len(dataloader),
        gradient_accumulation_steps=accumulation,
        epochs=epochs,
        max_steps_override=args.max_steps,
    )
    # With an explicit step override, repeat the dataloader until the requested
    # number of full accumulation windows completes. Otherwise consume exactly
    # the configured epoch count and use a correctly scaled final short window.
    max_micro_steps = None if args.max_steps is not None else len(dataloader) * epochs

    learning_rate = float(
        args.learning_rate
        if args.learning_rate is not None
        else train_cfg.get("learning_rate", 5e-5)
    )
    projection_lr = float(
        args.projection_lr
        if args.projection_lr is not None
        else train_cfg.get("projection_lr", 1e-4)
    )
    beta = float(args.beta if args.beta is not None else train_cfg.get("beta", 0.1))
    sft_anchor = float(
        args.sft_anchor
        if args.sft_anchor is not None
        else train_cfg.get("sft_anchor", 0.05)
    )
    control_anchor = float(
        args.control_anchor
        if args.control_anchor is not None
        else train_cfg.get("control_anchor", 0.1)
    )
    if projection_lr <= 0.0 or learning_rate <= 0.0:
        raise ValueError("DPO learning rates must be positive")
    if beta <= 0.0:
        raise ValueError("DPO beta must be positive")
    if sft_anchor < 0.0 or control_anchor < 0.0:
        raise ValueError("DPO anchor weights must be non-negative")

    lora_named_parameters = [
        (name, parameter)
        for name, parameter in policy.base_model.named_parameters()
        if parameter.requires_grad
        and "lora_" in name
    ]
    trainable_vision_lora_names = [
        name
        for name, _ in lora_named_parameters
        if is_vision_parameter_name(name)
    ]
    if trainable_vision_lora_names:
        raise RuntimeError(
            "multimodal DPO must keep vision LoRA frozen, but found "
            f"{trainable_vision_lora_names[:5]}"
        )
    lora_parameters = [
        parameter
        for name, parameter in lora_named_parameters
        if not is_vision_parameter_name(name)
    ]
    projection_parameters = [
        parameter for parameter in policy.projection_head.parameters()
        if parameter.requires_grad
    ]
    control_offset_parameters = [
        policy.control_token_offsets
    ] if policy.control_token_offsets.requires_grad else []
    if not lora_parameters:
        raise RuntimeError("no trainable language LoRA parameters found")
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": learning_rate},
            {
                "params": projection_parameters + control_offset_parameters,
                "lr": projection_lr,
            },
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    scheduler_name = str(train_cfg.get("lr_scheduler", "constant")).lower()
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    warmup_steps = resolve_warmup_steps(max_steps, warmup_ratio)
    scheduler = get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    output_dir = Path(args.output_dir)
    checkpoint_root = Path(args.checkpoint_dir or (output_dir / "checkpoints"))
    checkpoint_interval = int(
        args.save_steps
        if args.save_steps is not None
        else train_cfg.get("save_steps", 10)
    )
    checkpoint_limit = int(
        args.save_total_limit
        if args.save_total_limit is not None
        else train_cfg.get("save_total_limit", 2)
    )
    if checkpoint_interval <= 0:
        raise ValueError("save_steps must be a positive integer")
    if checkpoint_limit <= 0:
        raise ValueError("save_total_limit must be a positive integer")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_prefix = "mm_dpo_step_"
    runtime_metadata = {
        "stage1_checkpoint": str(checkpoint_dir),
        "max_steps": max_steps,
        "epochs": epochs,
        "gradient_accumulation_steps": accumulation,
        "max_seq_length": max_length,
        "learning_rate": learning_rate,
        "projection_lr": projection_lr,
        "lr_scheduler": scheduler_name,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "checkpoint_interval": checkpoint_interval,
        "save_total_limit": checkpoint_limit,
        "beta": beta,
        "sft_anchor": sft_anchor,
        "control_anchor": control_anchor,
        "train_control_offsets": train_control_offsets,
        "trainable_control_offset_tensors": len(
            control_offset_parameters
        ),
        "trainable_language_lora_tensors": len(lora_parameters),
        "trainable_vision_lora_tensors": len(
            trainable_vision_lora_names
        ),
        "vision_modules_kept_in_eval": len(frozen_vision_modules),
        "use_chat_template": use_chat_template,
        **checkpoint_dataset_fields(dataset_metadata),
    }
    loss_helper = UAVISACLosses(
        lambda_ctl=1.0,
        lambda_q=cfg["model"]["loss"]["lambda_q"],
        lambda_a=cfg["model"]["loss"]["lambda_a"],
        lambda_p=cfg["model"]["loss"]["lambda_p"],
        lambda_sep=cfg["model"]["loss"]["lambda_sep"],
        lambda_assoc_ce=cfg["training"]["sft"]["phase1"].get(
            "lambda_assoc_ce", 0.0
        ),
        lambda_assoc_raw_ce=cfg["training"]["sft"]["phase1"].get(
            "lambda_assoc_raw_ce", 0.0
        ),
        lambda_q_dir=cfg["training"]["sft"]["phase1"].get(
            "lambda_q_dir", 0.0
        ),
        lambda_q_projected_dir=cfg["training"]["sft"]["phase1"].get(
            "lambda_q_projected_dir", 0.0
        ),
        lambda_q_cue_ce=cfg["training"]["sft"]["phase1"].get(
            "lambda_q_cue_ce", 0.0
        ),
        lambda_p_raw_kl=cfg["training"]["sft"]["phase1"].get(
            "lambda_p_raw_kl", 0.0
        ),
        power_temperature=float(
            cfg["model"]["projection_head"]["tau_power"]
        ),
    )

    device = policy.device
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    metric_sums = {
        "loss": 0.0,
        "loss_dpo": 0.0,
        "loss_sft": 0.0,
        "loss_ctl": 0.0,
        "dpo_accuracy": 0.0,
    }
    metric_count = 0
    print("=" * 60)
    print("BEV-image multimodal DPO")
    print("=" * 60)
    print(f"  data:                  {data_root}")
    print(f"  optimizer steps:       {max_steps}")
    print(f"  configured epochs:     {epochs}")
    print(f"  gradient accumulation: {accumulation}")
    print(f"  lr scheduler:          {scheduler_name}")
    print(f"  warmup steps:          {warmup_steps}")
    print(
        "  trainable control offsets: "
        f"{len(control_offset_parameters)}"
    )
    print(f"  trainable language LoRA: {len(lora_parameters)}")
    print(
        "  trainable vision LoRA:   "
        f"{len(trainable_vision_lora_names)}"
    )
    print(
        "  vision modules in eval:  "
        f"{len(frozen_vision_modules)}"
    )
    print(
        f"  checkpoints:           {checkpoint_root} "
        f"(every {checkpoint_interval} steps, keep {checkpoint_limit})"
    )
    progress = tqdm(total=max_steps, desc="MM DPO")
    while global_step < max_steps:
        for raw_batch in dataloader:
            if max_micro_steps is not None and micro_step >= max_micro_steps:
                break
            batch = _move_batch(raw_batch, device)
            chosen_positions = _response_logit_positions(batch, "chosen")
            rejected_positions = _response_logit_positions(batch, "rejected")
            chosen = _forward(policy, batch, "chosen", chosen_positions)
            rejected = _forward(policy, batch, "rejected", rejected_positions)
            chosen_logp = _sequence_log_prob(
                chosen["logits"], batch["labels_chosen"], chosen_positions
            )
            rejected_logp = _sequence_log_prob(
                rejected["logits"], batch["labels_rejected"], rejected_positions
            )
            with torch.no_grad():
                ref_chosen = _forward(
                    reference, batch, "chosen", chosen_positions
                )
                ref_chosen_logp = _sequence_log_prob(
                    ref_chosen["logits"],
                    batch["labels_chosen"],
                    chosen_positions,
                )
                del ref_chosen
                ref_rejected = _forward(
                    reference, batch, "rejected", rejected_positions
                )
                ref_rejected_logp = _sequence_log_prob(
                    ref_rejected["logits"],
                    batch["labels_rejected"],
                    rejected_positions,
                )
                del ref_rejected

            preference_logit = beta * (
                (chosen_logp - rejected_logp)
                - (ref_chosen_logp - ref_rejected_logp)
            )
            loss_dpo = -F.logsigmoid(preference_logit).mean()
            chosen_token_count = batch["label_mask_chosen"].sum(dim=1).clamp_min(1.0)
            loss_sft = (-(chosen_logp / chosen_token_count)).mean()
            delta_hat = {
                "delta_q": chosen["delta_q"],
                "delta_a": chosen["delta_a"],
                "delta_p": chosen["delta_p"],
            }
            for key in ("delta_q_raw", "delta_a_raw", "delta_p_raw"):
                if key in chosen:
                    delta_hat[key] = chosen[key]
            if "q_cue_logits" in chosen:
                delta_hat["q_cue_logits"] = chosen["q_cue_logits"]
            delta_target = {
                "delta_q": batch["delta_q_target"],
                "delta_a": batch["delta_a_target"],
                "delta_p": batch["delta_p_target"],
                "q_current": batch["q_current"],
                "q_geometry_cues": batch["q_geometry_cues"],
                "q_geometry_mask": batch["q_geometry_mask"],
            }
            loss_ctl, _ = loss_helper.compute_phase1_total(
                delta_hat, delta_target, phase1_lambda_ctl=1.0
            )
            loss = (
                loss_dpo
                + sft_anchor * loss_sft
                + control_anchor * loss_ctl
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite multimodal DPO loss")
            if max_micro_steps is None:
                window_size = accumulation
            else:
                window_start = (micro_step // accumulation) * accumulation
                window_size = min(accumulation, max_micro_steps - window_start)
            (loss / window_size).backward()
            metric_sums["loss"] += float(loss.detach().item())
            metric_sums["loss_dpo"] += float(loss_dpo.detach().item())
            metric_sums["loss_sft"] += float(loss_sft.detach().item())
            metric_sums["loss_ctl"] += float(loss_ctl.detach().item())
            metric_sums["dpo_accuracy"] += float(
                (preference_logit.detach() > 0.0).float().mean().item()
            )
            metric_count += 1
            micro_step += 1
            is_full_window = micro_step % accumulation == 0
            is_final_short_window = (
                max_micro_steps is not None and micro_step == max_micro_steps
            )
            if not (is_full_window or is_final_short_window):
                continue

            torch.nn.utils.clip_grad_norm_(
                lora_parameters, float(cfg["hardware"].get("max_grad_norm", 1.0))
            )
            torch.nn.utils.clip_grad_norm_(
                projection_parameters,
                float(cfg["hardware"].get("max_grad_norm", 1.0)),
            )
            if control_offset_parameters:
                torch.nn.utils.clip_grad_norm_(
                    control_offset_parameters,
                    float(
                        cfg["hardware"].get("max_grad_norm", 1.0)
                    ),
                )
            step_lora_lr = float(optimizer.param_groups[0]["lr"])
            step_projection_lr = float(optimizer.param_groups[1]["lr"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            window_metrics = {
                key: value / metric_count for key, value in metric_sums.items()
            }
            metric_sums = {
                "loss": 0.0,
                "loss_dpo": 0.0,
                "loss_sft": 0.0,
                "loss_ctl": 0.0,
                "dpo_accuracy": 0.0,
            }
            metric_count = 0
            progress.update(1)
            progress.write(
                f"step={global_step} micro_step={micro_step} "
                f"loss={window_metrics['loss']:.6f} "
                f"loss_dpo={window_metrics['loss_dpo']:.6f} "
                f"loss_sft={window_metrics['loss_sft']:.6f} "
                f"loss_ctl={window_metrics['loss_ctl']:.6f} "
                f"dpo_accuracy={window_metrics['dpo_accuracy']:.6f} "
                f"lr_lora={step_lora_lr:.9g} "
                f"lr_projection={step_projection_lr:.9g}"
            )
            if global_step % checkpoint_interval == 0:
                checkpoint_metadata = _dpo_checkpoint_metadata(
                    stage1_metadata,
                    runtime_metadata,
                    global_step=global_step,
                    micro_step=micro_step,
                )
                _save(
                    policy,
                    checkpoint_root / f"{checkpoint_prefix}{global_step}",
                    checkpoint_metadata,
                )
                removed_checkpoints = rotate_step_checkpoints(
                    checkpoint_root,
                    prefix=checkpoint_prefix,
                    save_total_limit=checkpoint_limit,
                )
                for removed_checkpoint in removed_checkpoints:
                    progress.write(f"removed_old_checkpoint={removed_checkpoint}")
            if global_step >= max_steps:
                break
    progress.close()
    _save(
        policy,
        output_dir,
        _dpo_checkpoint_metadata(
            stage1_metadata,
            runtime_metadata,
            global_step=global_step,
            micro_step=micro_step,
        ),
    )
    print(f"OK: multimodal DPO complete\n  final_checkpoint: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train image-conditioned multimodal DPO")
    parser.add_argument("--config", default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="optimizer-step override; by default derive updates from configured epochs",
    )
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--projection_lr", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--sft_anchor", type=float, default=None)
    parser.add_argument("--control_anchor", type=float, default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    train_multimodal_dpo(parser.parse_args())


if __name__ == "__main__":
    main()
