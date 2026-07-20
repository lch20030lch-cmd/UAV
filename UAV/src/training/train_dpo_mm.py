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
from transformers import get_cosine_schedule_with_warmup, set_seed

from src.data.multimodal_dataset import (
    MultimodalDPODataset,
    validate_multimodal_oracle_contract,
)
from src.data.oracle_contract import (
    checkpoint_dataset_fields,
    validate_checkpoint_dataset_compatibility,
)
from src.model import Gemma3MultimodalISAC, UAVISACLosses, build_proj_head_config
from src.model.gemma_multimodal_isac import is_vision_parameter_name


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
    model.projection_head.load_state_dict(
        torch.load(projection_path, map_location="cpu"), strict=True
    )
    model.load_control_token_embeddings(checkpoint_dir)
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
    policy.base_model.train()
    policy.projection_head.train()

    max_length = int(args.max_length or train_cfg["max_seq_length"])
    dataset = MultimodalDPODataset(
        data_path=str(data_root / data_cfg.get("dpo_file", "dpo_dataset.jsonl")),
        data_dir=str(data_root),
        processor=policy.processor,
        max_length=max_length,
        num_control_tokens=cfg["model"]["control_token"]["num_tokens"],
        use_chat_template=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("per_device_batch_size", 1)),
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    lora_parameters = [
        parameter
        for name, parameter in policy.base_model.named_parameters()
        if parameter.requires_grad
        and "lora_" in name
        and not is_vision_parameter_name(name)
    ]
    projection_parameters = [
        parameter for parameter in policy.projection_head.parameters()
        if parameter.requires_grad
    ]
    if not lora_parameters:
        raise RuntimeError("no trainable language LoRA parameters found")
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "lr": float(args.learning_rate)},
            {"params": projection_parameters, "lr": float(args.projection_lr)},
        ],
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    accumulation = int(
        args.gradient_accumulation_steps
        or train_cfg.get("gradient_accumulation_steps", 1)
    )
    max_steps = int(args.max_steps)
    warmup_steps = int(max_steps * float(train_cfg.get("warmup_ratio", 0.03)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, max_steps
    )
    loss_helper = UAVISACLosses(
        lambda_ctl=1.0,
        lambda_q=cfg["model"]["loss"]["lambda_q"],
        lambda_a=cfg["model"]["loss"]["lambda_a"],
        lambda_p=cfg["model"]["loss"]["lambda_p"],
        lambda_sep=cfg["model"]["loss"]["lambda_sep"],
    )

    device = policy.device
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    progress = tqdm(total=max_steps, desc="MM DPO")
    while global_step < max_steps:
        for raw_batch in dataloader:
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

            preference_logit = float(args.beta) * (
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
            delta_target = {
                "delta_q": batch["delta_q_target"],
                "delta_a": batch["delta_a_target"],
                "delta_p": batch["delta_p_target"],
            }
            loss_ctl, _ = loss_helper.compute_phase1_total(
                delta_hat, delta_target, phase1_lambda_ctl=1.0
            )
            loss = (
                loss_dpo
                + float(args.sft_anchor) * loss_sft
                + float(args.control_anchor) * loss_ctl
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite multimodal DPO loss")
            (loss / accumulation).backward()
            micro_step += 1
            if micro_step % accumulation:
                continue

            torch.nn.utils.clip_grad_norm_(
                lora_parameters, float(cfg["hardware"].get("max_grad_norm", 1.0))
            )
            torch.nn.utils.clip_grad_norm_(
                projection_parameters,
                float(cfg["hardware"].get("max_grad_norm", 1.0)),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            progress.update(1)
            progress.write(
                f"step={global_step} loss={loss.item():.6f} "
                f"loss_dpo={loss_dpo.item():.6f} loss_sft={loss_sft.item():.6f} "
                f"loss_ctl={loss_ctl.item():.6f}"
            )
            if global_step >= max_steps:
                break
    progress.close()
    output_dir = Path(args.output_dir)
    _save(
        policy,
        output_dir,
        {
            **stage1_metadata,
            "stage": "multimodal_dpo",
            "stage1_checkpoint": str(checkpoint_dir),
            "max_steps": max_steps,
            "max_seq_length": max_length,
            "beta": float(args.beta),
            "sft_anchor": float(args.sft_anchor),
            "control_anchor": float(args.control_anchor),
            "use_chat_template": True,
            **checkpoint_dataset_fields(dataset_metadata),
        },
    )
    print(f"OK: multimodal DPO complete\n  final_checkpoint: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train image-conditioned multimodal DPO")
    parser.add_argument("--config", default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--stage1_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--projection_lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft_anchor", type=float, default=0.05)
    parser.add_argument("--control_anchor", type=float, default=0.1)
    train_multimodal_dpo(parser.parse_args())


if __name__ == "__main__":
    main()
