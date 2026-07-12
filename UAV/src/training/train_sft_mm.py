#!/usr/bin/env python
"""
Multimodal SFT smoke for BEV-image Gemma3.

This first smoke is intentionally conservative for RTX 5090 32GB:
  - freeze the Gemma3 multimodal backbone
  - train only the projection head
  - optimize CTL loss only

It verifies the trainable loop after the successful forward smoke:
  dataset -> multimodal forward -> projection head -> control loss
  -> backward -> optimizer step -> checkpoint
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.env_setup import setup_env

setup_env()

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

from src.data.multimodal_dataset import MultimodalSFTDataset
from src.model import Gemma3MultimodalISAC, UAVISACLosses, build_proj_head_config


def _move_batch(batch, device):
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += param.grad.detach().float().norm().item() ** 2
    return total ** 0.5


def _save_projection_smoke(model, save_dir: Path, metadata: dict):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.projection_head.state_dict(), save_dir / "projection_head.pt")
    model.processor.save_pretrained(save_dir / "processor")
    with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def train_mm_sft_smoke(
    config_path: str,
    data_dir: str = None,
    model_path: str = None,
    max_steps: int = None,
    max_length: int = None,
    output_dir: str = None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    model_cfg = cfg["model"]
    sim_cfg = cfg["simulation"]
    train_cfg = cfg["training"]["sft"]
    data_cfg = cfg["data"]

    data_root = Path(data_dir or data_cfg["output_dir"])
    sft_path = data_root / data_cfg.get("sft_file", "sft_dataset.jsonl")
    model_name = model_path or model_cfg["backbone"]
    max_seq_length = int(max_length or train_cfg["max_seq_length"])
    steps_limit = int(max_steps or train_cfg.get("phase1", {}).get("max_steps", 30))
    out_root = Path(output_dir or cfg.get("output_dir", "/root/autodl-tmp/outputs/mm_smoke"))
    ckpt_root = Path(cfg.get("checkpoint_dir", "/root/autodl-tmp/checkpoints/mm_smoke"))
    ckpt_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BEV-image multimodal SFT smoke")
    print("=" * 60)
    print(f"  data:       {sft_path}")
    print(f"  model:      {model_name}")
    print(f"  max_length: {max_seq_length}")
    print(f"  steps:      {steps_limit}")
    print("  trainable:  projection_head only")
    print()

    model = Gemma3MultimodalISAC(
        model_name_or_path=model_name,
        use_4bit=cfg["hardware"].get("use_4bit", True),
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        proj_head_config=build_proj_head_config(model_cfg, sim_cfg),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        freeze_vision_tower=model_cfg.get("freeze_vision_tower", True),
    )

    for param in model.base_model.parameters():
        param.requires_grad = False
    model.base_model.eval()
    model.projection_head.train()

    dataset = MultimodalSFTDataset(
        data_path=str(sft_path),
        data_dir=str(data_root),
        processor=model.processor,
        max_length=max_seq_length,
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["per_device_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    loss_fn = UAVISACLosses(
        lambda_ctl=model_cfg["loss"]["lambda_ctl"],
        lambda_q=model_cfg["loss"]["lambda_q"],
        lambda_a=model_cfg["loss"]["lambda_a"],
        lambda_p=model_cfg["loss"]["lambda_p"],
        lambda_sep=model_cfg["loss"]["lambda_sep"],
    )
    optimizer = torch.optim.AdamW(
        model.projection_head.parameters(),
        lr=train_cfg.get("learning_rate", 2e-4),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    device = model.device
    global_step = 0
    epoch = 0
    pbar = tqdm(total=steps_limit, desc="MM SFT smoke")

    while global_step < steps_limit:
        epoch += 1
        for batch in dataloader:
            if global_step >= steps_limit:
                break
            batch = _move_batch(batch, device)

            forward_keys = {
                key: value for key, value in batch.items()
                if key not in {
                    "labels",
                    "label_mask",
                    "has_q_current",
                    "delta_q_target",
                    "delta_a_target",
                    "delta_p_target",
                }
            }

            outputs = model(**forward_keys)
            delta_hat = {
                "delta_q": outputs["delta_q"],
                "delta_a": outputs["delta_a"],
                "delta_p": outputs["delta_p"],
            }
            delta_target = {
                "delta_q": batch["delta_q_target"],
                "delta_a": batch["delta_a_target"],
                "delta_p": batch["delta_p_target"],
            }
            total_loss, metrics = loss_fn.compute_phase1_total(
                delta_hat=delta_hat,
                delta_target=delta_target,
                phase1_lambda_ctl=train_cfg.get("phase1", {}).get("lambda_ctl", 1.0),
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = _grad_norm(model.projection_head.parameters())
            torch.nn.utils.clip_grad_norm_(
                model.projection_head.parameters(),
                cfg["hardware"].get("max_grad_norm", 1.0),
            )
            optimizer.step()

            global_step += 1
            pbar.update(1)
            pbar.write(
                f"step={global_step} epoch={epoch} "
                f"loss_ctl={metrics['loss_ctl']:.6f} "
                f"loss_total={metrics['loss_total']:.6f} "
                f"grad_norm_proj={grad_norm:.6f}"
            )

            if torch.isnan(total_loss):
                raise RuntimeError("NaN loss detected in multimodal SFT smoke.")

            if global_step % train_cfg.get("save_steps", 10) == 0:
                _save_projection_smoke(
                    model,
                    ckpt_root / f"mm_sft_smoke_step_{global_step}",
                    {
                        "global_step": global_step,
                        "loss_ctl": metrics["loss_ctl"],
                        "loss_total": metrics["loss_total"],
                        "grad_norm_proj": grad_norm,
                        "trainable": "projection_head_only",
                    },
                )

    pbar.close()

    final_dir = out_root / "mm_sft_smoke_final"
    _save_projection_smoke(
        model,
        final_dir,
        {
            "global_step": global_step,
            "max_steps": steps_limit,
            "max_seq_length": max_seq_length,
            "trainable": "projection_head_only",
        },
    )
    print()
    print("OK: multimodal SFT smoke complete")
    print(f"  final_checkpoint: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BEV-image multimodal SFT smoke")
    parser.add_argument("--config", type=str, default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    train_mm_sft_smoke(
        config_path=args.config,
        data_dir=args.data_dir,
        model_path=args.model,
        max_steps=args.max_steps,
        max_length=args.max_length,
        output_dir=args.output_dir,
    )
