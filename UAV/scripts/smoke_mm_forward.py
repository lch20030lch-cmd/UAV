#!/usr/bin/env python
"""
Single-batch BEV-image Gemma3 forward smoke.

This verifies the minimal model-facing multimodal loop:
  data + image -> processor/dataset -> Gemma3 multimodal forward
  -> control-token states -> projection head -> delta_q/a/p
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.multimodal_dataset import MultimodalSFTDataset
from src.model import Gemma3MultimodalISAC, build_proj_head_config


def _move_batch(batch, device):
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description="Run one multimodal model forward smoke")
    parser.add_argument("--config", type=str, default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="Override model.backbone from config")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    with (PROJECT_ROOT / args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    sim_cfg = cfg["simulation"]
    train_cfg = cfg["training"]["sft"]
    data_cfg = cfg["data"]

    model_name = args.model or model_cfg["backbone"]
    data_dir = Path(args.data_dir or data_cfg["output_dir"])
    data_path = data_dir / data_cfg.get("sft_file", "sft_dataset.jsonl")
    max_length = args.max_length or train_cfg["max_seq_length"]

    model = Gemma3MultimodalISAC(
        model_name_or_path=model_name,
        use_4bit=cfg["hardware"].get("use_4bit", True) and not args.no_4bit,
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        proj_head_config=build_proj_head_config(model_cfg, sim_cfg),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        freeze_vision_tower=model_cfg.get("freeze_vision_tower", True),
    )
    model.eval()

    dataset = MultimodalSFTDataset(
        data_path=str(data_path),
        data_dir=str(data_dir),
        processor=model.processor,
        max_length=max_length,
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    batch = _move_batch(batch, model.device)

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

    with torch.no_grad():
        outputs = model(**forward_keys)

    print("OK: multimodal model forward smoke")
    print(f"  data: {data_path}")
    print(f"  max_length: {max_length}")
    print(f"  input_ids: {tuple(batch['input_ids'].shape)}")
    print(f"  attention_mask: {tuple(batch['attention_mask'].shape)}")
    if "pixel_values" in batch:
        print(f"  pixel_values: {tuple(batch['pixel_values'].shape)}")
    print(f"  control_token_count: {int(batch['control_mask'].sum().item())}")
    print(f"  control_states: {tuple(outputs['control_states'].shape)}")
    print(f"  delta_q: {tuple(outputs['delta_q'].shape)}")
    print(f"  delta_a: {tuple(outputs['delta_a'].shape)}")
    print(f"  delta_p: {tuple(outputs['delta_p'].shape)}")

    for name in ["delta_q", "delta_a", "delta_p"]:
        tensor = outputs[name]
        if torch.isnan(tensor).any():
            raise RuntimeError(f"{name} contains NaN")


if __name__ == "__main__":
    main()
