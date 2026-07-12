#!/usr/bin/env python
"""
BEV-image 多模态烟雾测试 checkpoint 的 delta 输出诊断脚本。

该脚本复用现有多模态数据集/模型前向传播路径，统计
delta_q / delta_a / delta_p 的跨样本多样性。它不运行 SCA-FP，
因此成本远低于完整评估。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.multimodal_dataset import MultimodalSFTDataset
from src.model import Gemma3MultimodalISAC, build_proj_head_config


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _entropy_from_probs(probs: np.ndarray, axis: int = 0) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum(axis=axis, keepdims=True)
    return -(probs * np.log(probs)).sum(axis=axis)


def _summarize_tensor(name: str, values: np.ndarray) -> Dict:
    flat = values.reshape(values.shape[0], -1)
    per_dim_std = flat.std(axis=0)
    per_sample_norm = np.linalg.norm(flat, axis=1)
    return {
        f"{name}_shape": list(values.shape),
        f"{name}_mean": float(flat.mean()),
        f"{name}_std_all": float(flat.std()),
        f"{name}_mean_abs": float(np.abs(flat).mean()),
        f"{name}_per_dim_std_mean": float(per_dim_std.mean()),
        f"{name}_per_dim_std_max": float(per_dim_std.max()),
        f"{name}_per_sample_norm_mean": float(per_sample_norm.mean()),
        f"{name}_per_sample_norm_std": float(per_sample_norm.std()),
        f"{name}_min": float(flat.min()),
        f"{name}_max": float(flat.max()),
    }


def _summarize_deltas(delta_q: np.ndarray, delta_a: np.ndarray, delta_p: np.ndarray) -> Dict:
    summary = {}
    summary.update(_summarize_tensor("delta_q", delta_q))
    summary.update(_summarize_tensor("delta_a", delta_a))
    summary.update(_summarize_tensor("delta_p", delta_p))

    # 关联矩阵的 argmax 如果长期不变，说明模型还没有学到“按场景换 UAV”的能力。
    assoc_choice = np.argmax(delta_a, axis=1)
    assoc_unique_counts = [
        int(np.unique(assoc_choice[:, k]).size)
        for k in range(assoc_choice.shape[1])
    ]
    summary["delta_a_argmax_unique_per_user_mean"] = float(np.mean(assoc_unique_counts))
    summary["delta_a_argmax_unique_per_user_min"] = int(np.min(assoc_unique_counts))
    summary["delta_a_argmax_unique_per_user_max"] = int(np.max(assoc_unique_counts))

    assoc_entropy = _entropy_from_probs(delta_a, axis=1)
    summary["delta_a_entropy_mean"] = float(assoc_entropy.mean())
    summary["delta_a_entropy_std"] = float(assoc_entropy.std())

    p = np.clip(delta_p, 0.0, None)
    p_sum = p.sum(axis=2, keepdims=True)
    p_norm = p / np.maximum(p_sum, 1e-12)
    power_entropy = _entropy_from_probs(p_norm, axis=2)
    summary["delta_p_entropy_mean"] = float(power_entropy.mean())
    summary["delta_p_entropy_std"] = float(power_entropy.std())

    warnings = []
    if summary["delta_q_per_dim_std_mean"] < 1e-3:
        warnings.append("delta_q_low_cross_sample_variance")
    if summary["delta_a_per_dim_std_mean"] < 1e-3:
        warnings.append("delta_a_low_cross_sample_variance")
    if summary["delta_p_per_dim_std_mean"] < 1e-4:
        warnings.append("delta_p_low_cross_sample_variance")
    if summary["delta_a_argmax_unique_per_user_mean"] <= 1.2:
        warnings.append("delta_a_argmax_nearly_constant")
    summary["warnings"] = warnings
    return summary


def _move_batch(batch, device):
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def _load_projection_head(model: Gemma3MultimodalISAC, checkpoint: str):
    if not checkpoint:
        return None
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "projection_head.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"projection_head checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.projection_head.load_state_dict(state)
    return str(ckpt_path)


def _collect_deltas(
    model: Gemma3MultimodalISAC,
    dataset: MultimodalSFTDataset,
    num_samples: int,
) -> Dict[str, np.ndarray]:
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = model.device

    delta_qs: List[np.ndarray] = []
    delta_as: List[np.ndarray] = []
    delta_ps: List[np.ndarray] = []

    for idx, batch in enumerate(tqdm(dataloader, desc="MM delta inference")):
        if idx >= num_samples:
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
        with torch.no_grad():
            outputs = model(**forward_keys)
        delta_qs.append(_as_np(outputs["delta_q"].squeeze(0)))
        delta_as.append(_as_np(outputs["delta_a"].squeeze(0)))
        delta_ps.append(_as_np(outputs["delta_p"].squeeze(0)))

    return {
        "delta_q": np.stack(delta_qs, axis=0),
        "delta_a": np.stack(delta_as, axis=0),
        "delta_p": np.stack(delta_ps, axis=0),
    }


def main():
    parser = argparse.ArgumentParser(description="分析 BEV-image 多模态 delta 输出")
    parser.add_argument("--config", type=str, default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="覆盖配置文件中的 model.backbone")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="包含 projection_head.pt 的目录，或 projection_head.pt 文件本身")
    parser.add_argument("--name", type=str, default="mm_sft_smoke")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--save_raw", action="store_true")
    args = parser.parse_args()

    with (PROJECT_ROOT / args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    sim_cfg = cfg["simulation"]
    train_cfg = cfg["training"]["sft"]
    data_cfg = cfg["data"]

    data_dir = Path(args.data_dir or data_cfg["output_dir"])
    data_path = data_dir / data_cfg.get("sft_file", "sft_dataset.jsonl")
    model_name = args.model or model_cfg["backbone"]
    max_length = args.max_length or train_cfg["max_seq_length"]

    model = Gemma3MultimodalISAC(
        model_name_or_path=model_name,
        use_4bit=cfg["hardware"].get("use_4bit", True),
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        proj_head_config=build_proj_head_config(model_cfg, sim_cfg),
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        freeze_vision_tower=model_cfg.get("freeze_vision_tower", True),
    )
    loaded_projection = _load_projection_head(model, args.checkpoint)
    model.eval()

    dataset = MultimodalSFTDataset(
        data_path=str(data_path),
        data_dir=str(data_dir),
        processor=model.processor,
        max_length=max_length,
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
    )

    num_samples = min(args.num_samples, len(dataset))
    deltas = _collect_deltas(model, dataset, num_samples)
    summary = _summarize_deltas(deltas["delta_q"], deltas["delta_a"], deltas["delta_p"])

    result = {
        "name": args.name,
        "config": args.config,
        "data_path": str(data_path),
        "model": model_name,
        "checkpoint": args.checkpoint,
        "loaded_projection": loaded_projection,
        "num_samples": num_samples,
        "max_length": max_length,
        "summary": summary,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved summary to {output_path}")

    if args.save_raw:
        raw_path = output_path.with_suffix(".npz")
        np.savez_compressed(
            raw_path,
            delta_q=deltas["delta_q"],
            delta_a=deltas["delta_a"],
            delta_p=deltas["delta_p"],
        )
        print(f"Saved raw deltas to {raw_path}")

    print("\n=== Multimodal Delta Diagnostic Summary ===")
    for key in (
        "delta_q_per_dim_std_mean",
        "delta_a_per_dim_std_mean",
        "delta_p_per_dim_std_mean",
        "delta_a_argmax_unique_per_user_mean",
        "delta_a_entropy_mean",
        "delta_p_entropy_mean",
        "warnings",
    ):
        print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    main()
