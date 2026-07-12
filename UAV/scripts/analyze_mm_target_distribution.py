#!/usr/bin/env python
"""
分析多模态 smoke 数据中的 oracle delta 标签分布。

该脚本不加载大模型，主要回答一个问题：
当前 association argmax 固定，到底是模型塌缩，还是 oracle 标签本身就很单一？
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml


def _load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _entropy_from_probs(probs: np.ndarray, axis: int = 0) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum(axis=axis, keepdims=True)
    return -(probs * np.log(probs)).sum(axis=axis)


def _summarize_tensor(name: str, values: np.ndarray) -> Dict:
    flat = values.reshape(values.shape[0], -1)
    per_dim_std = flat.std(axis=0)
    return {
        f"{name}_shape": list(values.shape),
        f"{name}_mean": float(flat.mean()),
        f"{name}_std_all": float(flat.std()),
        f"{name}_mean_abs": float(np.abs(flat).mean()),
        f"{name}_per_dim_std_mean": float(per_dim_std.mean()),
        f"{name}_per_dim_std_max": float(per_dim_std.max()),
        f"{name}_min": float(flat.min()),
        f"{name}_max": float(flat.max()),
    }


def _summarize_association(prefix: str, delta_a: np.ndarray) -> Dict:
    # delta_a shape: (N, M, K)，axis=1 上 argmax 表示每个用户选择哪个 UAV。
    assoc_choice = np.argmax(delta_a, axis=1)
    num_samples, num_users = assoc_choice.shape

    unique_counts = []
    dominant_ratios = []
    per_user_hist = []
    for user_idx in range(num_users):
        choices = assoc_choice[:, user_idx].tolist()
        counts = Counter(choices)
        unique_counts.append(len(counts))
        dominant_ratios.append(max(counts.values()) / num_samples)
        per_user_hist.append({str(k): int(v) for k, v in sorted(counts.items())})

    flat_hist = Counter(assoc_choice.reshape(-1).tolist())
    assoc_entropy = _entropy_from_probs(delta_a, axis=1)

    return {
        f"{prefix}_argmax_unique_per_user_mean": float(np.mean(unique_counts)),
        f"{prefix}_argmax_unique_per_user_min": int(np.min(unique_counts)),
        f"{prefix}_argmax_unique_per_user_max": int(np.max(unique_counts)),
        f"{prefix}_argmax_fixed_user_count": int(sum(v == 1 for v in unique_counts)),
        f"{prefix}_argmax_dominant_ratio_mean": float(np.mean(dominant_ratios)),
        f"{prefix}_argmax_dominant_ratio_max": float(np.max(dominant_ratios)),
        f"{prefix}_argmax_uav_hist": {str(k): int(v) for k, v in sorted(flat_hist.items())},
        f"{prefix}_argmax_per_user_hist": per_user_hist,
        f"{prefix}_entropy_mean": float(assoc_entropy.mean()),
        f"{prefix}_entropy_std": float(assoc_entropy.std()),
    }


def _load_targets(records: List[Dict]) -> Dict[str, np.ndarray]:
    return {
        "delta_q": np.asarray([r["delta_q"] for r in records], dtype=np.float32),
        "delta_a": np.asarray([r["delta_a"] for r in records], dtype=np.float32),
        "delta_p": np.asarray([r["delta_p"] for r in records], dtype=np.float32),
    }


def _load_predictions(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if not path:
        return None
    data = np.load(path)
    return {
        "delta_q": np.asarray(data["delta_q"], dtype=np.float32),
        "delta_a": np.asarray(data["delta_a"], dtype=np.float32),
        "delta_p": np.asarray(data["delta_p"], dtype=np.float32),
    }


def _compare_argmax(target_delta_a: np.ndarray, pred_delta_a: np.ndarray) -> Dict:
    n = min(target_delta_a.shape[0], pred_delta_a.shape[0])
    target_choice = np.argmax(target_delta_a[:n], axis=1)
    pred_choice = np.argmax(pred_delta_a[:n], axis=1)
    match = target_choice == pred_choice
    per_user_match = match.mean(axis=0)
    return {
        "argmax_compare_num_samples": int(n),
        "argmax_match_rate_mean": float(match.mean()),
        "argmax_match_rate_per_user_mean": float(per_user_match.mean()),
        "argmax_match_rate_per_user_min": float(per_user_match.min()),
        "argmax_match_rate_per_user_max": float(per_user_match.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="分析多模态 smoke 数据的 oracle delta 标签分布")
    parser.add_argument("--config", type=str, default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--sft_file", type=str, default=None)
    parser.add_argument("--prediction_npz", type=str, default=None,
                        help="可选：analyze_mm_delta_outputs.py --save_raw 生成的 npz")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with (PROJECT_ROOT / args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    data_dir = Path(args.data_dir or data_cfg["output_dir"])
    data_path = data_dir / (args.sft_file or data_cfg.get("sft_file", "sft_dataset.jsonl"))
    records = _load_jsonl(data_path)
    targets = _load_targets(records)

    summary = {
        "data_path": str(data_path),
        "num_samples": len(records),
    }
    for name, values in targets.items():
        summary.update(_summarize_tensor(f"target_{name}", values))
    summary.update(_summarize_association("target_delta_a", targets["delta_a"]))

    predictions = _load_predictions(args.prediction_npz)
    if predictions is not None:
        for name, values in predictions.items():
            summary.update(_summarize_tensor(f"pred_{name}", values))
        summary.update(_summarize_association("pred_delta_a", predictions["delta_a"]))
        summary.update(_compare_argmax(targets["delta_a"], predictions["delta_a"]))
        summary["prediction_npz"] = args.prediction_npz

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved target distribution summary to {output_path}")

    print("\n=== Multimodal Target Distribution Summary ===")
    for key in (
        "num_samples",
        "target_delta_q_per_dim_std_mean",
        "target_delta_a_per_dim_std_mean",
        "target_delta_p_per_dim_std_mean",
        "target_delta_a_argmax_unique_per_user_mean",
        "target_delta_a_argmax_fixed_user_count",
        "target_delta_a_argmax_dominant_ratio_mean",
        "target_delta_a_entropy_mean",
    ):
        print(f"  {key}: {summary[key]}")

    if predictions is not None:
        print("\n=== Prediction vs Target Association Argmax ===")
        for key in (
            "pred_delta_a_argmax_unique_per_user_mean",
            "pred_delta_a_argmax_fixed_user_count",
            "argmax_match_rate_mean",
            "argmax_match_rate_per_user_min",
            "argmax_match_rate_per_user_max",
        ):
            print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    main()
