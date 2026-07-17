#!/usr/bin/env python
"""
分析 oracle delta_q 与 multimodal prompt 中几何提示的方向对齐程度。

用途：
1. 不加载大模型、不需要 GPU，直接检查当前 prompt/image 线索是否足以解释 q 方向。
2. 可选读取 analyze_mm_delta_outputs.py --save_raw 生成的 npz，对比模型预测 q 与 oracle q。
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml


UAV_LINE_RE = re.compile(r"^\s*UAV\s+(\d+):")
CUE_PATTERNS = {
    "weighted_center": re.compile(r"weighted_center:d=([\d.eE+-]+)m,dir=\[([^\]]+)\]"),
    "nearest_user": re.compile(r"nearest_user=u(\d+):d=([\d.eE+-]+)m,dir=\[([^\]]+)\]"),
    "nearest_target": re.compile(r"nearest_target=t(\d+):d=([\d.eE+-]+)m,dir=\[([^\]]+)\]"),
}


def _load_jsonl(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _parse_vec2(text: str) -> np.ndarray:
    values = [float(v.strip()) for v in text.split(",")]
    if len(values) != 2:
        raise ValueError(f"Expected 2D vector, got: {text}")
    return np.asarray(values, dtype=np.float32)


def _parse_prompt_geometry(prompt: str, num_uavs: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    dirs = {
        "weighted_center": np.full((num_uavs, 2), np.nan, dtype=np.float32),
        "nearest_user": np.full((num_uavs, 2), np.nan, dtype=np.float32),
        "nearest_target": np.full((num_uavs, 2), np.nan, dtype=np.float32),
    }
    dists = {
        "weighted_center": np.full((num_uavs,), np.nan, dtype=np.float32),
        "nearest_user": np.full((num_uavs,), np.nan, dtype=np.float32),
        "nearest_target": np.full((num_uavs,), np.nan, dtype=np.float32),
    }

    for line in prompt.splitlines():
        m = UAV_LINE_RE.match(line)
        if not m:
            continue
        uav_idx = int(m.group(1))
        if uav_idx < 0 or uav_idx >= num_uavs:
            continue

        match = CUE_PATTERNS["weighted_center"].search(line)
        if match:
            dists["weighted_center"][uav_idx] = float(match.group(1))
            dirs["weighted_center"][uav_idx] = _parse_vec2(match.group(2))

        match = CUE_PATTERNS["nearest_user"].search(line)
        if match:
            dists["nearest_user"][uav_idx] = float(match.group(2))
            dirs["nearest_user"][uav_idx] = _parse_vec2(match.group(3))

        match = CUE_PATTERNS["nearest_target"].search(line)
        if match:
            dists["nearest_target"][uav_idx] = float(match.group(2))
            dirs["nearest_target"][uav_idx] = _parse_vec2(match.group(3))

    return dirs, dists


def _normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    valid = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
    out = np.full(a.shape[:-1], np.nan, dtype=np.float32)
    if valid.any():
        aa = _normalize(a[valid])
        bb = _normalize(b[valid])
        out[valid] = np.sum(aa * bb, axis=-1)
    return out


def _safe_stats(prefix: str, values: np.ndarray) -> Dict:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_valid_count": 0,
        }
    return {
        f"{prefix}_mean": float(valid.mean()),
        f"{prefix}_std": float(valid.std()),
        f"{prefix}_min": float(valid.min()),
        f"{prefix}_max": float(valid.max()),
        f"{prefix}_valid_count": int(valid.size),
    }


def _summarize_norm(prefix: str, values: np.ndarray) -> Dict:
    norms = np.linalg.norm(values, axis=-1)
    flat = norms.reshape(-1)
    return {
        f"{prefix}_norm_mean": float(flat.mean()),
        f"{prefix}_norm_std": float(flat.std()),
        f"{prefix}_norm_min": float(flat.min()),
        f"{prefix}_norm_max": float(flat.max()),
    }


def _load_predictions(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if not path:
        return None
    data = np.load(path)
    result = {}
    for key in ("delta_q", "delta_q_raw"):
        if key in data:
            result[key] = np.asarray(data[key], dtype=np.float32)
    return result


def main():
    parser = argparse.ArgumentParser(description="分析 q 目标方向与 prompt 几何提示的对齐度")
    parser.add_argument("--config", type=str, default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--sft_file", type=str, default=None)
    parser.add_argument("--prediction_npz", type=str, default=None,
                        help="可选：analyze_mm_delta_outputs.py --save_raw 生成的 npz")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with (PROJECT_ROOT / args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    data_dir = Path(args.data_dir or data_cfg["output_dir"])
    data_path = data_dir / (args.sft_file or data_cfg.get("sft_file", "sft_dataset.jsonl"))
    records = _load_jsonl(data_path)

    delta_q = np.asarray([r["delta_q"] for r in records], dtype=np.float32)
    num_samples, num_uavs, _ = delta_q.shape
    target_xy_dir = _normalize(delta_q[..., :2])

    cue_dirs = {name: [] for name in CUE_PATTERNS}
    cue_dists = {name: [] for name in CUE_PATTERNS}
    prompt_types = Counter()
    parse_ok = 0

    for record in records:
        prompt_types[str(record.get("prompt_type", "unknown"))] += 1
        dirs, dists = _parse_prompt_geometry(record.get("prompt", ""), num_uavs)
        if all(np.isfinite(dirs[name]).all() for name in cue_dirs):
            parse_ok += 1
        for name in cue_dirs:
            cue_dirs[name].append(dirs[name])
            cue_dists[name].append(dists[name])

    cue_dirs = {name: np.stack(values, axis=0) for name, values in cue_dirs.items()}
    cue_dists = {name: np.stack(values, axis=0) for name, values in cue_dists.items()}
    cue_cosines = {name: _cosine(target_xy_dir, cue_dirs[name]) for name in cue_dirs}
    stacked_cos = np.stack([cue_cosines[name] for name in cue_cosines], axis=-1)
    valid_any = np.isfinite(stacked_cos).any(axis=-1)
    stacked_for_argmax = np.where(np.isfinite(stacked_cos), stacked_cos, -np.inf)
    best_idx = np.argmax(stacked_for_argmax, axis=-1)
    best_cos = np.full(valid_any.shape, np.nan, dtype=np.float32)
    best_cos[valid_any] = np.max(stacked_for_argmax[valid_any], axis=-1)
    cue_names = list(cue_cosines.keys())
    best_hist = Counter(cue_names[int(i)] for i in best_idx[valid_any].reshape(-1))

    summary = {
        "data_path": str(data_path),
        "num_samples": int(num_samples),
        "num_uavs": int(num_uavs),
        "prompt_type_counts": dict(prompt_types),
        "prompt_geometry_parse_rate": float(parse_ok / max(num_samples, 1)),
        **_summarize_norm("target_delta_q_3d", delta_q),
        **_summarize_norm("target_delta_q_xy", delta_q[..., :2]),
        "target_delta_q_abs_dh_mean": float(np.abs(delta_q[..., 2]).mean()),
        "target_delta_q_near_15m_ratio": float((np.linalg.norm(delta_q, axis=-1) > 14.5).mean()),
        "best_geometry_cue_hist": dict(best_hist),
    }

    for name, cos in cue_cosines.items():
        summary.update(_safe_stats(f"target_q_vs_{name}_xy_cosine", cos))
        summary.update(_safe_stats(f"{name}_distance_m", cue_dists[name]))
    summary.update(_safe_stats("target_q_vs_best_geometry_xy_cosine", best_cos))

    predictions = _load_predictions(args.prediction_npz)
    if predictions:
        n = min(num_samples, next(iter(predictions.values())).shape[0])
        target_3d_dir = _normalize(delta_q[:n])
        target_xy_dir_n = _normalize(delta_q[:n, :, :2])
        summary["prediction_npz"] = args.prediction_npz
        summary["prediction_compare_num_samples"] = int(n)
        for name, pred in predictions.items():
            pred = pred[:n]
            summary.update(_summarize_norm(f"pred_{name}_3d", pred))
            summary.update(_safe_stats(
                f"pred_{name}_vs_target_q_3d_cosine",
                _cosine(_normalize(pred), target_3d_dir),
            ))
            summary.update(_safe_stats(
                f"pred_{name}_vs_target_q_xy_cosine",
                _cosine(_normalize(pred[..., :2]), target_xy_dir_n),
            ))

    warnings = []
    if summary["prompt_geometry_parse_rate"] < 0.95:
        warnings.append("prompt_geometry_parse_rate_low")
    best_mean = summary.get("target_q_vs_best_geometry_xy_cosine_mean")
    if best_mean is not None and best_mean < 0.35:
        warnings.append("target_q_weakly_explained_by_prompt_geometry")
    summary["warnings"] = warnings

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved q geometry alignment summary to {output_path}")

    print("\n=== Q Geometry Alignment Summary ===")
    for key in (
        "num_samples",
        "prompt_geometry_parse_rate",
        "target_delta_q_3d_norm_mean",
        "target_delta_q_near_15m_ratio",
        "target_q_vs_weighted_center_xy_cosine_mean",
        "target_q_vs_nearest_user_xy_cosine_mean",
        "target_q_vs_nearest_target_xy_cosine_mean",
        "target_q_vs_best_geometry_xy_cosine_mean",
        "best_geometry_cue_hist",
    ):
        print(f"  {key}: {summary[key]}")

    if predictions:
        print("\n=== Prediction Q Alignment ===")
        for key in (
            "pred_delta_q_vs_target_q_3d_cosine_mean",
            "pred_delta_q_vs_target_q_xy_cosine_mean",
            "pred_delta_q_raw_vs_target_q_3d_cosine_mean",
            "pred_delta_q_raw_vs_target_q_xy_cosine_mean",
        ):
            if key in summary:
                print(f"  {key}: {summary[key]}")

    print(f"  warnings: {warnings}")


if __name__ == "__main__":
    main()
