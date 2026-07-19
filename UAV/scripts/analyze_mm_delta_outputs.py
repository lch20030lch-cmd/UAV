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


def _summarize_association(prefix: str, delta_a: np.ndarray) -> Dict:
    assoc_choice = np.argmax(delta_a, axis=1)
    assoc_unique_counts = [
        int(np.unique(assoc_choice[:, k]).size)
        for k in range(assoc_choice.shape[1])
    ]
    assoc_entropy = _entropy_from_probs(delta_a, axis=1)
    return {
        f"{prefix}_argmax_unique_per_user_mean": float(np.mean(assoc_unique_counts)),
        f"{prefix}_argmax_unique_per_user_min": int(np.min(assoc_unique_counts)),
        f"{prefix}_argmax_unique_per_user_max": int(np.max(assoc_unique_counts)),
        f"{prefix}_argmax_fixed_user_count": int(sum(v == 1 for v in assoc_unique_counts)),
        f"{prefix}_entropy_mean": float(assoc_entropy.mean()),
        f"{prefix}_entropy_std": float(assoc_entropy.std()),
    }


def _summarize_association_alignment(
    delta_a: np.ndarray,
    delta_a_target: np.ndarray,
) -> Dict:
    """对比预测关联与 oracle，并给出固定用户多数选择基线。"""
    if delta_a.shape != delta_a_target.shape:
        raise ValueError(
            "delta_a prediction/target shapes differ: "
            f"{delta_a.shape} != {delta_a_target.shape}"
        )
    if delta_a.ndim != 3:
        raise ValueError(f"delta_a must have shape (N, M, K), got {delta_a.shape}")

    pred_idx = np.argmax(delta_a, axis=1)
    target_idx = np.argmax(delta_a_target, axis=1)
    accuracy = float((pred_idx == target_idx).mean())

    num_samples, num_uavs, num_users = delta_a.shape
    majority_correct = 0
    for user_idx in range(num_users):
        counts = np.bincount(target_idx[:, user_idx], minlength=num_uavs)
        majority_correct += int(counts.max())
    fixed_user_majority_accuracy = float(
        majority_correct / max(num_samples * num_users, 1)
    )

    probs = np.clip(delta_a.astype(np.float64), 0.0, None)
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    oracle_probs = np.take_along_axis(
        probs,
        target_idx[:, None, :],
        axis=1,
    ).squeeze(1)
    return {
        "delta_a_argmax_accuracy": accuracy,
        "delta_a_fixed_user_majority_accuracy": fixed_user_majority_accuracy,
        "delta_a_accuracy_gain_over_fixed_user_majority": (
            accuracy - fixed_user_majority_accuracy
        ),
        "delta_a_oracle_probability_mean": float(oracle_probs.mean()),
        "delta_a_oracle_probability_std": float(oracle_probs.std()),
    }


def _summarize_q_alignment(
    delta_q: np.ndarray,
    delta_q_target: np.ndarray,
    q_max_norm: float = None,
) -> Dict:
    """统计投影后 Q 的方向、位移范数与移动约束。"""
    if delta_q.shape != delta_q_target.shape:
        raise ValueError(
            "delta_q prediction/target shapes differ: "
            f"{delta_q.shape} != {delta_q_target.shape}"
        )
    if delta_q.ndim != 3 or delta_q.shape[-1] != 3:
        raise ValueError(f"delta_q must have shape (N, M, 3), got {delta_q.shape}")

    pred_norm = np.linalg.norm(delta_q, axis=-1)
    target_norm = np.linalg.norm(delta_q_target, axis=-1)
    pred_dir = delta_q / np.maximum(pred_norm[..., None], 1e-8)
    target_dir = delta_q_target / np.maximum(target_norm[..., None], 1e-8)
    cosine_3d = (pred_dir * target_dir).sum(axis=-1)

    pred_xy = delta_q[..., :2]
    target_xy = delta_q_target[..., :2]
    pred_xy_dir = pred_xy / np.maximum(
        np.linalg.norm(pred_xy, axis=-1, keepdims=True),
        1e-8,
    )
    target_xy_dir = target_xy / np.maximum(
        np.linalg.norm(target_xy, axis=-1, keepdims=True),
        1e-8,
    )
    cosine_xy = (pred_xy_dir * target_xy_dir).sum(axis=-1)
    flat_pred_dir = pred_dir.reshape(pred_dir.shape[0], -1)
    flat_target_dir = target_dir.reshape(target_dir.shape[0], -1)

    result = {
        "delta_q_norm_mean": float(pred_norm.mean()),
        "delta_q_norm_std": float(pred_norm.std()),
        "delta_q_target_norm_mean": float(target_norm.mean()),
        "delta_q_target_norm_std": float(target_norm.std()),
        "delta_q_norm_mae": float(np.abs(pred_norm - target_norm).mean()),
        "delta_q_vs_target_3d_cosine_mean": float(cosine_3d.mean()),
        "delta_q_vs_target_3d_cosine_std": float(cosine_3d.std()),
        "delta_q_vs_target_xy_cosine_mean": float(cosine_xy.mean()),
        "delta_q_vs_target_xy_cosine_std": float(cosine_xy.std()),
        "delta_q_direction_per_dim_std_mean": float(
            flat_pred_dir.std(axis=0).mean()
        ),
        "delta_q_target_direction_per_dim_std_mean": float(
            flat_target_dir.std(axis=0).mean()
        ),
    }
    if q_max_norm is not None:
        tolerance = max(1e-4, float(q_max_norm) * 1e-4)
        result["delta_q_mobility_violation_ratio"] = float(
            (pred_norm > float(q_max_norm) + tolerance).mean()
        )
        result["delta_q_near_max_radius_ratio"] = float(
            (np.abs(pred_norm - float(q_max_norm)) <= 0.1).mean()
        )
    return result


def _summarize_q_cues(
    q_cue_logits: np.ndarray,
    q_geometry_cues: np.ndarray,
    delta_q_target: np.ndarray,
) -> Dict:
    target_xy = delta_q_target[..., :2]
    target_dir = target_xy / np.maximum(np.linalg.norm(target_xy, axis=-1, keepdims=True), 1e-8)
    cue_dir = q_geometry_cues / np.maximum(np.linalg.norm(q_geometry_cues, axis=-1, keepdims=True), 1e-8)
    cue_cos = (cue_dir * target_dir[:, :, None, :]).sum(axis=-1)
    target_idx = np.argmax(cue_cos, axis=-1)
    pred_idx = np.argmax(q_cue_logits, axis=-1)
    pred_probs = np.exp(q_cue_logits - q_cue_logits.max(axis=-1, keepdims=True))
    pred_probs = pred_probs / np.maximum(pred_probs.sum(axis=-1, keepdims=True), 1e-12)
    names = ["weighted_center", "nearest_user", "nearest_target"]
    target_hist = {names[i]: int((target_idx == i).sum()) for i in range(3)}
    pred_hist = {names[i]: int((pred_idx == i).sum()) for i in range(3)}
    chosen_cos = np.take_along_axis(cue_cos, pred_idx[..., None], axis=-1).squeeze(-1)
    best_cos = np.max(cue_cos, axis=-1)
    return {
        "q_cue_accuracy": float((pred_idx == target_idx).mean()),
        "q_cue_pred_hist": pred_hist,
        "q_cue_target_hist": target_hist,
        "q_cue_pred_prob_mean": pred_probs.reshape(-1, 3).mean(axis=0).tolist(),
        "q_cue_chosen_geometry_cosine_mean": float(chosen_cos.mean()),
        "q_cue_best_geometry_cosine_mean": float(best_cos.mean()),
    }


def _summarize_fixed_q_geometry(
    q_geometry_cues: np.ndarray,
    delta_q_target: np.ndarray,
    fixed_weights,
) -> Dict:
    """Evaluate the train-derived fixed geometry mixture without model predictions."""
    if q_geometry_cues.shape[:2] != delta_q_target.shape[:2]:
        raise ValueError("q geometry cues and q targets must align on sample/UAV axes")
    weights = np.asarray(fixed_weights, dtype=np.float32)
    if weights.shape != (3,) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("fixed q cue weights must be three non-negative values with positive sum")
    weights = weights / weights.sum()
    fixed_xy = (q_geometry_cues * weights.reshape(1, 1, 3, 1)).sum(axis=2)
    fixed_xy = fixed_xy / np.maximum(
        np.linalg.norm(fixed_xy, axis=-1, keepdims=True),
        1e-8,
    )
    target_xy = delta_q_target[..., :2]
    target_xy = target_xy / np.maximum(
        np.linalg.norm(target_xy, axis=-1, keepdims=True),
        1e-8,
    )
    fixed_xy_cosine = (fixed_xy * target_xy).sum(axis=-1)

    fixed_3d = np.concatenate([fixed_xy, np.zeros_like(fixed_xy[..., :1])], axis=-1)
    target_3d = delta_q_target / np.maximum(
        np.linalg.norm(delta_q_target, axis=-1, keepdims=True),
        1e-8,
    )
    fixed_3d_cosine = (fixed_3d * target_3d).sum(axis=-1)
    return {
        "q_fixed_geometry_weights": weights.tolist(),
        "q_fixed_geometry_vs_target_xy_cosine_mean": float(fixed_xy_cosine.mean()),
        "q_fixed_geometry_vs_target_3d_cosine_mean": float(fixed_3d_cosine.mean()),
    }


def _summarize_power_alignment(
    delta_p: np.ndarray,
    delta_p_target: np.ndarray,
    delta_a_target: np.ndarray,
) -> Dict:
    """分开统计有效通信、未关联泄漏与感知功率误差。"""
    if delta_p.shape != delta_p_target.shape:
        raise ValueError(
            f"delta_p prediction/target shapes differ: {delta_p.shape} != {delta_p_target.shape}"
        )
    if delta_a_target.shape != delta_p.shape[:-1] + (delta_p.shape[-1] - 1,):
        raise ValueError(
            "delta_a_target must align with communication power entries: "
            f"got {delta_a_target.shape} for delta_p {delta_p.shape}"
        )

    pred_comm = delta_p[..., :-1]
    target_comm = delta_p_target[..., :-1]
    active = delta_a_target > 0.5
    inactive = ~active
    comm_sq_error = (pred_comm - target_comm) ** 2
    sense_sq_error = (delta_p[..., -1:] - delta_p_target[..., -1:]) ** 2

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        selected = values[mask]
        return float(selected.mean()) if selected.size else 0.0

    pred_total = delta_p.sum(axis=-1)
    target_total = delta_p_target.sum(axis=-1)
    return {
        "delta_p_mse": float(((delta_p - delta_p_target) ** 2).mean()),
        "delta_p_active_comm_mse": masked_mean(comm_sq_error, active),
        "delta_p_inactive_comm_mse": masked_mean(comm_sq_error, inactive),
        "delta_p_sensing_mse": float(sense_sq_error.mean()),
        "delta_p_active_comm_pred_mean": masked_mean(pred_comm, active),
        "delta_p_active_comm_target_mean": masked_mean(target_comm, active),
        "delta_p_inactive_power_leakage_mean": masked_mean(pred_comm, inactive),
        "delta_p_total_per_uav_pred_mean": float(pred_total.mean()),
        "delta_p_total_per_uav_target_mean": float(target_total.mean()),
        "delta_p_total_per_uav_mae": float(np.abs(pred_total - target_total).mean()),
    }


def _summarize_deltas(
    delta_q: np.ndarray,
    delta_a: np.ndarray,
    delta_p: np.ndarray,
    delta_q_raw: np.ndarray = None,
    delta_a_raw: np.ndarray = None,
    delta_p_raw: np.ndarray = None,
    delta_q_target: np.ndarray = None,
    delta_a_target: np.ndarray = None,
    delta_p_target: np.ndarray = None,
    q_cue_logits: np.ndarray = None,
    q_geometry_cues: np.ndarray = None,
    q_fixed_cue_weights=None,
    control_states: np.ndarray = None,
    delta_raw: np.ndarray = None,
    q_max_norm: float = None,
) -> Dict:
    summary = {}
    summary.update(_summarize_tensor("delta_q", delta_q))
    summary.update(_summarize_tensor("delta_a", delta_a))
    summary.update(_summarize_tensor("delta_p", delta_p))

    # 关联矩阵的 argmax 如果长期不变，说明模型还没有学到“按场景换 UAV”的能力。
    summary.update(_summarize_association("delta_a", delta_a))
    if delta_q_target is not None:
        summary.update(_summarize_q_alignment(delta_q, delta_q_target, q_max_norm))
    if delta_a_target is not None:
        summary.update(_summarize_association_alignment(delta_a, delta_a_target))
    if delta_q_raw is not None:
        summary.update(_summarize_tensor("delta_q_raw", delta_q_raw))
    if delta_q_raw is not None and delta_q_target is not None:
        pred = delta_q_raw / np.maximum(np.linalg.norm(delta_q_raw, axis=-1, keepdims=True), 1e-8)
        target = delta_q_target / np.maximum(np.linalg.norm(delta_q_target, axis=-1, keepdims=True), 1e-8)
        cos = (pred * target).sum(axis=-1)
        mse = ((pred - target) ** 2).mean(axis=-1)
        summary["delta_q_raw_dir_cosine_mean"] = float(cos.mean())
        summary["delta_q_raw_dir_cosine_std"] = float(cos.std())
        summary["delta_q_raw_dir_mse_mean"] = float(mse.mean())
    if q_cue_logits is not None and q_geometry_cues is not None and delta_q_target is not None:
        summary.update(_summarize_q_cues(q_cue_logits, q_geometry_cues, delta_q_target))
    if (
        q_geometry_cues is not None
        and delta_q_target is not None
        and q_fixed_cue_weights is not None
    ):
        summary.update(
            _summarize_fixed_q_geometry(
                q_geometry_cues,
                delta_q_target,
                q_fixed_cue_weights,
            )
        )
    if delta_a_raw is not None:
        summary.update(_summarize_tensor("delta_a_raw", delta_a_raw))
        summary.update(_summarize_association("delta_a_raw", delta_a_raw))
    if delta_p_raw is not None:
        summary.update(_summarize_tensor("delta_p_raw", delta_p_raw))
    if delta_p_target is not None:
        summary.update(_summarize_tensor("delta_p_target", delta_p_target))
    if delta_p_target is not None and delta_a_target is not None:
        summary.update(_summarize_power_alignment(delta_p, delta_p_target, delta_a_target))
    if control_states is not None:
        summary.update(_summarize_tensor("control_states", control_states))
    if delta_raw is not None:
        summary.update(_summarize_tensor("delta_raw", delta_raw))

    p = np.clip(delta_p, 0.0, None)
    p_sum = p.sum(axis=2, keepdims=True)
    p_norm = p / np.maximum(p_sum, 1e-12)
    power_entropy = _entropy_from_probs(p_norm, axis=2)
    summary["delta_p_entropy_mean"] = float(power_entropy.mean())
    summary["delta_p_entropy_std"] = float(power_entropy.std())

    warnings = []
    if summary["delta_q_per_dim_std_mean"] < 1e-3:
        warnings.append("delta_q_low_cross_sample_variance")
    if summary.get("delta_q_mobility_violation_ratio", 0.0) > 0.0:
        warnings.append("delta_q_mobility_violation")
    if (
        "q_fixed_geometry_vs_target_xy_cosine_mean" in summary
        and summary.get("delta_q_vs_target_xy_cosine_mean", -1.0)
        < summary["q_fixed_geometry_vs_target_xy_cosine_mean"] - 0.01
    ):
        warnings.append("delta_q_below_fixed_geometry_baseline")
    if summary["delta_a_per_dim_std_mean"] < 1e-3:
        warnings.append("delta_a_low_cross_sample_variance")
    if summary["delta_p_per_dim_std_mean"] < 1e-4:
        warnings.append("delta_p_low_cross_sample_variance")
    if summary.get("delta_p_inactive_power_leakage_mean", 0.0) > 1e-2:
        warnings.append("delta_p_inactive_power_leakage")
    if summary["delta_a_argmax_unique_per_user_mean"] <= 1.2:
        warnings.append("delta_a_argmax_nearly_constant")
    if (
        "delta_a_accuracy_gain_over_fixed_user_majority" in summary
        and summary["delta_a_accuracy_gain_over_fixed_user_majority"] <= 0.0
    ):
        warnings.append("delta_a_not_above_fixed_user_majority")
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
    load_result = model.projection_head.load_state_dict(state, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(
            "Projection head loaded with non-strict key match: "
            f"missing={list(load_result.missing_keys)}, "
            f"unexpected={list(load_result.unexpected_keys)}"
        )
    return str(ckpt_path)


def _load_control_token_embeddings(model: Gemma3MultimodalISAC, checkpoint: str):
    if not checkpoint:
        return {}
    ckpt_path = Path(checkpoint)
    ckpt_root = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    return model.load_control_token_embeddings(ckpt_root)


def _resolve_lora_checkpoint(checkpoint: str, lora_checkpoint: str):
    if lora_checkpoint:
        return str(Path(lora_checkpoint))
    if not checkpoint:
        return None
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_file():
        return None
    candidate = ckpt_path / "lora"
    if (candidate / "adapter_config.json").exists():
        return str(candidate)
    return None


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
    delta_q_raws: List[np.ndarray] = []
    delta_a_raws: List[np.ndarray] = []
    delta_p_raws: List[np.ndarray] = []
    delta_q_targets: List[np.ndarray] = []
    delta_a_targets: List[np.ndarray] = []
    delta_p_targets: List[np.ndarray] = []
    q_cue_logits_list: List[np.ndarray] = []
    q_geometry_cues_list: List[np.ndarray] = []
    control_states_list: List[np.ndarray] = []
    delta_raws: List[np.ndarray] = []

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
                "q_geometry_mask",
            }
        }
        with torch.no_grad():
            outputs = model(**forward_keys)
        delta_qs.append(_as_np(outputs["delta_q"].squeeze(0)))
        delta_as.append(_as_np(outputs["delta_a"].squeeze(0)))
        delta_ps.append(_as_np(outputs["delta_p"].squeeze(0)))
        if "delta_q_raw" in outputs:
            delta_q_raws.append(_as_np(outputs["delta_q_raw"].squeeze(0)))
        if "delta_a_raw" in outputs:
            delta_a_raws.append(_as_np(outputs["delta_a_raw"].squeeze(0)))
        if "delta_p_raw" in outputs:
            delta_p_raws.append(_as_np(outputs["delta_p_raw"].squeeze(0)))
        if "delta_q_target" in batch:
            delta_q_targets.append(_as_np(batch["delta_q_target"].squeeze(0)))
        if "delta_a_target" in batch:
            delta_a_targets.append(_as_np(batch["delta_a_target"].squeeze(0)))
        if "delta_p_target" in batch:
            delta_p_targets.append(_as_np(batch["delta_p_target"].squeeze(0)))
        if "q_cue_logits" in outputs:
            q_cue_logits_list.append(_as_np(outputs["q_cue_logits"].squeeze(0)))
        if "q_geometry_cues" in batch:
            q_geometry_cues_list.append(_as_np(batch["q_geometry_cues"].squeeze(0)))
        if "control_states" in outputs:
            control_states_list.append(_as_np(outputs["control_states"].squeeze(0)))
        if "delta_raw" in outputs:
            delta_raws.append(_as_np(outputs["delta_raw"].squeeze(0)))

    result = {
        "delta_q": np.stack(delta_qs, axis=0),
        "delta_a": np.stack(delta_as, axis=0),
        "delta_p": np.stack(delta_ps, axis=0),
    }
    if delta_a_raws:
        result["delta_a_raw"] = np.stack(delta_a_raws, axis=0)
    if delta_q_raws:
        result["delta_q_raw"] = np.stack(delta_q_raws, axis=0)
    if delta_p_raws:
        result["delta_p_raw"] = np.stack(delta_p_raws, axis=0)
    if delta_q_targets:
        result["delta_q_target"] = np.stack(delta_q_targets, axis=0)
    if delta_a_targets:
        result["delta_a_target"] = np.stack(delta_a_targets, axis=0)
    if delta_p_targets:
        result["delta_p_target"] = np.stack(delta_p_targets, axis=0)
    if q_cue_logits_list:
        result["q_cue_logits"] = np.stack(q_cue_logits_list, axis=0)
    if q_geometry_cues_list:
        result["q_geometry_cues"] = np.stack(q_geometry_cues_list, axis=0)
    if control_states_list:
        result["control_states"] = np.stack(control_states_list, axis=0)
    if delta_raws:
        result["delta_raw"] = np.stack(delta_raws, axis=0)
    return result


def main():
    parser = argparse.ArgumentParser(description="分析 BEV-image 多模态 delta 输出")
    parser.add_argument("--config", type=str, default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="覆盖配置文件中的 model.backbone")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="包含 projection_head.pt 的目录，或 projection_head.pt 文件本身")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="LoRA adapter 目录；不填时会尝试从 --checkpoint/lora 自动发现")
    parser.add_argument("--name", type=str, default="mm_sft_smoke")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--projection_head_type", type=str, choices=["shared", "split"], default=None,
                        help="可选 projection head 类型；分析 split checkpoint 时需要传 split")
    parser.add_argument("--q_projection_mode", type=str, choices=["clip", "direction"], default=None,
                        help="可选 q 投影模式；分析 direction checkpoint 时需要传 direction")
    parser.add_argument(
        "--q_geometry_mode",
        type=str,
        choices=["none", "cue_xy", "fixed_residual_xy"],
        default=None,
        help="可选：分析 cue_xy 或 fixed_residual_xy checkpoint 时传入对应模式",
    )
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
    lora_checkpoint = _resolve_lora_checkpoint(args.checkpoint, args.lora_checkpoint)
    proj_head_config = build_proj_head_config(model_cfg, sim_cfg)
    if args.projection_head_type is not None:
        proj_head_config["head_type"] = args.projection_head_type
    if args.q_projection_mode is not None:
        proj_head_config["q_projection_mode"] = args.q_projection_mode
    if args.q_geometry_mode is not None:
        proj_head_config["q_geometry_mode"] = args.q_geometry_mode

    model = Gemma3MultimodalISAC(
        model_name_or_path=model_name,
        use_4bit=cfg["hardware"].get("use_4bit", True),
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        proj_head_config=proj_head_config,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        freeze_vision_tower=model_cfg.get("freeze_vision_tower", True),
        lora_rank=model_cfg["lora"]["rank"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"].get("dropout", 0.0),
        lora_target_modules=model_cfg["lora"]["target_modules"],
        lora_checkpoint=lora_checkpoint,
    )
    loaded_projection = _load_projection_head(model, args.checkpoint)
    loaded_control_embeddings = _load_control_token_embeddings(model, args.checkpoint)
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
    summary = _summarize_deltas(
        delta_q=deltas["delta_q"],
        delta_a=deltas["delta_a"],
        delta_p=deltas["delta_p"],
        delta_q_raw=deltas.get("delta_q_raw"),
        delta_a_raw=deltas.get("delta_a_raw"),
        delta_p_raw=deltas.get("delta_p_raw"),
        delta_q_target=deltas.get("delta_q_target"),
        delta_a_target=deltas.get("delta_a_target"),
        delta_p_target=deltas.get("delta_p_target"),
        q_cue_logits=deltas.get("q_cue_logits"),
        q_geometry_cues=deltas.get("q_geometry_cues"),
        q_fixed_cue_weights=proj_head_config.get("q_fixed_cue_weights"),
        control_states=deltas.get("control_states"),
        delta_raw=deltas.get("delta_raw"),
        q_max_norm=(
            float(sim_cfg["uav_max_speed_ms"])
            * float(sim_cfg["slot_duration_s"])
        ),
    )

    result = {
        "name": args.name,
        "config": args.config,
        "data_path": str(data_path),
        "model": model_name,
        "checkpoint": args.checkpoint,
        "projection_head_type": proj_head_config.get("head_type", "shared"),
        "q_projection_mode": proj_head_config.get("q_projection_mode", "clip"),
        "q_geometry_mode": proj_head_config.get("q_geometry_mode", "none"),
        "loaded_projection": loaded_projection,
        "loaded_control_embeddings": loaded_control_embeddings,
        "lora_checkpoint": lora_checkpoint,
        "loaded_lora_checkpoint": model.loaded_lora_checkpoint,
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
            **deltas,
        )
        print(f"Saved raw deltas to {raw_path}")

    print("\n=== Multimodal Delta Diagnostic Summary ===")
    print(f"  loaded_projection: {loaded_projection}")
    print(f"  loaded_control_embeddings: {loaded_control_embeddings}")
    print(f"  loaded_lora_checkpoint: {model.loaded_lora_checkpoint}")
    for key in (
        "delta_q_per_dim_std_mean",
        "delta_a_per_dim_std_mean",
        "delta_p_per_dim_std_mean",
        "delta_p_raw_per_dim_std_mean",
        "delta_p_target_per_dim_std_mean",
        "delta_p_mse",
        "delta_p_active_comm_mse",
        "delta_p_inactive_comm_mse",
        "delta_p_sensing_mse",
        "delta_p_inactive_power_leakage_mean",
        "delta_p_total_per_uav_pred_mean",
        "delta_p_total_per_uav_target_mean",
        "delta_p_total_per_uav_mae",
        "delta_q_raw_per_dim_std_mean",
        "delta_q_norm_mean",
        "delta_q_target_norm_mean",
        "delta_q_norm_mae",
        "delta_q_near_max_radius_ratio",
        "delta_q_mobility_violation_ratio",
        "delta_q_vs_target_3d_cosine_mean",
        "delta_q_vs_target_xy_cosine_mean",
        "delta_q_direction_per_dim_std_mean",
        "delta_q_target_direction_per_dim_std_mean",
        "q_fixed_geometry_weights",
        "q_fixed_geometry_vs_target_xy_cosine_mean",
        "q_fixed_geometry_vs_target_3d_cosine_mean",
        "delta_q_raw_dir_cosine_mean",
        "delta_q_raw_dir_mse_mean",
        "q_cue_accuracy",
        "q_cue_target_hist",
        "q_cue_pred_hist",
        "q_cue_chosen_geometry_cosine_mean",
        "q_cue_best_geometry_cosine_mean",
        "delta_a_argmax_unique_per_user_mean",
        "delta_a_argmax_fixed_user_count",
        "delta_a_argmax_accuracy",
        "delta_a_fixed_user_majority_accuracy",
        "delta_a_accuracy_gain_over_fixed_user_majority",
        "delta_a_oracle_probability_mean",
        "delta_a_entropy_mean",
        "delta_a_raw_per_dim_std_mean",
        "delta_a_raw_argmax_unique_per_user_mean",
        "delta_a_raw_argmax_fixed_user_count",
        "delta_a_raw_entropy_mean",
        "control_states_per_dim_std_mean",
        "control_states_per_dim_std_max",
        "delta_raw_per_dim_std_mean",
        "delta_raw_per_dim_std_max",
        "delta_p_entropy_mean",
        "warnings",
    ):
        if key in summary:
            print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    main()
