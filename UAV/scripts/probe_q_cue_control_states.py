#!/usr/bin/env python
"""用缓存的 control states 诊断 q-cue 选择信息是否可读。

该脚本读取 ``analyze_mm_delta_outputs.py --save_raw`` 生成的 NPZ，
不加载 Gemma，也不重新编码图像。它复用线上 q-cue 头的 ControlReadout
结构，在训练环境上拟合，并在未参与训练的样本上评估；既支持同一 NPZ
内部切分，也支持用 ``--validation_npz`` 指定独立 seed 的验证集：

1. frozen control states 能否被当前 q-cue 头读出；
2. dynamic cue mixture 是否优于 fixed / shuffled mixture；
3. hard cue CE 的多数类基线与真实训练准确率之间有多大差距。
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.projection_head import ControlReadout


CUE_NAMES = ("weighted_center", "nearest_user", "nearest_target")


class CompactCueReadout(nn.Module):
    """保留 per-UAV attention query，但用小瓶颈替代 2560→1280 大读出。"""

    def __init__(
        self,
        hidden_dim: int,
        num_queries: int,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}")
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.attn_queries = nn.Parameter(torch.empty(1, num_queries, hidden_dim))
        nn.init.normal_(self.attn_queries, std=0.02)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.LayerNorm(bottleneck_dim),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, len(CUE_NAMES)),
        )

    def forward(self, control_states: torch.Tensor) -> torch.Tensor:
        batch_size = control_states.shape[0]
        queries = self.attn_queries.expand(batch_size, -1, -1)
        scores = torch.bmm(queries, control_states.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights, control_states)
        logits = self.readout(pooled.reshape(-1, self.hidden_dim))
        return logits.reshape(batch_size, self.num_queries * len(CUE_NAMES))


def _unit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(eps)


def _build_targets(
    cues: torch.Tensor,
    delta_q_target: torch.Tensor,
    cue_mask: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回每个 UAV 的最佳 cue 标签及有效位置 mask。"""
    target_dir = _unit(delta_q_target[..., :2])
    cue_dir = _unit(cues)
    cosine = (cue_dir * target_dir.unsqueeze(2)).sum(dim=-1)

    if cue_mask is None:
        valid = torch.ones_like(cosine, dtype=torch.bool)
    else:
        valid = cue_mask.to(dtype=torch.bool)
        cosine = cosine.masked_fill(~valid, -1e4)

    valid_uav = valid.any(dim=-1)
    target_idx = cosine.argmax(dim=-1)
    return target_idx, valid_uav


def _hist(values: torch.Tensor, mask: torch.Tensor) -> Dict[str, int]:
    valid_values = values[mask]
    counts = torch.bincount(valid_values, minlength=len(CUE_NAMES)).cpu().tolist()
    return {name: int(counts[idx]) for idx, name in enumerate(CUE_NAMES)}


def _accuracy(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> float:
    if not valid.any():
        return float("nan")
    pred = logits.argmax(dim=-1)
    return float((pred[valid] == targets[valid]).float().mean().item())


def _majority_accuracy(targets: torch.Tensor, valid: torch.Tensor) -> float:
    counts = torch.bincount(targets[valid], minlength=len(CUE_NAMES))
    return float(counts.max().float().div(counts.sum().clamp_min(1)).item())


def _compute_probe_loss(
    logits: torch.Tensor,
    cues: torch.Tensor,
    delta_q_target: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    loss_mode: str,
    ce_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 hard CE、连续 mixture-direction loss 及实际优化目标。"""
    loss_ce = F.cross_entropy(logits[valid], targets[valid])
    cue_weights = F.softmax(logits, dim=-1)
    pred_dir = _unit((cue_weights.unsqueeze(-1) * cues).sum(dim=2))
    target_dir = _unit(delta_q_target[..., :2])
    cosine = (pred_dir * target_dir).sum(dim=-1)
    loss_direction = (1.0 - cosine[valid]).mean()

    if loss_mode == "ce":
        total = loss_ce
    elif loss_mode == "direction":
        total = loss_direction
    elif loss_mode == "hybrid":
        total = loss_direction + ce_weight * loss_ce
    else:
        raise ValueError(f"Unsupported loss mode: {loss_mode}")
    return total, loss_ce, loss_direction


def _select_probe_indices(
    targets: torch.Tensor,
    valid: torch.Tensor,
    train_samples: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """确定性寻找一个覆盖全部 cue 类别的环境级训练子集。"""
    num_samples = targets.shape[0]
    if train_samples <= 0 or train_samples >= num_samples:
        raise ValueError(
            f"train_samples must be in [1, {num_samples - 1}], got {train_samples}"
        )

    generator = torch.Generator().manual_seed(seed)
    all_classes = set(range(len(CUE_NAMES)))
    selected = None
    for _ in range(1000):
        candidate = torch.randperm(num_samples, generator=generator)[:train_samples]
        candidate_device = candidate.to(targets.device)
        labels = targets[candidate_device][valid[candidate_device]].cpu().tolist()
        if set(labels) == all_classes:
            selected = candidate.cpu().tolist()
            break
    if selected is None:
        raise RuntimeError(
            f"Could not find {train_samples} samples covering all q-cue classes. "
            "Increase --train_samples."
        )

    selected_set = set(selected)
    validation = [idx for idx in range(num_samples) if idx not in selected_set]
    return selected, validation


def _mixture_cosine(
    weights: torch.Tensor,
    cues: torch.Tensor,
    delta_q_target: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    cue_mix = _unit((weights.unsqueeze(-1) * cues).sum(dim=2))
    target_dir = _unit(delta_q_target[..., :2])
    cosine = (cue_mix * target_dir).sum(dim=-1)
    if not valid.any():
        return float("nan")
    return float(cosine[valid].mean().item())


def _evaluate_split(
    name: str,
    readout: ControlReadout,
    states: torch.Tensor,
    cues: torch.Tensor,
    delta_q_target: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    fixed_weights: torch.Tensor,
    shuffle_seed: int,
) -> Dict:
    with torch.no_grad():
        logits = readout(states).reshape(states.shape[0], targets.shape[1], len(CUE_NAMES))
        weights = F.softmax(logits, dim=-1)
        pred = logits.argmax(dim=-1)

        fixed = fixed_weights.view(1, 1, -1).expand_as(weights)
        flat_weights = weights.reshape(-1, len(CUE_NAMES))
        generator = torch.Generator(device=weights.device).manual_seed(shuffle_seed)
        permutation = torch.randperm(flat_weights.shape[0], generator=generator, device=weights.device)
        shuffled = flat_weights[permutation].reshape_as(weights)

        result = {
            "name": name,
            "num_environment_samples": int(states.shape[0]),
            "num_valid_uavs": int(valid.sum().item()),
            "accuracy": _accuracy(logits, targets, valid),
            "majority_accuracy": _majority_accuracy(targets, valid),
            "target_hist": _hist(targets, valid),
            "pred_hist": _hist(pred, valid),
            "probability_mean": weights[valid].mean(dim=0).cpu().tolist(),
            "probability_std": weights[valid].std(dim=0, unbiased=False).cpu().tolist(),
            "dynamic_mixture_cosine": _mixture_cosine(weights, cues, delta_q_target, valid),
            "fixed_mixture_cosine": _mixture_cosine(fixed, cues, delta_q_target, valid),
            "shuffled_mixture_cosine": _mixture_cosine(shuffled, cues, delta_q_target, valid),
        }
    return result


def _print_split(result: Dict):
    print(f"\n=== {result['name']} ===")
    print(f"  environment samples:      {result['num_environment_samples']}")
    print(f"  valid UAV labels:         {result['num_valid_uavs']}")
    print(f"  accuracy:                 {result['accuracy']:.4f}")
    print(f"  majority baseline:        {result['majority_accuracy']:.4f}")
    print(f"  target hist:              {result['target_hist']}")
    print(f"  pred hist:                {result['pred_hist']}")
    print(f"  probability mean:         {result['probability_mean']}")
    print(f"  probability std:          {result['probability_std']}")
    print(f"  dynamic mixture cosine:   {result['dynamic_mixture_cosine']:.4f}")
    print(f"  fixed mixture cosine:     {result['fixed_mixture_cosine']:.4f}")
    print(f"  shuffled mixture cosine:  {result['shuffled_mixture_cosine']:.4f}")


def _format_indices(indices: List[int], preview: int = 12) -> str:
    if len(indices) <= preview:
        return str(indices)
    return f"{indices[:preview]} ... ({len(indices)} total)"


def _load_probe_npz(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"probe NPZ not found: {path}")
    data = np.load(path)
    required = ("control_states", "q_geometry_cues", "delta_q_target")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"prediction NPZ {path} is missing required arrays: {missing}")

    states = torch.from_numpy(np.asarray(data["control_states"], dtype=np.float32)).to(device)
    cues = torch.from_numpy(np.asarray(data["q_geometry_cues"], dtype=np.float32)).to(device)
    delta_q_target = torch.from_numpy(
        np.asarray(data["delta_q_target"], dtype=np.float32)
    ).to(device)
    cue_mask = None
    if "q_geometry_mask" in data:
        cue_mask = torch.from_numpy(
            np.asarray(data["q_geometry_mask"], dtype=np.bool_)
        ).to(device)

    if states.ndim != 3:
        raise ValueError(f"control_states must have shape (N, C, H), got {tuple(states.shape)}")
    if cues.ndim != 4 or cues.shape[2:] != (len(CUE_NAMES), 2):
        raise ValueError(f"q_geometry_cues must have shape (N, M, 3, 2), got {tuple(cues.shape)}")
    if delta_q_target.shape[:2] != cues.shape[:2] or delta_q_target.shape[-1] != 3:
        raise ValueError(
            "delta_q_target must have shape (N, M, 3) aligned with q_geometry_cues, "
            f"got {tuple(delta_q_target.shape)}"
        )
    if states.shape[0] != cues.shape[0]:
        raise ValueError(
            f"control_states and q_geometry_cues sample counts differ: "
            f"{states.shape[0]} != {cues.shape[0]}"
        )

    targets, valid = _build_targets(cues, delta_q_target, cue_mask)
    return {
        "states": states,
        "cues": cues,
        "delta_q_target": delta_q_target,
        "targets": targets,
        "valid": valid,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe whether cached multimodal control states contain q-cue selection information."
    )
    parser.add_argument("--prediction_npz", type=str, required=True)
    parser.add_argument("--validation_npz", type=str, default=None,
                        help="可选独立 seed 验证 NPZ；提供后不会从训练 NPZ 内部切验证集")
    parser.add_argument("--train_samples", type=int, default=10,
                        help="训练环境数；提供 --validation_npz 时传 0 表示使用全部训练 NPZ")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--loss_mode", type=str, choices=["ce", "direction", "hybrid"],
                        default="ce",
                        help="ce 保持旧探针；direction 直接优化 cue 混合方向；hybrid 组合两者")
    parser.add_argument("--ce_weight", type=float, default=0.1,
                        help="hybrid 模式下 hard CE 的辅助权重")
    parser.add_argument("--probe_hidden_dim", type=int, default=0,
                        help="0 使用原始 2560→1280 ControlReadout；正数使用紧凑瓶颈，建议 64/128")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="紧凑 probe 的 dropout；原始 ControlReadout 不使用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default=None,
                        help="例如 cuda 或 cpu；默认优先使用 CUDA")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    npz_path = Path(args.prediction_npz)
    train_data = _load_probe_npz(npz_path, device)
    states = train_data["states"]
    cues = train_data["cues"]
    delta_q_target = train_data["delta_q_target"]
    targets = train_data["targets"]
    valid = train_data["valid"]

    validation_npz_path = Path(args.validation_npz) if args.validation_npz else None
    if validation_npz_path is not None:
        validation_data = _load_probe_npz(validation_npz_path, device)
        if args.train_samples == 0:
            train_indices = list(range(states.shape[0]))
        else:
            train_indices, _ = _select_probe_indices(
                targets, valid, args.train_samples, args.seed
            )
        val_indices = list(range(validation_data["states"].shape[0]))
    else:
        if args.train_samples == 0:
            raise ValueError("--train_samples 0 requires --validation_npz")
        train_indices, val_indices = _select_probe_indices(
            targets, valid, args.train_samples, args.seed
        )
        validation_data = {
            "states": states,
            "cues": cues,
            "delta_q_target": delta_q_target,
            "targets": targets,
            "valid": valid,
        }
    train_index = torch.tensor(train_indices, device=device, dtype=torch.long)
    val_index = torch.tensor(val_indices, device=device, dtype=torch.long)

    if validation_data["states"].shape[1:] != states.shape[1:]:
        raise ValueError(
            "Training and validation control state shapes differ: "
            f"{tuple(states.shape[1:])} != {tuple(validation_data['states'].shape[1:])}"
        )
    if validation_data["cues"].shape[1:] != cues.shape[1:]:
        raise ValueError(
            "Training and validation q geometry cue shapes differ: "
            f"{tuple(cues.shape[1:])} != {tuple(validation_data['cues'].shape[1:])}"
        )

    num_control_tokens = states.shape[1]
    hidden_dim = states.shape[2]
    num_uavs = cues.shape[1]
    if args.probe_hidden_dim > 0:
        readout = CompactCueReadout(
            hidden_dim=hidden_dim,
            num_queries=num_uavs,
            bottleneck_dim=args.probe_hidden_dim,
            dropout=args.dropout,
        ).to(device)
        readout_type = f"compact_{args.probe_hidden_dim}"
    else:
        readout = ControlReadout(
            hidden_dim=hidden_dim,
            num_control_tokens=num_control_tokens,
            out_dim=num_uavs * len(CUE_NAMES),
            num_queries=num_uavs,
        ).to(device)
        readout_type = "original_control_readout"
    optimizer = torch.optim.AdamW(
        readout.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_states = states[train_index]
    train_cues = cues[train_index]
    train_delta_q_target = delta_q_target[train_index]
    train_targets = targets[train_index]
    train_valid = valid[train_index]

    print("=" * 72)
    print("Q-cue cached-control-state probe")
    print("=" * 72)
    print(f"  training NPZ:           {npz_path}")
    print(f"  validation NPZ:         {validation_npz_path or 'internal split'}")
    print(f"  device:                 {device}")
    print(f"  training states:        {tuple(states.shape)}")
    print(f"  validation states:      {tuple(validation_data['states'].shape)}")
    print(f"  q geometry cues:        {tuple(cues.shape)}")
    print(f"  train sample indices:   {_format_indices(train_indices)}")
    print(f"  train target hist:      {_hist(train_targets, train_valid)}")
    print(f"  validation samples:     {len(val_indices)}")
    print(f"  readout type:           {readout_type}")
    print(f"  loss mode:              {args.loss_mode}")
    print(f"  trainable parameters:   {sum(p.numel() for p in readout.parameters()):,}")

    final_loss = None
    for step in range(1, args.steps + 1):
        logits = readout(train_states).reshape(
            train_states.shape[0], num_uavs, len(CUE_NAMES)
        )
        final_loss, loss_ce, loss_direction = _compute_probe_loss(
            logits,
            train_cues,
            train_delta_q_target,
            train_targets,
            train_valid,
            args.loss_mode,
            args.ce_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        final_loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_steps == 0 or step == args.steps:
            accuracy = _accuracy(logits.detach(), train_targets, train_valid)
            pred_hist = _hist(logits.detach().argmax(dim=-1), train_valid)
            train_mix_cosine = 1.0 - float(loss_direction.detach().item())
            print(
                f"step={step} loss={final_loss.item():.6f} "
                f"loss_ce={loss_ce.item():.6f} "
                f"loss_direction={loss_direction.item():.6f} "
                f"train_accuracy={accuracy:.4f} "
                f"train_mix_cosine={train_mix_cosine:.4f} "
                f"pred_hist={pred_hist}"
            )

    readout.eval()
    with torch.no_grad():
        train_logits = readout(train_states).reshape(
            train_states.shape[0], num_uavs, len(CUE_NAMES)
        )
        train_weights = F.softmax(train_logits, dim=-1)
        fixed_weights = train_weights[train_valid].mean(dim=0)

    train_result = _evaluate_split(
        "TRAIN / OVERFIT",
        readout,
        train_states,
        train_cues,
        train_delta_q_target,
        train_targets,
        train_valid,
        fixed_weights,
        args.seed + 1,
    )
    val_result = _evaluate_split(
        "VALIDATION / UNSEEN",
        readout,
        validation_data["states"][val_index],
        validation_data["cues"][val_index],
        validation_data["delta_q_target"][val_index],
        validation_data["targets"][val_index],
        validation_data["valid"][val_index],
        fixed_weights,
        args.seed + 2,
    )

    _print_split(train_result)
    _print_split(val_result)

    train_mixture_gain = (
        train_result["dynamic_mixture_cosine"] - train_result["fixed_mixture_cosine"]
    )
    validation_accuracy_gain = val_result["accuracy"] - val_result["majority_accuracy"]
    validation_mixture_gain = (
        val_result["dynamic_mixture_cosine"] - val_result["fixed_mixture_cosine"]
    )
    if args.loss_mode == "ce":
        train_fit_pass = bool(train_result["accuracy"] >= 0.90)
        generalization_pass = bool(
            validation_accuracy_gain > 0.0 and validation_mixture_gain > 0.01
        )
    else:
        train_fit_pass = bool(train_mixture_gain > 0.05)
        generalization_pass = bool(validation_mixture_gain > 0.01)

    if generalization_pass:
        conclusion = (
            "GENERALIZATION PASS: dynamic cue weights outperform the fixed mixture on "
            "unseen environments. The probe design is a candidate for integration into "
            "the main multimodal training path."
        )
    elif train_fit_pass:
        conclusion = (
            "MEMORIZATION ONLY: the probe improves or fits the training split but does not "
            "beat the fixed mixture on unseen environments. Do not integrate this selector; "
            "reduce capacity, regularize, or increase data first."
        )
    else:
        conclusion = (
            "UNDERFIT: the probe does not improve enough even on the training split. "
            "Check optimization and representation quality before main-model integration."
        )
    print(f"\nConclusion: {conclusion}")

    result = {
        "prediction_npz": str(npz_path),
        "validation_npz": str(validation_npz_path) if validation_npz_path else None,
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss_mode": args.loss_mode,
        "ce_weight": args.ce_weight,
        "readout_type": readout_type,
        "probe_hidden_dim": args.probe_hidden_dim,
        "dropout": args.dropout,
        "train_sample_indices": train_indices,
        "validation_sample_indices": val_indices,
        "final_train_loss": float(final_loss.item()),
        "train_fit_pass": train_fit_pass,
        "generalization_pass": generalization_pass,
        "train_mixture_gain": train_mixture_gain,
        "validation_accuracy_gain": validation_accuracy_gain,
        "validation_mixture_gain": validation_mixture_gain,
        "train": train_result,
        "validation": val_result,
        "conclusion": conclusion,
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved probe report to {output_path}")


if __name__ == "__main__":
    main()
