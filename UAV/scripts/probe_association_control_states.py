#!/usr/bin/env python
"""Probe whether cached multimodal control states contain readable A labels.

The probe consumes NPZ files written by ``analyze_mm_delta_outputs.py
--save_raw``. It compares the online-equivalent association readout with a
flattened linear upper bound without loading Gemma or changing the main model.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.projection_head import ControlReadout, ResidualMLP


class OnlineEquivalentAssociationReadout(nn.Module):
    """The same ControlReadout + ResidualMLP used by the split A branch."""

    def __init__(self, num_tokens: int, hidden_dim: int, num_uavs: int, num_users: int):
        super().__init__()
        output_dim = num_uavs * num_users
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.readout = ControlReadout(
            hidden_dim,
            num_tokens,
            output_dim,
            num_queries=num_uavs,
        )
        self.mlp = ResidualMLP(output_dim, [256, 256])

    def forward(self, control_states: torch.Tensor) -> torch.Tensor:
        raw = self.readout(control_states)
        return self.mlp(raw).reshape(-1, self.num_uavs, self.num_users)


class FlattenedLinearAssociationReadout(nn.Module):
    """High-capacity linear upper bound that retains every control-token slot."""

    def __init__(self, num_tokens: int, hidden_dim: int, num_uavs: int, num_users: int):
        super().__init__()
        self.num_uavs = num_uavs
        self.num_users = num_users
        self.readout = nn.Linear(num_tokens * hidden_dim, num_uavs * num_users)

    def forward(self, control_states: torch.Tensor) -> torch.Tensor:
        flat = control_states.reshape(control_states.shape[0], -1)
        return self.readout(flat).reshape(-1, self.num_uavs, self.num_users)


def _load_cached_arrays(path: str):
    data = np.load(path)
    required = ("control_states", "delta_a_target")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} is missing required arrays: {missing}")

    states = np.asarray(data["control_states"], dtype=np.float32)
    targets = np.asarray(data["delta_a_target"], dtype=np.float32)
    if states.ndim != 3:
        raise ValueError(f"control_states must have shape (N, C, H), got {states.shape}")
    if targets.ndim != 3:
        raise ValueError(f"delta_a_target must have shape (N, M, K), got {targets.shape}")
    if states.shape[0] != targets.shape[0]:
        raise ValueError(
            "control_states and delta_a_target sample counts differ: "
            f"{states.shape[0]} != {targets.shape[0]}"
        )
    return states, targets


def _state_summary(states: torch.Tensor) -> Dict:
    flat = states.reshape(states.shape[0], -1)
    per_dim_std = flat.std(dim=0, unbiased=False)
    if flat.shape[0] <= 1:
        return {
            "per_dim_std_mean": float(per_dim_std.mean().item()),
            "per_dim_std_max": float(per_dim_std.max().item()),
            "nearest_cosine_mean": None,
            "nearest_cosine_max": None,
            "duplicate_pair_count": 0,
        }

    normalized = F.normalize(flat, dim=1, eps=1e-8)
    cosine = normalized @ normalized.T
    cosine.fill_diagonal_(-float("inf"))
    nearest_cosine = cosine.max(dim=1).values
    distances = torch.cdist(flat, flat)
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    duplicate_pairs = int(((distances <= 1e-6) & upper).sum().item())
    return {
        "per_dim_std_mean": float(per_dim_std.mean().item()),
        "per_dim_std_max": float(per_dim_std.max().item()),
        "nearest_cosine_mean": float(nearest_cosine.mean().item()),
        "nearest_cosine_max": float(nearest_cosine.max().item()),
        "duplicate_pair_count": duplicate_pairs,
    }


def _association_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict:
    target_idx = targets.argmax(dim=1)
    pred_idx = logits.argmax(dim=1)
    num_samples, num_uavs, num_users = logits.shape
    accuracy = (pred_idx == target_idx).float()

    probs = F.softmax(logits, dim=1)
    oracle_probs = probs.gather(1, target_idx.unsqueeze(1)).squeeze(1)
    top_k = min(2, num_uavs)
    top_indices = probs.topk(top_k, dim=1).indices
    top2_accuracy = (
        top_indices == target_idx.unsqueeze(1)
    ).any(dim=1).float().mean()
    sorted_probs = probs.sort(dim=1).values
    if num_uavs >= 2:
        top1_margin = sorted_probs[:, -1, :] - sorted_probs[:, -2, :]
    else:
        top1_margin = sorted_probs[:, -1, :]

    majority_correct = 0
    for user_idx in range(num_users):
        counts = torch.bincount(target_idx[:, user_idx], minlength=num_uavs)
        majority_correct += int(counts.max().item())
    fixed_majority = majority_correct / max(num_samples * num_users, 1)

    pred_hist = torch.bincount(pred_idx.reshape(-1), minlength=num_uavs)
    target_hist = torch.bincount(target_idx.reshape(-1), minlength=num_uavs)
    per_user_accuracy = accuracy.mean(dim=0)
    return {
        "accuracy": float(accuracy.mean().item()),
        "top2_accuracy": float(top2_accuracy.item()),
        "fixed_user_majority_accuracy": float(fixed_majority),
        "gain_over_fixed_user_majority": float(accuracy.mean().item() - fixed_majority),
        "oracle_probability_mean": float(oracle_probs.mean().item()),
        "top1_margin_mean": float(top1_margin.mean().item()),
        "accuracy_per_user_min": float(per_user_accuracy.min().item()),
        "accuracy_per_user_max": float(per_user_accuracy.max().item()),
        "pred_hist": {
            str(index): int(count) for index, count in enumerate(pred_hist.tolist())
        },
        "target_hist": {
            str(index): int(count) for index, count in enumerate(target_hist.tolist())
        },
    }


def _evaluate(model: nn.Module, states: torch.Tensor, targets: torch.Tensor) -> Dict:
    model.eval()
    with torch.no_grad():
        logits = model(states)
        target_idx = targets.argmax(dim=1)
        loss = F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, logits.shape[1]),
            target_idx.reshape(-1),
        )
    metrics = _association_metrics(logits, targets)
    metrics["cross_entropy"] = float(loss.item())
    return metrics


def _train_probe(
    model: nn.Module,
    train_states: torch.Tensor,
    train_targets: torch.Tensor,
    validation_states: torch.Tensor,
    validation_targets: torch.Tensor,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    log_every: int,
) -> Dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    target_idx = train_targets.argmax(dim=1)
    history = []

    for step in range(1, steps + 1):
        model.train()
        logits = model(train_states)
        loss = F.cross_entropy(
            logits.permute(0, 2, 1).reshape(-1, logits.shape[1]),
            target_idx.reshape(-1),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            train_metrics = _evaluate(model, train_states, train_targets)
            validation_metrics = _evaluate(model, validation_states, validation_targets)
            row = {
                "step": step,
                "gradient_norm": float(grad_norm.item()),
                "train_cross_entropy": train_metrics["cross_entropy"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": validation_metrics["accuracy"],
            }
            history.append(row)
            print(
                f"step={step} loss={row['train_cross_entropy']:.6f} "
                f"train_accuracy={row['train_accuracy']:.4f} "
                f"validation_accuracy={row['validation_accuracy']:.4f} "
                f"grad_norm={row['gradient_norm']:.6f}"
            )

    return {
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "history": history,
        "train": _evaluate(model, train_states, train_targets),
        "validation": _evaluate(model, validation_states, validation_targets),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Probe corrected association labels from cached control states"
    )
    parser.add_argument("--prediction_npz", required=True)
    parser.add_argument("--validation_npz", required=True)
    parser.add_argument(
        "--readout_types",
        nargs="+",
        choices=("online_equivalent", "flat_linear"),
        default=("online_equivalent", "flat_linear"),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit_threshold", type=float, default=0.9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.steps <= 0 or args.log_every <= 0:
        raise ValueError("steps and log_every must be positive")
    if not 0.0 < args.overfit_threshold <= 1.0:
        raise ValueError("overfit_threshold must be in (0, 1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_states_np, train_targets_np = _load_cached_arrays(args.prediction_npz)
    val_states_np, val_targets_np = _load_cached_arrays(args.validation_npz)
    if train_states_np.shape[1:] != val_states_np.shape[1:]:
        raise ValueError("training and validation control-state shapes are incompatible")
    if train_targets_np.shape[1:] != val_targets_np.shape[1:]:
        raise ValueError("training and validation association target shapes are incompatible")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_states = torch.from_numpy(train_states_np).to(device)
    train_targets = torch.from_numpy(train_targets_np).to(device)
    val_states = torch.from_numpy(val_states_np).to(device)
    val_targets = torch.from_numpy(val_targets_np).to(device)
    _, num_tokens, hidden_dim = train_states.shape
    _, num_uavs, num_users = train_targets.shape

    print("=" * 72)
    print("Association cached-control-state probe")
    print("=" * 72)
    print(f"  training NPZ:       {args.prediction_npz}")
    print(f"  validation NPZ:     {args.validation_npz}")
    print(f"  device:             {device}")
    print(f"  training states:    {tuple(train_states.shape)}")
    print(f"  validation states:  {tuple(val_states.shape)}")
    print(f"  association target: {tuple(train_targets.shape)}")

    report = {
        "training_npz": args.prediction_npz,
        "validation_npz": args.validation_npz,
        "device": str(device),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "overfit_threshold": args.overfit_threshold,
        "training_state_summary": _state_summary(train_states),
        "validation_state_summary": _state_summary(val_states),
        "probes": {},
    }

    for readout_type in args.readout_types:
        print(f"\n=== {readout_type} ===")
        torch.manual_seed(args.seed)
        if readout_type == "online_equivalent":
            model = OnlineEquivalentAssociationReadout(
                num_tokens, hidden_dim, num_uavs, num_users
            )
        else:
            model = FlattenedLinearAssociationReadout(
                num_tokens, hidden_dim, num_uavs, num_users
            )
        model.to(device)
        report["probes"][readout_type] = _train_probe(
            model,
            train_states,
            train_targets,
            val_states,
            val_targets,
            args.steps,
            args.learning_rate,
            args.weight_decay,
            args.log_every,
        )

    exact_accuracy = report["probes"].get("online_equivalent", {}).get("train", {}).get("accuracy")
    flat_accuracy = report["probes"].get("flat_linear", {}).get("train", {}).get("accuracy")
    if exact_accuracy is not None and exact_accuracy >= args.overfit_threshold:
        conclusion = (
            "ONLINE OPTIMIZATION BOTTLENECK: the online-equivalent A head can overfit "
            "cached states with full-batch optimization."
        )
    elif flat_accuracy is not None and flat_accuracy >= args.overfit_threshold:
        conclusion = (
            "A-HEAD BOTTLENECK: cached states distinguish the training environments, "
            "but the online-equivalent A readout cannot fit them."
        )
    else:
        conclusion = (
            "FROZEN-STATE BOTTLENECK: neither tested readout can reach the train overfit "
            "threshold; do not extend projection-only training."
        )
    report["conclusion"] = conclusion

    print("\n=== FINAL ===")
    for name, probe in report["probes"].items():
        print(f"  {name} train accuracy:      {probe['train']['accuracy']:.4f}")
        print(f"  {name} validation accuracy: {probe['validation']['accuracy']:.4f}")
        print(f"  {name} train CE:            {probe['train']['cross_entropy']:.6f}")
    print(f"  conclusion: {conclusion}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved probe report to {output_path}")


if __name__ == "__main__":
    main()
