#!/usr/bin/env python
"""Probe Q-direction readability from cached multimodal control states.

The probe consumes NPZ files written by ``analyze_mm_delta_outputs.py
--save_raw``.  It never loads Gemma and never changes a training checkpoint.
Its purpose is to distinguish four failure modes:

1. the online Q readout only needs a better-conditioned/longer optimization;
2. the learned attention readout is the bottleneck;
3. flattening all control-token slots makes Q linearly or nonlinearly readable;
4. the frozen control states remain unreadable even for the tested upper bounds.
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.probe_association_control_states import (
    _normalize_cached_states,
    _state_summary,
)
from src.model.projection_head import ControlReadout, ResidualMLP


class OnlineEquivalentQReadout(nn.Module):
    """The same ControlReadout + ResidualMLP used by the split Q branch."""

    def __init__(self, num_tokens: int, hidden_dim: int, num_uavs: int):
        super().__init__()
        output_dim = num_uavs * 3
        self.num_uavs = num_uavs
        self.readout = ControlReadout(
            hidden_dim,
            num_tokens,
            output_dim,
            num_queries=num_uavs,
        )
        self.mlp = ResidualMLP(output_dim, [256, 256])

    def forward(self, control_states: torch.Tensor) -> torch.Tensor:
        raw = self.readout(control_states)
        return self.mlp(raw).reshape(-1, self.num_uavs, 3)


class MeanLinearQReadout(nn.Module):
    """Linear readout after mean pooling, used to test pooling alone."""

    def __init__(self, hidden_dim: int, num_uavs: int):
        super().__init__()
        self.num_uavs = num_uavs
        self.readout = nn.Linear(hidden_dim, num_uavs * 3)

    def forward(self, control_states: torch.Tensor) -> torch.Tensor:
        pooled = control_states.mean(dim=1)
        return self.readout(pooled).reshape(-1, self.num_uavs, 3)


class FlattenedLinearQReadout(nn.Module):
    """Linear upper bound that retains every control-token slot."""

    def __init__(self, num_tokens: int, hidden_dim: int, num_uavs: int):
        super().__init__()
        self.num_uavs = num_uavs
        self.readout = nn.Linear(num_tokens * hidden_dim, num_uavs * 3)

    def forward(self, control_states: torch.Tensor) -> torch.Tensor:
        flat = control_states.reshape(control_states.shape[0], -1)
        return self.readout(flat).reshape(-1, self.num_uavs, 3)


class FlattenedMLPQReadout(nn.Module):
    """Nonlinear upper bound that retains every control-token slot."""

    def __init__(
        self,
        num_tokens: int,
        hidden_dim: int,
        num_uavs: int,
        probe_hidden_dim: int,
    ):
        super().__init__()
        self.num_uavs = num_uavs
        self.readout = nn.Sequential(
            nn.Linear(num_tokens * hidden_dim, probe_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(probe_hidden_dim),
            nn.Linear(probe_hidden_dim, num_uavs * 3),
        )

    def forward(self, control_states: torch.Tensor) -> torch.Tensor:
        flat = control_states.reshape(control_states.shape[0], -1)
        return self.readout(flat).reshape(-1, self.num_uavs, 3)


def _load_cached_arrays(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        required = ("control_states", "delta_q_target")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        states = np.asarray(data["control_states"], dtype=np.float32)
        targets = np.asarray(data["delta_q_target"], dtype=np.float32)

    if states.ndim != 3:
        raise ValueError(
            f"control_states must have shape (N, C, H), got {states.shape}"
        )
    if targets.ndim != 3 or targets.shape[-1] != 3:
        raise ValueError(
            f"delta_q_target must have shape (N, M, 3), got {targets.shape}"
        )
    if states.shape[0] != targets.shape[0]:
        raise ValueError(
            "control_states and delta_q_target sample counts differ: "
            f"{states.shape[0]} != {targets.shape[0]}"
        )
    return states, targets


def _direction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    valid = target_norm > 1e-6
    if not torch.any(valid):
        raise ValueError("delta_q_target contains no non-zero direction")
    prediction_dir = F.normalize(prediction, p=2, dim=-1, eps=1e-6)
    target_dir = F.normalize(target, p=2, dim=-1, eps=1e-6)
    return F.mse_loss(prediction_dir[valid], target_dir[valid])


def _direction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> Dict:
    target_norm = torch.linalg.vector_norm(target, dim=-1)
    valid = target_norm > 1e-6
    if not torch.any(valid):
        raise ValueError("delta_q_target contains no non-zero direction")

    prediction_norm = torch.linalg.vector_norm(prediction, dim=-1)
    prediction_dir = F.normalize(prediction, p=2, dim=-1, eps=1e-6)
    target_dir = F.normalize(target, p=2, dim=-1, eps=1e-6)
    cosine = torch.sum(prediction_dir * target_dir, dim=-1)[valid]
    per_uav_cosine = []
    for uav_index in range(target.shape[1]):
        uav_valid = valid[:, uav_index]
        values = torch.sum(
            prediction_dir[:, uav_index] * target_dir[:, uav_index],
            dim=-1,
        )
        per_uav_cosine.append(
            float(values[uav_valid].mean().item())
            if torch.any(uav_valid)
            else None
        )

    return {
        "direction_mse": float(
            F.mse_loss(prediction_dir[valid], target_dir[valid]).item()
        ),
        "cosine_mean": float(cosine.mean().item()),
        "cosine_min": float(cosine.min().item()),
        "cosine_per_uav": per_uav_cosine,
        "prediction_direction_per_dim_std_mean": float(
            prediction_dir.std(dim=0, unbiased=False).mean().item()
        ),
        "target_direction_per_dim_std_mean": float(
            target_dir.std(dim=0, unbiased=False).mean().item()
        ),
        "prediction_raw_norm_mean": float(prediction_norm[valid].mean().item()),
        "valid_direction_count": int(valid.sum().item()),
    }


def _evaluate(
    model: nn.Module,
    states: torch.Tensor,
    targets: torch.Tensor,
) -> Dict:
    model.eval()
    with torch.no_grad():
        prediction = model(states)
    return _direction_metrics(prediction, targets)


def _train_probe(
    model: nn.Module,
    train_states: torch.Tensor,
    train_targets: torch.Tensor,
    validation_states: Optional[torch.Tensor],
    validation_targets: Optional[torch.Tensor],
    steps: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    log_every: int,
) -> Dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history = []
    best_logged_state = None
    best_logged_step = None
    best_logged_loss = float("inf")

    for step in range(1, steps + 1):
        model.train()
        prediction = model(train_states)
        loss = _direction_loss(prediction, train_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                gradient_clip_norm,
            )
        else:
            grad_norm = torch.sqrt(
                sum(
                    parameter.grad.detach().pow(2).sum()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            train_metrics = _evaluate(model, train_states, train_targets)
            validation_metrics = (
                _evaluate(model, validation_states, validation_targets)
                if validation_states is not None
                else None
            )
            row = {
                "step": step,
                "gradient_norm": float(grad_norm.item()),
                "train_direction_mse": train_metrics["direction_mse"],
                "train_cosine_mean": train_metrics["cosine_mean"],
            }
            if validation_metrics is not None:
                row["validation_cosine_mean"] = validation_metrics["cosine_mean"]
            history.append(row)
            if row["train_direction_mse"] < best_logged_loss:
                best_logged_loss = row["train_direction_mse"]
                best_logged_step = step
                best_logged_state = copy.deepcopy(model.state_dict())
            validation_text = (
                f" validation_cosine={validation_metrics['cosine_mean']:.6f}"
                if validation_metrics is not None
                else ""
            )
            print(
                f"step={step} "
                f"train_mse={row['train_direction_mse']:.6f} "
                f"train_cosine={row['train_cosine_mean']:.6f}"
                f"{validation_text} "
                f"grad_norm={row['gradient_norm']:.6f}"
            )

    final_iterate_train = _evaluate(model, train_states, train_targets)
    final_iterate_validation = (
        _evaluate(model, validation_states, validation_targets)
        if validation_states is not None
        else None
    )
    if best_logged_state is not None:
        model.load_state_dict(best_logged_state)

    return {
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "history": history,
        "selected_step": best_logged_step,
        "selection_metric": "lowest_logged_train_direction_mse",
        "final_iterate_train": final_iterate_train,
        "final_iterate_validation": final_iterate_validation,
        "train": _evaluate(model, train_states, train_targets),
        "validation": (
            _evaluate(model, validation_states, validation_targets)
            if validation_states is not None
            else None
        ),
    }


def _classify_bottleneck(probes: Dict, threshold: float) -> str:
    cosine = {
        name: result["train"]["cosine_mean"]
        for name, result in probes.items()
    }
    if cosine.get("online_equivalent", -1.0) >= threshold:
        return (
            "ONLINE OPTIMIZATION BOTTLENECK: the exact Q readout can fit the "
            "cached states; repair the formal training schedule/conditioning "
            "before enabling LoRA."
        )
    if cosine.get("mean_linear", -1.0) >= threshold:
        return (
            "Q-READOUT BOTTLENECK: mean-pooled states are readable, but the "
            "online-equivalent attention/residual readout cannot fit them."
        )
    if cosine.get("flat_linear", -1.0) >= threshold:
        return (
            "CONTROL-TOKEN POOLING BOTTLENECK: Q is linearly readable only when "
            "all token slots are retained."
        )
    if cosine.get("flat_mlp", -1.0) >= threshold:
        return (
            "NONLINEAR Q-READOUT BOTTLENECK: flattened states contain train-fit "
            "information, but neither the online nor linear readout exposes it."
        )
    return (
        "FROZEN-STATE BOTTLENECK: none of the tested readouts can fit cached "
        "training Q directions; change the representation before extending "
        "projection-only training."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Probe Q directions from cached multimodal control states"
    )
    parser.add_argument("--prediction_npz", required=True)
    parser.add_argument("--validation_npz")
    parser.add_argument(
        "--readout_types",
        nargs="+",
        choices=(
            "online_equivalent",
            "mean_linear",
            "flat_linear",
            "flat_mlp",
        ),
        default=(
            "online_equivalent",
            "mean_linear",
            "flat_linear",
            "flat_mlp",
        ),
    )
    parser.add_argument("--probe_hidden_dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_clip_norm", type=float, default=5.0)
    parser.add_argument(
        "--state_normalization",
        choices=("none", "hidden_feature"),
        default="hidden_feature",
        help="Use training-only statistics; hidden_feature improves conditioning",
    )
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit_cosine_threshold", type=float, default=0.95)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.steps <= 0 or args.log_every <= 0:
        raise ValueError("steps and log_every must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if args.weight_decay < 0 or args.gradient_clip_norm < 0:
        raise ValueError("weight_decay and gradient_clip_norm must be non-negative")
    if args.probe_hidden_dim <= 0:
        raise ValueError("probe_hidden_dim must be positive")
    if not -1.0 <= args.overfit_cosine_threshold <= 1.0:
        raise ValueError("overfit_cosine_threshold must be in [-1, 1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_states_np, train_targets_np = _load_cached_arrays(args.prediction_npz)
    if args.validation_npz:
        validation_states_np, validation_targets_np = _load_cached_arrays(
            args.validation_npz
        )
        if train_states_np.shape[1:] != validation_states_np.shape[1:]:
            raise ValueError(
                "training and validation control-state shapes are incompatible"
            )
        if train_targets_np.shape[1:] != validation_targets_np.shape[1:]:
            raise ValueError(
                "training and validation Q target shapes are incompatible"
            )
    else:
        validation_states_np = None
        validation_targets_np = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_states = torch.from_numpy(train_states_np).to(device)
    train_targets = torch.from_numpy(train_targets_np).to(device)
    validation_states = (
        torch.from_numpy(validation_states_np).to(device)
        if validation_states_np is not None
        else None
    )
    validation_targets = (
        torch.from_numpy(validation_targets_np).to(device)
        if validation_targets_np is not None
        else None
    )
    raw_training_state_summary = _state_summary(train_states)
    raw_validation_state_summary = (
        _state_summary(validation_states)
        if validation_states is not None
        else None
    )
    train_states, normalization_validation, normalization_summary = (
        _normalize_cached_states(
            train_states,
            validation_states if validation_states is not None else train_states,
            args.state_normalization,
        )
    )
    if validation_states is not None:
        validation_states = normalization_validation

    _, num_tokens, hidden_dim = train_states.shape
    _, num_uavs, _ = train_targets.shape
    print("=" * 72)
    print("Q cached-control-state direction probe")
    print("=" * 72)
    print(f"  training NPZ:        {args.prediction_npz}")
    print(f"  validation NPZ:      {args.validation_npz}")
    print(f"  device:              {device}")
    print(f"  training states:     {tuple(train_states.shape)}")
    print(f"  Q targets:           {tuple(train_targets.shape)}")
    print(f"  state normalization: {args.state_normalization}")

    report = {
        "training_npz": args.prediction_npz,
        "validation_npz": args.validation_npz,
        "device": str(device),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip_norm": args.gradient_clip_norm,
        "state_normalization": args.state_normalization,
        "normalization_summary": normalization_summary,
        "seed": args.seed,
        "overfit_cosine_threshold": args.overfit_cosine_threshold,
        "training_state_summary": raw_training_state_summary,
        "validation_state_summary": raw_validation_state_summary,
        "probe_training_state_summary": _state_summary(train_states),
        "probe_validation_state_summary": (
            _state_summary(validation_states)
            if validation_states is not None
            else None
        ),
        "probes": {},
    }

    for readout_type in args.readout_types:
        print(f"\n=== {readout_type} ===")
        torch.manual_seed(args.seed)
        if readout_type == "online_equivalent":
            model = OnlineEquivalentQReadout(
                num_tokens,
                hidden_dim,
                num_uavs,
            )
        elif readout_type == "mean_linear":
            model = MeanLinearQReadout(hidden_dim, num_uavs)
        elif readout_type == "flat_linear":
            model = FlattenedLinearQReadout(
                num_tokens,
                hidden_dim,
                num_uavs,
            )
        else:
            model = FlattenedMLPQReadout(
                num_tokens,
                hidden_dim,
                num_uavs,
                args.probe_hidden_dim,
            )
        model.to(device)
        report["probes"][readout_type] = _train_probe(
            model,
            train_states,
            train_targets,
            validation_states,
            validation_targets,
            args.steps,
            args.learning_rate,
            args.weight_decay,
            args.gradient_clip_norm,
            args.log_every,
        )

    conclusion = _classify_bottleneck(
        report["probes"],
        args.overfit_cosine_threshold,
    )
    report["conclusion"] = conclusion

    print("\n=== FINAL ===")
    for name, probe in report["probes"].items():
        validation_text = (
            f"{probe['validation']['cosine_mean']:.6f}"
            if probe["validation"] is not None
            else "not provided"
        )
        print(f"  {name} selected step:       {probe['selected_step']}")
        print(
            f"  {name} train cosine:        "
            f"{probe['train']['cosine_mean']:.6f}"
        )
        print(f"  {name} validation cosine:   {validation_text}")
        print(
            f"  {name} train direction MSE: "
            f"{probe['train']['direction_mse']:.6f}"
        )
    print(f"  conclusion: {conclusion}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved probe report to {output_path}")


if __name__ == "__main__":
    main()
