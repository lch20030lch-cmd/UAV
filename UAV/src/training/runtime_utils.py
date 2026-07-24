"""Small, framework-independent helpers shared by training entry points."""

import math
import re
import shutil
from pathlib import Path
from typing import Optional


def resolve_optimizer_steps(
    *,
    num_batches: int,
    gradient_accumulation_steps: int,
    epochs: int,
    max_steps_override: Optional[int] = None,
) -> int:
    """Resolve optimizer updates from epochs unless an explicit limit is given."""
    if num_batches <= 0:
        raise ValueError("training dataloader must contain at least one batch")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if max_steps_override is not None:
        if max_steps_override <= 0:
            raise ValueError("max_steps must be positive")
        return int(max_steps_override)
    return math.ceil(
        (num_batches * int(epochs)) / gradient_accumulation_steps
    )


def resolve_warmup_steps(max_steps: int, warmup_ratio: float) -> int:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0, 1]")
    return int(max_steps * warmup_ratio)


def resolve_optimizer_controls(
    train_cfg: dict,
    *,
    configured_max_grad_norm: float,
    lr_scheduler_override: Optional[str] = None,
    warmup_ratio_override: Optional[float] = None,
    weight_decay_override: Optional[float] = None,
    max_grad_norm_override: Optional[float] = None,
):
    """Resolve auditable optimizer controls without mutating shared config."""
    scheduler_name = str(
        lr_scheduler_override
        if lr_scheduler_override is not None
        else train_cfg.get("lr_scheduler", "constant")
    ).strip().lower()
    warmup_ratio = float(
        warmup_ratio_override
        if warmup_ratio_override is not None
        else train_cfg.get("warmup_ratio", 0.0)
    )
    weight_decay = float(
        weight_decay_override
        if weight_decay_override is not None
        else train_cfg.get("weight_decay", 0.01)
    )
    max_grad_norm = float(
        max_grad_norm_override
        if max_grad_norm_override is not None
        else configured_max_grad_norm
    )

    if not scheduler_name:
        raise ValueError("lr_scheduler must not be empty")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be in [0, 1]")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    return {
        "lr_scheduler": scheduler_name,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "max_grad_norm": max_grad_norm,
    }


def rotate_step_checkpoints(
    checkpoint_root,
    *,
    prefix: str,
    save_total_limit: Optional[int],
):
    """Delete only the oldest exact-match step directories under one root."""
    if save_total_limit is None:
        return []
    limit = int(save_total_limit)
    if limit <= 0:
        raise ValueError("save_total_limit must be a positive integer")
    if not prefix:
        raise ValueError("checkpoint prefix must not be empty")

    root = Path(checkpoint_root).resolve()
    if not root.is_dir():
        return []
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    candidates = []
    for child in root.iterdir():
        match = pattern.fullmatch(child.name)
        if match and child.is_dir() and not child.is_symlink():
            candidates.append((int(match.group(1)), child))
    candidates.sort(key=lambda item: item[0])

    removed = []
    for _, child in candidates[:-limit]:
        resolved_child = child.resolve()
        if resolved_child.parent != root:
            raise RuntimeError(
                f"refusing to remove checkpoint outside root: {resolved_child}"
            )
        shutil.rmtree(resolved_child)
        removed.append(str(resolved_child))
    return removed
