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
