"""
从 multimodal v3 prompt 中解析 q 几何候选方向。

v3 prompt 和 BEV 图像表达同一组三类候选方向：
- weighted user center
- nearest user
- nearest target

训练时把这些方向作为结构化张量交给 projection head，避免 q 分支完全靠
hidden state 自由回归方向。
"""

import re
from typing import Tuple

import numpy as np


CUE_NAMES = ("weighted_center", "nearest_user", "nearest_target")
UAV_LINE_RE = re.compile(r"^\s*UAV\s+(\d+):")
CUE_PATTERNS = {
    "weighted_center": re.compile(r"weighted_center:d=([\d.eE+-]+)m,dir=\[([^\]]+)\]"),
    "nearest_user": re.compile(r"nearest_user=u(\d+):d=([\d.eE+-]+)m,dir=\[([^\]]+)\]"),
    "nearest_target": re.compile(r"nearest_target=t(\d+):d=([\d.eE+-]+)m,dir=\[([^\]]+)\]"),
}


def _parse_vec2(text: str) -> np.ndarray:
    values = [float(v.strip()) for v in text.split(",")]
    if len(values) != 2:
        raise ValueError(f"Expected 2D vector, got: {text}")
    return np.asarray(values, dtype=np.float32)


def parse_q_geometry_cues(prompt: str, num_uavs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    解析 prompt 中每架 UAV 的三类候选方向。

    Returns:
        cues: (M, 3, 2)，三类候选方向的 xy 单位向量。
        mask: (M, 3)，对应候选是否成功解析。
    """
    cues = np.zeros((num_uavs, len(CUE_NAMES), 2), dtype=np.float32)
    mask = np.zeros((num_uavs, len(CUE_NAMES)), dtype=np.float32)

    for line in prompt.splitlines():
        match_uav = UAV_LINE_RE.match(line)
        if not match_uav:
            continue
        uav_idx = int(match_uav.group(1))
        if uav_idx < 0 or uav_idx >= num_uavs:
            continue

        match = CUE_PATTERNS["weighted_center"].search(line)
        if match:
            cues[uav_idx, 0] = _parse_vec2(match.group(2))
            mask[uav_idx, 0] = 1.0

        match = CUE_PATTERNS["nearest_user"].search(line)
        if match:
            cues[uav_idx, 1] = _parse_vec2(match.group(3))
            mask[uav_idx, 1] = 1.0

        match = CUE_PATTERNS["nearest_target"].search(line)
        if match:
            cues[uav_idx, 2] = _parse_vec2(match.group(3))
            mask[uav_idx, 2] = 1.0

    return cues, mask
