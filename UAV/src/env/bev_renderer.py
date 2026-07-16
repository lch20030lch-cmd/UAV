"""
UAV-ISAC 多模态数据的 BEV 鸟瞰图渲染器。

渲染逻辑刻意保持简单、稳定：图像用于表达空间几何关系，而不是做展示图。
输出图像为正方形、坐标轴固定，并避免大段文字 legend，让视觉模型主要看到
UAV、用户、目标之间的相对位置。
"""

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


def _as_array(value, dtype=np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def render_bev_image(
    q_positions: Sequence,
    user_positions: Sequence,
    target_positions: Sequence,
    save_path: str,
    area_size: Tuple[float, float] = (1000.0, 1000.0),
    association: Optional[Sequence] = None,
    user_weights: Optional[Sequence] = None,
    channel_gains_users: Optional[Sequence] = None,
    sensing_sinrs: Optional[Sequence] = None,
    target_detected: Optional[Sequence] = None,
    image_size: int = 224,
    coverage_radius: float = 250.0,
    movement_radius: float = 15.0,
    draw_association: bool = True,
    draw_coverage: bool = True,
    draw_guidance: bool = True,
) -> str:
    """将 UAV / 用户 / 目标几何关系渲染为 BEV PNG。

    Args:
        q_positions: UAV 位置，shape (M, 3)。
        user_positions: 地面用户位置，shape (K, 2)。
        target_positions: 感知目标位置，shape (T, 2)。
        save_path: 输出 PNG 路径。
        area_size: 服务区域宽高，单位米。
        association: 可选关联矩阵，shape (M, K)。
        user_weights: 可选用户需求权重，shape (K,)，用于调整用户点大小。
        channel_gains_users: 可选 UAV-用户信道增益，shape (M, K)，用于画候选通信线。
        sensing_sinrs: 可选 UAV-目标感知 SINR，shape (M, T)，用于画候选感知线。
        target_detected: 可选目标可见性 bool mask，shape (T,)。
        image_size: 输出图片宽高，单位像素。
        coverage_radius: UAV 覆盖圆的可视化半径。
        movement_radius: 单槽移动半径，通常为 v_max * Δt = 15m。
        draw_association: 是否绘制当前关联线。
        draw_coverage: 是否绘制 UAV 覆盖圆。
        draw_guidance: 是否绘制候选移动方向视觉线索。

    Returns:
        输出路径字符串。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    q = _as_array(q_positions)
    users = _as_array(user_positions)
    targets = _as_array(target_positions)
    assoc = None if association is None else _as_array(association)
    weights = None if user_weights is None else _as_array(user_weights)
    gains = None if channel_gains_users is None else _as_array(channel_gains_users)
    sensing = None if sensing_sinrs is None else _as_array(sensing_sinrs)

    if target_detected is None:
        detected = np.ones((targets.shape[0],), dtype=bool)
    else:
        detected = np.asarray(target_detected, dtype=bool)

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dpi = 100
    figsize = (image_size / dpi, image_size / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f8f5")

    area_w, area_h = float(area_size[0]), float(area_size[1])
    ax.set_xlim(0, area_w)
    ax.set_ylim(0, area_h)
    ax.set_aspect("equal", adjustable="box")

    if draw_coverage:
        for x, y, _h in q:
            ax.add_patch(
                Circle(
                    (float(x), float(y)),
                    coverage_radius,
                    facecolor="#3b82f6",
                    edgecolor="#1d4ed8",
                    alpha=0.08,
                    linewidth=0.8,
                )
            )

    if draw_guidance and q.size > 0:
        # 真实移动半径很小，但它直接对应 delta_q 的物理上限。
        for x, y, _h in q:
            ax.add_patch(
                Circle(
                    (float(x), float(y)),
                    movement_radius,
                    facecolor="none",
                    edgecolor="#1d4ed8",
                    alpha=0.9,
                    linewidth=1.0,
                    linestyle="--",
                    zorder=6,
                )
            )

        # 紧凑通信几何线: 每架 UAV 指向加权用户中心和最近用户。
        if users.size > 0:
            if weights is not None and weights.size == users.shape[0]:
                w = np.maximum(weights.astype(np.float32), 1e-6)
                weighted_center = (users * w[:, None]).sum(axis=0) / w.sum()
                ax.scatter(
                    [weighted_center[0]],
                    [weighted_center[1]],
                    s=46,
                    marker="*",
                    c="#7c3aed",
                    edgecolors="white",
                    linewidths=0.45,
                    alpha=0.95,
                    zorder=4,
                )
            else:
                weighted_center = users.mean(axis=0)

            for m in range(q.shape[0]):
                q_xy = q[m, :2]
                d_user = np.linalg.norm(users - q_xy[None, :], axis=1)
                nearest_user = int(np.argmin(d_user))
                ax.plot(
                    [q[m, 0], weighted_center[0]],
                    [q[m, 1], weighted_center[1]],
                    color="#7c3aed",
                    linewidth=0.75,
                    alpha=0.38,
                    linestyle=":",
                    zorder=2,
                )
                ax.plot(
                    [q[m, 0], users[nearest_user, 0]],
                    [q[m, 1], users[nearest_user, 1]],
                    color="#22c55e",
                    linewidth=0.8,
                    alpha=0.45,
                    linestyle="-",
                    zorder=2,
                )

        # 紧凑感知几何线: 每架 UAV 只指向最近目标。
        if targets.size > 0:
            for m in range(q.shape[0]):
                q_xy = q[m, :2]
                d_target = np.linalg.norm(targets - q_xy[None, :], axis=1)
                nearest_t = int(np.argmin(d_target))
                ax.plot(
                    [q[m, 0], targets[nearest_t, 0]],
                    [q[m, 1], targets[nearest_t, 1]],
                    color="#f97316",
                    linewidth=0.8,
                    alpha=0.48,
                    linestyle="--",
                    zorder=2,
                )

    if draw_association and assoc is not None and assoc.size > 0:
        best_uav = np.argmax(assoc, axis=0)
        for k, m in enumerate(best_uav):
            if k >= users.shape[0] or m >= q.shape[0]:
                continue
            ax.plot(
                [q[m, 0], users[k, 0]],
                [q[m, 1], users[k, 1]],
                color="#9ca3af",
                linewidth=0.45,
                alpha=0.35,
                zorder=1,
            )

    if users.size > 0:
        if weights is not None and weights.size == users.shape[0]:
            w = weights.astype(np.float32)
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)
            user_sizes = 14.0 + 18.0 * w
        else:
            user_sizes = 18
        ax.scatter(
            users[:, 0],
            users[:, 1],
            s=user_sizes,
            marker="o",
            c="#16a34a",
            edgecolors="white",
            linewidths=0.35,
            alpha=0.95,
            zorder=3,
        )

    if targets.size > 0:
        visible_targets = targets[detected]
        hidden_targets = targets[~detected]
        if visible_targets.size > 0:
            ax.scatter(
                visible_targets[:, 0],
                visible_targets[:, 1],
                s=42,
                marker="X",
                c="#dc2626",
                edgecolors="white",
                linewidths=0.45,
                alpha=0.95,
                zorder=4,
            )
        if hidden_targets.size > 0:
            ax.scatter(
                hidden_targets[:, 0],
                hidden_targets[:, 1],
                s=34,
                marker="x",
                c="#991b1b",
                linewidths=1.0,
                alpha=0.45,
                zorder=4,
            )

    if q.size > 0:
        ax.scatter(
            q[:, 0],
            q[:, 1],
            s=72,
            marker="^",
            c="#2563eb",
            edgecolors="white",
            linewidths=0.6,
            alpha=0.98,
            zorder=5,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")
        spine.set_linewidth(0.8)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    fig.savefig(out_path, format="png", dpi=dpi)
    plt.close(fig)
    return str(out_path)


def render_bev_sample(
    env_sample,
    save_path: str,
    area_size: Tuple[float, float] = (1000.0, 1000.0),
    image_size: int = 224,
    movement_radius: float = 15.0,
) -> str:
    """将 EnvironmentSample 渲染为 BEV PNG。"""
    return render_bev_image(
        q_positions=env_sample.q_current,
        user_positions=env_sample.u_positions,
        target_positions=env_sample.s_positions,
        association=env_sample.association,
        user_weights=env_sample.user_weights,
        channel_gains_users=env_sample.channel_gains_users,
        sensing_sinrs=env_sample.sensing_sinrs,
        save_path=save_path,
        area_size=area_size,
        image_size=image_size,
        movement_radius=movement_radius,
        draw_association=False,
    )
