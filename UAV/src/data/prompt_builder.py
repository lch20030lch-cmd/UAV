"""
Prompt 构造器
论文 Section 3 — 构造多模态 prompt Π(t)

模态:
  - 通信摘要 c(t): 每用户 SINR, UAV 负载, 速率压力
  - 感知摘要 r(t): 每目标 SINR, 定位难度, 未覆盖目标
  - BEV 地图 V(t): 文本网格或图像
  - 系统指令: 优化目标 + 约束说明 + 输出格式
"""
import numpy as np


SYSTEM_INSTRUCTION = """You are a UAV-ISAC decision controller for low-altitude IoT networks.

Your task: Given the current network state (communication summary, sensing summary, and bird's-eye-view map), propose a warm-start decision prior δ = [δ_q, δ_a, δ_p] that will be used to initialize a numerical optimizer.

## Decision Variables
- δ_q: UAV displacement suggestions (shape {M}×3). Each row is [dx, dy, dh] in meters. dx,dy are horizontal moves; dh is altitude change. Clamp dx,dy to [-v_max*Δt, +v_max*Δt] ≈ [-15, 15]m per slot. Clamp altitude to [{H_min}, {H_max}]m.
- δ_a: User association proposals (shape {M}×{K}). A soft assignment matrix where each column sums to 1. Higher values mean stronger recommendation for that UAV to serve that user.
- δ_p: Power allocation hints (shape {M}×({K}+1)). First {K} entries per UAV are communication power for each user; last entry is sensing probe power. Each UAV row must satisfy sum(δ_p[m]) ≤ P_max = {P_max_W:.4f}W.

## Constraints (must respect)
1. Per-UAV power budget: sum of communication + sensing power ≤ P_max
2. Altitude: {H_min}m ≤ H_m ≤ {H_max}m
3. Single association: each user served by exactly one UAV
4. Per-UAV load cap: at most {K_max} users per UAV
5. UAV minimum separation: ≥ {d_min}m
6. Communication SINR ≥ {sinr_c_min_dB}dB for associated users
7. Sensing SINR ≥ {sinr_s_min_dB}dB for detected targets

## Output Format
Return ONLY a valid JSON object with the following structure:
```json
{{
  "delta_q": [[dx1, dy1, dh1], [dx2, dy2, dh2], ...],
  "delta_a": [[a_11, a_12, ...], [a_21, a_22, ...], ...],
  "delta_p": [[p_c11, p_c12, ..., p_s1], [p_c21, ..., p_s2], ...]
}}
```

Propose a warm start that maximizes: weighted sum-rate + λ_s × sensing SINR − λ_f × idle UAV penalty."""


def build_system_prompt(config: dict) -> str:
    """
    构造系统指令 (填入具体参数值)

    Args:
        config: 仿真配置 dict

    Returns:
        参数化的系统指令字符串
    """
    params = {
        "M": config.get("num_uavs", 4),
        "K": config.get("num_users", 20),
        "H_min": config.get("altitude_min_m", 50),
        "H_max": config.get("altitude_max_m", 300),
        "v_max": config.get("uav_max_speed_ms", 15),
        "K_max": config.get("load_cap_per_uav", 10),
        "d_min": config.get("uav_min_separation_m", 10),
        "sinr_c_min_dB": config.get("sinr_c_min_db", 0),
        "sinr_s_min_dB": config.get("sinr_s_min_db", 10),
        "P_max_W": 10 ** ((config.get("p_max_dbm", 30) - 30) / 10),
    }
    return SYSTEM_INSTRUCTION.format(**params)


def build_communication_summary_str(summary: dict) -> str:
    """格式化通信摘要 c(t) 为文本"""
    lines = ["[Communication Summary c(t)]"]
    lines.append(f"  Per-user SINR (dB): {summary['per_user_sinr_db']}")
    lines.append(f"  Per-UAV load (#users): {summary['per_uav_load']}")
    lines.append(f"  Rate pressure (req/achievable): {summary['rate_pressure']}")
    return "\n".join(lines)


def build_sensing_summary_str(summary: dict) -> str:
    """格式化感知摘要 r(t) 为文本"""
    lines = ["[Sensing Summary r(t)]"]
    lines.append(f"  Per-target sensing SINR (dB): {summary['per_target_sinr_db']}")
    lines.append(f"  Localization difficulty (CRB/ε_max): {summary['localization_difficulty']}")
    lines.append(f"  Uncovered targets (< Γ_s^min): {summary['uncovered_targets']}")
    lines.append(f"  Best UAV per target: {summary['best_uav_per_target']}")
    return "\n".join(lines)


def _unit_direction(src_xy: np.ndarray, dst_xy: np.ndarray) -> tuple:
    """返回 src 指向 dst 的单位方向和距离。"""
    vec = np.asarray(dst_xy, dtype=np.float32) - np.asarray(src_xy, dtype=np.float32)
    dist = float(np.linalg.norm(vec) + 1e-8)
    unit = vec / dist
    return unit, dist


def _fmt_vec(vec, ndigits: int = 3) -> str:
    return "[" + ", ".join(f"{float(v):.{ndigits}f}" for v in vec) + "]"


def build_geometry_guidance_str(env_sample, config: dict) -> str:
    """
    构造与 BEV 图像互补的紧凑几何提示。

    v2 曾把 top-k 用户、信道强用户和多个目标都写入 prompt，信息量大但噪声也大。
    v3 只保留高信噪比方向：加权用户中心、最近用户、最近目标。
    """
    q = np.asarray(env_sample.q_current, dtype=np.float32)
    users = np.asarray(env_sample.u_positions, dtype=np.float32)
    targets = np.asarray(env_sample.s_positions, dtype=np.float32)
    assoc = np.asarray(env_sample.association, dtype=np.float32)
    weights = np.asarray(env_sample.user_weights, dtype=np.float32)

    max_move = float(config.get("uav_max_speed_ms", 15.0)) * float(config.get("slot_duration_s", 1.0))
    lines = ["[Geometry Guidance g(t)]"]
    lines.append(
        f"  Compact movement cues: delta_q is a {max_move:.1f}m direction choice. "
        "Use the same three cues shown in the BEV image: weighted user center, nearest user, nearest target."
    )

    weighted_centroid = None
    if users.size > 0:
        user_centroid = users.mean(axis=0)
        weights_safe = np.maximum(weights, 1e-6)
        weighted_centroid = (users * weights_safe[:, None]).sum(axis=0) / weights_safe.sum()
        lines.append(f"  User centroid xy: {_fmt_vec(user_centroid, 1)}")
        lines.append(f"  Weighted-demand centroid xy: {_fmt_vec(weighted_centroid, 1)}")

    loads = assoc.sum(axis=1).astype(int).tolist() if assoc.size else [0 for _ in range(q.shape[0])]
    for m in range(q.shape[0]):
        q_xy = q[m, :2]

        if weighted_centroid is not None:
            center_dir, center_dist = _unit_direction(q_xy, weighted_centroid)
            center_text = f"weighted_center:d={center_dist:.1f}m,dir={_fmt_vec(center_dir, 3)}"
        else:
            center_text = "weighted_center:n/a"

        if users.size > 0:
            d_user = np.linalg.norm(users - q_xy[None, :], axis=1)
            nearest_user = int(np.argmin(d_user))
            user_dir, user_dist = _unit_direction(q_xy, users[nearest_user])
            user_text = (
                f"nearest_user=u{nearest_user}:d={user_dist:.1f}m,"
                f"dir={_fmt_vec(user_dir, 3)},w={float(weights[nearest_user]):.2f}"
            )
        else:
            user_text = "nearest_user:n/a"

        if targets.size > 0:
            d_target = np.linalg.norm(targets - q_xy[None, :], axis=1)
            nearest_t = int(np.argmin(d_target))
            target_dir, target_dist = _unit_direction(q_xy, targets[nearest_t])
            target_text = f"nearest_target=t{nearest_t}:d={target_dist:.1f}m,dir={_fmt_vec(target_dir, 3)}"
        else:
            target_text = "nearest_target:n/a"

        lines.append(
            f"  UAV {m}: xy={_fmt_vec(q_xy, 1)}, h={float(q[m, 2]):.1f}m, load={loads[m]}, "
            f"{center_text}, {user_text}, {target_text}"
        )

    return "\n".join(lines)


def build_indexed_association_str(env_sample) -> str:
    """为 association 输出补齐用户列索引与候选 UAV 链路信息。"""
    users = np.asarray(env_sample.u_positions, dtype=np.float64)
    weights = np.asarray(env_sample.user_weights, dtype=np.float64)
    gains = np.asarray(env_sample.channel_gains_users, dtype=np.float64)

    if users.ndim != 2 or users.shape[1] != 2:
        raise ValueError(f"u_positions must have shape (K, 2), got {users.shape}")
    num_users = users.shape[0]
    if weights.shape != (num_users,):
        raise ValueError(
            f"user_weights must have shape {(num_users,)}, got {weights.shape}"
        )
    if gains.ndim != 2 or gains.shape[1] != num_users:
        raise ValueError(
            "channel_gains_users must have shape (M, K) aligned with users, "
            f"got {gains.shape}"
        )

    lines = [
        "[Indexed Association Map]",
        "  delta_a rows follow UAV IDs m0..; columns follow user IDs u0.. in the order below.",
        "  rank lists candidate UAVs from strongest to weakest channel; rel_db uses the same rank order.",
    ]
    per_user_sinr = env_sample.comm_summary.get("per_user_sinr_db", [])
    for k in range(num_users):
        gain_k = np.maximum(gains[:, k], 1e-30)
        rank = np.argsort(-gain_k)
        best_gain = gain_k[rank[0]]
        rel_db = 10.0 * np.log10(gain_k[rank] / best_gain)
        rank_text = ">".join(f"m{int(m)}" for m in rank)
        rel_text = ",".join(f"{float(value):.1f}" for value in rel_db)
        sinr_text = (
            f"{float(per_user_sinr[k]):.1f}"
            if k < len(per_user_sinr)
            else "n/a"
        )
        lines.append(
            f"  u{k}:xy={_fmt_vec(users[k], 1)},w={float(weights[k]):.2f},"
            f"best_sinr_db={sinr_text},rank={rank_text},rel_db=[{rel_text}]"
        )

    return "\n".join(lines)


def build_full_prompt(
    env_sample,
    config: dict,
) -> str:
    """
    构造完整的多模态 prompt Π(t)

    格式: System Instruction + Communication Summary + Sensing Summary + BEV Grid

    Args:
        env_sample: EnvironmentSample 对象
        config: 仿真配置 dict

    Returns:
        完整 prompt 字符串
    """
    parts = []

    # 1. 系统指令
    parts.append(build_system_prompt(config))

    # 2. 通信摘要
    parts.append(build_communication_summary_str(env_sample.comm_summary))

    # 3. 感知摘要
    parts.append(build_sensing_summary_str(env_sample.sensing_summary))

    # 4. BEV 文本网格
    parts.append(env_sample.bev_grid_text)

    # 5. 最终指令
    parts.append("\nNow propose the warm-start decision prior δ in JSON format.")

    return "\n\n".join(parts)


def build_multimodal_prompt(
    env_sample,
    config: dict,
) -> str:
    """
    Build the text part of a BEV-image multimodal prompt.

    This keeps c(t) and r(t) as text, but replaces the full text-grid BEV with a
    short description of the attached image. Model-specific image placeholders
    are intentionally left to the processor/chat-template layer.
    """
    parts = []

    parts.append(build_system_prompt(config))
    parts.append(build_communication_summary_str(env_sample.comm_summary))
    parts.append(build_sensing_summary_str(env_sample.sensing_summary))
    parts.append(build_geometry_guidance_str(env_sample, config))
    parts.append(build_indexed_association_str(env_sample))
    parts.append(
        "[Bird's-Eye-View Image]\n"
        "The attached BEV image uses the same compact geometry cues: blue triangles are UAVs, "
        "green users are scaled by demand weight, red X markers are sensing targets, "
        "blue rings show the per-slot mobility radius, purple lines point to the weighted user center, "
        "green lines point to nearest users, and orange dashed lines point to nearest sensing targets. "
        "Visual markers are intentionally uncluttered; use the Indexed Association Map for exact user IDs."
    )
    parts.append("\nNow propose the warm-start decision prior delta in JSON format.")

    return "\n\n".join(parts)


def format_oracle_response(sample_id: int, delta_q, delta_a, delta_p) -> str:
    """
    将 Oracle prior 序列化为 JSON 响应字符串

    对应论文中的 Ξ(Ω*) → δ (公式 14-16)

    Args:
        delta_q: (M, 3) UAV 位移
        delta_a: (M, K) 关联矩阵
        delta_p: (M, K+1) 功率分配

    Returns:
        JSON 格式的响应字符串 (浮点数截断至 4 位小数)
    """
    import json

    def _trunc(obj, ndigits=4):
        """递归截断浮点数精度。
        np.round 对 float32 不够：0.191 在 IEEE 754 中无法精确表示，
        .tolist() 会还原为 0.19099999964237213 这种 17 位噪声。
        Python round() 在 float64 下配合 json.dumps 则输出干净的 "0.191"。
        """
        if isinstance(obj, float):
            return round(obj, ndigits)
        if isinstance(obj, list):
            return [_trunc(v, ndigits) for v in obj]
        return obj

    response_dict = {
        "delta_q": _trunc(np.round(delta_q, 4).tolist()),
        "delta_a": _trunc(np.round(delta_a, 4).tolist()),
        "delta_p": _trunc(np.round(delta_p, 4).tolist()),
    }

    # Compact JSON — no indent. With 176 floats, indent=2 adds ~1400 chars
    # of whitespace/newlines that BPE tokenizer wastes tokens on (>200 tokens).
    return json.dumps(response_dict, indent=None, separators=(",", ":"))
