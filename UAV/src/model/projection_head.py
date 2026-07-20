"""
可微约束投影头
论文 Section 5 — Differentiable Constraint-Projection Head

h_Φ(Z_c) → δ̂ = [Proj_Q(δ̃_q'), Proj_A(δ̃_a'), Proj_P(δ̃_p')]

架构 (公式 21-23):
  Z_c  (控制 token hidden states)
   ↓  Linear Readout
  δ̃   (raw continuous prior)
   ↓  Residual MLP f_Φ (公式 22)
  δ̃'  (corrected prior)
   ↓  Structured Projections (公式 23)
  δ̂   (feasible warm-start prior)

三个投影模块:
  Proj_Q: Clipping + tanh (高度/区域/移动性)
  Proj_P: Softmax + 功率预算  (per-UAV power budget)
  Proj_A: Sinkhorn + 容量     (capacitated transport)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualMLP(nn.Module):
    """
    残差 MLP 修正器 f_Φ (公式 22)

    δ̃' = δ̃ + MLP(δ̃)

    两层 hidden: 256 → 256
    """

    def __init__(self, in_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        prev_dim = in_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.GELU(),
                nn.LayerNorm(hd),
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, in_dim))  # 残差: 输出 = 输入维度
        self.net = nn.Sequential(*layers)

        # 初始化: 近乎 identity
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # 最后一层设为零 (初始时残差≈0)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ControlReadout(nn.Module):
    """
    控制 Token 读出 (公式 21)

    Z_c (control token hidden states) → δ̃ (raw continuous prior)

    Multi-Query Attention Pooling:
      - M 个独立 query (每架 UAV 一个)，各自从 control token 中提取专属信息
      - 单 query 方案被证伪: softmax 强制互斥，一个 query 无法同时关注 4 架无人机的独立状态
      - 32-token 扩容也被证伪: 出口瓶子仍是单个 query 向量，梯度被 softmax 稀释 32×
      - M 个 query 独立计算 attention，每个 UAV 有自己的"视角"
      - 共享 readout MLP 作用于每个 UAV 的 pooled vector
      - 初始化 query ~ N(0, 0.02) → 初始注意力接近均匀
    """

    def __init__(self, hidden_dim: int, num_control_tokens: int, out_dim: int, num_queries: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_control_tokens
        self.num_queries = num_queries

        # M 个独立可学习 query — 每个 UAV 一个，各自关注不同 control token
        self.attn_queries = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        nn.init.normal_(self.attn_queries, std=0.02)

        # 每个 query 输出的维度 = 总输出 / M
        # total_out = M*3 (pos) + M*K (assoc) + M*(K+1) (power), 能被 M 整除
        per_query_out = out_dim // num_queries
        self.per_query_out = per_query_out

        # 共享 readout: 对每个 UAV 的 pooled vector 独立作用
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, per_query_out),
        )

    def forward(self, control_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            control_states: (batch, num_control_tokens, hidden_dim)
                             控制 token 位置的 hidden states

        Returns:
            raw_prior: (batch, out_dim)  连续的原始 prior (M 个 UAV 的输出拼接)
        """
        B = control_states.shape[0]
        M = self.num_queries
        H = self.hidden_dim

        # M 个 query 独立 attention: queries (B,M,H) × keys^T (B,H,N) → (B,M,N)
        queries = self.attn_queries.expand(B, -1, -1)                     # (B, M, hidden_dim)
        scale = H ** 0.5
        attn_scores = torch.bmm(queries, control_states.transpose(1, 2)) / scale  # (B, M, N)
        attn_weights = F.softmax(attn_scores, dim=-1)                     # (B, M, N)

        # 每个 query 独立池化: (B,M,N) × (B,N,H) → (B, M, H)
        pooled = torch.bmm(attn_weights, control_states)                  # (B, M, hidden_dim)

        # 对每个 UAV 的 pooled vector 应用共享 readout
        pooled_flat = pooled.reshape(B * M, H)                            # (B*M, H)
        out_flat = self.readout(pooled_flat)                              # (B*M, per_query_out)
        out = out_flat.reshape(B, M * self.per_query_out)                # (B, total_out_dim)

        return out


class DeploymentProjection(nn.Module):
    """
    部署投影 Proj_Q
    公式 (24-26)

    强制执行:
      - 高度约束: H_min ≤ Ĥ ≤ H_max
      - 区域约束: x,y ∈ A
      - 移动性: ||Δq|| ≤ v_max * Δt
    """

    def __init__(
        self,
        area_w: float = 1000.0,
        area_h: float = 1000.0,
        h_min: float = 50.0,
        h_max: float = 300.0,
        v_max_dt: float = 15.0,
    ):
        super().__init__()
        self.register_buffer("area_w", torch.tensor(area_w))
        self.register_buffer("area_h", torch.tensor(area_h))
        self.register_buffer("h_min", torch.tensor(h_min))
        self.register_buffer("h_max", torch.tensor(h_max))
        self.register_buffer("v_max_dt", torch.tensor(v_max_dt))

    def forward(
        self,
        delta_tilde: torch.Tensor,     # (B, M, 3) — 位移 [dx, dy, dh]
        q_current: Optional[torch.Tensor] = None,  # (B, M, 3)
    ) -> torch.Tensor:
        """
        投影位移到约束空间

        如果提供 q_current:
          新位置 = clip(q_current + displacement)
        否则:
          只 clip 位移幅度
        """
        B, M, _ = delta_tilde.shape

        # 3D 移动性约束 (论文公式 28): ||Δq||_2 ≤ v_max * Δt
        # 对整个 3D 位移向量做范数裁剪, 而非仅裁剪水平分量
        displacement_3d = delta_tilde  # (B, M, 3)
        norms_3d = torch.norm(displacement_3d, dim=-1, keepdim=True) + 1e-8
        scale_3d = torch.clamp(self.v_max_dt / norms_3d, max=1.0)
        clipped_3d = displacement_3d * scale_3d

        if q_current is not None:
            # 从当前位置计算新坐标
            new_pos = q_current + clipped_3d  # (B, M, 3)

            # 区域裁剪 (x, y)
            new_pos_xy = torch.stack([
                torch.clamp(new_pos[..., 0], 0.0, self.area_w),
                torch.clamp(new_pos[..., 1], 0.0, self.area_h),
            ], dim=-1)

            # 高度裁剪
            new_pos_h = torch.clamp(new_pos[..., 2:3], self.h_min, self.h_max)

            # 拼接后转回位移
            new_pos_full = torch.cat([new_pos_xy, new_pos_h], dim=-1)
            result = new_pos_full - q_current
        else:
            result = clipped_3d

        return result


class PowerProjection(nn.Module):
    """
    功率投影 Proj_P
    公式 (28)

    p̂_m = P_max * softmax(p̃_m' / τ_p)

    默认严格实现论文中的 simplex 投影:
      Σ(通信 + 感知) = P_max
      每个条目 ≥ 0

    可选 ``p_min_ratio > 0`` 时，通信功率下界必须由 association 软门控，
    避免给未关联用户强制分配功率。默认关闭该可选下界，由下游 SCA-FP
    对关联条件下的最小用户功率做最终可行化。

    """

    def __init__(self, p_max: float = 1.0, tau: float = 0.5, p_min_ratio: float = 0.0):
        super().__init__()
        self.register_buffer("p_max", torch.tensor(p_max))
        self.p_min = p_min_ratio * p_max  # absolute floor
        self.tau = tau

    def forward(
        self,
        p_tilde: torch.Tensor,
        association: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            p_tilde: (B, M, K+1) — 原始功率得分
                     [:, :, :K] = 通信功率
                     [:, :, K]  = 感知功率

        Returns:
            p_hat: (B, M, K+1) — 投影后的功率分配
        """
        B, M, D = p_tilde.shape
        K_comm = D - 1  # number of communication users

        # softmax 归一化 (沿最后一维, 即 UAV 内部)
        p_soft = F.softmax(p_tilde / self.tau, dim=-1)  # (B, M, D)

        # 缩放到功率预算
        p_hat = self.p_max * p_soft

        if self.p_min <= 0:
            return p_hat

        if association is None:
            raise ValueError(
                "PowerProjection with p_min_ratio > 0 requires association weights."
            )
        if association.shape != p_tilde.shape[:-1] + (K_comm,):
            raise ValueError(
                "association must align with communication power entries: "
                f"expected {p_tilde.shape[:-1] + (K_comm,)}, got {tuple(association.shape)}"
            )

        # 关联感知的通信下界。association=0 时下界也是 0，避免与 oracle 中
        # 未关联用户的零功率标签冲突；感知功率不受该下界约束。
        assoc = association.to(device=p_tilde.device, dtype=p_tilde.dtype).clamp(0.0, 1.0)
        base_comm = self.p_min * assoc
        base_total = base_comm.sum(dim=-1, keepdim=True)

        # 极端情况下 association 软负载可能使下界总和超过预算；先整体缩放，
        # 再把剩余预算按原 softmax 权重分配，严格保持总功率不超过 P_max。
        base_scale = torch.clamp(
            self.p_max.to(dtype=p_tilde.dtype) / base_total.clamp_min(1e-8),
            max=1.0,
        )
        base_comm = base_comm * base_scale
        base_total = base_comm.sum(dim=-1, keepdim=True)
        remaining = (self.p_max.to(dtype=p_tilde.dtype) - base_total).clamp_min(0.0)
        extra = remaining * p_soft
        return torch.cat(
            [base_comm + extra[..., :K_comm], extra[..., K_comm:]],
            dim=-1,
        )


class AssociationProjection(nn.Module):
    """
    关联投影 Proj_A
    公式 (29)

    训练时: Sinkhorn 归一化 → 软关联 Ẑ ∈ [0,1]^{M×K}
    推理时: 离散化为二进制 Â

    保证:
      Σ_m Ẑ_{m,k} ≈ 1   (每列近1 — single association)
      Σ_k Ẑ_{m,k} ≤ K_max (每行容量上限)
    """

    def __init__(self, K_max: int = 10, tau: float = 0.5, n_iters: int = 20):
        super().__init__()
        self.K_max = K_max
        self.tau = tau
        self.n_iters = n_iters

    def forward(self, a_tilde: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a_tilde: (B, M, K) — 原始关联得分

        Returns:
            z_hat: (B, M, K) — 软关联矩阵
        """
        B, M, K = a_tilde.shape

        # Step 1: 稳定的 exp 缩放。减去每个用户列内最大值不改变后续列归一化结果，
        # 但能避免 raw logits 变大后 exp 溢出为 inf/nan。
        scaled = a_tilde / self.tau
        scaled = scaled - scaled.amax(dim=1, keepdim=True)
        S = torch.exp(scaled)  # (B, M, K)

        # Step 2: Sinkhorn 迭代 (列归一化 + 行容量裁剪)
        for _ in range(self.n_iters):
            # 列归一化 (每用户 ≈ 1)
            col_sum = S.sum(dim=1, keepdim=True) + 1e-8  # (B, 1, K)
            S = S / col_sum

            # 行容量裁剪 (每 UAV ≤ K_max)
            row_sum = S.sum(dim=2, keepdim=True) + 1e-8  # (B, M, 1)
            scale = torch.where(
                row_sum > self.K_max,
                self.K_max / row_sum,
                torch.ones_like(row_sum),
            )
            S = S * scale

        return torch.nan_to_num(S, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    @staticmethod
    def discretize(z_hat: torch.Tensor) -> torch.Tensor:
        """
        离散化软关联 → 二进制关联 (推理时使用)

        贪心算法: 每列取最大值, 容量超出时移到次优
        简化版: 直接 argmax + 容量后处理
        """
        # 简化为 argmax (列方向)
        best_m = torch.argmax(z_hat, dim=1)  # (B, K)
        B, M, K = z_hat.shape
        A = torch.zeros_like(z_hat)
        for b in range(B):
            for k in range(K):
                A[b, best_m[b, k], k] = 1.0
        return A


class ConstraintProjectionHead(nn.Module):
    """
    完整投影头 h_Φ (公式 23)

    默认 shared 模式保持原论文链路:
      Z_c → ControlReadout → δ̃ → ResidualMLP → δ̃'
      δ̃' → [Proj_Q, Proj_A, Proj_P] → δ̂

    split 模式用于多模态 smoke 中的分阶段训练:
      Z_c → q/a/p 三个独立 ControlReadout → 三个独立 ResidualMLP
      → [Proj_Q, Proj_A, Proj_P] → δ̂

    split 模式把离散关联任务和连续 q/p 回归任务解耦，避免 Stage B 训练 q/p 时
    通过共享 MLP 破坏 Stage A 已学到的 association 读出能力。
    """

    def __init__(
        self,
        hidden_dim: int = 2560,  # Gemma 3 4B hidden_size
        num_control_tokens: int = 8,
        mlp_hidden: list = None,
        readout_out_dim: int = 128,
        M: int = 4,
        K: int = 20,
        area_w: float = 1000.0,
        area_h: float = 1000.0,
        h_min: float = 50.0,
        h_max: float = 300.0,
        v_max_dt: float = 15.0,
        p_max: float = 1.0,
        K_max: int = 10,
        tau_power: float = 0.5,
        tau_assoc: float = 0.5,
        sinkhorn_iters: int = 20,
        head_type: str = "shared",
        q_projection_mode: str = "clip",
        q_geometry_mode: str = "none",
        q_fixed_cue_weights: list = None,
        q_residual_max_scale: float = 0.5,
    ):
        super().__init__()
        self.M = M
        self.K = K
        if head_type not in {"shared", "split"}:
            raise ValueError(f"Unsupported projection head_type: {head_type}")
        if q_projection_mode not in {"clip", "direction"}:
            raise ValueError(f"Unsupported q_projection_mode: {q_projection_mode}")
        if q_geometry_mode not in {"none", "cue_xy", "fixed_residual_xy"}:
            raise ValueError(f"Unsupported q_geometry_mode: {q_geometry_mode}")
        if q_geometry_mode == "fixed_residual_xy" and q_projection_mode != "direction":
            raise ValueError("fixed_residual_xy requires q_projection_mode='direction'")
        if q_residual_max_scale <= 0:
            raise ValueError("q_residual_max_scale must be positive")
        self.head_type = head_type
        self.q_projection_mode = q_projection_mode
        self.q_geometry_mode = q_geometry_mode
        self.q_residual_max_scale = float(q_residual_max_scale)

        if q_geometry_mode == "fixed_residual_xy" and q_fixed_cue_weights is None:
            raise ValueError(
                "fixed_residual_xy requires explicit q_fixed_cue_weights "
                "calibrated on the current training dataset"
            )
        fixed_weights = q_fixed_cue_weights or [1.0, 1.0, 1.0]
        if len(fixed_weights) != 3 or any(float(weight) < 0 for weight in fixed_weights):
            raise ValueError("q_fixed_cue_weights must contain three non-negative values")
        fixed_weights_tensor = torch.tensor(fixed_weights, dtype=torch.float32)
        if float(fixed_weights_tensor.sum()) <= 0:
            raise ValueError("q_fixed_cue_weights must have a positive sum")
        normalized_fixed_weights = fixed_weights_tensor / fixed_weights_tensor.sum()
        if q_geometry_mode == "fixed_residual_xy":
            self.register_buffer("q_fixed_cue_weights", normalized_fixed_weights)
            self.q_residual_adapter = nn.Linear(3, 3)
            nn.init.zeros_(self.q_residual_adapter.weight)
            nn.init.zeros_(self.q_residual_adapter.bias)
        else:
            # Keep old checkpoint state_dicts unchanged when the new geometry mode is unused.
            self.register_buffer(
                "q_fixed_cue_weights",
                normalized_fixed_weights,
                persistent=False,
            )
            self.q_residual_adapter = None

        # 读出维度 = M*3 (位移) + M*K (关联) + M*(K+1) (功率)
        self.q_dim = M * 3
        self.a_dim = M * K
        self.p_dim = M * (K + 1)
        self.total_delta_dim = self.q_dim + self.a_dim + self.p_dim

        if self.head_type == "shared":
            # 旧结构: 三个任务共用一个读出和一个残差修正器。
            self.readout = ControlReadout(hidden_dim, num_control_tokens, self.total_delta_dim, num_queries=M)
            self.mlp = ResidualMLP(self.total_delta_dim, mlp_hidden or [256, 256])
        else:
            # 新结构: q/a/p 分支完全解耦，便于分阶段冻结和诊断。
            self.readout_q = ControlReadout(hidden_dim, num_control_tokens, self.q_dim, num_queries=M)
            self.readout_a = ControlReadout(hidden_dim, num_control_tokens, self.a_dim, num_queries=M)
            self.readout_p = ControlReadout(hidden_dim, num_control_tokens, self.p_dim, num_queries=M)
            self.q_mlp = ResidualMLP(self.q_dim, mlp_hidden or [256, 256])
            self.a_mlp = ResidualMLP(self.a_dim, mlp_hidden or [256, 256])
            self.p_mlp = ResidualMLP(self.p_dim, mlp_hidden or [256, 256])

        # 可选 q 几何候选方向选择头。它只在传入 q_geometry_cues 且
        # q_geometry_mode=cue_xy 时参与输出，默认不改变旧路径。
        self.readout_q_cue = ControlReadout(hidden_dim, num_control_tokens, M * 3, num_queries=M)

        # 投影模块
        self.proj_q = DeploymentProjection(area_w, area_h, h_min, h_max, v_max_dt)
        self.proj_a = AssociationProjection(K_max, tau_assoc, sinkhorn_iters)
        self.proj_p = PowerProjection(p_max, tau_power)

    def _unflatten(self, delta_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将 (B, total_dim) 拆分为 (δ_q, δ_a, δ_p)
        """
        B = delta_flat.shape[0]
        M, K = self.M, self.K
        idx = 0

        delta_q = delta_flat[:, idx:idx + M * 3].reshape(B, M, 3)
        idx += M * 3

        delta_a = delta_flat[:, idx:idx + M * K].reshape(B, M, K)
        idx += M * K

        delta_p = delta_flat[:, idx:idx + M * (K + 1)].reshape(B, M, K + 1)

        return delta_q, delta_a, delta_p

    def _prepare_delta_q(self, delta_q_raw: torch.Tensor) -> torch.Tensor:
        """
        准备送入 Proj_Q 的 q 先验。

        clip 模式保持旧逻辑：raw delta 直接交给 Proj_Q 做移动半径裁剪。
        direction 模式面向 15m 边界饱和数据：raw delta 只表达方向，
        先归一化到 v_max_dt 半径，再交给 Proj_Q 做区域/高度等物理裁剪。
        """
        if self.q_projection_mode == "clip":
            return delta_q_raw
        direction = F.normalize(delta_q_raw, p=2, dim=-1, eps=1e-6)
        radius = self.proj_q.v_max_dt.to(device=delta_q_raw.device, dtype=delta_q_raw.dtype)
        return direction * radius

    def _compose_q_from_geometry_cues(
        self,
        delta_q_raw: torch.Tensor,
        q_geometry_cues: Optional[torch.Tensor],
        q_cue_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        用 prompt/BEV 的三类候选 xy 方向组合 q 位移。

        q_geometry_cues: (B, M, 3, 2)，顺序为 weighted center / nearest user /
        nearest target。xy 方向由 cue 权重加权，z 方向暂时保留 raw q 的 dh，
        因为当前 v3 几何提示主要表达水平移动方向。
        """
        if self.q_geometry_mode == "none":
            return self._prepare_delta_q(delta_q_raw)

        if q_geometry_cues is None:
            raise ValueError(f"q_geometry_cues are required for q_geometry_mode={self.q_geometry_mode}")

        cues = q_geometry_cues.to(device=delta_q_raw.device, dtype=delta_q_raw.dtype)
        radius = self.proj_q.v_max_dt.to(device=delta_q_raw.device, dtype=delta_q_raw.dtype)

        if self.q_geometry_mode == "fixed_residual_xy":
            weights = self.q_fixed_cue_weights.to(device=cues.device, dtype=cues.dtype)
            fixed_xy = torch.sum(weights.view(1, 1, 3, 1) * cues, dim=2)
            fixed_xy = F.normalize(fixed_xy, p=2, dim=-1, eps=1e-6)
            fixed_direction = torch.cat([fixed_xy, torch.zeros_like(fixed_xy[..., :1])], dim=-1)
            residual = torch.tanh(self.q_residual_adapter(delta_q_raw))
            combined_direction = F.normalize(
                fixed_direction + self.q_residual_max_scale * residual,
                p=2,
                dim=-1,
                eps=1e-6,
            )
            return combined_direction * radius

        if q_cue_logits is None:
            raise ValueError("q_cue_logits are required for q_geometry_mode=cue_xy")
        weights = F.softmax(q_cue_logits.to(dtype=delta_q_raw.dtype), dim=-1)
        cue_xy = torch.sum(weights.unsqueeze(-1) * cues, dim=2)
        cue_xy = F.normalize(cue_xy, p=2, dim=-1, eps=1e-6)
        q_xy = cue_xy * radius
        return torch.cat([q_xy, delta_q_raw[..., 2:3]], dim=-1)

    def forward(
        self,
        control_states: torch.Tensor,      # (B, num_control_tokens, hidden_dim)
        q_current: Optional[torch.Tensor] = None,  # (B, M, 3)
        q_geometry_cues: Optional[torch.Tensor] = None,  # (B, M, 3, 2)
    ) -> dict:
        """
        前向传播

        Args:
            control_states: 控制 token 位置的 hidden states
            q_current: 当前 UAV 位置 (用于部署投影)

        Returns:
            dict with:
              "delta_q": (B, M, 3)   UAV 位移先验
              "delta_a": (B, M, K)   关联先验 (软)
              "delta_p": (B, M, K+1) 功率先验
              "delta_raw": (B, total_dim) 未投影的原始 prior
        """
        if self.head_type == "shared":
            # Step 1: 读出
            delta_raw = self.readout(control_states)  # (B, total_dim)

            # Step 2: MLP 修正
            delta_corrected = self.mlp(delta_raw)     # (B, total_dim)

            # Step 3: 拆分 + 分别投影
            dq, da, dp = self._unflatten(delta_corrected)
        else:
            # split 模式下三条分支独立读出、独立修正。
            dq_raw = self.readout_q(control_states)
            da_raw = self.readout_a(control_states)
            dp_raw = self.readout_p(control_states)
            delta_raw = torch.cat([dq_raw, da_raw, dp_raw], dim=-1)

            dq = self.q_mlp(dq_raw).reshape(control_states.shape[0], self.M, 3)
            da = self.a_mlp(da_raw).reshape(control_states.shape[0], self.M, self.K)
            dp = self.p_mlp(dp_raw).reshape(control_states.shape[0], self.M, self.K + 1)

        q_cue_logits = None
        q_cue_weights = None
        if self.q_geometry_mode == "cue_xy" and q_geometry_cues is not None:
            q_cue_logits = self.readout_q_cue(control_states).reshape(control_states.shape[0], self.M, 3)
            q_cue_weights = F.softmax(q_cue_logits, dim=-1)

        dq_for_projection = self._compose_q_from_geometry_cues(dq, q_geometry_cues, q_cue_logits)
        dq_proj = self.proj_q(dq_for_projection, q_current)
        da_proj = self.proj_a(da)
        dp_proj = self.proj_p(dp, association=da_proj)

        result = {
            "delta_q": dq_proj,
            "delta_a": da_proj,
            "delta_p": dp_proj,
            "delta_raw": delta_raw,
            "delta_q_raw": dq,
            "delta_a_raw": da,
            "delta_p_raw": dp,
        }
        if q_cue_logits is not None:
            result["q_cue_logits"] = q_cue_logits
            result["q_cue_weights"] = q_cue_weights
        if self.q_geometry_mode == "fixed_residual_xy":
            result["q_fixed_cue_weights"] = self.q_fixed_cue_weights
        return result
