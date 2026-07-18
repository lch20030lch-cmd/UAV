"""
损失函数
论文 Section 4 & 5

损失汇总:
  Stage I:
    L_I = L_SFT + λ_ctl * L_ctl              (公式 30)

  Stage II:
    L_II = L_DPO + μ * L_SFT + λ_ctl * L_ctl (公式 37)

  Total:
    L = L_II + λ_sep * L_sep                  (公式 39)

其中:
  L_SFT: causal LM cross-entropy (公式 27)
  L_ctl: continuous warm-start regression (公式 28)
  L_DPO: direct preference optimization (公式 34)
  L_sep: UAV separation penalty (公式 27)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class UAVISACLosses:
    """
    UAV-ISAC 训练损失计算器

    封装所有论文中的损失函数, 解耦 token-level 和 continuous losses
    """

    def __init__(
        self,
        lambda_ctl: float = 0.5,
        lambda_q: float = 1.0,
        lambda_a: float = 0.5,
        lambda_p: float = 0.3,
        lambda_sep: float = 0.1,
        lambda_assoc_ce: float = 0.0,
        lambda_assoc_raw_ce: float = 0.0,
        lambda_q_dir: float = 0.0,
        lambda_q_cue_ce: float = 0.0,
        lambda_p_raw_kl: float = 0.0,
        power_temperature: float = 0.5,
        dpo_beta: float = 0.1,
        sft_anchor_mu: float = 0.05,
    ):
        self.lambda_ctl = lambda_ctl
        self.lambda_q = lambda_q
        self.lambda_a = lambda_a
        self.lambda_p = lambda_p
        self.lambda_sep = lambda_sep
        self.lambda_assoc_ce = lambda_assoc_ce
        self.lambda_assoc_raw_ce = lambda_assoc_raw_ce
        self.lambda_q_dir = lambda_q_dir
        self.lambda_q_cue_ce = lambda_q_cue_ce
        self.lambda_p_raw_kl = lambda_p_raw_kl
        if power_temperature <= 0:
            raise ValueError(f"power_temperature must be positive, got {power_temperature}")
        self.power_temperature = power_temperature
        self.dpo_beta = dpo_beta
        self.sft_anchor_mu = sft_anchor_mu

    def compute_association_ce_loss(
        self,
        delta_a_hat: torch.Tensor,
        delta_a_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        按用户计算 association 分类损失。

        delta_a 的列表示用户，行表示 UAV。oracle target 是近似 one-hot，
        因此可以把每个用户的 UAV 选择视为 M 类分类问题。
        """
        pred = torch.clamp(delta_a_hat, min=1e-8, max=1.0)
        pred = pred.permute(0, 2, 1).contiguous()  # (B, K, M)
        target_idx = torch.argmax(delta_a_target, dim=1).contiguous()  # (B, K)
        return F.nll_loss(
            torch.log(pred).view(-1, pred.shape[-1]),
            target_idx.view(-1),
        )

    def compute_association_raw_ce_loss(
        self,
        delta_a_raw: torch.Tensor,
        delta_a_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        直接对 projection head 输出的 association logits 计算分类损失。

        该项绕开 Sinkhorn/概率投影，专门检查 readout 是否能学出正确排序。
        """
        logits = delta_a_raw.permute(0, 2, 1).contiguous()  # (B, K, M)
        target_idx = torch.argmax(delta_a_target, dim=1).contiguous()  # (B, K)
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            target_idx.view(-1),
        )

    def compute_q_direction_loss(
        self,
        delta_q_raw: torch.Tensor,
        delta_q_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        对 UAV 位移方向做归一化监督。

        mm_smoke_100 中 delta_q target 几乎全部贴着 15m 移动边界，
        因此 q 学习本质上更接近方向学习，而不是自由位移回归。
        该损失绕开 Proj_Q 的半径裁剪，直接约束 raw q 分支输出方向。
        """
        pred_dir = F.normalize(delta_q_raw, p=2, dim=-1, eps=1e-6)
        target_dir = F.normalize(delta_q_target, p=2, dim=-1, eps=1e-6)
        return F.mse_loss(pred_dir, target_dir)

    def compute_q_cue_ce_loss(
        self,
        q_cue_logits: torch.Tensor,
        q_geometry_cues: torch.Tensor,
        delta_q_target: torch.Tensor,
        q_geometry_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        几何 cue 分类损失。

        对每架 UAV，先选出与 oracle delta_q 水平移动方向 cosine 最大的 cue，
        再监督 q_cue_logits 预测该 cue。这样 q 分支先学会“沿 prompt/BEV 中
        哪条候选线移动”，而不是直接自由回归 dx/dy。
        """
        logits = q_cue_logits.to(dtype=torch.float32)
        cues = q_geometry_cues.to(device=logits.device, dtype=torch.float32)
        target_xy = delta_q_target[..., :2].to(device=logits.device, dtype=torch.float32)
        target_dir = F.normalize(target_xy, p=2, dim=-1, eps=1e-6)
        cue_dir = F.normalize(cues, p=2, dim=-1, eps=1e-6)
        cosine = torch.sum(cue_dir * target_dir.unsqueeze(2), dim=-1)
        if q_geometry_mask is not None:
            mask = q_geometry_mask.to(device=logits.device, dtype=torch.bool)
            cosine = cosine.masked_fill(~mask, -1e4)
        target_idx = torch.argmax(cosine, dim=-1)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_idx.reshape(-1),
        )

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """按逻辑组求均值，避免大量未关联零条目淹没有效功率监督。"""
        squared_error = (prediction - target).pow(2)
        mask = mask.to(device=prediction.device, dtype=torch.bool)
        if not mask.any():
            return squared_error.new_tensor(0.0)
        return squared_error[mask].mean()

    def compute_power_loss(
        self,
        delta_p_hat: torch.Tensor,
        delta_p_target: torch.Tensor,
        delta_a_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """关联感知的功率回归损失。

        oracle ``delta_p`` 的通信部分只在 ``delta_a_target=1`` 的位置非零。
        若直接对全部 ``M*K`` 条目做一个 MSE，大量未关联零条目会主导梯度，
        使最容易的解退化为近常数小功率。这里分别归一化三类监督：

        1. 已关联用户通信功率；
        2. 未关联用户零功率/泄漏；
        3. 感知功率。

        三组等权平均，使损失不随未关联条目数量线性放大。
        """
        if delta_p_hat.shape != delta_p_target.shape:
            raise ValueError(
                "delta_p prediction/target shapes differ: "
                f"{tuple(delta_p_hat.shape)} != {tuple(delta_p_target.shape)}"
            )
        num_comm_users = delta_p_hat.shape[-1] - 1
        if delta_a_target.shape != delta_p_hat.shape[:-1] + (num_comm_users,):
            raise ValueError(
                "delta_a_target must align with communication power entries: "
                f"expected {delta_p_hat.shape[:-1] + (num_comm_users,)}, "
                f"got {tuple(delta_a_target.shape)}"
            )

        pred_comm = delta_p_hat[..., :num_comm_users]
        target_comm = delta_p_target[..., :num_comm_users]
        active_mask = delta_a_target > 0.5
        inactive_mask = ~active_mask

        loss_active = self._masked_mse(pred_comm, target_comm, active_mask)
        loss_inactive = self._masked_mse(pred_comm, target_comm, inactive_mask)
        loss_sensing = F.mse_loss(
            delta_p_hat[..., num_comm_users:],
            delta_p_target[..., num_comm_users:],
        )
        loss = (loss_active + loss_inactive + loss_sensing) / 3.0
        return loss, {
            "loss_p_active": loss_active,
            "loss_p_inactive": loss_inactive,
            "loss_p_sensing": loss_sensing,
        }

    def compute_power_raw_kl_loss(
        self,
        delta_p_raw: torch.Tensor,
        delta_p_target: torch.Tensor,
    ) -> torch.Tensor:
        """在 PowerProjection 前用 soft-target KL 提供不饱和的功率梯度。

        projected MSE 在 softmax 错误饱和为 one-hot 后梯度可能接近 0。
        KL 对 raw logits 的梯度为 ``softmax(logits/tau) - target``，即使当前
        分布已经错误饱和也能继续纠正。oracle 每架 UAV 的功率和约为 1，
        这里仍显式归一化以兼容小的数值/舍入误差。
        """
        if delta_p_raw.shape != delta_p_target.shape:
            raise ValueError(
                "delta_p_raw/target shapes differ: "
                f"{tuple(delta_p_raw.shape)} != {tuple(delta_p_target.shape)}"
            )
        target = delta_p_target.clamp_min(0.0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        log_prediction = F.log_softmax(
            delta_p_raw / self.power_temperature,
            dim=-1,
        )
        target_log = torch.where(
            target > 0,
            torch.log(target.clamp_min(1e-12)),
            torch.zeros_like(target),
        )
        return torch.sum(target * (target_log - log_prediction), dim=-1).mean()

    def compute_control_loss(
        self,
        delta_hat: Dict[str, torch.Tensor],
        delta_target: Dict[str, torch.Tensor],
        return_components: bool = False,
    ):
        """
        连续控制损失 L_ctl (公式 28)

        L_ctl = λ_q ||δ̂_q - δ*_q||² + λ_a BCE(δ̂_a, δ*_a) + λ_p ||δ̂_p - δ*_p||²

        其中:
          - δ̂_q, δ̂_p → MSE (连续回归)
          - δ̂_a → BCE (软关联 vs 二值 oracle)
        """
        # Auto-align dtypes (projection head may be f32, model bf16, etc.)
        common_dtype = torch.float32
        dq_hat = delta_hat["delta_q"].to(dtype=common_dtype)
        da_hat = delta_hat["delta_a"].to(dtype=common_dtype)
        dp_hat = delta_hat["delta_p"].to(dtype=common_dtype)
        dq_tgt = delta_target["delta_q"].to(dtype=common_dtype)
        da_tgt = delta_target["delta_a"].to(dtype=common_dtype)
        dp_tgt = delta_target["delta_p"].to(dtype=common_dtype)

        # 位移 loss (MSE)。权重为 0 时跳过实际计算，避免无关分支的数值问题影响 smoke。
        loss_q = F.mse_loss(dq_hat, dq_tgt) if self.lambda_q != 0 else dq_hat.new_tensor(0.0)

        # 关联 loss (BCE: 软关联 vs 二值 oracle)。投影输出理论上在 [0,1]，
        # 这里仍做一次 clamp，防止 Sinkhorn 数值误差触发 CUDA BCE assert。
        da_prob = torch.clamp(torch.nan_to_num(da_hat, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        loss_a = (
            F.binary_cross_entropy(da_prob, da_tgt)
            if self.lambda_a != 0
            else da_hat.new_tensor(0.0)
        )

        # 可选辅助项: 按用户做 UAV 分类，专门约束 association argmax。
        loss_a_ce = (
            self.compute_association_ce_loss(da_prob, da_tgt)
            if self.lambda_assoc_ce != 0
            else da_hat.new_tensor(0.0)
        )
        if "delta_a_raw" in delta_hat:
            da_raw = delta_hat["delta_a_raw"].to(dtype=common_dtype)
            loss_a_raw_ce = self.compute_association_raw_ce_loss(da_raw, da_tgt)
        else:
            loss_a_raw_ce = da_hat.new_tensor(0.0)

        if "delta_q_raw" in delta_hat and self.lambda_q_dir != 0:
            dq_raw = delta_hat["delta_q_raw"].to(dtype=common_dtype)
            loss_q_dir = self.compute_q_direction_loss(dq_raw, dq_tgt)
        else:
            loss_q_dir = dq_hat.new_tensor(0.0)

        if (
            "q_cue_logits" in delta_hat
            and "q_geometry_cues" in delta_target
            and self.lambda_q_cue_ce != 0
        ):
            loss_q_cue_ce = self.compute_q_cue_ce_loss(
                delta_hat["q_cue_logits"],
                delta_target["q_geometry_cues"],
                dq_tgt,
                delta_target.get("q_geometry_mask"),
            )
        else:
            loss_q_cue_ce = dq_hat.new_tensor(0.0)

        # 功率 loss：按关联有效通信、未关联泄漏、感知功率三组分别归一化。
        if self.lambda_p != 0:
            loss_p, power_parts = self.compute_power_loss(dp_hat, dp_tgt, da_tgt)
        else:
            loss_p = dp_hat.new_tensor(0.0)
            power_parts = {
                "loss_p_active": dp_hat.new_tensor(0.0),
                "loss_p_inactive": dp_hat.new_tensor(0.0),
                "loss_p_sensing": dp_hat.new_tensor(0.0),
            }

        if "delta_p_raw" in delta_hat and self.lambda_p_raw_kl != 0:
            dp_raw = delta_hat["delta_p_raw"].to(dtype=common_dtype)
            loss_p_raw_kl = self.compute_power_raw_kl_loss(dp_raw, dp_tgt)
        else:
            loss_p_raw_kl = dp_hat.new_tensor(0.0)

        total = (
            self.lambda_q * loss_q
            + self.lambda_a * loss_a
            + self.lambda_p * loss_p
            + self.lambda_assoc_ce * loss_a_ce
            + self.lambda_assoc_raw_ce * loss_a_raw_ce
            + self.lambda_q_dir * loss_q_dir
            + self.lambda_q_cue_ce * loss_q_cue_ce
            + self.lambda_p_raw_kl * loss_p_raw_kl
        )

        if return_components:
            return total, {
                "loss_q": loss_q,
                "loss_a_bce": loss_a,
                "loss_a_ce": loss_a_ce,
                "loss_a_raw_ce": loss_a_raw_ce,
                "loss_q_dir": loss_q_dir,
                "loss_q_cue_ce": loss_q_cue_ce,
                "loss_p": loss_p,
                "loss_p_raw_kl": loss_p_raw_kl,
                **power_parts,
            }
        return total

    def compute_separation_penalty(
        self,
        q_hat: torch.Tensor,         # (B, M, 3) — 投影后的 UAV 位置
        d_min: float = 10.0,
    ) -> torch.Tensor:
        """
        UAV 分离惩罚 L_sep (公式 27)

        L_sep = Σ_{m<m'} [max(0, d_min - ||q̂_m - q̂_m'||_2)]²

        非凸约束转化为可微惩罚项
        """
        B, M, _ = q_hat.shape
        if M < 2:
            return torch.tensor(0.0, device=q_hat.device)

        total_penalty = 0.0
        for m in range(M):
            for mp in range(m + 1, M):
                diff = q_hat[:, m, :2] - q_hat[:, mp, :2]  # (B, 2) — 仅水平
                dist = torch.norm(diff, dim=-1)             # (B,)
                penalty = F.relu(d_min - dist) ** 2         # (B,)
                total_penalty += penalty.mean()

        return self.lambda_sep * total_penalty

    def compute_dpo_loss(
        self,
        logp_chosen: torch.Tensor,       # (B,)
        logp_rejected: torch.Tensor,     # (B,)
        logp_ref_chosen: torch.Tensor,   # (B,) — 冻结参考模型
        logp_ref_rejected: torch.Tensor, # (B,)
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        """
        DPO 损失 L_DPO (公式 34)

        L_DPO = -E[ log σ( β * log(π_θ(chosen)/π_ref(chosen))
                            - β * log(π_θ(rejected)/π_ref(rejected)) ) ]

        Args:
            logp_chosen: log π_θ(chosen|Π)
            logp_rejected: log π_θ(rejected|Π)
            logp_ref_chosen: log π_0(chosen|Π)
            logp_ref_rejected: log π_0(rejected|Π)
        """
        # 对数比 (相对于参考)
        chosen_ratio = logp_chosen - logp_ref_chosen       # (B,)
        rejected_ratio = logp_rejected - logp_ref_rejected # (B,)

        # DPO 目标
        logits = self.dpo_beta * (chosen_ratio - rejected_ratio)

        # label_smoothing (可选)
        if label_smoothing > 0:
            targets = 1.0 - label_smoothing
        else:
            targets = 1.0

        loss = -F.logsigmoid(logits)
        loss = loss.mean()

        # 准确率监控
        with torch.no_grad():
            accuracy = (logits > 0).float().mean()

        return loss, accuracy

    def compute_sft_loss(
        self,
        logits: torch.Tensor,        # (B, seq_len, vocab_size)
        labels: torch.Tensor,        # (B, seq_len)
        label_mask: Optional[torch.Tensor] = None,  # (B, seq_len)
    ) -> torch.Tensor:
        """
        SFT 损失 L_SFT (公式 27)

        标准 causal LM cross-entropy
        可用 label_mask 只计算 response 部分的 token

        纯 PyTorch 原生实现:
          直接用 F.cross_entropy 展平计算, 不依赖 Unsloth (会产生
          CheckpointError — 局部 import 仍触发全局 monkey-patch, 导致
          forward/recompute 张量数不一致 68≠65) 也不依赖梯度检查点.
          bs=1 时单步 CE 的 fp32 梯度约 4GB, 通过 grad_accum=16
          保持有效 batch=16 且显存安全.
        """
        # 右移: predict next token
        shift_logits = logits[:, :-1, :].contiguous()    # (B, S-1, V)
        shift_labels = labels[:, 1:].clone()              # (B, S-1)

        if label_mask is not None:
            shift_mask = label_mask[:, 1:]                # (B, S-1)
            shift_labels[shift_mask == 0] = -100

        # 展平为 2D (N, V) 匹配 F.cross_entropy 标准输入
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss

    def compute_phase1_total(
        self,
        delta_hat: Dict[str, torch.Tensor],
        delta_target: Dict[str, torch.Tensor],
        phase1_lambda_ctl: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Phase 1 CTL-only 损失: L = λ_ctl * L_ctl

        关闭 CE，强制 LoRA 学会将环境信息编码到 control token hidden states。
        用于 Phase 1 → Phase 2 分阶段训练。

        Returns:
            total_loss, metrics_dict
        """
        lambda_ctl = phase1_lambda_ctl if phase1_lambda_ctl is not None else self.lambda_ctl
        loss_ctl, ctl_parts = self.compute_control_loss(
            delta_hat,
            delta_target,
            return_components=True,
        )
        total = lambda_ctl * loss_ctl

        metrics = {
            "loss_ctl": loss_ctl.item(),
            "loss_total": total.item(),
            "phase": "phase1",
            "loss_q": ctl_parts["loss_q"].item(),
            "loss_a_bce": ctl_parts["loss_a_bce"].item(),
            "loss_a_ce": ctl_parts["loss_a_ce"].item(),
            "loss_a_raw_ce": ctl_parts["loss_a_raw_ce"].item(),
            "loss_q_dir": ctl_parts["loss_q_dir"].item(),
            "loss_q_cue_ce": ctl_parts["loss_q_cue_ce"].item(),
            "loss_p": ctl_parts["loss_p"].item(),
            "loss_p_raw_kl": ctl_parts["loss_p_raw_kl"].item(),
            "loss_p_active": ctl_parts["loss_p_active"].item(),
            "loss_p_inactive": ctl_parts["loss_p_inactive"].item(),
            "loss_p_sensing": ctl_parts["loss_p_sensing"].item(),
            "lambda_assoc_ce": self.lambda_assoc_ce,
            "lambda_assoc_raw_ce": self.lambda_assoc_raw_ce,
            "lambda_q_dir": self.lambda_q_dir,
            "lambda_q_cue_ce": self.lambda_q_cue_ce,
            "lambda_p_raw_kl": self.lambda_p_raw_kl,
        }
        return total, metrics

    def compute_stage1_total(
        self,
        delta_hat: Dict[str, torch.Tensor],
        delta_target: Dict[str, torch.Tensor],
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None,
        q_hat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Stage I 总损失: L_I = L_SFT + λ_ctl * L_ctl  (+ λ_sep * L_sep)

        Returns:
            total_loss, metrics_dict
        """
        loss_sft = self.compute_sft_loss(logits, labels, label_mask)
        loss_ctl = self.compute_control_loss(delta_hat, delta_target)

        total = loss_sft + self.lambda_ctl * loss_ctl

        metrics = {
            "loss_sft": loss_sft.item(),
            "loss_ctl": loss_ctl.item(),
        }

        if q_hat is not None:
            loss_sep = self.compute_separation_penalty(q_hat)
            total = total + loss_sep
            metrics["loss_sep"] = loss_sep.item()

        metrics["loss_total"] = total.item()
        return total, metrics

    def compute_stage2_total(
        self,
        delta_hat: Dict[str, torch.Tensor],
        delta_target: Dict[str, torch.Tensor],
        logp_chosen: torch.Tensor,
        logp_rejected: torch.Tensor,
        logp_ref_chosen: torch.Tensor,
        logp_ref_rejected: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        q_hat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Stage II 总损失: L = L_DPO + μ*L_SFT + λ_ctl*L_ctl + λ_sep*L_sep

        Returns:
            total_loss, metrics_dict
        """
        loss_dpo, dpo_acc = self.compute_dpo_loss(
            logp_chosen, logp_rejected, logp_ref_chosen, logp_ref_rejected
        )

        loss_ctl = self.compute_control_loss(delta_hat, delta_target)

        total = loss_dpo + self.lambda_ctl * loss_ctl

        metrics = {
            "loss_dpo": loss_dpo.item(),
            "dpo_accuracy": dpo_acc.item(),
            "loss_ctl": loss_ctl.item(),
        }

        # SFT anchor (防遗忘)
        if self.sft_anchor_mu > 0 and logits is not None and labels is not None:
            loss_sft = self.compute_sft_loss(logits, labels, label_mask)
            total = total + self.sft_anchor_mu * loss_sft
            metrics["loss_sft_anchor"] = loss_sft.item()

        # 分离惩罚
        if q_hat is not None:
            loss_sep = self.compute_separation_penalty(q_hat)
            total = total + loss_sep
            metrics["loss_sep"] = loss_sep.item()

        metrics["loss_total"] = total.item()
        return total, metrics
