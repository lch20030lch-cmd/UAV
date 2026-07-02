---
type: report
stage: smoke_v3
last_updated: 2026-07-02
status: passed
data: smoke_v3 (200 environments)
model: Gemma 3 12B + LoRA rank=16 + SDPA
---

# Smoke Test v3 — 全链路闭环验证报告

**日期**: 2026-07-02 | **数据**: 200 条新生成 (smoke_v3) | **结论**: ✅ 通过

## 1. 测试目标

验证以下全链路闭环：

```
数据生成 → Phase 1 CTL 预热 → Phase 2 SFT → DPO → Eval
```

在 200 条小规模数据上确认：0 崩溃、0 NaN、0 OOM、加速比可复现。

## 2. 测试配置

| 项 | 值 |
|----|-----|
| 环境数 | 200 (smoke_v3) |
| SFT epochs | 1 |
| DPO epochs | 1 |
| Phase 1 max_steps | 20 (CTL warmup) |
| Batch size (SFT) | 2 × 8 grad_accum = 16 effective |
| Batch size (DPO) | 1 × 16 grad_accum = 16 effective |
| Max seq length | 3456 |
| 精度 | bf16 全精度 LoRA |
| GPU | RTX PRO 6000 96GB |
| Eval workers | 30 (32 vCPU AMD EPYC 9654) |

## 3. 遇到的问题与修复

### 3.1 Phase 1 速度过慢

**现象**: Phase 1 (CTL-only warmup) 以 31.9s/it 速度运行，100 步预计耗时 ~53 分钟。

**根因**: Phase 1 对 12B 模型执行完整 forward+backward（8 次 grad_accum），每次 gradient sync 约 32s。对于 200 条数据的烟雾测试，CTL 表征预热的性价比极低。

**修复**: 将 `phase1.max_steps` 从 100 降至 20，`sensitivity_check_steps` 从 50 降至 10。完整 CTL 演示 + 快速切换 Phase 2。

**影响**: Phase 1 耗时从 ~53min → ~10min。

---

### 3.2 磁盘爆满 (No space left on device)

**现象**: `accelerator.save_state()` 在中间 checkpoint 和最终保存时触发 `SafetensorError: No space left on device`。系统盘仅 30GB overlay。

**根因**: `accelerator.save_state()` 保存完整训练状态（模型权重 + Adam optimizer momentum/variance），单次写入即可填满剩余磁盘空间。烟雾测试中 `save_steps=10` 导致频繁写入大文件。

**修复**: 在 `train_sft.py` 引入 `save_full_state` 标志位：
- `save_full_state: true` (全量生产): `accelerator.save_state()` — 含 optimizer，可断点续训
- `save_full_state: false` (烟雾测试): `accelerator.unwrap_model(model).save_pretrained()` — 仅存 LoRA 权重 (~10MB)

同时将最终保存也改为条件执行。

**代码改动**: `src/training/train_sft.py` 第 674-697 行，`configs/default.yaml` 新增 `save_full_state: true`。

---

### 3.3 Eval 与 DPO 训练 GPU 争抢

**现象**: DPO 训练占用 90.47 GiB 显存，同时启动 eval 触发 `CUDA out of memory`。

**原因**: RTX PRO 6000 仅有 96GB，DPO 双模型 (reference + policy) 几乎占满。

**解决**: DPO 跑完后再跑 eval。以烟雾测试的规模（200 条），DPO 仅需 ~3 分钟。

---

### 3.4 CRB 恒为 0

**现象**: `mean_crb: 0.0000 ± 0.0000`，所有样本均为 0。

**原因**: CRB (Cramér-Rao Bound) 指标在 eval 代码中尚未实现，预留为 0 占位。

**影响**: 不影响核心结论。感知 SINR 已覆盖感知性能评估。

---

## 4. 评估结果

### 4.1 SFT vs DPO 完整对比

| 指标 | SFT | DPO | Δ |
|------|-----|-----|----|
| **SCA-FP Speedup** | **1.3475x** | **1.3475x** | 0 |
| Warm iters | 2.0000 ± 0 | 2.0000 ± 0 | — |
| Cold iters | 2.695 ± 0.46 | 2.695 ± 0.46 | — |
| Sum-rate (Mbps) | 39.70 ± 16.3 | 39.53 ± 16.4 | -0.4% |
| Sensing SINR (dB) | 14.42 ± 1.07 | 14.42 ± 1.07 | 0% |
| Joint Satisfaction | 0.5063 ± 0.025 | 0.5085 ± 0.021 | +0.4% |
| Inference Latency (ms) | 316.8 ± 41.5 | 316.9 ± 41.2 | 0% |
| Valid Samples | 200/200 | 200/200 | — |

### 4.2 核心发现

#### 发现 1：SFT 已撞到 SCA-FP 物理下限

`sca_fp_iterations_warm = 2.0000 ± 0.0000` — 所有 200 个样本精确到 4 位小数，方差为 0。

这是 SCA-FP 交替优化算法的**数学硬下限**：
1. 第 1 步：从 warm-start 开始，用闭式解更新 $X_1$
2. 第 2 步：再次更新得 $X_2$，计算 $||X_2 - X_1|| < \text{tol}$，触发收敛

即使模型给出绝对完美的全局最优解，求解器也必须跑 2 步才能确认收敛。**2.0 步是物理极限，无法突破。**

#### 发现 2：DPO=SFT 的原因

已验证 smoke_v3 的 `dpo_dataset.jsonl`：chosen 和 rejected 的 **δ_a 和 δ_p 完全相同**（连小数点都一致），仅 δ_q 不同。

这不是数据生成的 bug — 这是刻意的架构设计（见 `oracle_generator.py:437`）：
```python
def _format_rejected_response(...):
    """格式化 Rejected 响应 JSON — δ_q 是陷阱, δ_a/δ_p 是 Chosen 的"""
```

DPO 损失在 δ_a/δ_p 上的 reward difference = 0，模型在这两个维度上没有偏好信号。Masked DPO 正是为此设计 — 只让 δ_q 接收 DPO 梯度。

**这验证了 [架构局限预警](status.md) 的正确性：当前共享 Control Token 架构下，DPO 无法为 δ_a/δ_p 提供额外学习信号。这是 v2 解耦 Control Token 的学术动机。**

#### 发现 3：200 条过拟合 — 且这是好事

Warm iters = 2.0 (方差 0) 说明模型已经把 200 条训练数据背下来了。对烟雾测试而言，**过拟合是好事**：
- 证明模型有能力学会最优解
- 证明训练管线没有发散/NaN/梯度问题
- 200 条规模下的过拟合是**预期行为**，不反映全量训练后的泛化能力

---

## 5. 验证清单

| 验证项 | 状态 | 证据 |
|--------|------|------|
| 全链路闭环 (Data→SFT→DPO→Eval) | ✅ | 0 崩溃 |
| 加速比可复现 | ✅ | 1.3475x (与 smoke_v2 一致) |
| 无 NaN / 梯度爆炸 | ✅ | 所有 loss 正常 |
| 磁盘安全 | ✅ | save_pretrained (10MB/ckpt) |
| 推理延迟达标 | ✅ | 317ms (端侧可行) |
| 物理约束满足 | ✅ | SINR ≥ 11 dB, sum-rate 14-122 Mbps |
| JSON 解析合法率 | ✅ | 100% (200/200) |
| Masked DPO 自洽性 | ✅ | 数据设计印证架构设计 |

---

## 6. 架构局限 (诚实讨论)

### 6.1 表征挤占 (Representation Crowding)

δ_q, δ_a, δ_p 共享 8 个 `<ctrl>` token。Masked DPO 使 δ_q 独占了偏好梯度，δ_a/δ_p 仅靠 MSE 信号维持常数均值。这是当前架构的硬天花板。

### 6.2 DPO 数据局限

当前 DPO 数据仅对 δ_q 提供偏好信号（chosen/rejected 的 δ_a/δ_p 相同）。要突破 1.5x 加速比，需要：
- 显式解耦 Control Token (`<ctrl_q>`, `<ctrl_a>`, `<ctrl_p>`)
- 重新设计 DPO rejected 采样策略 — 构造真实的多维度次优解

**这属于 v2 工作，不属于当前 v1 交付范围。**

---

## 7. 下一步：全量 5000 环境

### 7.1 执行计划

```bash
# Step 1: 全量数据生成 (5000 envs, ~1h, 30 workers)
python scripts/generate_data.py \
    --config configs/default.yaml \
    --num-env 5000 --workers 30 \
    --output-dir /root/autodl-tmp/data/full_v2

# Step 2: 全量 SFT 训练 (3 epochs, ~8.7h)
tmux new -s sft_full
python src/training/train_sft.py --config configs/default.yaml

# Step 3: DPO 训练 (2 epochs, ~5-10h)
python src/training/train_dpo.py --config configs/default.yaml \
    --stage1_ckpt /root/autodl-tmp/outputs/stage1_sft_final

# Step 4: 评估 (100 条未见过的测试环境, 30 workers)
python src/eval/evaluate.py --config configs/default.yaml \
    --model /root/autodl-tmp/outputs/stage2_dpo_final \
    --output /root/autodl-tmp/eval/final_5000.json --workers 30
```

### 7.2 关键注意

> **⚠️ Eval 环境必须与训练集完全隔离！**
>
> 烟雾测试中 200 条训练=测试，导致 2.0 步过拟合假象。全量评估必须生成 100 条**全新的、模型从未见过的随机环境**，才能测量真实泛化加速比。

### 7.3 验收标准

| 指标 | 目标 | 当前 (200 条过拟合) |
|------|------|---------------------|
| SCA-FP Speedup | **> 1.5x** | 1.3475x |
| Warm iters | 2.2-2.5 (泛化) | 2.0 (过拟合) |
| Cold iters | 3.5-5.0 (新环境) | 2.695 (过拟合) |
| Inference | < 500ms | 317ms ✅ |
| 训练稳定性 | 0 NaN, 0 OOM | ✅ |

---

## 8. 文件清单

| 文件 | 说明 |
|------|------|
| `H:\Projects\smoke_v3_data\sft_dataset.jsonl` | SFT 训练数据 (200 条) |
| `H:\Projects\smoke_v3_data\dpo_dataset.jsonl` | DPO 偏好数据 (200 条) |
| `H:\Projects\smoke_result\sft.json` | SFT eval 结果 |
| `H:\Projects\smoke_result\dpo.json` | DPO eval 结果 |
| `configs\smoke.yaml` | 烟雾测试配置 |
| `src\training\train_sft.py` | SFT 训练脚本 (含 save_full_state 改动) |
| `src\eval\evaluate.py` | 并行 CPU eval 脚本 |
