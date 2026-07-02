---
type: status
status: current
stage: data_regeneration
last_updated: 2026-07-02
related: [onboarding, quickstart, canonical_config]
---

# 项目当前状态

**最后更新**: 2026-07-02 | **阶段**: 🟡 全量数据重生 (5000 envs) → SFT → DPO

## 一句话

Smoke v3 全链路闭环验证通过 (1.347x speedup, 仅 200 条数据)。两个根因已修复，管线可完整跑通。下一步：全量 5000 环境数据重生 → SFT 3 epochs → DPO 2 epochs → 评估。

## 🔴 双重根因 (均已修复)

### 根因 1: 数据退化 — 地面杂波缺失

SCA-FP 求解器缺乏地面杂波建模 → 97.4% 向下、84.7% 满速、84.3% 满功率的退化解。修复：`ground_clutter_db=12.0` 创造非凸 trade-off。

→ 详见 [data_degeneracy.md](../03_bugs/resolved/data_degeneracy.md)

### 根因 2: q_current 缺失 — Mode Collapse

旧代码不写 `q_current` 字段，`numel()>0` silent pass 掩盖了缺失 → loss_sep=0 → 模型输出恒值模板 (0.893x)。修复：`has_q_current` boolean flag + 统一 tensor shape。

→ 详见 [q_current_missing.md](../03_bugs/resolved/q_current_missing.md)

**后果**：2026-07-02 之前生成的全部数据 (19,925 SFT + DPO) 作废。

## 📦 当前数据资产

```
⚠️ 旧数据全部作废 (缺失 q_current):
/root/autodl-tmp/data/cache/
├── sft_dataset.jsonl      19,925 条 ❌
├── dpo_dataset.jsonl      19,925 条 ❌
├── dpo_top5000.jsonl       5,000 条 ❌

✅ 新数据 (含 q_current, 200 条 smoke test):
/root/autodl-tmp/data/smoke_v3/
├── sft_dataset.jsonl         200 条 ✅
├── dpo_dataset.jsonl         200 条 ✅

🟡 待生成:
/root/autodl-tmp/data/full_v2/ (5000 envs, 全量生产)
```

## 🏗️ 架构终态

```
Control Token (8 个, <ctrl_0>..<ctrl_7>)
       ↓ Gemma 3 12B (LoRA, rank=16, SDPA, bf16)
Control Hidden States [B, 8, 3840]
       ↓ Multi-Query Attention Pooling (4 queries → 4 UAV pools)
       ↓ Shared Readout MLP (3840→1920→960→44 per UAV)
       ↓ ResidualMLP → Unflatten → Constraint Projections
  δ_q [B,4,3] + δ_a [B,4,20] + δ_p [B,4,21]
```

**关键参数**: bs=2, grad_accum=8, seq=3456, bf16 全精度, SDPA attention, 0 Unsloth

→ 详见 [01_architecture/](../01_architecture/)

### ⚠️ 已知架构局限：表征饥饿 (Representation Starvation)

Masked DPO 下 δ_q/δ_a/δ_p 共享 control token embedding → δ_q 独占偏好梯度 → δ_a/δ_p 退化为常数均值。这是当前 1.3-1.5x 加速比的天花板。

→ 详见 [adr_007_dpo_masking_strategy.md](../06_decisions/adr_007_dpo_masking_strategy.md)

## ✅ 已解决

| 类别 | 项 |
|------|-----|
| 代码 | 7 轮审查闭合，全部源码完成 |
| OOM | #1-#7 全部修复 (省 ~54 GB, bs=2 稳定) |
| 架构 | Plan A (纯 PyTorch), Multi-Query Pooling, 地面杂波修复 |
| 数据 | 数据退化 + q_current 两个根因修复 |
| 验证 | Smoke v3 全链路闭环通过 (1.347x speedup @ 200 条) |
| Bug | 20 个 bug 全部关闭 ([registry](../03_bugs/README.md)) |

## ⏭️ 下一步

### 1. 全量数据生成 (5000 envs, ~1h)

```bash
cd /root/UAV-ISAC-MLLM && git pull
python scripts/generate_data.py \
    --config configs/default.yaml \
    --num-env 5000 --workers 30 \
    --output-dir /root/autodl-tmp/data/full_v2
```

### 2. Stage I SFT (3 epochs, ~8.7h)

```bash
tmux new -s sft_full
python src/training/train_sft.py --config configs/default.yaml
```

### 3. Stage II DPO (2 epochs, ~5-10h)

```bash
python src/training/train_dpo.py --config configs/default.yaml \
    --stage1_ckpt /root/autodl-tmp/outputs/stage1_sft_final
```

### 4. 评估 (100 条全新测试环境)

```bash
python src/eval/evaluate.py --config configs/default.yaml \
    --model /root/autodl-tmp/outputs/stage2_dpo_final \
    --data_dir /root/autodl-tmp/data/test_v1
```

## 🔑 关键经验 (精选)

1. **数据分布是真正的天花板** — 先看数据，再看模型。两个根因都是数据问题。
2. **CE Loss 对连续物理量有硬天花板** — 没有距离度量，这是 SFT 在物理回归上的根本局限。
3. **MSE 代理指标会背叛你** — loss_ctl 和 sens 在训练后期背离。只有 SCA-FP 加速比是真实判据。
4. **单 Attention Query 是回归读出的隐形杀手** — softmax 互斥性使一个 query 无法同时关注多个独立目标。
5. **共享 Control Token 是表征瓶颈** — δ_q/δ_a/δ_p 共享 embedding 时，DPO 梯度会挤占 δ_a/δ_p 的表征空间。
6. **奥卡姆剃刀优先** — 先修数据 → 再换训练方法 → 最后改架构。
7. **Unsloth 不存在"局部借用"** — 即使函数体内 import，仍全局 monkey-patch。与 Gemma 3 + SDPA 不可共存。

## 🃏 后备方案 (若 DPO 失败)

| 优先级 | 方案 | 改动量 | 原理 |
|--------|------|--------|------|
| **P0** | CoT 注入 | 改 prompt | 让模型先推理再输出数值 |
| **P1** | Regression Head | ~200 行 | MSE 回归替代 CE 离散化 |
| **P2** | Online RL (PPO) | 大改动 | 直接用 SCA-FP 加速比做 reward |

→ 详见 [adr_006](../06_decisions/adr_006_data_regeneration.md)
