# Smoke Test 2: q_current 缺失导致 Mode Collapse — 完整诊断与修复

**日期**: 2026-07-02
**严重程度**: 🔴 Critical — 导致全量 SFT 训练完全无效
**影响范围**: 全部已生成数据 (旧代码 19,925 SFT + 19,925 DPO) 作废

## 摘要

Stage I SFT smoke test (200 条, 2 epochs) 完成后 eval 发现 **SCA-FP speedup = 0.893x** — model warmstart 比 cold-start 还慢。根因为训练数据中完全没有 `q_current` 字段，导致投影头输入退化、分离惩罚始终为零、模型输出恒值模板。

修复后重跑 smoke test 恢复为 **1.347x speedup**，确认 `q_current` 为 mode collapse 的唯一原因。

## 时间线

| 阶段 | 事件 | 结果 |
|------|------|------|
| T1 | SFT smoke test (旧数据 sft_smoke.jsonl) | 训练完成, Phase 1→Phase 2 顺利切换 |
| T2 | step 25 checkpoint eval | SCA-FP speedup **0.893x** — mode collapse |
| T3 | 数据诊断 | `q_current` 字段完全缺失 |
| T4 | 根因分析 | 旧代码未写 `q_current`; `numel()>0` 检查 silent pass |
| T5 | 数据集修复 (`has_q_current` flag + 占位 zeros) | 消除 DataLoader collation 崩溃 |
| T6 | 数据重生 (新代码 200 envs, 含 `q_current`) | 验证通过 |
| T7 | Smoke test 重跑 | 训练正常 |
| T8 | step 200 eval | SCA-FP speedup **1.347x** — 修复确认 |

## 根因分析

### Bug 1: 数据生成代码未写 `q_current`

**文件**: `src/data/oracle_generator.py:164-173`

旧代码生成 SFT 样本时漏了 `q_current`:
```python
# 旧代码 (缺失 q_current)
sft_sample = {
    "id": f"env_{sample_id}",
    "prompt": prompt,
    "response": response_chosen,
    "delta_q": delta_q_chosen.tolist(),
    "delta_a": delta_a_chosen.tolist(),
    "delta_p": delta_p_chosen.tolist(),
}
# → sft_dataset.jsonl 中 d['q_current'] 报 KeyError
```

新代码已包含:
```python
# 新代码
sft_sample = {
    ...
    "q_current": q_current.tolist(),  # ← 此行修复
    ...
}
```

### Bug 2: `numel() > 0` 防御性检查掩盖了缺失

**文件**: `src/training/train_sft.py:411,529,549` 和 `src/training/train_dpo.py:331,340,350,357,399`

旧检查逻辑:
```python
q_current = batch["q_current"] if batch["q_current"].numel() > 0 else None
```

当 `q_current` 为空 tensor `torch.tensor([])` shape `[0]` 时, `numel()` = 0 → 直接传 `None` 给投影头, **不报错不告警**。

后果:
- `loss_sep` (UAV 分离惩罚) 永远为 0 (因为 `q_hat = q_current + delta_q = None + ... = None`)
- 投影头虽然从 control token 读取 `delta_q`/`delta_a`/`delta_p`, 但没有 `q_current` 做物理约束, 输出毫无意义
- CE loss 正常下降 → 训练看起来 "收敛良好" → **极具欺骗性**

### Bug 3: 旧 smoke test 在 eval 前未暴露

第一次全量 SFT 训练 (旧数据) 跑了 8.7h 到 step 150 才第一次 eval。
如果 smoke test 后立刻 eval, 30 分钟就能发现 collapse。

## 修复方案

### 修复 1: Dataset 层 — 统一 tensor shape + 真值标记

**文件**: `src/data/dataset.py:157-174` (SFT) 和 `src/data/dataset.py:231-242` (DPO)

```python
# 无论数据是否有 q_current, 都生成 shape [4, 3] 的 tensor
qc = item.get("q_current", None)
if qc and len(qc) > 0:
    result["q_current"] = torch.tensor(qc, dtype=torch.float32)
    result["has_q_current"] = torch.tensor(True)
else:
    result["q_current"] = torch.zeros(4, 3, dtype=torch.float32)
    result["has_q_current"] = torch.tensor(False)
```

**关键**: 用 `has_q_current` boolean flag 替代 `numel() > 0` 检查。
- `has_q_current = True` → 数据含真值, 正常使用
- `has_q_current = False` → 占位 zeros, 投影头回退 (loss_sep=0)

同时确保 DataLoader `default_collate` 不会因 shape 不一致崩溃 (之前 `[4,3]` 和 `[0]` 混在一起时 `torch.stack` 报错)。

### 修复 2: 训练层 — 用 `has_q_current` 替代 `numel()`

**文件**: `src/training/train_sft.py` (3 处) 和 `src/training/train_dpo.py` (5 处)

```python
# 旧代码
q_current=batch["q_current"] if batch["q_current"].numel() > 0 else None

# 新代码
q_current=batch["q_current"] if batch["has_q_current"].all() else None
```

### 修复 3: 数据重生

```bash
python scripts/generate_data.py --config configs/default.yaml \
    --num-env 200 --output-dir /root/autodl-tmp/data/smoke_v2 --workers 20
```

验证:
```python
>>> d.get('q_current')
present=True, len=4, first_UAV=[310.7, 574.0, 238.9]  # ✓ 4×3 真值
```

## 评估结果对比

| 指标 | 旧 Smoke (无 q_current) | 新 Smoke (含 q_current) |
|------|------------------------|------------------------|
| SCA-FP speedup | **0.893x** | **1.347x** |
| Warm iterations | 3.x | **2.00** |
| Cold iterations | 2.7 | 2.695 |
| Joint satisfaction | — | 0.506 |
| Mode collapse | ⚠️ δ_q 仅 Z 轴, δ_a/δ_p 均匀常数 | ✅ 正常 |
| Control sensitivity L2 | 0.0000 (全零) | — |

**结论**: `q_current` 修复后, 仅 200 条训练数据 warmstart 就将 SCA-FP 从 2.7→2.0 迭代 (节省 26%)。全量 5000 条预期进一步提升。

## 资产盘点

| 资产 | 状态 | 处置 |
|------|------|------|
| `/root/autodl-tmp/data/full5000/sft_dataset.jsonl` (19,925 条) | ❌ 无 q_current | 作废 |
| `/root/autodl-tmp/data/full5000/dpo_dataset.jsonl` (19,925 对) | ❌ 无 q_current | 作废 |
| `/root/autodl-tmp/data/smoke_v2/sft_dataset.jsonl` (200 条) | ✅ 含 q_current | 保留 |
| `/root/autodl-tmp/checkpoints/stage1_step_*` (旧 step150 最佳) | ❌ 无 q_current | 作废 |
| 代码修复 (has_q_current flag) | ✅ 已提交 | `270b707` |
| evaluate.py attn_implementation 修复 | ✅ 已提交 | `5ac8fc0` |

## 教训

1. **Loss 数值有欺骗性, eval 才是唯一真相**: CE loss 降到 0.000x 时模型可能只是输出 template, 必须用下游任务 (SCA-FP speedup) 验证
2. **Smoke test 后立刻 eval**: 不要等到全量训练结束。200 条 30 分钟就能发现 collapse, 全量 8.7h 再发现浪费算力
3. **防御性 silent pass 是隐患**: `numel() > 0` 把数据缺失变成了 "无动作" 而非报错。应该用 explicit boolean flag (`has_q_current`) 并加 warn log
4. **不可恢复的字段缺失 vs 可后补的字段**: `q_current` 编码在 prompt 文本里, 但提取需要重跑 solver。如果当时也存了 `uav_initial_positions` 可能有后补余地。数据 schema 设计中应考虑不可恢复字段的完整性校验

## 后续

- 🟢 生成全量 5000 环境数据 (新代码, 含 q_current)
- 🟢 全量 SFT 训练 (3 epochs, bs=2, grad_accum=8)
- 🟢 评估最佳 checkpoint → Stage II DPO
