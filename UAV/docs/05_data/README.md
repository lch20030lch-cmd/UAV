---
type: reference
status: current
stage: data_regeneration
last_updated: 2026-07-02
---

# Data Layer

## ⚠️ 旧数据全部作废

2026-07-02 之前生成的全部数据 (Run 1-4) 已作废，原因：

1. **q_current 字段缺失** — 旧 `oracle_generator.py` 不写 `q_current`，导致分离惩罚永远为 0 → mode collapse (0.893x speedup)
2. **数据退化** — 旧 SCA-FP 求解器缺乏地面杂波建模 → 97.4% 向下 + 84.7% 满速退化解

两个根因均已修复。详见 [data_degeneracy.md](../03_bugs/resolved/data_degeneracy.md) 和 [q_current_missing.md](../03_bugs/resolved/q_current_missing.md)。

## 当前有效数据

| 数据集 | 位置 | 数量 | 状态 |
|--------|------|------|------|
| Smoke v3 | `/root/autodl-tmp/data/smoke_v3/` | 200 SFT + 200 DPO | ✅ 验证通过 (1.347x) |
| Full v2 | `/root/autodl-tmp/data/full_v2/` | 5000 envs | 🟡 待生成 |

## 数据格式

标准格式定义见 [data_schema.md](data_schema.md)。

## 历史数据 (已归档)

| Run | 数量 | 归档位置 | 作废原因 |
|-----|------|----------|----------|
| Run 1 | 5000 SFT | [result_v1_failed.md](../99_archive/old_results/result_v1_failed.md) | RNG 崩溃 + Token 截断 |
| Run 2 | 70 SFT | [result_v2_trial.md](../99_archive/old_results/result_v2_trial.md) | 试运行，已过时 |
| Run 3 | 5 SFT + 196 DPO | [result_v3_trial.md](../99_archive/old_results/result_v3_trial.md) | 试运行，已过时 |
| Run 4 | 5000 SFT + 186,896 DPO | [data_validation_v1.md](../99_archive/data_validation_v1.md) | q_current 缺失 + 数据退化 |

## 验证工具

- `scripts/validate_data.py`: 物理正确性 + 格式完整性
- `scripts/eda_data.py`: 统计多样性 + 分布检查 (3-section EDA)
- `scripts/quick_validate_fix.py`: 求解器修复快速验证
