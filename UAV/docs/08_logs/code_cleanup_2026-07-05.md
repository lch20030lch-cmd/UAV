# 代码冗余清理日志

> 日期: 2026-07-05
> 基准: `CODE_REDUNDANCY.md` 审计报告
> 目标: 消除 ~465 行冗余代码

---

## 清理计划

| # | 条目 | 预计节省 | 风险 | 状态 |
|---|------|:--:|:--:|:--:|
| 1 | 删死代码（无用函数 + import） | ~93 行 | 零 | ✅ |
| 2 | 提取防爆盾环境变量 | ~35 行 | 低 | ⏳ |
| 3 | 提取 `build_proj_head_config()` | ~52 行 | 低 | ⏳ |
| 4 | 提取 OOM6 防护 `_ensure_gc_and_freeze_lm_head()` | ~75 行 | 低 | ⏳ |
| 5 | Dataset 基类 `BaseISACDataset` | ~20 行 | 低 | ⏳ |
| 6 | `__init__` vs `from_pretrained` 逻辑合并 | ~120 行 | 中高 | ⏳ |

---

## 第 1 项: 删除死代码 ✅

### 删除的未调用函数 (4 个, ~82 行)

| 文件 | 函数 | 行数 |
|------|------|:--:|
| `src/data/oracle_generator.py` | `_compute_baseline_utility` | 12 |
| `src/data/oracle_generator.py` | `_run_snapback_test` | 47 |
| `src/env/uav_channel.py` | `generate_random_beamformers` | 10 |
| `src/solver/sca_fp.py` | `compute_utility_from_solution` | 13 |

### 删除的无用 import (6 个, ~6 行)

| 文件 | 删除的 import |
|------|------|
| `src/training/train_sft.py` | `import torch.nn as nn` |
| `src/training/train_sft.py` | `import json` |
| `src/training/train_dpo.py` | `from typing import Dict` |
| `src/training/train_dpo.py` | `import json` |
| `src/model/losses.py` | `import torch.nn as nn` |
| `src/eval/evaluate.py` | `Dict, List, Tuple` (仅保留 `Optional`) |

**节省: ~88 行**