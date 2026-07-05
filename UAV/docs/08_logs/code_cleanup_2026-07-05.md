# 代码冗余清理日志

> 日期: 2026-07-05
> 基准: `CODE_REDUNDANCY.md` 审计报告
> 目标: 消除 ~465 行冗余代码

---

## 清理计划

| # | 条目 | 预计节省 | 风险 | 状态 |
|---|------|:--:|:--:|:--:|
| 1 | 删死代码（无用函数 + import） | ~93 行 | 零 | ✅ |
| 2 | 提取防爆盾环境变量 | ~35 行 | 低 | ✅ |
| 3 | 提取 `build_proj_head_config()` | ~52 行 | 低 | ✅ |
| 4 | 提取 OOM6 防护函数 | ~75 行 | 低 | ✅ |
| 5 | Dataset 基类 `BaseISACDataset` | ~20 行 | 低 | ⏸ 暂缓 |
| 6 | `__init__` vs `from_pretrained` 合并 | ~120 行 | 中高 | ⏸ 暂缓 |

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

---

## 第 2 项: 提取防爆盾环境变量 ✅

**新建**: `src/training/env_setup.py`

- `setup_env()`: 统一管理 OMP/MKL/PYTORCH_CUDA_ALLOC_CONF/HF_ENDPOINT/FlexAttention 等环境变量
- `train_sft.py` / `train_dpo.py`: 各删除 ~35 行重复代码，改为 `from src.training.env_setup import setup_env; setup_env()`

**节省: ~35 行**

---

## 第 3 项: 提取 `build_proj_head_config()` ✅

**新建**: `src/model/__init__.py` → `build_proj_head_config(model_cfg, sim_cfg) -> dict`

| 文件 | 旧代码 | 新代码 |
|------|------|------|
| `train_sft.py` | 15 行字典 + 变量 | 1 行函数调用 |
| `train_dpo.py` | 2×15 行字典 (train + ref) | 2×1 行函数调用 |
| `evaluate.py` | 10 行不完整字典 (缺 5 key) | 1 行函数调用 |

**节省: ~52 行** + evaluate.py 获得完整的 projection_head 配置

---

## 第 4 项: 提取 OOM6 防护函数 ✅

**新建**: `src/model/gemma_isac.py` → `ensure_gc_and_freeze_lm_head(peft_model)`

| 位置 | 旧代码 | 新代码 |
|------|------|------|
| `__init__` | ~22 行 GC + lm_head | 1 行调用 |
| `from_pretrained` | ~34 行 GC + lm_head | 1 行调用 |
| `train_sft.py` resume | ~46 行 GC + lm_head | 1 行调用 + 1 行日志 |

**节省: ~98 行**（含更防御性的 GC 硬加固逻辑统一应用到所有路径）

---

## 汇总

| 阶段 | 内容 | 累计节省 | commit |
|------|------|:--:|------|
| 1 | 删死代码 | ~88 行 | `4e75ef3` |
| 2 | 防爆盾变量 | ~123 行 | `58b1b6d` |
| 3 | proj_head_config | ~175 行 | `bbb12a6` |
| 4 | OOM6 防护 | ~273 行 | `de22c3f` |

**总计消除: ~273 行冗余代码，9 个文件变更。**

剩余的 Dataset 基类和 __init__/from_pretrained 合并因风险较高暂缓，待 DPO 训练完成后再做。