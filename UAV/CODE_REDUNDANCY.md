# 代码冗余审计报告

> 扫描范围：`src/` 下 14 个 Python 模块，共 ~5,257 行
> 预计可消除：**~500+ 行**

---

## 🔴 高优先级

### 1. "防爆盾"环境变量块 (3 处 → 1 处)

`train_sft.py:30-60` 和 `train_dpo.py:19-47` 顶部 30 行**完全复制**：

- 5 个线程控制变量 (`OMP/MKL/OPENBLAS/VECLIB/NUMEXPR`)
- `PYTORCH_CUDA_ALLOC_CONF`
- 防爆盾 0/1/2/3 (HF 镜像、FlexAttention、Unsloth 肃清注释、Inductor 禁用)

**方案**：提取为 `src/training/env_setup.py`，两个训练脚本各一行 `from src.training.env_setup import setup_env; setup_env()`

**可省**：~35 行

---

### 2. `proj_head_config` 字典构造 (4 处 → 1 处)

15 行的投影头配置字典在 **3 个文件里出现 4 次**：

| 文件 | 行号 | 备注 |
|------|------|------|
| `train_sft.py` | 170-187 | 提取为变量 `_proj_head_cfg` |
| `train_dpo.py` | 179-196 | 内联 dict (train model) |
| `train_dpo.py` | 212-229 | 内联 dict (ref model，同上一个完全一样) |
| `evaluate.py` | 249-261 | 缺 5 个 key，依赖默认值 |

**方案**：提取 `build_proj_head_config(model_cfg, sim_cfg) -> dict`，放在 `src/model/__init__.py`

**可省**：~52 行

---

### 3. OOM6 防护代码 (3 处 → 1 处)

gradient checkpointing 强制启用 + lm_head 解绑冻结，**同一段逻辑出现 3 次**：

| 文件 | 行号 | 上下文 |
|------|------|------|
| `train_sft.py` | 208-253 | resume 路径 (46 行) |
| `gemma_isac.py` | 513-546 | `from_pretrained` PeftModel 加载分支 (34 行) |
| `gemma_isac.py` | 564-593 | `from_pretrained` fresh PEFT 分支 (30 行) |

三者均执行：
1. `gradient_checkpointing_enable()` + 底层验证
2. 检查 `lm_head` 与 `embed_tokens` 权重绑定 (`data_ptr()`)
3. 若绑定 → clone 解绑 + `requires_grad=False`

**方案**：提取 `_ensure_gc_and_freeze_lm_head(model, logger) -> bool`

**可省**：~75 行

---

### 4. 死代码 (12 处)

#### 4a. 从未调用的函数

| 文件 | 函数 | 行号 | 行数 |
|------|------|------|:--:|
| `oracle_generator.py` | `_compute_baseline_utility` | 235-246 | 12 |
| `oracle_generator.py` | `_run_snapback_test` | 252-297 | 47 |
| `uav_channel.py` | `generate_random_beamformers` | 258-267 | 10 |
| `sca_fp.py` | `compute_utility_from_solution` | 561-573 | 13 |

#### 4b. 无用 import

| 文件 | 行号 | 内容 |
|------|------|------|
| `train_sft.py` | 69 | `import torch.nn as nn` |
| `train_sft.py` | 85 | `import json` |
| `train_dpo.py` | 53 | `from typing import Dict` |
| `train_dpo.py` | 69 | `import json` |
| `losses.py` | 23 | `import torch.nn as nn` |
| `evaluate.py` | 23 | `from typing import Dict, List, Tuple` (仅 Optional 用到) |

#### 4c. `**kwargs` 黑洞

| 文件 | 行号 | 说明 |
|------|------|------|
| `gemma_isac.py` | 59 | `**kwargs` 标注"兼容旧 BnB 参数"但所有调用方都不传 |

**可省**：~93 行

---

### 5. Unsloth 4-bit 代码路径 (2 处)

`gemma_isac.py` 中 `use_4bit=True` 的两个分支仍保留 Unsloth 加载逻辑：

| 方法 | 行号 |
|------|------|
| `__init__` | 67-91 |
| `from_pretrained` | 454-470 |

项目文档写"已彻底肃清 Unsloth"，`use_4bit` 在所有配置中为 `false`，但代码路径还在。**注意**：RTX 5090 适配后会重写为 bitsandbytes，此处的删除可并入那个改动。

**可省**：~70 行（但建议在 RTX 5090 适配时一并处理）

---

## 🟡 中优先级

### 6. `SFTDataset` 和 `DPODataset` 重复 (2 处 → 基类)

`dataset.py` 中两个类的 `__init__` 和 JSONL 加载逻辑几乎一样：

| 类 | 行号 | 共同逻辑 |
|------|------|------|
| `SFTDataset` | 137-152 | tokenizer 捕获、control_token_ids 构造、JSONL 加载 |
| `DPODataset` | 180-195 | 同上 |

**方案**：提取 `BaseISACDataset` 基类

**可省**：~20 行

---

### 7. `Gemma3ISAC.__init__` 与 `from_pretrained` 重复逻辑

同一类中两个方法共享大量相同逻辑：

- `hidden_dim` 提取 (各 8 行)
- `pad_token is None` 检查 (各 2 行)
- 控制 token 添加到 tokenizer (各 5 行)
- `resize_token_embeddings` (各 1 行)
- LoRA 注入 + GC + lm_head 冻结 (~30 行 × 2)

**方案**：`__init__` 委托给私有方法，`from_pretrained` 复用

**可省**：~120 行（**最大单次收益**，但改动量大）

---

### 8. 过长函数

| 函数 | 文件 | 行数 | 建议 |
|------|------|:--:|------|
| `train_stage1` | `train_sft.py` | **566** | 拆为 `_init_model`, `_run_phase1`, `_run_phase2` 等 |
| `train_stage2` | `train_dpo.py` | **356** | 拆为 `_init_model_pair`, `_run_training_loop` |
| `from_pretrained` | `gemma_isac.py` | **210** | 拆为 `_load_base`, `_restore_ckpt` |
| `run_evaluation` | `evaluate.py` | **237** | 拆为 `_run_inference`, `_run_solver` |

不省行数，但改善可读性和可测试性。

---

## 汇总

| 优先级 | 条目 | 可省行数 |
|:--:|------|:--:|
| 🔴 | 防爆盾环境变量 | ~35 |
| 🔴 | proj_head_config 字典 | ~52 |
| 🔴 | OOM6 防护代码 | ~75 |
| 🔴 | 死代码 (函数+import) | ~93 |
| 🔴 | Unsloth 4-bit 路径 | ~70 |
| 🟡 | Dataset 基类 | ~20 |
| 🟡 | __init__ vs from_pretrained | ~120 |
| 🟡 | 过长函数拆分 | 0 (可读性) |
| | **合计** | **~465 行** |

---

## 建议执行顺序

1. **删死代码** — 无风险，5 分钟，立即见效
2. **提防爆盾** — 低风险，两个 import 变一个
3. **提 proj_head_config** — 低风险，纯函数提取
4. **提 OOM6 防护** — 低风险，纯函数提取
5. **Dataset 基类** — 低风险，继承重构
6. **__init__ vs from_pretrained** — 高风险（影响面大），建议在 RTX 5090 适配时一并重构
