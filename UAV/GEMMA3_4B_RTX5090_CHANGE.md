# Gemma3-4B + RTX 5090 更改评估与落地清单

## 结论

把论文中的 Gemma3-12B 训练路线切换为 Gemma3-4B 是可行的，也是当前 RTX 5090 32GB 环境下更现实的方向。

当前代码的大部分训练框架已经与 Gemma3-4B 对齐，包括 LoRA、control token、projection head、SFT、DPO、Masked DPO 和 SCA-FP warm-start 接口。但 `RTX5090_PLAN.md` 中要求的 4-bit QLoRA 路径尚未真正落地：代码仍使用 Unsloth，计划要求改为 bitsandbytes NF4 + PEFT。

因此当前状态可以概括为：

- 方法可行。
- Gemma3-4B 方向正确。
- 当前配置已部分切到 4B。
- RTX 5090 32GB 的 4-bit 工程路径尚未完成。
- 应先完成 4-bit 加载栈切换，再做冗余代码整理。

## 论文方法与当前代码对应关系

论文核心方法是：

- Gemma backbone + LoRA
- control token hidden states
- differentiable constraint-projection head
- Stage I SFT
- Stage II DPO
- downstream SCA-FP warm start

当前代码中已经具备对应模块：

- `src/model/gemma_isac.py`: Gemma3ISAC 主模型，包含 LoRA、control token、projection head 调用。
- `src/model/projection_head.py`: `ConstraintProjectionHead`，输出 `delta_q / delta_a / delta_p`。
- `src/training/train_sft.py`: Stage I SFT。
- `src/training/train_dpo.py`: Stage II DPO。
- `src/data/dataset.py`: SFT/DPO dataset，包含 Masked DPO token label masking。

Gemma3-12B 到 Gemma3-4B 的替换不会改变 `delta_q / delta_a / delta_p` 标签定义，也不会改变 SCA-FP warm-start 接口。因此数据格式和优化器接口不需要因为模型尺寸变化而重写。

## 当前已对齐 Gemma3-4B 的部分

`configs/default.yaml` 已经使用：

```yaml
model:
  backbone: "/root/autodl-tmp/huggingface/models/gemma-3-4b-it"
  torch_dtype: "bfloat16"
  attn_implementation: "sdpa"

  control_token:
    num_tokens: 8
    hidden_dim: 2560

  projection_head:
    hidden_dim: 2560
```

`projection_head.py` 默认 hidden size 也是 `2560`，符合 Gemma3-4B。

这说明当前代码结构已经偏向 4B，不是从 12B 配置硬改过来的临时状态。

## 当前主要阻塞

### 1. 4-bit 路径仍是 Unsloth

`RTX5090_PLAN.md` 要求：

- 删除 Unsloth。
- 使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`。
- 使用 `prepare_model_for_kbit_training`。
- 使用 PEFT 注入 LoRA。

但当前代码仍然存在：

- `requirements.txt` 中仍有 `unsloth`。
- `src/model/gemma_isac.py` 的 `use_4bit=True` 分支仍 `from unsloth import FastLanguageModel`。
- `Gemma3ISAC.from_pretrained` 的 4-bit 分支仍使用 Unsloth。

这与 `RTX5090_PLAN.md` 冲突，也是 RTX 5090 32GB 落地前必须先修的点。

### 2. `configs/rtx5090.yaml` 尚未创建

`RTX5090_PLAN.md` 要求新增 `configs/rtx5090.yaml`，关键配置应为：

```yaml
hardware:
  gpu: "RTX 5090"
  vram_gb: 32
  use_4bit: true

training:
  sft:
    per_device_batch_size: 2
    gradient_accumulation_steps: 8
    max_seq_length: 3456
  dpo:
    per_device_batch_size: 1
    gradient_accumulation_steps: 16
    max_seq_length: 3456

model:
  backbone: "google/gemma-3-4b-it"
  torch_dtype: "bfloat16"
  attn_implementation: "sdpa"
```

当前 `configs/` 下只有 `default.yaml` 和 `smoke.yaml`。

### 3. DPO 显存风险仍需实测

Gemma3-4B 的 4-bit 权重本身可控，但 DPO 中会涉及：

- train model chosen forward
- train model rejected forward
- ref model chosen forward
- ref model rejected forward
- `seq_len x vocab_size` logits
- DPO log-prob 计算
- SFT anchor CE

因此 32GB 上的主要风险不是模型权重，而是 logits 和 DPO 多次 forward 的峰值显存。

如果 DPO OOM，优先降：

```yaml
training:
  dpo:
    max_seq_length: 2048
```

其次再考虑关闭或降低 SFT anchor。

## 建议更改顺序

### Step 1: 完成 bitsandbytes 4-bit 路径

修改 `src/model/gemma_isac.py`：

- 删除 `from unsloth import FastLanguageModel`。
- 删除 `FastLanguageModel.from_pretrained`。
- 删除 `FastLanguageModel.get_peft_model`。
- 新增 `BitsAndBytesConfig`。
- 新增 `prepare_model_for_kbit_training`。
- 4-bit 和 bf16 路径统一走 HF `AutoModelForCausalLM` + PEFT。

4-bit 初始化建议：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    attn_implementation=attn_implementation,
    trust_remote_code=True,
)

base_model = prepare_model_for_kbit_training(base_model)
```

LoRA 注入策略：

- 4-bit: `LoraConfig` 不加 `modules_to_save`。
- bf16: 保留 `modules_to_save=["embed_tokens"]`。
- 两条路径都复用 gradient checkpointing 和 `lm_head` 冻结逻辑。

### Step 2: 更新依赖

修改 `requirements.txt`：

```txt
bitsandbytes>=0.45.3
```

删除：

```txt
unsloth
```

同时更新注释，避免再次出现“代码说已删除 Unsloth，但依赖仍安装 Unsloth”的状态。

### Step 3: 新建 RTX 5090 配置

新增：

```txt
configs/rtx5090.yaml
```

建议从 `configs/default.yaml` 复制后只改硬件和 4-bit 相关参数，避免引入无关差异。

### Step 4: 先 smoke，再全量

建议验证顺序：

```bash
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "from src.model import Gemma3ISAC; m = Gemma3ISAC('google/gemma-3-4b-it', use_4bit=True, attn_implementation='sdpa'); print('load ok')"
python src/training/train_sft.py --config configs/rtx5090.yaml --data_dir /path/to/smoke_data
python src/training/train_dpo.py --config configs/rtx5090.yaml --stage1_ckpt /path/to/stage1_ckpt --data_dir /path/to/smoke_data
```

### Step 5: 再做冗余代码清理

`CODE_REDUNDANCY.md` 中的建议总体成立，但不建议在 4-bit 迁移前做大重构。优先级建议：

1. 提取 `proj_head_config` 构造函数。
2. 提取 OOM6 防护函数。
3. 提取训练脚本环境变量设置。
4. 清理无用 import。
5. 最后再考虑 `Gemma3ISAC.__init__` 和 `from_pretrained` 的结构性合并。

## 风险评估

### 可接受风险

- Gemma3-4B 性能可能低于 12B，但训练可行性显著提高。
- 4-bit QLoRA 可能略损精度，但 deadline 和 32GB 显存约束下是合理折中。
- 当前文本 BEV grid 不等价于真正多模态图像输入，但可作为第一阶段实验实现。

### 高风险点

- bitsandbytes 在具体服务器 CUDA/PyTorch 组合上可能仍有兼容问题。
- DPO 在 3456 seq length 下可能 OOM。
- 当前 DPO 一次保留多份 logits，32GB 上需要 smoke 实测。
- 如果从 12B checkpoint 继续训练到 4B，不可行；必须重新用 4B 初始化并训练 LoRA。

## 推荐最终路线

短期目标是先跑通论文主线，而不是先追求代码最优雅：

1. Gemma3-4B。
2. RTX 5090 使用 bitsandbytes NF4 QLoRA。
3. SFT batch size 2，grad accumulation 8。
4. DPO batch size 1，grad accumulation 16。
5. 如果 DPO OOM，先降 `max_seq_length` 到 2048。
6. smoke 通过后再全量 SFT。
7. 全量 SFT 稳定后再 DPO。
8. 最后根据评估结果决定是否补多模态图像分支。

## 一句话判断

Gemma3-4B 是当前最现实、最稳的训练方向；代码结构基本支持这个方向，但必须先把 `use_4bit=True` 从 Unsloth 改成 bitsandbytes NF4，否则 `RTX5090_PLAN.md` 仍停留在计划层，RTX 5090 32GB 的完整 SFT + DPO 路径还没有工程闭环。
