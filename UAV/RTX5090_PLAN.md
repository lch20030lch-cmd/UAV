# RTX 5090 32GB 适配 — 完整作战计划

## Context

- 项目从 RTX PRO 6000 96GB (bf16 全精度) 迁移到 RTX 5090 32GB
- 32GB 必须用 4-bit QLoRA，尤其 DPO 双模型
- 当前 4-bit 路径用 Unsloth（已肃清，SDPA 不兼容）→ 替换为 bitsandbytes NF4
- bitsandbytes >= 0.45.3 已确认支持 RTX 5090 (sm_120)，有 ~30% 吞吐损失但可用
- 论文要求两阶段全跑 (SFT + DPO)，Deadline 一周
- 数据在便宜 CPU 服务器另行生成，不在此计划内

## 改什么 (6 个文件)

| # | 文件 | 操作 |
|---|------|------|
| 1 | `src/model/gemma_isac.py` | **重写 4-bit 路径**：Unsloth → bitsandbytes NF4 + PEFT |
| 2 | `requirements.txt` | `unsloth` → `bitsandbytes>=0.45.3` |
| 3 | `configs/rtx5090.yaml` | **新建**：use_4bit=true, bs=2, backbone 指向 HF Hub |
| 4 | `configs/default.yaml` | 更新硬件注释（96GB → 32GB 说明） |
| 5 | `CLAUDE.md` | 更新 GPU/精度/约定 |
| 6 | `docs/00_system_state/server_ops.md` | 新增 RTX 5090 部署章节 |

## 核心改造：`gemma_isac.py`

### `__init__` — 4-bit 路径

```python
if use_4bit:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    self.tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True)

    self.base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    self.base_model = prepare_model_for_kbit_training(self.base_model)
```

### `__init__` — LoRA 注入 (双轨统一)

```
4-bit 路径 (新增):    LoraConfig 不加 modules_to_save
                      (prepare_model_for_kbit_training 已让 embed 可训练)

bf16 路径 (不变):     LoraConfig + modules_to_save=["embed_tokens"]
```

两者的 gc enable + lm_head 解绑冻结逻辑完全相同。

### `from_pretrained` — 4-bit 路径

```python
if use_4bit:
    # 1) bnb config 加载 base model
    # 2) prepare_model_for_kbit_training
    # 3) 如有已训练 LoRA → PeftModel.from_pretrained
    # 4) gc enable + lm_head 冻结 (复用 OOM6 防护)
    # 5) 恢复 ctrl_embed.pt → patch 回 embedding 表
```

### 删除的内容

- 两个 `from unsloth import FastLanguageModel` 及其 if/else 分支
- `FastLanguageModel.get_peft_model()` 调用
- Gemma3Processor unwrap 逻辑 (bnb 路径直接返回 tokenizer)
- `**kwargs` 中的 `bnb_4bit_compute_dtype` / `bnb_4bit_quant_type` 兼容参数（不再需要）

## `configs/rtx5090.yaml` — 关键参数

```yaml
hardware:
  gpu: "RTX 5090"
  vram_gb: 32
  use_4bit: true

training:
  sft:
    per_device_batch_size: 2        # bs=2，跟 PRO 6000 bf16 一样
    gradient_accumulation_steps: 8  # 有效 batch = 2×8 = 16
    max_seq_length: 3456
  dpo:
    per_device_batch_size: 1        # DPO 双模型，bs=1 保持不变
    gradient_accumulation_steps: 16 # 有效 batch = 1×16 = 16
    max_seq_length: 3456

model:
  backbone: "google/gemma-3-4b-it"  # HF Hub 直接下载
  torch_dtype: "bfloat16"
  attn_implementation: "sdpa"
```

其余超参数 (lr, LoRA, loss weights, simulation) 与 default.yaml 相同。

**batch size 策略说明：**

| | SFT | DPO |
|---|:--:|:--:|
| bs | **2** | 1 |
| grad_accum | **8** | 16 |
| 有效 batch | 16 | 16 |

SFT 用 bs=2 是因为内存够，forward 次数减半、训练更快。
DPO 必须 bs=1 是因为需要同时加载 train + ref 两个模型，32GB 已经没有余量再翻倍。

**如果 SFT bs=2 时 OOM，只需把 yaml 里两行改成：**
```yaml
per_device_batch_size: 1
gradient_accumulation_steps: 16   # 有效 batch 仍为 16
```
不需要改任何代码。

## VRAM 预算

### SFT (4-bit QLoRA, bs=2)

| 组件 | 显存 |
|------|------|
| 4-bit 权重 | ~4 GB |
| LoRA + Embed (fp32) | ~1.5 GB |
| Optimizer (LoRA+embed) | ~4.5 GB |
| Activations (gc) | ~5 GB |
| CE intermediates | ~3.5 GB |
| CUDA overhead | ~2 GB |
| **峰值** | **~20.5 GB / 32 GB** |

### DPO (4-bit QLoRA, bs=1)

| 组件 | 显存 |
|------|------|
| Train 模型 (4-bit + LoRA + embed) | ~10 GB |
| Ref 模型 (4-bit 冻结) | ~5 GB |
| 4x forward activations | ~3 GB |
| log_softmax 中间量 | ~2 GB |
| CUDA overhead | ~2 GB |
| **峰值** | **~22 GB / 32 GB** |

## 性能预期

| 阶段 | 配置 | 步速 | 步数/epoch | 单 epoch | 总耗时 |
|------|------|------|:--:|:--:|:--:|
| SFT | bs=2, grad_accum=8 | ~3.5s | 2500 | ~2.4h | **~7.3h** (3 epochs) |
| DPO | bs=1, grad_accum=16 | ~4.5s | 19925 | ~3.1h | **~6.2h** (2 epochs) |
| **训练总计** | | | | | **~13.5h** |
| Smoke test | bs=2, 200 条 | ~3s | 100 | ~5min | ~5min |

对比 PRO 6000 bf16 的 ~10h (SFT 8.5h + DPO 1.5h)，RTX 5090 4-bit 慢了约 35%，主因是 bnb dequant 开销。但 13.5h 在一周内绰绰有余。

## 风险与应对

| 风险 | 应对 |
|------|------|
| bnb 0.45.3 在 RTX 5090 上 crash | 备选：HQQ (纯 PyTorch，零 CUDA 编译) 或 torchao |
| SFT bs=2 OOM | 改配置 bs=1, grad_accum=16，训练时间 ~12.5h，仍在预算内 |
| DPO 双模型 OOM | 降 max_seq_length → 2048，或 ref model 用更低精度 |
| 4-bit 训练精度不够 | 开 double_quant、调 bnb_4bit_compute_dtype |
| Phase 1 sensitivity 不达标 | 增加 phase1 max_steps，或手动跳过 |

## 执行顺序

1. 改 `requirements.txt`
2. 改 `gemma_isac.py` (__init__ + from_pretrained)
3. 创建 `configs/rtx5090.yaml`
4. 更新 `CLAUDE.md` + `server_ops.md`
5. 服务器上：`pip install -r requirements.txt` → smoke test 验证

## 验证

```bash
# Step 1: 环境检查
python -c "import bitsandbytes; print('bnb', bitsandbytes.__version__)"
python -c "import torch; t=torch.zeros(1,device='cuda'); print(t.device, torch.version.cuda)"

# Step 2: 4-bit 加载测试 (只需几秒)
python -c "
from src.model import Gemma3ISAC
m = Gemma3ISAC('google/gemma-3-4b-it', use_4bit=True, attn_implementation='sdpa')
print('Load OK, params:', sum(p.numel() for p in m.parameters()))
"

# Step 3: Smoke SFT (~5min, bs=2)
python src/training/train_sft.py --config configs/rtx5090.yaml --data_dir /path/to/smoke_data

# Step 4: 过拟合测试
python scripts/test_sft_overfit.py --data-dir /path/to/smoke_data
```
