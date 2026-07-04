# 服务器操作速查

> **服务器**: AutoDL RTX PRO 6000 96GB (Blackwell sm_120)
> **环境**: Python 3.12, conda env `uavmllm`
> **模型**: Gemma 3 4B Instruct (bf16 LoRA, SDPA)

---

## 首次部署

```bash
# 1. clone 代码
cd /root
git clone https://github.com/lch20030lch-cmd/UAV.git Projects

# 2. 创建 conda 环境 (Python 3.12 + PyTorch CUDA 12.8)
conda create -n uavmllm python=3.12 -y
conda activate uavmllm

# 3. 安装 PyTorch (Blackwell sm_120 需要 CUDA 12.8+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 4. 安装项目依赖
cd /root/Projects
pip install transformers accelerate peft pyyaml tqdm tensorboard safetensors

# 5. 下载 Gemma 3 4B 模型 (~8GB)
huggingface-cli download google/gemma-3-4b-it \
  --local-dir /root/autodl-tmp/huggingface/models/gemma-3-4b-it

# 6. 验证模型可加载
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tok = AutoTokenizer.from_pretrained('/root/autodl-tmp/huggingface/models/gemma-3-4b-it')
print(f'Tokenizer OK, vocab={len(tok)}')
"
```

---

## 每次更新代码

```bash
cd /root/Projects && git pull
```

---

## 环境变量 (每次训练前必须设置)

```bash
# Blackwell sm_120: 禁止 FlexAttention (共享内存不足)
export TORCHINDUCTOR_FLEX_ATTENTION=0
export TORCH_COMPILE_DISABLE=1

# CUDA 内存: 允许动态释放缓存段
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 防止 DataLoader CPU 过载
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

---

## 数据生成

```bash
conda activate uavmllm
cd /root/Projects

# 全量数据: 5000 环境 × 10 restarts
python UAV/scripts/generate_data.py \
  --config UAV/configs/default.yaml \
  --output /root/autodl-tmp/data/full_v2

# Smoke 数据: 200 环境 (快速验证)
python UAV/scripts/generate_data.py \
  --config UAV/configs/smoke.yaml \
  --output /root/autodl-tmp/data/smoke_v3

# 数据校验
python UAV/scripts/validate_data.py \
  --data /root/autodl-tmp/data/full_v2/sft_dataset.jsonl

# EDA 探索
python UAV/scripts/eda_data.py \
  --data /root/autodl-tmp/data/full_v2/sft_dataset.jsonl
```

---

## Smoke Test (4B, ~20min)

快速验证完整管线 (SFT → eval) 无 bug：

```bash
conda activate uavmllm
cd /root/Projects

# Smoke SFT (1 epoch, phase1 max 20 steps)
python UAV/src/training/train_sft.py --config UAV/configs/smoke.yaml

# 评估加速比 (替换 step 号为实际 checkpoint)
python UAV/scripts/eval_generation.py \
  --checkpoint /root/autodl-tmp/checkpoints/smoke/stage1_step_XX \
  --config UAV/configs/smoke.yaml --n_scafp 50
```

---

## 全量训练 (5000 条)

```bash
conda activate uavmllm
cd /root/Projects

# Stage I: SFT (~2h, 3 epochs, bs=2×grad_accum=8)
python UAV/src/training/train_sft.py --config UAV/configs/default.yaml

# Stage II: DPO (~1.5h, 2 epochs, bs=1×grad_accum=16)
python UAV/src/training/train_dpo.py --config UAV/configs/default.yaml \
  --stage1_ckpt /root/autodl-tmp/outputs/stage1_sft_final

# 评估 (200 test envs, 9 baselines)
python UAV/src/eval/evaluate.py --config UAV/configs/default.yaml
```

---

## 断点续训

### 从 Phase 1 checkpoint 继续 (跳过 Phase 1, 直接 Phase 2)

```bash
python UAV/src/training/train_sft.py --config UAV/configs/default.yaml \
  --resume_from /root/autodl-tmp/checkpoints/phase1_smoke_150
```

### 从完整训练状态恢复 (含 optimizer/scheduler)

```bash
python UAV/src/training/train_sft.py --config UAV/configs/default.yaml \
  --resume_from_checkpoint /root/autodl-tmp/checkpoints/stage1_step_300
```

---

## 过拟合测试 (验证训练代码无 bug)

```bash
# 16 条数据, 10 epochs → 应该 loss → 0
python UAV/scripts/test_sft_overfit.py \
  --data-dir /root/autodl-tmp/data/full_v2
```

---

## 监控

```bash
# GPU 实时监控
watch -n 1 nvidia-smi

# TensorBoard (本地端口转发后访问 localhost:6006)
tensorboard --logdir /root/autodl-tmp/logs --bind_all

# 磁盘使用
df -h /root/autodl-tmp
du -sh /root/autodl-tmp/checkpoints/*
```

---

## Checkpoint 管理

```bash
# 查看所有 checkpoint 大小
ls -lh /root/autodl-tmp/checkpoints/

# 手动清理旧 checkpoint (config 中 save_total_limit=3 会自动清理)
rm -rf /root/autodl-tmp/checkpoints/stage1_step_100

# Checkpoint 结构:
# save_pretrained (save_full_state=false):
#   lora/adapter_model.safetensors  ← LoRA 权重 (~100MB)
#   ctrl_embed.pt                   ← 8 行 control token embedding (~60KB)
#   projection_head.pt              ← 投影头权重 (~2MB)
#   tokenizer/                      ← tokenizer 文件
#
# save_state (save_full_state=true):
#   上述 + optimizer.pt + scheduler.pt + scaler.pt (~12GB)
```

---

## 常见故障排查

### OOM (Out of Memory)

```bash
# 症状: CUDA out of memory, 峰值 > 96GB
# 解决:
#   1. 确认 FlexAttention 已禁用
python -c "import torch._inductor.config as c; print(getattr(c,'flex_attention',None))"
#   2. 降 bs: SFT bs=2→1, grad_accum=8→16
#   3. 降 seq: 3456→2048
#   4. 确认 gradient_checkpointing 已启用
```

### FlexAttention 相关崩溃

```bash
# 症状: "shared memory 101KB < required 114KB"
# 确认环境变量:
echo $TORCHINDUCTOR_FLEX_ATTENTION  # 应为 0
echo $TORCH_COMPILE_DISABLE         # 应为 1
```

### CheckpointError (forward/recompute 张量数不一致)

```bash
# 症状: "number of tensors 68 != 65"
# 根因: Unsloth 全局 monkey-patch 与 grad ckpt 冲突
# 确认: grep -r "unsloth" UAV/src/  # 应为空
```

### DataLoader 卡死 (CPU 100%)

```bash
# 症状: 加载数据时进程卡住, CPU 满载
# 解决: train_sft.py 已设 OMP_NUM_THREADS=1
# 如果仍卡: 改 num_workers=0 (已在 SFT 默认, DPO 见下方)
```

---

## 本地更新流程

```bash
# Windows 本地 (Git Bash)
cd C:\Users\Shardeom-PC\Desktop\Projects
git add . && git commit -m "写清楚改了什么" && git push
```

---

## 路径速查

| 内容 | 路径 |
|------|------|
| 代码 | `/root/Projects/UAV/` |
| 4B 模型 | `/root/autodl-tmp/huggingface/models/gemma-3-4b-it/` |
| 数据 (全量) | `/root/autodl-tmp/data/full_v2/` |
| 数据 (smoke) | `/root/autodl-tmp/data/smoke_v3/` |
| Checkpoints | `/root/autodl-tmp/checkpoints/` |
| 训练输出 | `/root/autodl-tmp/outputs/` |
| TensorBoard 日志 | `/root/autodl-tmp/logs/` |
| 系统盘 (30GB) | `/root/` — **不要放数据/模型** |
| 数据盘 (可扩容) | `/root/autodl-tmp/` — 所有大文件放这里 |
