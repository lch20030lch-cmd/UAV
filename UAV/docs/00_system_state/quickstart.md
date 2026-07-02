---
type: reference
status: current
stage: data_regeneration
last_updated: 2026-07-02
related: [canonical_config, status, hardware_adaptation]
---

# Quickstart — 从零到训练

**目标受众**: 刚接手项目的工程师，需要在服务器上跑起来。

> 📋 **第一次接触项目？先读 [status.md](status.md)** — 包含当前状态、blocker、已知问题。

**预计时间**: 20 min (不含数据生成和训练时间)

## ⚠️ 重要：旧数据全部作废

两个独立根因导致 2026-07-02 之前生成的全部数据无效：

1. **数据退化**: SCA-FP 求解器缺乏地面杂波建模 → 97.4% 向下 + 84.7% 满速退化解
2. **q_current 缺失**: 旧代码不写 `q_current` 字段 → 分离惩罚永远为 0 → mode collapse (0.893x)

修复后的求解器 (`ground_clutter_db=12.0` + `has_q_current` flag) 已在 smoke v3 验证通过 (1.347x speedup)。

→ 详见 [data_degeneracy.md](../03_bugs/resolved/data_degeneracy.md) 和 [q_current_missing.md](../03_bugs/resolved/q_current_missing.md)

## 前置条件

- AutoDL RTX PRO 6000 96GB 服务器 (或同等 Blackwell GPU)
- GitHub 访问权限 (repo: `Lampotaku/UAV-ISAC-MLLM`, private)
- HuggingFace 认证 (Gemma 3 是 gated model)

## Step 1: 环境搭建

```bash
cd /root
git clone git@github.com:Lampotaku/UAV-ISAC-MLLM.git
cd UAV-ISAC-MLLM
bash scripts/autodl_setup.sh

# 或手动安装 (纯 PyTorch — 不使用 Unsloth)
conda create -n uavmllm python=3.12 -y
conda activate uavmllm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers trl datasets accelerate peft
pip install scipy numpy matplotlib pyyaml wandb
```

> ⚠️ **不再安装 Unsloth**。Plan A 使用纯 PyTorch CE + SDPA。

## Step 2: HuggingFace 认证

```bash
huggingface-cli login
```

## Step 3: 验证环境

```bash
conda activate uavmllm
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import AutoModel; print('HF OK')"
python -m compileall -q src scripts
```

## Step 4: 数据准备

### 第 1 步：快速验证求解器修复 (5 min)

```bash
python scripts/quick_validate_fix.py
```

**验收标准**：
- 满速飞行比例 < 40%（原 84.7%）
- 精细微调 (<5m) 比例 > 10%（原 0%）
- **上升比例 > 15%（原 0%）** — 核心红线

### 第 2 步：全量数据生成 (~1h, 30 workers)

```bash
python scripts/generate_data.py \
    --num-envs 5000 \
    --num-restarts 10 \
    --output-dir /root/autodl-tmp/data/full_v2 \
    --num-workers 30
```

### 第 3 步：数据验证 + EDA 验收

```bash
python scripts/validate_data.py --data-dir /root/autodl-tmp/data/full_v2
python scripts/eda_data.py --data-dir /root/autodl-tmp/data/full_v2
```

**EDA Section 3 三条红线全部通过才能进入训练。**

## Step 5: Stage I SFT 训练 (~8.7h)

```bash
tmux new -s sft_full
export TORCHINDUCTOR_FLEX_ATTENTION=0
conda activate uavmllm

python src/training/train_sft.py \
    --config configs/default.yaml \
    --data_dir /root/autodl-tmp/data/full_v2
```

## Step 6: Stage II DPO 训练 (~5-10h)

```bash
tmux new -s dpo_train
export TORCHINDUCTOR_FLEX_ATTENTION=0
conda activate uavmllm

python src/training/train_dpo.py \
    --config configs/default.yaml \
    --stage1_ckpt /root/autodl-tmp/outputs/stage1_sft_final \
    --data_dir /root/autodl-tmp/data/full_v2
```

**预期 VRAM**: ~65-75 GB / 96 GB (bs=1, 双模型)

## Step 7: 评估

```bash
# 生成测试数据 (不与训练集重叠)
python scripts/generate_data.py \
    --config configs/default.yaml \
    --num-env 100 --workers 30 \
    --output-dir /root/autodl-tmp/data/test_v1

# 评估
python src/eval/evaluate.py --config configs/default.yaml \
    --model /root/autodl-tmp/outputs/stage2_dpo_final \
    --data_dir /root/autodl-tmp/data/test_v1
```

**验收标准**：
- SCA-FP 加速比 ≥ 1.3×（smoke v3 baseline）
- 目标 ≥ 1.5×

---

## ⚠️ Blackwell RTX PRO 6000 陷阱

| 陷阱 | 症状 | 解决 |
|------|------|------|
| FlexAttention OOM | "shared memory" error | `export TORCHINDUCTOR_FLEX_ATTENTION=0` |
| bitsandbytes 不支持 | ImportError | bf16 全精度 (96GB 无需量化) |
| Flash Attention 2 不可用 | 无预编译 sm_120 wheel | 用 SDPA (`attn_implementation="sdpa"`) |
| Triton 未调优 sm_120 | 性能下降 | 接受 — 等待上游更新 |
| **Unsloth 全局劫持** | SDPA 被覆盖为 eager | **彻底移除 Unsloth (Plan A)** |

→ 详见 [hardware_adaptation.md](../01_architecture/hardware_adaptation.md)

## 工作流速查

```
git clone → autodl_setup.sh → HF login → verify env
  → quick_validate_fix.py (5 min)
  → generate_data.py 5000 envs (~1h) → validate + EDA
  → SFT training (~8.7h) → DPO training (~5-10h) → evaluate
```

## 本地开发流程

```
Windows (h:\Projects\UAV) → git push → AutoDL (git pull) → 执行训练
```

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| OOM | 降 bs=1, 确认 gradient checkpointing 生效 |
| 数据分布退化 | 用新版 solver (ground_clutter_db=12.0) 重新生成 |
| DPO OOM | 降 bs=1, 确认 reference model 用独立 load (非 deepcopy) |
| q_current 缺失 | 确认代码包含 `has_q_current` flag (commit 270b707+) |
