---
type: onboarding
status: current
stage: data_regeneration
last_updated: 2026-07-02
related: [status, quickstart, canonical_config]
---

# 新成员接手文档 — UAV-ISAC-MLLM

**最后更新**: 2026-07-02 | **上一任**: Lampota

## 1. 项目一句话

用 **Gemma 3 12B + LoRA + 约束投影头** 为无人机通信感知一体化 (UAV-ISAC) 的 SCA-FP 数值优化器提供**智能热启动**——即用神经网络预测一个接近最优的初始解，让传统优化器从该点开始迭代，从而减少迭代次数。

**类比**: 传统优化器是从随机点爬山找山顶；我们的模型看一张"地形图"后直接指一个离山顶很近的位置。

## 2. 当前状态

详见 [status.md](status.md)。摘要：

- ✅ Smoke v3 全链路闭环验证通过 (1.347x speedup)
- ✅ 两个根因修复：数据退化 + q_current 缺失
- 🔴 旧数据全部作废，需重新生成
- 🟡 待执行：全量 5000 环境数据重生 → SFT → DPO → 评估

## 3. 接手第一步

### Step 1: 登录服务器

```
平台: AutoDL
GPU: RTX PRO 6000 96GB (Blackwell sm_120)
实例: 需要从 Lampota 获取 SSH 连接信息
```

### Step 2: 拉最新代码

```bash
cd /root/UAV-ISAC-MLLM && git pull
conda activate uavmllm
```

### Step 3: 验证环境

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
python -c "from transformers import AutoModel; print('HF OK')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
```

### Step 4: 生成全量数据 + 训练

详见 [quickstart.md](quickstart.md) 和 [status.md](status.md) 的"下一步"章节。

## 4. 服务器环境

| 项 | 值 |
|----|-----|
| 平台 | AutoDL |
| GPU | RTX PRO 6000 96GB (Blackwell sm_120) |
| CUDA | 13.0, Driver 595.58.03 |
| 系统盘 | 30GB (`/root/`) — **不要写大文件到这里！** |
| 数据盘 | `/root/autodl-tmp/` — 所有数据、checkpoint、输出放这里 |
| Python | 3.12, conda env: `uavmllm` |
| 本地开发 | Windows, `h:\Projects\UAV` → git push → 服务器 git pull |

### 关键路径

```
/root/UAV-ISAC-MLLM/                     # 代码
/root/autodl-tmp/huggingface/models/     # Gemma 3 12B 权重 (~24 GB)
/root/autodl-tmp/data/                   # 训练数据
/root/autodl-tmp/checkpoints/            # 模型 checkpoint
/root/autodl-tmp/outputs/                # 训练输出 / 日志
```

## 5. 架构速览

→ 完整参考: [01_architecture/](../01_architecture/)

```
Input Prompt (文本 + <ctrl_0>...<ctrl_7>)
       ↓ Gemma 3 12B (LoRA, rank=16, SDPA, bf16)
Control Hidden States [B, 8, 3840]
       ↓ Multi-Query Attention Pooling (4 queries → 4 UAVs)
       ↓ Shared Readout MLP → Constraint Projections
  δ_q [B,4,3] + δ_a [B,4,20] + δ_p [B,4,21]
```

### 源码目录

```
src/
├── env/          # UAV 仿真 (拓扑、信道、场景生成)
├── solver/       # SCA-FP 数值优化器 (Oracle)
├── data/         # Prompt 构造、Oracle 标注、Dataset
├── model/        # ★ 核心: gemma_isac.py, projection_head.py, losses.py
├── training/     # train_sft.py (Stage I), train_dpo.py (Stage II)
└── eval/         # evaluate.py
scripts/          # 辅助脚本 (数据生成、验证、过拟合测试)
configs/          # default.yaml (所有超参数)
```

## 6. 常见问题

### Q: 训练 OOM 了？

1. 确认 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
2. 终极方案：降 `per_device_batch_size: 1` + 升 `gradient_accumulation_steps: 16`
3. 确认 gradient checkpointing 生效

### Q: 能在本地 Windows 跑吗？

**不行。** Gemma 3 12B 需要 ~24 GB 仅加载权重，加上训练状态需要 ~76 GB。代码在本地修改，git push，服务器 pull 并运行。

### Q: 怎么在服务器上 debug？

```bash
# 快速显存检查
python -c "
import torch
from transformers import AutoModel
model = AutoModel.from_pretrained(
    '/root/autodl-tmp/huggingface/models/gemma-3-12b-it',
    torch_dtype=torch.bfloat16, attn_implementation='sdpa',
).to('cuda')
print(f'Base model: {torch.cuda.memory_allocated()/1e9:.1f} GB')
"

# 看 GPU
watch -n 1 nvidia-smi

# 看训练日志
tail -f /root/autodl-tmp/outputs/stage1_sft_final/logs/*.log
```

### Q: checkpoint 在哪里？怎么恢复训练？

```bash
/root/autodl-tmp/checkpoints/
├── phase1_step_150/    ← Phase 1 最佳

# 恢复训练
python src/training/train_sft.py \
    --config configs/default.yaml \
    --resume_from /root/autodl-tmp/checkpoints/phase1_step_150
```

### Q: 怎么判断 Phase 1 学得怎么样？

两个指标：
- **`loss_ctl`**: 控制损失，越低越好
- **`sensitivity`**: 跨环境区分度。> 0.05 = 有效，> 0.08 = 良好

**关键**: loss_ctl 和 sensitivity 在训练后期背离——选 sens 最高的 checkpoint，不选 loss 最低的。

## 7. 日常操作

### 代码修改 → 服务器运行

```bash
# 本地 Windows (h:\Projects\UAV)
git add -A && git commit -m "描述你的改动" && git push

# 服务器
cd /root/UAV-ISAC-MLLM && git pull
```

### 数据验证

```bash
python scripts/validate_data.py --data-dir /root/autodl-tmp/data/full_v2
python scripts/eda_data.py --data-dir /root/autodl-tmp/data/full_v2
```

### 过拟合测试

```bash
export TORCHINDUCTOR_FLEX_ATTENTION=0
python scripts/test_sft_overfit.py --data-dir /root/autodl-tmp/data/full_v2
```

## 8. 关键文件速查

| 你想知道... | 看这个 |
|-------------|--------|
| 项目当前状态 | [status.md](status.md) |
| 怎么跑起来 | [quickstart.md](quickstart.md) |
| 所有配置参数 | [configs/default.yaml](../configs/default.yaml) |
| 模型定义 | [src/model/gemma_isac.py](../src/model/gemma_isac.py) |
| 训练循环 | [src/training/train_sft.py](../src/training/train_sft.py) |
| 所有 bug 记录 | [03_bugs/README.md](../03_bugs/README.md) |
| 为什么不用 Unsloth | [adr_001](../06_decisions/adr_001_unsloth_removal.md) |
| 问题数学定义 | [problem_formulation.md](../01_architecture/problem_formulation.md) |
| 模块拓扑 | [system_design.md](../01_architecture/system_design.md) |

## 9. 禁忌清单

1. ❌ **不要 `import unsloth`** — 全局 monkey-patch 破坏一切
2. ❌ **不要在系统盘 (`/root/`) 写大文件** — 只有 30GB
3. ❌ **不要 `copy.deepcopy(model)`** — 双份 24GB 直接 OOM
4. ❌ **不要改 `modules_to_save` 加回 `lm_head`** — 会解绑权重 + ~12 GB Adam
5. ❌ **不要关闭 gradient checkpointing** — activations ~60 GB
6. ❌ **不要用 `output_hidden_states=True`** — 存 48 层 ~6 GB
7. ❌ **不要在 CausalLM wrapper 上传 `labels`** — HF 内部 CE 产生 fp32 logits ~16 GB
8. ❌ **不要用 Flash Attention 2** — Blackwell sm_120 无预编译 wheel，用 SDPA

## 10. 联系人

| 角色 | 人 | 联系 |
|------|----|------|
| 上一任 | Lampota | GitHub: `Lampotaku` |
| GitHub | `Lampotaku/UAV-ISAC-MLLM` | private repo |

遇到 Block 级问题：先查 [03_bugs/](../03_bugs/)、读 [status.md](status.md)、联系 Lampota。
