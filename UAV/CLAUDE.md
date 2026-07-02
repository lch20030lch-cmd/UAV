# UAV-ISAC-MLLM — Constraint-Aware MLLM for UAV-ISAC

**目标**: 用 Gemma 3 4B (LoRA + 约束投影头) 为 UAV-ISAC 的 SCA-FP 数值优化器提供智能热启动。

**文档导航 (新成员必读 — 2026-06-26 重构)**:
- [docs/README.md](docs/README.md) — 文档总索引
- [docs/00_current/status.md](docs/00_current/status.md) — **当前状态** ⭐
- [docs/00_current/quickstart.md](docs/00_current/quickstart.md) — 从零到训练
- [docs/01_architecture/problem_formulation.md](docs/01_architecture/problem_formulation.md) — 问题与数学框架
- [docs/01_architecture/system_design.md](docs/01_architecture/system_design.md) — 模块拓扑与数据流
- [docs/02_training_log/oom_incidents.md](docs/02_training_log/oom_incidents.md) — OOM 五连杀根因分析 ⭐
- [docs/06_decisions/adr_001_unsloth_removal.md](docs/06_decisions/adr_001_unsloth_removal.md) — Plan A 决策

## 当前状态

- ✅ 全部源码完成，7 轮审查闭合 + 一审修复闭合
- ✅ GitHub 仓库: `lch20030lch-cmd/UAV`
- ✅ Plan A: 纯 PyTorch CE + SDPA, 0 Unsloth 引用
- ✅ 终极配置: bs=2, grad_accum=8, seq=3456, bf16 全精度
- 🔴 旧数据全作废 (19,925 SFT + DPO) — q_current 缺失 → mode collapse (0.893x)
- ✅ q_current Bug 修复: has_q_current flag + 统一 tensor shape (commit 270b707)
- ✅ Smoke test v2 通过: SCA-FP speedup 1.347x (200 条含 q_current 新数据)
- 🟡 **全量数据重生 (5000 envs) → SFT 训练 → DPO**

## 关键环境信息

| 项 | 值 |
|----|-----|
| 本地 | Windows, `C:\Users\Shardeom-PC\Desktop\Projects\UAV` |
| 服务器 | AutoDL RTX PRO 6000 96GB, `/root/Projects/UAV` |
| 数据盘 | `/root/autodl-tmp/` (系统盘仅 30GB) |
| GPU | Blackwell sm_120, CUDA 13.0, Driver 595.58.03 |
| 精度 | bf16 全精度 LoRA (96GB 无需量化) |
| Python | 3.12, conda env: `uavmllm` |

## 架构速览

```
src/env/        → 仿真环境 (UAV 拓扑 + 物理信道 + 场景生成)
src/solver/     → SCA-FP 数值优化器 (交替优化)
src/data/       → 数据层 (Prompt 构造 + Oracle 生成 + Dataset)
src/model/      → Gemma3ISAC + ProjectionHead + Losses
src/training/   → Stage I SFT + Stage II DPO
src/eval/       → 评估 (6 指标 × 9 基线)
scripts/        → generate_data.py, validate_data.py, eda_data.py, test_sft_overfit.py, autodl_setup.sh
configs/        → default.yaml (全部超参数)
```

## 工作流

```
✅ git clone → autodl_setup.sh → smoke test (5 envs)
✅ validate → full generation (5000 envs) → EDA
✅ overfitting test → OOM 1-5 修复 → Plan A (纯 PyTorch) → bs=2 终极配置
🟢 Stage I SFT (3 epochs, ~8.7h) → ⏳ Stage II DPO → ⏳ evaluate
```

### SFT 完成后 (服务器上执行)

```bash
cd /root/Projects && git pull
conda activate uavmllm
# Step 1: Stage II DPO (2 epochs, ~5-10h)
python src/training/train_dpo.py --config configs/default.yaml --data_dir /root/autodl-tmp/data/full5000
# Step 2: 评估 (200 test envs, 9 baselines)
python src/eval/evaluate.py --config configs/default.yaml
```

## 关键约定

- 所有路径用 `/root/autodl-tmp/`，不写系统盘
- 代码修改在本地 Windows，git push/pull 同步
- **永远不在项目中 `import unsloth`** — 全局 monkey-patch 与 SDPA + grad ckpt 不兼容
- DPO reference model 独立加载（不 deepcopy，会 OOM）
- 数据生成支持 Ctrl+C 断点续跑
- SFT: bs=2/grad_accum=8; DPO: bs=1/grad_accum=16 (有效 batch 均 16)
