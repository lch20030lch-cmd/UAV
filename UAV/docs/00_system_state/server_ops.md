# 服务器操作速查

## 首次部署

```bash
# 1. clone 代码
cd /root
git clone https://github.com/lch20030lch-cmd/UAV.git Projects

# 2. 下载 4B 模型 (~8GB)
huggingface-cli download google/gemma-3-4b-it \
  --local-dir /root/autodl-tmp/huggingface/models/gemma-3-4b-it

# 3. 激活环境
conda activate uavmllm
```

## 每次更新代码

```bash
cd /root/Projects && git pull
```

## Smoke Test（4B，~20min）

```bash
conda activate uavmllm
cd /root/Projects

# 训练
python UAV/src/training/train_sft.py --config UAV/configs/smoke.yaml

# 评估加速比（替换 step 号）
python UAV/scripts/eval_generation.py \
  --checkpoint /root/autodl-tmp/checkpoints/smoke/stage1_step_XX \
  --config UAV/configs/smoke.yaml --n_scafp 50
```

## 全量训练（5000 条）

```bash
conda activate uavmllm
cd /root/Projects

# SFT（~2h）
python UAV/src/training/train_sft.py --config UAV/configs/default.yaml

# DPO（~1.5h）
python UAV/src/training/train_dpo.py --config UAV/configs/default.yaml \
  --stage1_ckpt /root/autodl-tmp/outputs/stage1_sft_final
```

## 本地更新流程

```bash
cd C:\Users\Shardeom-PC\Desktop\Projects
git add . && git commit -m "写清楚改了什么" && git push
```

## 路径速查

| 内容 | 路径 |
|------|------|
| 代码 | `/root/Projects/UAV/` |
| 4B 模型 | `/root/autodl-tmp/huggingface/models/gemma-3-4b-it/` |
| 数据 | `/root/autodl-tmp/data/full_v2/` |
| Checkpoints | `/root/autodl-tmp/checkpoints/` |
| 日志 | `/root/autodl-tmp/logs/` |
