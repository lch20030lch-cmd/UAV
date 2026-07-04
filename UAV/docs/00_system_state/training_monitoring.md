# 训练监控指南 — 如何判断训练是否正常

> 适用: RTX 5090 32GB + Gemma3-4B + 4-bit QLoRA
> 配置: `configs/rtx5090.yaml`
> 更新: 2026-07-04

---

## SFT (Stage I) — 关键信号

### Phase 1 → Phase 2 切换（最重要）

训练日志会显示：

```
step 50: loss_ctl=0.48, sens=0.0205
step 100: loss_ctl=0.21, sens=0.0873
step 150: loss_ctl=0.15, sens=0.1234  ← 超过 0.1!
Phase 1 → Phase 2: sensitivity > threshold
```

| 信号 | 健康 | 异常 |
|------|:--:|------|
| `sens` 趋势 | 逐步上升 | 一直 < 0.05 不涨 |
| Phase 1 步数 | 100-250 步 | 50 步就切 (太早) 或 250 步满了还不切 |

如果 250 步满了 sensitivity 还不达标 → CTL-only warmup 没学好。可能原因：lr 太低或数据噪声太大。

### Phase 2 Loss 曲线

```
step 200: loss_sft=1.85, loss_ctl=0.35
step 500: loss_sft=1.42, loss_ctl=0.28
step 1000: loss_sft=1.15, loss_ctl=0.22
step 2000: loss_sft=0.92, loss_ctl=0.18
```

| 信号 | 健康 | 异常 |
|------|:--:|------|
| `loss_sft` | 单调下降到 < 1.0 | 不降 或 剧烈震荡 |
| `loss_ctl` | 平稳下降到 < 0.3 | 突然飙升 (> 2.0) |
| 两 loss 比例 | ctl:sft ≈ 1:5~8 | ctl > sft (投影头崩了) |

### 显存

每步峰值应稳定在 **~25 GB**。持续上涨说明有内存泄漏。

### SFT 验收清单

- [ ] Phase 1 在 250 步内成功切换到 Phase 2
- [ ] Phase 2 `loss_sft` 降到 < 1.0
- [ ] `loss_ctl` < 0.3
- [ ] 无 OOM，显存稳定
- [ ] 3 epochs 正常完成

---

## DPO (Stage II) — 关键信号

### 核心指标

```
[Step 100] loss_dpo=0.45, dpo_accuracy=0.72, loss_ctl=3.21, grad_norm=85.3
```

| 信号 | 健康 | 异常 |
|------|:--:|------|
| `loss_dpo` | 从 ~0.69 逐步降到 ~0.3-0.5 | 卡在 0.6931 (随机水平) |
| `dpo_accuracy` | 逐步升到 > 0.6 | 一直在 0.5 附近 (抛硬币) |
| `loss_ctl` | 平稳或微降 | 暴涨 (投影头退化) |
| `grad_norm_lora_total` | 20-200 | > 500 (梯度爆炸) 或 < 1 (梯度消失) |

### dpo_accuracy 含义

chosen 样本的 log-prob > rejected 样本的比例：
- 0.5 = 模型分不清好坏
- > 0.65 = 模型在学习偏好，正常
- > 0.8 = 学得不错

### DPO 验收清单

- [ ] `loss_dpo` 从 ~0.69 下降到 < 0.4
- [ ] `dpo_accuracy` 稳定在 > 0.65
- [ ] `loss_ctl` 不暴涨
- [ ] 无 NaN/Inf
- [ ] 2 epochs 正常完成

---

## 实用监控命令

### 实时看日志

```bash
tail -f sft.log    # SFT
tail -f dpo.log    # DPO
```

### 快速抽取指标

```bash
# SFT loss 趋势
grep "loss_sft\|loss_ctl" sft.log | tail -30

# DPO 核心
grep "loss_dpo\|dpo_accuracy" dpo.log | tail -30

# 检查 OOM
grep -i "out of memory\|OOM" sft.log dpo.log

# 检查 NaN
grep -i "nan\|inf" sft.log dpo.log
```

### 显存监控（另开终端）

```bash
watch -n 5 nvidia-smi
```

### nohup 后台运行

```bash
# SFT (~12-15h)
nohup python src/training/train_sft.py \
  --config configs/rtx5090.yaml \
  --data_dir /root/autodl-tmp/data/full5000 \
  > sft.log 2>&1 &

# 记下 PID 方便 kill
echo $!

# DPO (~10-12h, SFT 完成后)
nohup python src/training/train_dpo.py \
  --config configs/rtx5090.yaml \
  --stage1_ckpt /root/autodl-tmp/outputs/stage1_sft_final \
  --data_dir /root/autodl-tmp/data/full5000 \
  > dpo.log 2>&1 &

echo $!
```

---

## 异常处理速查

| 症状 | 可能原因 | 处理 |
|------|------|------|
| Phase 1 sens 不涨 | LR 太低 / 数据差 | 增大 `phase1.lr_lora` → 1e-3 |
| Phase 2 loss 震荡 | LR 太高 | 降 `sft.learning_rate` → 1e-4 |
| loss_ctl 突然暴涨 | 投影头数值不稳定 | 检查数据 `delta_q/a/p` 标签是否合法 |
| loss_dpo 卡在 0.693 | 模型学不到偏好 | 检查 DPO 数据 chosen/rejected 质量 |
| OOM | 显存不够 | SFT: 降 bs/seq_len; DPO: 降 seq_len/mu |
| grad_norm > 500 | 梯度爆炸 | 降低 LR 或增大 `max_grad_norm` |
| grad_norm < 1 | 梯度消失 | 检查 LoRA 是否被冻结 |
