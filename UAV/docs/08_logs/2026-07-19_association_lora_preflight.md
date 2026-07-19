---
type: log
status: a4_lora_preflight_complete_validation_pending
stage: multimodal_association_lora_preflight
last_updated: 2026-07-19
---

# 2026-07-19 A4 Association + LoRA 预检

## 目标

A3 projection-only 在 train500/独立 val100 上能够泛化，但绝对准确率仍偏低：

```text
train accuracy: 0.4372
val accuracy:   0.4280
val fixed-user majority: 0.3115
```

因此从 A3 checkpoint 初始化，执行保守的小学习率 A+LoRA 预检：

```text
max_steps: 200
freeze_qp_branch: true
train_lora: true
projection_lr: 0.0003
lora_lr: 0.00005
lambda_assoc_ce: 0.2
lambda_assoc_raw_ce: 1.0
lambda_q/a/p: 0 / 0 / 0
```

## 训练完成

服务器已完成训练并保存：

```text
OK: multimodal SFT smoke complete
final_checkpoint:
/root/autodl-tmp/outputs/mm_geom_v3_stage_a4_assoc_lora_preflight200/mm_sft_lora_smoke_final
```

当前只能确认训练闭环与 LoRA checkpoint 保存成功，状态为
`training_complete_validation_pending`。

## 验收要求

LoRA 会改变 q/a/p 共享的 control states。即使 `freeze_qp_branch` 冻结了 q/p 投影
参数，也不能保证 q/p 的实际输出保持不变。因此必须同时诊断：

1. A 的 train500 与独立 val100 accuracy、baseline gain、oracle probability；
2. P 的 val100 active/inactive/sensing MSE、leakage、输出方差；
3. Q 的 val100 direction cosine 与输出方差；
4. `loaded_lora_checkpoint` 必须指向本次 checkpoint 的 `lora` 目录；
5. train/val gap 不能明显扩大。

只有 A 提升且 P/Q 没有不可接受回退，才允许从 200-step 预检进入更长 A+LoRA 训练。
