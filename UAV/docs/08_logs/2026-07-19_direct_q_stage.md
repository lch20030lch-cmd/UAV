---
type: log
status: direct_q_lora_preflight_complete_validation_pending
stage: multimodal_direct_q_direction
last_updated: 2026-07-19
---

# 2026-07-19 Direct Q 主线准备

## 当前阻塞

A4 独立 val100 上的 Q 仍明显不足：

```text
delta_q_per_dim_std_mean: 0.179109
target_delta_q_per_dim_std_mean: 8.350011
delta_q_raw_dir_cosine_mean: 0.087146
delta_q_raw_dir_mse_mean: 0.608570
```

历史 direct-Q 最好方向 cosine 约为 0.274；dynamic cue selector 又不能在独立环境上
超过 fixed mixture。因此 Q 主线回到：

```text
q_geometry_mode = none
q_projection_mode = direction
direct q-direction loss + small-LR LoRA
```

dynamic cue 只保留为失败消融，不能接入主方法。

## 新增 Q-only 投影隔离

`train_sft_mm.py` 新增：

```text
--freeze_all_except_q
```

它只保留以下 projection 参数可训练：

```text
readout_q
q_mlp
```

并冻结：

```text
readout_q_cue
readout_a / a_mlp
readout_p / p_mlp
```

同时修正分支前缀匹配为模块边界匹配，避免 `readout_q` 前缀意外把
`readout_q_cue` 也设为可训练。checkpoint metadata 新增：

```text
freeze_all_except_q
isolated_projection_branch = q
```

新增 `tests/test_training_branch_freeze.py` 验证 direct-Q 隔离不会打开 q-cue/A/P。

## LoRA retention 原则

LoRA 会改变共享 control states，单纯冻结 A/P projection 参数仍可能让 A/P 功能回退。
因此 direct-Q 训练需要保留小权重 A/P 监督，让 retention loss 通过冻结的 A/P 读出
反向约束 LoRA，同时重点优化 `loss_q_dir`。

服务器单元测试通过后，先执行短预检，不直接长训；验收必须同时比较 Q 提升与 A/P
回退幅度。

## Q1 200-step Direct-Q + LoRA 预检完成

服务器已完成训练：

```text
OK: multimodal SFT smoke complete
final checkpoint:
/root/autodl-tmp/outputs/mm_geom_v3_stage_q1_direct_direction_lora_preflight200/mm_sft_lora_smoke_final
```

本轮设置：

```text
q_projection_mode: direction
q_geometry_mode: none
freeze_all_except_q: true
train_lora: true
projection_lr: 0.001
lora_lr: 0.00002
lambda_q_dir: 1.0
lambda_assoc_ce/raw_ce: 0.05 / 0.2
lambda_p/raw_kl: 0.05 / 0.2
```

当前状态为 `training_complete_validation_pending`。必须同时诊断 train500 和独立
val100，并以 raw Q direction cosine 为主指标；`direction` 投影会机械地把 Q 幅度
拉到 15m，因此不能仅凭 projected Q 方差增大判定成功。A/P 指标也必须与 A4 基线
比较，以确认 retention loss 是否有效。
