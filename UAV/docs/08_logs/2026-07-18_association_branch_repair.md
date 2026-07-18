---
type: log
status: a_only_generalizes_but_underfit_lora_preflight_pending
stage: multimodal_association_branch_repair
last_updated: 2026-07-18
---

# 2026-07-18 Association 分支 P1 修复

## 触发原因

P0.1 raw-KL 功率分支在独立 val100 上显著降低了功率误差，但仍存在：

```text
delta_p_inactive_power_leakage_mean: 0.0277360305
delta_p_per_dim_std_mean: 0.0051548160
delta_p_target_per_dim_std_mean: 0.0673355162
```

进一步采用 association-aware 功率约束之前，必须先确认 association 不仅会随场景
变化，而且确实命中 oracle UAV。旧诊断只有 `unique_per_user`，不能区分“动态且正确”
与“动态但选错”。

## 新增诊断

`analyze_mm_delta_outputs.py` 已新增：

```text
delta_a_argmax_accuracy
delta_a_fixed_user_majority_accuracy
delta_a_accuracy_gain_over_fixed_user_majority
delta_a_oracle_probability_mean
delta_a_oracle_probability_std
warning: delta_a_not_above_fixed_user_majority
```

并新增 `tests/test_delta_diagnostics.py`，验证 accuracy、固定用户多数基线和形状检查。

## Stage A2 在独立 val100 上的真实正确性

当前 P0.1 checkpoint 的 association 参数继承自旧 geom-v3 Stage A2，在独立
`val100_seed2026` 上得到：

```text
delta_a_argmax_unique_per_user_mean: 3.45
delta_a_argmax_fixed_user_count: 0
delta_a_argmax_accuracy: 0.3760
delta_a_fixed_user_majority_accuracy: 0.3115
delta_a_accuracy_gain_over_fixed_user_majority: 0.0645
delta_a_oracle_probability_mean: 0.3414420
warnings: ['delta_p_inactive_power_leakage']
```

判定：

```text
PASS: association 不是固定模板，并且高于固定用户多数基线；
FAIL: 37.6% 的准确率不足以作为 hard power mask；
FAIL: 若立即 hard mask，约 62.4% 用户的 oracle active 功率位置可能被错误屏蔽。
```

因此不能根据旧的 diversity 指标宣称 association 已稳定，也不能立即把预测
association 写进 PowerProjection 的硬掩码路径。

## 下一步：A-only train500

使用当前 P0.1 完整 checkpoint 初始化，以保留已修好的 P 参数；通过
`--freeze_qp_branch` 冻结 q、q-cue、p，只训练：

```text
readout_a
a_mlp
```

建议参数：

```text
data: train500_seed42
max_steps: 1000
projection_lr: 0.001
lambda_assoc_ce: 0.2
lambda_assoc_raw_ce: 1.0
lambda_q/a/p: 0 / 0 / 0
lambda_p_raw_kl: 0
```

训练后必须重新检查 train500 和独立 val100 的 accuracy、固定多数基线、oracle
probability 与 train/val gap。只有独立验证准确率显著提升后，才讨论 soft/hard
association-aware power projection。

## 2026-07-18 A-only train500 完成

服务器已完成 1000-step A-only 训练：

```text
OK: multimodal SFT smoke complete
final checkpoint:
/root/autodl-tmp/outputs/mm_geom_v3_stage_a3_assoc_only_train500_1000step/mm_sft_smoke_final
```

该 checkpoint 从 P0.1 完整功率 checkpoint 初始化，通过 `--freeze_qp_branch` 只更新
association 分支。当前状态为 `training_complete_validation_pending`；必须同时检查完整
train500 和独立 val100 后才能判定是否通过。

## 2026-07-18 A-only train500 / 独立 val100 结果

完整 train500：

```text
delta_a_argmax_unique_per_user_mean: 4.0
delta_a_argmax_fixed_user_count: 0
delta_a_argmax_accuracy: 0.4372
delta_a_fixed_user_majority_accuracy: 0.3173
delta_a_accuracy_gain_over_fixed_user_majority: 0.1199
delta_a_oracle_probability_mean: 0.3739552
```

独立 val100：

```text
delta_a_argmax_unique_per_user_mean: 4.0
delta_a_argmax_fixed_user_count: 0
delta_a_argmax_accuracy: 0.4280
delta_a_fixed_user_majority_accuracy: 0.3115
delta_a_accuracy_gain_over_fixed_user_majority: 0.1165
delta_a_oracle_probability_mean: 0.3559454
```

分支隔离检查：q/p 的 train500 与 val100 指标均与 A-only 训练前一致，说明
`--freeze_qp_branch` 没有更新 q/p 投影参数。

判定：

```text
PASS: train/val accuracy gap 仅 0.0092，没有明显过拟合；
PASS: val accuracy 相比旧 Stage A2 的 0.376 提升到 0.428；
PASS: train/val 均比各自 fixed-user majority 高约 0.12；
PASS: unique=4、fixed=0，association 没有退化为固定模板；
FAIL: train accuracy 也只有 0.4372，projection-only 仍明显欠拟合；
FAIL: 0.428 的 val accuracy 仍不足以安全执行 hard power mask。
```

## 下一步：保守 A+LoRA 预检

历史 20-sample staged 实验显示，小学习率 LoRA 曾把 association match 从 0.50
提升到 0.5525；但直接 LoRA 训练也曾出现固定解。因此不立即长训，先执行 200-step
预检：

```text
init: A3 association-only train500 checkpoint
train_lora: true
freeze_qp_branch: true
projection_lr: 0.0003
lora_lr: 0.00005
lambda_assoc_ce: 0.2
lambda_assoc_raw_ce: 1.0
```

注意：冻结 q/p 投影参数不能阻止 LoRA 改变它们读取的 control states。因此预检验收
必须同时检查：

1. association train/val accuracy 是否继续提升；
2. q/p 指标是否因 LoRA 产生不可接受的功能回退；
3. train/val gap 是否仍然较小。
