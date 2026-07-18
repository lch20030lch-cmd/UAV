---
type: log
status: a_only_train500_pending
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
