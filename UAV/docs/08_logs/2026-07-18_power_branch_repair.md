---
type: log
status: code_and_target_gates_pass_p_training_pending
stage: multimodal_power_branch_repair
last_updated: 2026-07-18
---

# 2026-07-18 功率分支 P0 修复

## 优先级结论

完整 SFT/DPO 之前按以下顺序处理阻塞项：

1. P0：修复功率投影、功率损失和诊断指标；
2. P1：在 train500/val100 上验证 association；
3. P2：关闭 dynamic cue，用 direct q-direction + LoRA 验证 q；
4. P3：三个分支均通过后才允许联合 SFT/DPO。

本日志只记录 P0。

## 已确认的问题

旧 `PowerProjection` 默认对全部 `M*K` 个通信条目施加
`0.01 * P_max` 下界，即使该 UAV 没有关联对应用户。

oracle 数据的 `delta_p` 来自 SCA-FP beamformer 功率，未关联位置通常为 0。
因此旧实现存在直接冲突：

```text
oracle inactive communication power = 0
old projection inactive communication power >= 0.01 * P_max
```

旧功率损失又对全部 `M*(K+1)` 条目直接做一次 MSE。对于 `M=4, K=20`，
大量未关联零条目会压过已关联通信功率和感知功率监督，容易得到近常数小功率解。
历史 Stage B2 已出现：

```text
delta_p_per_dim_std_mean: 6.2088170160734535e-09
warnings: ['delta_p_low_cross_sample_variance']
```

## 代码修改

### 1. PowerProjection 与论文公式对齐

默认路径现在只执行：

```text
p_hat = P_max * softmax(p_raw / tau_p)
```

默认 `p_min_ratio` 从 `0.01` 改为 `0.0`，不再给未关联用户强制分配功率。

若未来显式启用 `p_min_ratio > 0`，必须传入 association 权重；下界变为：

```text
p_floor[m,k] = p_min * association[m,k]
```

并对剩余预算重新归一化，保证总功率不超过 `P_max`。

### 2. 关联感知的分组功率损失

新损失分别计算：

```text
loss_p_active    # 已关联用户通信功率
loss_p_inactive  # 未关联用户功率泄漏
loss_p_sensing   # 感知功率
loss_p = (active + inactive + sensing) / 3
```

三组分别求均值后等权组合，避免未关联零条目的数量支配梯度。

### 3. 日志与诊断

训练日志新增：

```text
loss_p
loss_p_active
loss_p_inactive
loss_p_sensing
```

`analyze_mm_delta_outputs.py` 新增保存和输出：

```text
delta_p_raw
delta_a_target
delta_p_target
delta_p_active_comm_mse
delta_p_inactive_comm_mse
delta_p_sensing_mse
delta_p_inactive_power_leakage_mean
delta_p_total_per_uav_mae
```

新增 warning：

```text
delta_p_inactive_power_leakage
```

### 4. 回归测试

新增：

```text
tests/test_power_branch.py
```

覆盖默认 simplex、可选 association-aware floor、预算守恒、分组损失和反向梯度。

本机已通过 `py_compile` 和 `git diff --check`。本机 Python 环境没有安装
PyTorch，运行时单元测试必须在服务器 `uavmllm` 环境执行。

## 服务器验证顺序

### A. 单元测试

```bash
python -m unittest tests.test_power_branch -v
```

### B. 先检查 train500/val100 oracle 功率标签

```bash
python scripts/analyze_mm_target_distribution.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_geom_v3_train500_seed42 \
  --output /root/autodl-tmp/outputs/mm_geom_v3_probe/train500_target_distribution.json

python scripts/analyze_mm_target_distribution.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_geom_v3_val100_seed2026 \
  --output /root/autodl-tmp/outputs/mm_geom_v3_probe/val100_target_distribution.json
```

必须重点确认：

```text
target_delta_p_per_dim_std_mean > 0
target_delta_p_inactive_comm_mean ≈ 0
target_delta_p_inactive_nonzero_ratio ≈ 0
```

只有标签本身满足这些条件，才进入 P-only 训练。

## P0 验收门槛

代码级门槛：

```text
server unittest PASS
PowerProjection sum(delta_p, dim=-1) == P_max
```

训练级门槛将在标签检查后执行：

```text
不再出现 delta_p_low_cross_sample_variance
validation delta_p_active_comm_mse 相比初始化下降
validation delta_p_sensing_mse 相比初始化下降
delta_p_inactive_power_leakage_mean < 0.01
```

初始代码提交后，P0 状态为 `code_complete_runtime_pending`；目标分布和
单元测试均通过前不进入完整联合训练。

## 2026-07-18 服务器标签验证结果

train500：

```text
target_delta_p_per_dim_std_mean: 0.0716962442
target_delta_p_active_comm_mean: 0.0996794403
target_delta_p_inactive_comm_mean: 0.0
target_delta_p_inactive_nonzero_ratio: 0.0
target_delta_p_sensing_mean: 0.5016000271
target_delta_p_total_per_uav_mean: 0.9999971986
target_delta_p_total_per_uav_std: 0.0000649001
```

val100（独立 `seed=2026`）：

```text
target_delta_p_per_dim_std_mean: 0.0673355162
target_delta_p_active_comm_mean: 0.1000996977
target_delta_p_inactive_comm_mean: 0.0
target_delta_p_inactive_nonzero_ratio: 0.0
target_delta_p_sensing_mean: 0.4995000064
target_delta_p_total_per_uav_mean: 0.9999984503
target_delta_p_total_per_uav_std: 0.0000640142
```

判定：

```text
PASS: 功率标签具有非零跨环境变化；
PASS: 未关联通信功率严格为 0；
PASS: 每架 UAV 的 oracle 总功率约为 P_max=1；
PASS: train500 与独立 val100 的 active/sensing/total 分布高度一致。
```

因此历史 p 常数坍缩不能归因于 oracle 标签单一或 train/validation
分布偏移。修复 PowerProjection 的无条件下界及重新平衡功率损失是必要的。
当前只剩服务器单元测试 gate；通过后进入 P-only 初始化基线与训练。

## 2026-07-18 服务器单元测试结果

服务器 `uavmllm` 环境执行：

```text
test_exact_power_target_has_zero_grouped_loss ... ok
test_inactive_entries_do_not_dominate_active_and_sensing_groups ... ok
test_default_projection_is_simplex_without_unconditional_floor ... ok
test_optional_floor_is_association_aware_and_budget_safe ... ok
test_optional_floor_requires_association ... ok

Ran 5 tests in 0.181s
OK
```

判定：

```text
PASS: 默认 simplex 无条件下界已移除；
PASS: 可选下界受 association 控制且保持预算；
PASS: 分组功率损失数值与梯度正确；
PASS: P0 代码 gate 与 target-data gate 均通过。
```

下一阶段先测旧 Stage A2 checkpoint 在新投影公式下的 val100 初始化基线，
然后进行 P-only 训练，并用同一独立验证集检查是否超过初始化基线。
