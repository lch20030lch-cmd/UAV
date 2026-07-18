---
type: log
status: p_only_generalizes_but_underfit_association_gate_pending
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

## 2026-07-18 Stage A2 独立验证初始化基线

使用旧 geom-v3 Stage A2 split-association checkpoint，在修复后的默认
PowerProjection 下对独立 val100 推理：

```text
delta_p_per_dim_std_mean: 0.0219130013
delta_p_raw_per_dim_std_mean: 0.1918728501
delta_p_target_per_dim_std_mean: 0.0673355162
delta_p_mse: 0.0299952198
delta_p_active_comm_mse: 0.0198893212
delta_p_inactive_comm_mse: 0.0131981177
delta_p_sensing_mse: 0.3324812353
delta_p_inactive_power_leakage_mean: 0.0491444282
delta_p_total_per_uav_pred_mean: 1.0
delta_p_total_per_uav_target_mean: 0.9999984503
delta_p_total_per_uav_mae: 0.0000355333
warnings: ['delta_p_inactive_power_leakage']
```

冻结的 association 初始化状态：

```text
delta_a_argmax_unique_per_user_mean: 3.45
delta_a_argmax_fixed_user_count: 0
```

判定：

```text
PASS: 移除无条件 floor 后，未训练 p 分支已不再是近零方差常数输出；
PASS: simplex 严格保持总功率约 1；
FAIL: inactive leakage 0.04914 > 0.01；
FAIL: sensing MSE 0.33248 很高；
P-only 训练有必要，且已有独立验证基线可比较。
```

P-only 训练后的独立 val100 验收标准：

```text
delta_p_active_comm_mse < 0.0198893212
delta_p_inactive_comm_mse < 0.0131981177
delta_p_sensing_mse < 0.3324812353
delta_p_inactive_power_leakage_mean < 0.01
delta_p_per_dim_std_mean > 1e-4
delta_a_argmax_unique_per_user_mean ≈ 3.45
delta_a_argmax_fixed_user_count = 0
warnings 不包含 delta_p_low_cross_sample_variance
warnings 不包含 delta_p_inactive_power_leakage
```

## P-only 参数隔离

为避免 `--freeze_assoc_branch + lambda_q=0` 仍把无梯度 q 参数放进优化器，
训练脚本新增：

```text
--freeze_all_except_p
```

它只保留以下参数可训练：

```text
readout_p
p_mlp
```

训练启动时必须看到：

```text
isolated projection branch:  power
isolated trainable tensors:   > 0
trainable LoRA tensors:       0
lambda_q/a/p:                 0.0 / 0.0 / 1.0
```

这样 Stage A2 的 q、association 和实验性 q-cue 参数均不会被 P-only 更新。

## 2026-07-18 P-only train500 完成

服务器完成 500-step P-only 训练：

```text
OK: multimodal SFT smoke complete
final_checkpoint:
/root/autodl-tmp/outputs/mm_geom_v3_stage_p0_power_only_train500_500step/mm_sft_smoke_final
```

该 checkpoint 的继承关系为：

```text
geom-v3 Stage A2 split association
  -> freeze_all_except_p
  -> train500 / 500 steps / projection_lr=0.001
  -> P-only checkpoint
```

训练完成本身不等于 P0 通过。下一步必须分别诊断 train500 与独立 val100，
并与本日志中的 Stage A2 val100 初始化基线逐项比较。

## 2026-07-18 第一次 P-only train500 诊断：失败

第一次 P-only checkpoint 在 train500 上的关键结果：

```text
delta_p_per_dim_std_mean: 1.3321270842e-07
delta_p_raw_per_dim_std_mean: 0.2548862994
delta_p_target_per_dim_std_mean: 0.0716962442
delta_p_mse: 0.0663626939
delta_p_active_comm_mse: 0.0583080761
delta_p_inactive_comm_mse: 0.0499998517
delta_p_sensing_mse: 0.3520784378
delta_p_inactive_power_leakage_mean: 0.0499999262
delta_p_entropy_mean: 1.9797970356e-05
warnings:
  - delta_p_low_cross_sample_variance
  - delta_p_inactive_power_leakage
```

判定：

```text
FAIL: 投影后功率方差约为 1e-7，已经跨样本塌缩；
FAIL: 熵约为 2e-5，softmax 已经饱和为错误的近 one-hot 分配；
FAIL: active、inactive、sensing 三组误差均未改善；
FAIL: inactive leakage 仍约为 0.05。
```

`delta_p_raw` 仍有方差而投影后的 `delta_p` 几乎恒定，说明本轮主要问题不是
control states 完全无信息，而是只用投影后 MSE 时，错误饱和的 softmax 梯度接近
零。此 checkpoint 不进入独立 val100 验收，也不能作为后续训练的初始化。

## P0.1：raw power soft-target KL 修复

在 `PowerProjection` 之前新增 raw logits 辅助损失：

```text
target_prob = normalize(clamp(delta_p_target, min=0))
log_pred = log_softmax(delta_p_raw / tau_power)
loss_p_raw_kl = KL(target_prob || log_pred)
```

总功率监督更新为：

```text
lambda_p * grouped_projected_mse
  + lambda_p_raw_kl * raw_soft_target_kl
```

这样即使 softmax 已经错误饱和，raw KL 仍能直接对 logits 提供非零纠正梯度。
默认配置把 `lambda_p_raw_kl` 设为 `0.0`，因此旧实验和其他训练入口不会被静默
改变；P0.1 预检会显式设置该权重。

训练日志同时新增：

```text
loss_p_raw_kl
delta_p_entropy
delta_p_inactive_leakage
```

新增回归测试检查错误饱和 logits 上的 raw KL 仍为有限值且反向梯度非零。

新的执行顺序：

1. 服务器运行 6 项功率分支单元测试；
2. 从干净的 Stage A2 checkpoint 重新开始 50-step P-only 预检；
3. 预检确认熵未快速降到 0、raw KL 能下降后，再从 Stage A2 干净初始化运行完整 500 step；
4. 完整训练通过 train500 与独立 val100 验收后，才进入 association 的 P1 阶段。

## 2026-07-18 P0.1 raw-KL 50-step 预检结果

从干净 Stage A2 checkpoint 初始化，只训练 `readout_p / p_mlp`，使用：

```text
max_steps=50
projection_lr=0.0003
lambda_p=0.1
lambda_p_raw_kl=1.0
```

在 train500 的前 100 条样本上诊断：

```text
delta_p_per_dim_std_mean: 0.0180089530
delta_p_raw_per_dim_std_mean: 0.1297873259
delta_p_target_per_dim_std_mean: 0.0678294450
delta_p_mse: 0.0102780098
delta_p_active_comm_mse: 0.0137879374
delta_p_inactive_comm_mse: 0.0008396233
delta_p_sensing_mse: 0.1343041658
delta_p_inactive_power_leakage_mean: 0.0197597090
delta_p_entropy_mean: 1.6685179450
delta_p_total_per_uav_pred_mean: 1.0
delta_p_total_per_uav_mae: 0.0000330287
warnings: ['delta_p_inactive_power_leakage']
```

与第一次失败的 P-only 训练相比：

```text
projected variance: 1.33e-7 -> 0.01801
entropy:            1.98e-5 -> 1.66852
inactive MSE:       0.05000 -> 0.00084
sensing MSE:        0.35208 -> 0.13430
inactive leakage:   0.05000 -> 0.01976
```

判定：

```text
PASS: raw KL 阻止了错误 one-hot 饱和和跨样本常数塌缩；
PASS: active、inactive、sensing 三组监督均产生了有效改善；
PASS: association 仍保持 unique=3.9、fixed_user_count=0，P-only 隔离有效；
PENDING: inactive leakage 仍高于最终 0.01 门槛，需在完整 500-step 后复验。
```

因此允许从同一个干净 Stage A2 初始化运行完整 500-step P-only 实验。为保持实验
可复现性，不从 50-step 预检 checkpoint 续训，也不继承第一次失败的 checkpoint。

## 2026-07-18 P0.1 raw-KL 完整 500-step 训练完成

服务器已完成从干净 Stage A2 初始化的完整 P-only 训练：

```text
OK: multimodal SFT smoke complete
final_checkpoint:
/root/autodl-tmp/outputs/mm_geom_v3_stage_p0_power_raw_kl_train500_500step/mm_sft_smoke_final
```

当前状态为 `training_complete_validation_pending`。下一步依次检查训练末段日志、
完整 train500 诊断和独立 val100 诊断；在独立验证结果通过前，不能将该 checkpoint
并入联合 SFT，也不能进入 DPO。

## 2026-07-18 P0.1 完整 train500 / 独立 val100 诊断

完整 train500：

```text
delta_p_per_dim_std_mean: 0.0051668622
delta_p_target_per_dim_std_mean: 0.0716962442
delta_p_mse: 0.0081159258
delta_p_active_comm_mse: 0.0125617245
delta_p_inactive_comm_mse: 0.0009256767
delta_p_sensing_mse: 0.0937406793
delta_p_inactive_power_leakage_mean: 0.0274850018
delta_p_entropy_mean: 2.2866133624
```

独立 val100：

```text
delta_p_per_dim_std_mean: 0.0051548160
delta_p_target_per_dim_std_mean: 0.0673355162
delta_p_mse: 0.0078541981
delta_p_active_comm_mse: 0.0114227328
delta_p_inactive_comm_mse: 0.0009410096
delta_p_sensing_mse: 0.0937093645
delta_p_inactive_power_leakage_mean: 0.0277360305
delta_p_entropy_mean: 2.2955480158
warnings: ['delta_p_inactive_power_leakage']
```

相对 Stage A2 独立 val100 初始化基线：

```text
overall MSE:    0.0299952 -> 0.0078542  (-73.8%)
active MSE:     0.0198893 -> 0.0114227  (-42.6%)
inactive MSE:   0.0131981 -> 0.0009410  (-92.9%)
sensing MSE:    0.3324812 -> 0.0937094  (-71.8%)
inactive leak:  0.0491444 -> 0.0277360  (-43.6%)
```

判定：

```text
PASS: raw-KL 功率修复在独立环境上稳定泛化，train/val 没有可见过拟合间隙；
PASS: catastrophic one-hot collapse 已消失，总误差和三组误差显著下降；
FAIL: inactive leakage 0.02774 仍高于 0.01 最终门槛；
FAIL: val 输出方差仅为目标方差约 7.7%，样本级功率变化仍明显不足。
```

结果更接近跨环境平均功率模板，而不是充分的逐环境功率预测。继续盲目增加 P-only
步数不能直接证明会解决该问题。下一步先完成 association 正确性 gate：只有
association 不仅多样、而且在独立验证上确实命中 oracle，才允许尝试 association-aware
功率掩码或联合 A/P 训练。

为此，`analyze_mm_delta_outputs.py` 新增：

```text
delta_a_argmax_accuracy
delta_a_fixed_user_majority_accuracy
delta_a_accuracy_gain_over_fixed_user_majority
delta_a_oracle_probability_mean
delta_a_oracle_probability_std
warning: delta_a_not_above_fixed_user_majority
```

旧指标 `unique_per_user` 只能证明预测会变化，不能证明选对 UAV；上述新指标用于补齐
这一诊断缺口。
