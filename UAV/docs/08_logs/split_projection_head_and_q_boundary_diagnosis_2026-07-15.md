# split projection head 与 q 边界饱和诊断

日期：2026-07-15

## 背景

在 100 条 BEV-image MLLM smoke 数据上，shared projection head 的 Stage A / Stage B 结果显示：

| 实验 | association unique mean | fixed user count | argmax match rate |
| --- | ---: | ---: | ---: |
| Stage A shared association warmup | 2.35 | 3 | 0.373 |
| Stage B shared q/a/p mixed CTL | 2.25 | 4 | 0.344 |
| Stage B shared strong assoc | 2.10 | 5 | 0.3615 |

这说明 association 单独训练能学到一部分，但加入 q/p 混合训练后会回退；单纯提高 association raw CE 权重也无法恢复多样性。

因此新增 split projection head：

```text
control_states -> q_readout -> q_mlp -> Proj_Q
control_states -> a_readout -> a_mlp -> Proj_A
control_states -> p_readout -> p_mlp -> Proj_P
```

并新增分支冻结参数：

```bash
--projection_head_type split
--freeze_qp_branch
--freeze_assoc_branch
```

## Stage A2：split association warmup

命令要点：

```text
projection_head_type = split
freeze_qp_branch = true
lambda_q/a/p = 0 / 0 / 0
lambda_assoc_raw_ce = 1.0
projection_lr = 0.01
max_steps = 300
```

诊断结果：

```text
delta_a_argmax_unique_per_user_mean: 2.95
delta_a_argmax_fixed_user_count: 1
delta_a_raw_per_dim_std_mean: 13.115493774414062
delta_a_raw_argmax_unique_per_user_mean: 3.4
delta_a_raw_argmax_fixed_user_count: 0
warnings: []
```

target-vs-pred：

```text
pred_delta_a_argmax_unique_per_user_mean: 2.95
pred_delta_a_argmax_fixed_user_count: 1
argmax_match_rate_mean: 0.4315
argmax_match_rate_per_user_min: 0.33
argmax_match_rate_per_user_max: 0.51
```

对比 shared Stage A：

| 指标 | shared Stage A | split Stage A2 |
| --- | ---: | ---: |
| unique mean | 2.35 | 2.95 |
| fixed user count | 3 | 1 |
| match rate | 0.373 | 0.4315 |

结论：

```text
同一份 100 条数据、同一 prompt/image 输入下，仅修改 projection head 结构后，
association 多样性和匹配率都明显提升。
这支持 shared projection trunk 是 association 学习不稳的重要原因之一。
```

## Stage B2：冻结 association，只训练 q/p

命令要点：

```text
init_checkpoint = Stage A2
projection_head_type = split
freeze_assoc_branch = true
lambda_q/a/p = 1.0 / 0 / 0.3
lambda_assoc_raw_ce = 0
projection_lr = 0.002
max_steps = 300
```

诊断结果：

```text
delta_q_per_dim_std_mean: 0.1623062789440155
delta_a_per_dim_std_mean: 0.14648902416229248
delta_p_per_dim_std_mean: 6.2088170160734535e-09
delta_a_argmax_unique_per_user_mean: 2.95
delta_a_argmax_fixed_user_count: 1
delta_a_raw_argmax_unique_per_user_mean: 3.4
delta_a_raw_argmax_fixed_user_count: 0
warnings: ['delta_p_low_cross_sample_variance']
```

target-vs-pred：

```text
pred_delta_a_argmax_unique_per_user_mean: 2.95
pred_delta_a_argmax_fixed_user_count: 1
argmax_match_rate_mean: 0.4315
```

结论：

```text
freeze_assoc_branch 生效，Stage B2 没有破坏 association。
这说明 split head 已解决 q/p 训练直接改坏 association 分支的问题。

但 q 仅轻微变化，p 直接低方差塌缩。
因此 q/p 学习不足是另一个独立问题，不能再归因于 association 分支被干扰。
```

## Stage B2 q-only 诊断

为隔离 power 分支，进一步只训练 q：

```text
init_checkpoint = Stage A2
projection_head_type = split
freeze_assoc_branch = true
lambda_q/a/p = 1.0 / 0 / 0
projection_lr = 0.005
max_steps = 500
```

诊断结果：

```text
delta_q_per_dim_std_mean: 0.0560290701687336
delta_a_argmax_unique_per_user_mean: 2.95
delta_a_argmax_fixed_user_count: 1
argmax_match_rate_mean: 0.4315
warnings: []
```

结论：

```text
association 仍然稳定，但 q-only 没有拉起 q，反而比 Stage A2 / B2 更弱。
这说明 q 学习不足不是 power 分支拖累，而更可能是 q target 分布或 q loss 形式的问题。
```

## q target 分布分析

对 `/root/autodl-tmp/data/mm_smoke_100/sft_dataset.jsonl` 的 `delta_q` 做统计：

```text
shape: (100, 4, 3)
overall mean: 0.9380278587341309
overall std: 8.606552124023438
per dim mean: [-0.462178, -0.29670396, 3.5729644]
per dim std: [8.420082, 8.811498, 7.9529824]
per dim min: [-14.9467, -14.6809, -14.9621]
per dim max: [14.9647, 14.8699, 14.6665]
norm mean: 14.995257377624512
norm std: 0.012648346833884716
norm min/max: 14.783653259277344 / 15.000061988830566
near 15m ratio: 1.0
```

关键发现：

```text
100% 的 delta_q target 都接近 15m 移动上限。
```

这意味着 q target 并不是普通连续位移回归，而更接近：

```text
方向向量 + 固定半径 15m
```

当前直接对投影后的 `delta_q` 做 MSE，会受到 `Proj_Q` 移动性裁剪影响，容易使模型学到保守小位移或梯度不稳定方向，而不是稳定学习方向。

## 当前判断

问题已经拆成三层：

1. **association 固定 / 回退问题**
   - 已由 split projection head 明显改善。
   - Stage A2 从 `2.35 / fixed 3 / match 0.373` 提升到 `2.95 / fixed 1 / match 0.4315`。

2. **Stage B 中 association 被 q/p 带偏的问题**
   - 已由 `freeze_assoc_branch` 控制住。
   - Stage B2 后 association 指标与 Stage A2 完全一致。

3. **q/p 本身学习不足的问题**
   - q target 全部贴 15m 速度墙，直接 MSE 不适合作为唯一监督。
   - p 在 q/p 联训中出现低方差塌缩，需要后续单独分析。

## 下一步建议

优先改 q 的训练目标，而不是继续增加 step 或盲目调学习率。

建议新增 q 方向辅助损失：

```text
q_dir_target = delta_q_target / ||delta_q_target||
q_dir_pred = delta_q_raw / ||delta_q_raw||
loss_q_dir = MSE(q_dir_pred, q_dir_target)
```

新增训练参数建议：

```bash
--lambda_q_dir
```

下一轮实验：

```text
Stage B3 q-direction:
  init_checkpoint = Stage A2
  projection_head_type = split
  freeze_assoc_branch = true
  lambda_q = 0
  lambda_q_dir = 1.0
  lambda_p = 0
```

目标：

```text
先验证 q 分支能否学到 15m 边界上的方向信息。
如果 q-direction 能拉起方向多样性，再考虑恢复投影后 q MSE 或 q/p 联合训练。
```

对于老师提出的“增强 prompt 和图片信息量”方向，当前结论是：

```text
association 的提升来自 projection head 结构修改，说明输入信息不是完全不足。
但 q/p 仍可能需要更明确的输入提示，例如最近用户方向、UAV-target 几何关系、负载/信道排序、覆盖半径和候选关联边。
在完成 q-direction loss 验证后，再决定是否增强 prompt/image，以免同时改动导致归因不清。
```
