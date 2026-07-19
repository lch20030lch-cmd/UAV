---
type: log
status: fixed_residual_preflight_complete_joint_warmstart_selected
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

## Q1 训练趋势与原有诊断结果

前 20 / 后 20 step 平均：

```text
loss_q_dir:                 0.497985 -> 0.514331
loss_a_ce:                  1.179702 -> 1.037897
loss_a_raw_ce:              1.170218 -> 1.054717
loss_p_raw_kl:              0.989697 -> 0.933993
delta_p_inactive_leakage:   0.027847 -> 0.026233
grad_norm_proj:             1.465515 -> 0.238640
grad_norm_lora:             1.085328 -> 0.822210
```

单步 batch size 为 1，且 200 step 尚未遍历完整 train500，因此 `loss_q_dir` 的两段
均值不能单独证明收敛；但梯度持续非零，A/P retention loss 与 leakage 没有恶化。

Train500 / 独立 val100：

```text
Q raw dir cosine:       0.2412 / 0.2140
Q raw dir MSE:          0.5059 / 0.5240
A accuracy:             0.4695 / 0.4400
P overall MSE:          0.007868 / 0.007590
P leakage:              0.025185 / 0.025394
```

相对 A4 独立 val100：

```text
Q raw cosine:  0.0871 -> 0.2140
A accuracy:    0.4300 -> 0.4400
P leakage:     0.02688 -> 0.02539
```

判定：direct-Q 信号有实质提升且能泛化，A/P retention 有效；但 Q cosine 仍低于历史
最好约 0.274，更未达到阶段目标 0.4，暂不进入长训。

原诊断只打印 `delta_q_per_dim_std_mean`，不足以区分“位移长度不足”和“15m 固定方向
模板”。因此新增投影后 Q 诊断：

```text
delta_q_norm_mean / target_norm_mean / norm_mae
delta_q_near_max_radius_ratio
delta_q_mobility_violation_ratio
delta_q_vs_target_3d_cosine_mean
delta_q_vs_target_xy_cosine_mean
delta_q_direction_per_dim_std_mean
delta_q_target_direction_per_dim_std_mean
```

使用现有 Q1 checkpoint 重跑诊断即可确认，不需要重新训练。

## 投影后诊断结论：Q1 退化为固定方向模板

独立 val100 的完整投影后结果：

```text
delta_q_norm_mean:                         15.000000
delta_q_target_norm_mean:                  14.995847
delta_q_norm_mae:                          0.004158
delta_q_near_max_radius_ratio:             1.000000
delta_q_mobility_violation_ratio:          0.000000
delta_q_vs_target_3d_cosine_mean:           0.214046
delta_q_vs_target_xy_cosine_mean:          -0.088626
delta_q_direction_per_dim_std_mean:         0.008251
delta_q_target_direction_per_dim_std_mean:  0.556831
```

判定分为两部分：

1. `Proj_Q` 正常：位移范数贴近 15m，且没有移动约束违规；
2. direct-Q 方向学习失败：预测方向方差只有目标的约 1.48%，XY cosine 为负，
   3D cosine 的小幅正值不能证明学到了有用的场景水平方向。

因此 Q1 checkpoint 只保留为失败诊断/消融，不允许直接继续 500/1000 step 长训。
A/P retention 是有效的：独立 val100 的 A accuracy 为 0.4400，P MSE 为 0.007590，
P leakage 为 0.025394，均未因 Q1 回退。

## Q2 修复：固定几何先验 + 受限 MLLM 残差

独立 train500/val100 探针中，动态 selector 在验证集的 XY cosine 为 0.6440，
而 train500-only 固定混合为 0.6937。因此不再让模型自由选择 cue，也不再从零自由预测
整个方向。新增：

```text
q_geometry_mode = fixed_residual_xy
fixed weights = [0.31186843, 0.09240539, 0.59572625]
cue order = [weighted_center, nearest_user, nearest_target]
```

权重只由 train500 探针的平均输出得到，未使用 val100。前向路径为：

```text
fixed_xy = normalize(sum(fixed_weight_i * cue_i))
residual = normalize(q_raw_from_MLLM)
gate = sigmoid(trainable_logit), initial gate = 0.05, maximum = 1.0
q_direction = normalize([fixed_xy, 0] + gate * residual)
delta_q = 15m * q_direction -> Proj_Q
```

这样 MLLM 仍学习场景相关三维残差，但训练初始状态由独立验证更稳健的固定几何方向
托底。门控限制残差，防止模型在训练早期再次破坏水平方向。

同时新增：

```text
--lambda_q_projected_dir
loss_q_projected_dir
q_residual_gate
q_fixed_geometry_vs_target_xy_cosine_mean
q_fixed_geometry_vs_target_3d_cosine_mean
delta_q_below_fixed_geometry_baseline warning
```

`--freeze_all_except_q` 也会训练新增的 `q_residual_gate_logit`；旧的 `none/cue_xy`
checkpoint state dict 不新增无关 key，保持兼容。

## Q2 验收顺序

先用 A4/Q1 checkpoint 做不训练的 val100 基线前向，确认代码输出不低于 fixed geometry；
再做 200-step 预检，禁止直接长训。预检验收以独立 val100 为准：

```text
projected XY cosine >= fixed geometry XY cosine - 0.01   # 不破坏水平先验
projected 3D cosine >= fixed geometry 3D cosine + 0.02   # 残差主要补充三维信息
direction std 明显高于 Q1 的 0.00825
A accuracy >= 0.42
P MSE <= 0.009
P leakage <= 0.03
mobility violation ratio = 0
```

如果残差不能在保持 XY 的前提下超过 3D 固定基线，则将 gate 固定为 0，把 fixed mixture 作为最终 Q 几何分支；
这不否定整篇方法，A/P 仍由 MLLM 学习，Q 的动态残差则作为失败消融如实报告。

## Q2 无训练 val100 基线通过

使用 Q1 checkpoint、`fixed_residual_xy` 和默认 gate=0.05，在独立 val100 上完成
无训练前向：

```text
projected Q XY cosine:          0.693665
fixed geometry XY cosine:       0.693712
projected Q 3D cosine:          0.593184
fixed geometry 3D cosine:       0.582721
predicted direction std:        0.467004
target direction std:           0.556831
Q norm mean / violation ratio:  14.999999 / 0.0
q_residual_gate:                0.050000
A val accuracy:                 0.4400
P val MSE / leakage:            0.007590 / 0.025394
```

相对失败的 direct-Q 路径：

```text
XY cosine:       -0.088626 -> 0.693665
direction std:    0.008251 -> 0.467004
```

判定：固定几何前向实现正确，场景方向多样性恢复，物理投影正常，A/P 完全保持；
初始小残差没有破坏 XY，并已给 3D cosine 带来约 0.0105 增益。允许进入 200-step
小学习率预检，仍不允许直接长训。

## Q2 200-step train500/val100 最终判定

```text
                                    train500    val100
projected 3D cosine:                0.603803    0.593601
fixed geometry 3D cosine:           0.592064    0.582721
residual gain over fixed:          +0.011739   +0.010880
projected XY cosine:                0.719447    0.693635
fixed geometry XY cosine:           0.719435    0.693712
direction std:                      0.469801    0.466859
q residual gate:                    0.051953    0.051953
A accuracy:                         0.4728      0.4360
P MSE:                              0.007857    0.007581
P inactive leakage:                 0.025188    0.025418
mobility violation ratio:           0.0         0.0
```

残差相对固定基线的 train/val 增益差仅 0.00086，不是训练集记忆；但相对无训练
val100 的 3D cosine 只从 0.593184 增至 0.593601，200-step 新增收益仅约 0.00042。

最终判定：

1. Q2 作为联合训练 warm-start 通过：固定几何提供稳定水平移动，保守 MLLM 残差在
   独立环境带来约 0.0109 的三维增益；
2. Q-only 继续扩步不通过：当前损失/小 gate 下新增训练收益过低，不再单独跑
   500/1000 step；
3. canonical checkpoint 为
   `mm_geom_v3_stage_q2_fixed_residual_lora_preflight200/mm_sft_lora_smoke_final`；
4. 论文中必须把 fixed geometry 和 learned residual 分开消融，不把固定先验的收益
   全部表述为 MLLM 学习收益。
