---
type: log
status: current
stage: multimodal_smoke_q_optimization
last_updated: 2026-07-17
---

# q 几何候选方向头修改

## 背景

q 几何可学习性诊断显示，oracle `delta_q` 的水平移动方向与 prompt/BEV 中的候选几何方向高度对齐：

```text
target_q_vs_best_geometry_xy_cosine_mean: 0.8722
```

但模型预测 q 与 oracle q 的对齐很低：

```text
pred_delta_q_raw_vs_target_q_3d_cosine_mean: 0.2600
pred_delta_q_raw_vs_target_q_xy_cosine_mean: 0.1236
```

因此当前瓶颈不是 prompt/image 完全缺少信息，而是 projection head 的 q 分支没有把这些几何线索转化成 q 输出。

## 修改思路

新增一个可选的 q 几何候选方向选择机制：

```text
prompt/BEV 中三类候选方向:
1. weighted_center
2. nearest_user
3. nearest_target

projection head 预测每架 UAV 对三类 cue 的权重。
cue 权重加权组合出 q 的水平移动方向。
```

这样 q 分支不再完全自由回归 `dx, dy`，而是先学会“沿图文共同表达的哪条候选方向移动”。

默认行为不变。只有显式传入：

```text
--q_geometry_mode cue_xy
--lambda_q_cue_ce > 0
```

才启用新路径。

## 代码变更

新增：

```text
src/data/geometry_cues.py
```

用于从 v3 prompt 中解析：

```text
q_geometry_cues: (M, 3, 2)
q_geometry_mask: (M, 3)
```

修改：

```text
src/data/multimodal_dataset.py
src/model/projection_head.py
src/model/gemma_multimodal_isac.py
src/model/losses.py
src/model/__init__.py
src/training/train_sft_mm.py
scripts/analyze_mm_delta_outputs.py
scripts/smoke_mm_forward.py
```

关键新增参数：

```text
--q_geometry_mode {none,cue_xy}
--lambda_q_cue_ce FLOAT
```

新增诊断输出：

```text
q_cue_accuracy
q_cue_target_hist
q_cue_pred_hist
q_cue_chosen_geometry_cosine_mean
q_cue_best_geometry_cosine_mean
```

## 建议实验

基于 geom v3 Stage A2 association checkpoint，启动 Stage B5：

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke_100_geom_v3 \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --init_checkpoint /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_a2_split_assoc/mm_sft_smoke_final \
  --max_steps 1000 \
  --max_length 3072 \
  --output_dir /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b5_q_cue_xy_1000step \
  --projection_head_type split \
  --q_geometry_mode cue_xy \
  --freeze_assoc_branch \
  --projection_lr 0.005 \
  --lambda_q 0 \
  --lambda_q_dir 0 \
  --lambda_q_cue_ce 1.0 \
  --lambda_a 0 \
  --lambda_p 0 \
  --lambda_assoc_ce 0 \
  --lambda_assoc_raw_ce 0
```

诊断时需要传入相同的 `q_geometry_mode`：

```bash
python scripts/analyze_mm_delta_outputs.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke_100_geom_v3 \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --checkpoint /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b5_q_cue_xy_1000step/mm_sft_smoke_final \
  --projection_head_type split \
  --q_geometry_mode cue_xy \
  --name mm100_geom_v3_stage_b5_q_cue_xy_1000step \
  --num_samples 100 \
  --max_length 3072 \
  --output /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b5_q_cue_xy_1000step/delta_diag_mm100_geom_v3_stage_b5_q_cue_xy_1000step.json \
  --save_raw
```

## 判读标准

重点看：

```text
q_cue_accuracy
q_cue_chosen_geometry_cosine_mean
delta_q_raw_dir_cosine_mean
pred_delta_q_vs_target_q_xy_cosine_mean
argmax_match_rate_mean
```

如果 `q_cue_accuracy` 和 `q_cue_chosen_geometry_cosine_mean` 明显提升，但最终 `delta_q` 仍弱，说明 cue 选择学到了，但 q 输出组合/高度处理还要继续改。

如果 cue 本身也学不动，问题更偏向 control token readout 或 backbone 是否需要 LoRA 参与。

## 2026-07-17 追加：B6 只训练 q cue head

B5 结果说明 `cue_xy` 能把 `delta_q` 幅度拉起来：

```text
delta_q_per_dim_std_mean: 7.0771
```

但 cue 选择明显偏向 `weighted_center`：

```text
q_cue_accuracy: 0.31
q_cue_pred_hist: {'weighted_center': 360, 'nearest_user': 40, 'nearest_target': 0}
q_cue_target_hist: {'weighted_center': 123, 'nearest_user': 99, 'nearest_target': 178}
```

因此新增训练开关：

```text
--freeze_all_except_q_cue
```

作用：

```text
只训练 readout_q_cue
冻结 readout_q / q_mlp / readout_a / a_mlp / readout_p / p_mlp
冻结 base Gemma 和 control token 之外的其他 projection head 参数
```

B6 建议从 B5 checkpoint 初始化，只修 cue 选择器：

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke_100_geom_v3 \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --init_checkpoint /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b5_q_cue_xy_1000step/mm_sft_smoke_final \
  --max_steps 1000 \
  --max_length 3072 \
  --output_dir /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b6_q_cue_only_1000step \
  --projection_head_type split \
  --q_geometry_mode cue_xy \
  --freeze_all_except_q_cue \
  --projection_lr 0.003 \
  --lambda_q 0 \
  --lambda_q_dir 0 \
  --lambda_q_cue_ce 5.0 \
  --lambda_a 0 \
  --lambda_p 0 \
  --lambda_assoc_ce 0 \
  --lambda_assoc_raw_ce 0
```

训练启动时应看到：

```text
q-cue-only trainable tensors: > 0
```

诊断时继续传：

```text
--q_geometry_mode cue_xy
```

优先观察：

```text
q_cue_accuracy
q_cue_pred_hist
q_cue_chosen_geometry_cosine_mean
delta_q_per_dim_std_mean
delta_a_argmax_unique_per_user_mean
```

## 2026-07-18 修复：B6 loss_q_cue_ce 未进入训练

第一次 B6 跑完后日志显示：

```text
loss_ctl=0.000000
loss_total=0.000000
loss_q_cue_ce=0.000000
grad_norm_proj=0.000000
```

metadata 已确认：

```text
q_geometry_mode = cue_xy
freeze_all_except_q_cue = true
lambda_q_cue_ce = 5.0
q_cue_only_trainable_tensors > 0
```

因此不是命令参数未生效，而是训练脚本构造 `delta_hat` 时漏传了：

```text
outputs["q_cue_logits"]
```

导致 `UAVISACLosses.compute_control_loss()` 中的 q cue CE 分支没有触发，`loss_q_cue_ce` 被置为 0。

修复文件：

```text
src/training/train_sft_mm.py
```

修复内容：

```text
if "q_cue_logits" in outputs:
    delta_hat["q_cue_logits"] = outputs["q_cue_logits"]
```

第一次 B6 checkpoint 不应作为有效 q cue 训练结果使用。修复后建议重新跑一个新目录：

```text
/root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b6_q_cue_only_fix_1000step
```
