# projection head修改

日期：2026-07-14

## 背景

100条多模态 smoke 数据已经生成并完成 shared projection head 的 Stage A / Stage B 对照。

关键现象：

- Stage A 只训练 association raw CE 时，association 固定问题明显缓解。
- Stage B 加入 `q/a/p` 混合控制损失后，association 指标没有继续提升，反而出现轻微回退。
- 加强 `lambda_assoc_raw_ce` 后，`argmax_match_rate_mean` 略有恢复，但 `delta_a_argmax_unique_per_user_mean` 与 `fixed_user_count` 更差，说明模型更倾向保守模板，而不是真正学到更丰富的关联选择。

当前 100条结果摘要：

| 实验 | unique mean | fixed user count | match rate |
| --- | ---: | ---: | ---: |
| Stage A association raw CE | 2.35 | 3 | 0.373 |
| Stage B q/a/p mixed CTL | 2.25 | 4 | 0.344 |
| Stage B strong assoc | 2.10 | 5 | 0.3615 |

诊断判断：

- target association 本身不固定，`target_delta_a_argmax_unique_per_user_mean=4.0`，`target_delta_a_argmax_fixed_user_count=0`。
- control states 有跨样本差异，`control_states_per_dim_std_mean≈0.47`。
- association 在 Stage A 能学起来一部分，说明不是数据完全无信号，也不是模型完全读不出来。
- Stage B 回退更像 projection head 内部多任务梯度冲突：离散 association 与连续 q/p 共用同一组 readout / residual MLP，q/p 回归更新会冲淡 association 读出能力。

## 本次代码修改

### 1. projection head 增加 split 模式

文件：`src/model/projection_head.py`

保留旧的 shared 模式作为默认结构：

```text
control_states
  -> shared ControlReadout
  -> shared ResidualMLP
  -> split q/a/p
  -> structured projection
```

新增 split 模式：

```text
control_states
  -> q_readout -> q_mlp -> Proj_Q
  -> a_readout -> a_mlp -> Proj_A
  -> p_readout -> p_mlp -> Proj_P
```

目的：

- 让 association 分支和 q/p 分支在参数上解耦。
- 支持 Stage A2 只训练 association 分支。
- 支持 Stage B2 冻结 association 分支，只训练 q/p 分支，避免 q/p 把 association 重新带偏。

### 2. 配置构造支持 head_type

文件：`src/model/__init__.py`

`build_proj_head_config` 增加 `head_type` 字段，默认仍为 `shared`。

### 3. 多模态 SFT smoke 训练脚本增加分支控制

文件：`src/training/train_sft_mm.py`

新增参数：

```bash
--projection_head_type {shared,split}
--freeze_assoc_branch
--freeze_qp_branch
```

规则：

- 分支冻结参数只允许在 `--projection_head_type split` 下使用。
- `--freeze_assoc_branch` 与 `--freeze_qp_branch` 不能同时使用。
- checkpoint metadata 会记录 projection head 类型和冻结状态。

### 4. 诊断与前向 smoke 支持 split checkpoint

文件：

- `scripts/analyze_mm_delta_outputs.py`
- `scripts/smoke_mm_forward.py`

新增参数：

```bash
--projection_head_type split
```

加载 split checkpoint 做前向或诊断时必须带该参数，否则默认 shared 结构会与 split checkpoint 的 `state_dict` 不匹配。

## 验证

本地已完成 Python 语法检查：

```bash
python -m py_compile \
  UAV/src/model/projection_head.py \
  UAV/src/model/__init__.py \
  UAV/src/training/train_sft_mm.py \
  UAV/scripts/analyze_mm_delta_outputs.py \
  UAV/scripts/smoke_mm_forward.py
```

本机环境没有安装 `torch`，因此未在本地执行张量级 forward smoke。需要在 AutoDL 的 `uavmllm` 环境中继续验证。

## 下一步实验计划

使用同一份 100条数据：

```text
/root/autodl-tmp/data/mm_smoke_100
```

不要重新生成数据，保证与 shared head 的 Stage A/B 结果公平对比。

### Stage A2：split head 只训练 association 分支

目标：

- 验证 split head 是否能复现或超过 shared Stage A。
- 重点观察 `fixed_user_count` 是否小于等于 3，`match_rate` 是否不低于 0.373。

建议命令：

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke_100 \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_steps 300 \
  --max_length 3072 \
  --output_dir /root/autodl-tmp/outputs/mm_smoke_100_stage_a2_split_assoc \
  --projection_head_type split \
  --freeze_qp_branch \
  --projection_lr 0.01 \
  --lambda_q 0 \
  --lambda_a 0 \
  --lambda_p 0 \
  --lambda_assoc_ce 0 \
  --lambda_assoc_raw_ce 1.0
```

### Stage A2 诊断

```bash
python scripts/analyze_mm_delta_outputs.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke_100 \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --checkpoint /root/autodl-tmp/outputs/mm_smoke_100_stage_a2_split_assoc/mm_sft_smoke_final \
  --projection_head_type split \
  --name mm100_stage_a2_split_assoc \
  --num_samples 100 \
  --max_length 3072 \
  --output /root/autodl-tmp/outputs/mm_smoke_100_stage_a2_split_assoc/delta_diag_mm100_stage_a2_split_assoc.json \
  --save_raw
```

```bash
python scripts/analyze_mm_target_distribution.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke_100 \
  --prediction_npz /root/autodl-tmp/outputs/mm_smoke_100_stage_a2_split_assoc/delta_diag_mm100_stage_a2_split_assoc.npz \
  --output /root/autodl-tmp/outputs/mm_smoke_100_stage_a2_split_assoc/target_vs_pred_mm100_stage_a2_split_assoc.json
```

### Stage B2：冻结 association 分支，只训练 q/p 分支

只有 Stage A2 通过后再执行：

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke_100 \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --init_checkpoint /root/autodl-tmp/outputs/mm_smoke_100_stage_a2_split_assoc/mm_sft_smoke_final \
  --max_steps 300 \
  --max_length 3072 \
  --output_dir /root/autodl-tmp/outputs/mm_smoke_100_stage_b2_split_qp \
  --projection_head_type split \
  --freeze_assoc_branch \
  --projection_lr 0.002 \
  --lambda_q 1.0 \
  --lambda_a 0 \
  --lambda_p 0.3 \
  --lambda_assoc_ce 0 \
  --lambda_assoc_raw_ce 0
```

Stage B2 的目标不是提高 association，而是在 association 不回退的前提下拉起 `delta_q` 与 `delta_p` 的跨样本方差。

## 2026-07-17 追加：q direction projection 模式

### 新诊断结论

后续 100 条 smoke 实验进一步发现：

```text
delta_q target 的 norm mean ≈ 14.995m
near 15m ratio = 1.0
```

这说明当前数据中的 `delta_q` 不是普通自由连续位移回归，而更接近：

```text
方向向量 + 固定 15m 移动半径
```

在旧 `Proj_Q` 路径中：

```text
delta_q_raw -> Proj_Q clip(||delta_q|| <= 15m)
```

模型需要同时学习方向和幅度，而幅度又会被移动性约束裁剪。q-direction loss 能让 raw q 分支动起来，但 1000 step 后方向 cosine 仍只在 0.26-0.27 左右，说明只加方向损失还不够。

### 本次新增代码

文件：

```text
src/model/projection_head.py
src/model/__init__.py
src/training/train_sft_mm.py
scripts/analyze_mm_delta_outputs.py
scripts/smoke_mm_forward.py
```

新增参数：

```bash
--q_projection_mode {clip,direction}
```

默认值仍为：

```text
clip
```

保持旧 checkpoint 和旧实验兼容。

新 `direction` 模式：

```text
delta_q_raw -> normalize(delta_q_raw) -> 15m * direction -> Proj_Q
```

其中 `Proj_Q` 仍保留区域、高度和物理边界裁剪。区别是 raw q 只负责表达方向，半径由 `v_max_dt` 显式给定。

### 设计目的

```text
将 q 任务从“任意三维位移回归”改成“15m 边界上的方向学习”，
更符合当前 oracle delta_q 全部贴移动半径上限的数据事实。
```

### 下一步建议实验

沿用 compact geom v3 数据：

```text
/root/autodl-tmp/data/mm_smoke_100_geom_v3
```

从 v3 Stage A2 association checkpoint 继续：

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke_100_geom_v3 \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --init_checkpoint /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_a2_split_assoc/mm_sft_smoke_final \
  --max_steps 1000 \
  --max_length 3072 \
  --output_dir /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b4_split_q_direction_proj_1000step \
  --projection_head_type split \
  --q_projection_mode direction \
  --freeze_assoc_branch \
  --projection_lr 0.005 \
  --lambda_q 0 \
  --lambda_q_dir 1.0 \
  --lambda_a 0 \
  --lambda_p 0 \
  --lambda_assoc_ce 0 \
  --lambda_assoc_raw_ce 0
```

诊断时也必须带同样的 q 投影模式：

```bash
python scripts/analyze_mm_delta_outputs.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke_100_geom_v3 \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --checkpoint /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b4_split_q_direction_proj_1000step/mm_sft_smoke_final \
  --projection_head_type split \
  --q_projection_mode direction \
  --name mm100_geom_v3_stage_b4_split_q_direction_proj_1000step \
  --num_samples 100 \
  --max_length 3072 \
  --output /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b4_split_q_direction_proj_1000step/delta_diag_mm100_geom_v3_stage_b4_split_q_direction_proj_1000step.json \
  --save_raw
```

验收目标：

```text
1. association 保持 geom v3 Stage A2 水平附近：
   unique≈3.9, fixed=0, match≈0.445

2. q direction cosine 明显高于 0.27。

3. delta_q_per_dim_std_mean 继续向 target 8.37 靠近，而不是只扩大 raw q 幅度。
```
