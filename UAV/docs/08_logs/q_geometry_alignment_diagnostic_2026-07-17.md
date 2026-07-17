---
type: log
status: current
stage: multimodal_smoke_diagnostic
last_updated: 2026-07-17
---

# q 几何可学习性诊断

## 背景

geom v3 + split projection head 后，association 已经明显恢复：

```text
delta_a_argmax_unique_per_user_mean: 3.9
delta_a_argmax_fixed_user_count: 0
argmax_match_rate_mean: 0.445
```

但 Stage B3 / B4 的 q 方向仍然没有明显提升：

```text
delta_q_raw_dir_cosine_mean: 0.2600 左右
```

B4 的 `q_projection_mode=direction` 与旧 `clip` 结果几乎一致，说明瓶颈不在最后的 15m 投影裁剪，而在 raw q 方向本身没有学准。

## 本次修改

新增无卡诊断脚本：

```text
scripts/analyze_q_geometry_alignment.py
```

脚本直接读取 `sft_dataset.jsonl`，从 v3 prompt 中解析每架 UAV 的三类几何提示：

```text
- weighted_center direction
- nearest_user direction
- nearest_target direction
```

然后计算 oracle `delta_q` 的水平移动方向与这些几何提示的 cosine 对齐度。

如果传入 `analyze_mm_delta_outputs.py --save_raw` 生成的 npz，还会额外比较模型预测 q / raw q 与 oracle q 的方向对齐度。

## 服务器运行命令

只分析 oracle q 与 prompt 几何提示：

```bash
python scripts/analyze_q_geometry_alignment.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke_100_geom_v3 \
  --output /root/autodl-tmp/outputs/mm_smoke_100_geom_v3/q_geometry_alignment_geom_v3.json
```

结合 B3 预测结果一起分析：

```bash
python scripts/analyze_q_geometry_alignment.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke_100_geom_v3 \
  --prediction_npz /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b3_split_q_dir_1000step/delta_diag_mm100_geom_v3_stage_b3_split_q_dir_1000step.npz \
  --output /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b3_split_q_dir_1000step/q_geometry_alignment_mm100_geom_v3_stage_b3.json
```

结合 B4 预测结果一起分析：

```bash
python scripts/analyze_q_geometry_alignment.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke_100_geom_v3 \
  --prediction_npz /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b4_split_q_direction_proj_1000step/delta_diag_mm100_geom_v3_stage_b4_split_q_direction_proj_1000step.npz \
  --output /root/autodl-tmp/outputs/mm_smoke_100_geom_v3_stage_b4_split_q_direction_proj_1000step/q_geometry_alignment_mm100_geom_v3_stage_b4.json
```

## 判读标准

重点看：

```text
target_q_vs_weighted_center_xy_cosine_mean
target_q_vs_nearest_user_xy_cosine_mean
target_q_vs_nearest_target_xy_cosine_mean
target_q_vs_best_geometry_xy_cosine_mean
pred_delta_q_raw_vs_target_q_xy_cosine_mean
```

如果 `target_q_vs_best_geometry_xy_cosine_mean` 本身就低，说明当前 prompt/image 几何提示不足以解释 oracle q，下一步应增强数据中的文本和图像信息。

如果 oracle q 与几何提示对齐度高，但模型 q 仍然低，说明问题更偏向 q branch / control token 读出 / q loss，需要继续改训练结构。
