# q-direction 训练与几何增强记录

日期：2026-07-16

## 今日目标

延续 2026-07-15 的结论：

```text
split projection head 已经明显改善 association；
但 delta_q target 100% 贴近 15m 移动边界，直接 q MSE 不适合作为唯一监督。
```

今日主要验证两件事：

1. q-direction loss 是否能让 q 分支学到边界方向信息。
2. 若方向仍弱，是否需要按老师建议增强 prompt 与 BEV 图片的信息量，并让二者相辅相成。

## 已完成代码基础

此前已新增：

```bash
--lambda_q_dir
```

对应逻辑：

```text
q_dir_target = delta_q_target / ||delta_q_target||
q_dir_pred   = delta_q_raw / ||delta_q_raw||
loss_q_dir   = MSE(q_dir_pred, q_dir_target)
```

诊断脚本新增指标：

```text
delta_q_raw_per_dim_std_mean
delta_q_raw_dir_cosine_mean
delta_q_raw_dir_mse_mean
```

## Stage B3 q-direction：500 step

训练设置：

```text
init_checkpoint = /root/autodl-tmp/outputs/mm_smoke_100_stage_a2_split_assoc/mm_sft_smoke_final
projection_head_type = split
freeze_assoc_branch = true
lambda_q = 0
lambda_q_dir = 1.0
lambda_a = 0
lambda_p = 0
projection_lr = 0.005
max_steps = 500
```

诊断结果：

```text
delta_q_per_dim_std_mean: 0.39804720878601074
delta_q_raw_per_dim_std_mean: 4.676386833190918
delta_q_raw_dir_cosine_mean: 0.24615392088890076
delta_q_raw_dir_mse_mean: 0.5025640726089478

delta_a_argmax_unique_per_user_mean: 2.95
delta_a_argmax_fixed_user_count: 1
argmax_match_rate_mean: 0.4315
warnings: []
```

结论：

```text
q-direction loss 能让 q 分支开始动起来；
association 仍然完全保住；
但 q 方向对齐只有弱正相关，cosine 仍偏低。
```

## Stage B3 q-direction：1000 step

训练设置：

```text
同 500 step，但 max_steps = 1000
output_dir = /root/autodl-tmp/outputs/mm_smoke_100_stage_b3_split_q_dir_1000step
```

诊断结果：

```text
delta_q_per_dim_std_mean: 2.150986433029175
delta_q_raw_per_dim_std_mean: 24.268796920776367
delta_q_raw_dir_cosine_mean: 0.2742322087287903
delta_q_raw_dir_mse_mean: 0.48384520411491394

delta_a_argmax_unique_per_user_mean: 2.95
delta_a_argmax_fixed_user_count: 1
argmax_match_rate_mean: 0.4315
warnings: []
```

对比 500 step：

| 指标 | 500 step | 1000 step |
| --- | ---: | ---: |
| delta_q std | 0.3980 | 2.1510 |
| delta_q_raw std | 4.6764 | 24.2688 |
| q direction cosine | 0.2462 | 0.2742 |
| q direction MSE | 0.5026 | 0.4838 |
| association match | 0.4315 | 0.4315 |

结论：

```text
增加训练步数能显著拉大 q 输出方差；
但方向 cosine 只从 0.246 提升到 0.274，提升有限。
这说明 q 分支已经会响应样本差异，但当前输入信息或监督形式仍不足以稳定对齐 oracle q 方向。
```

## 当前判断

问题已经进一步拆清楚：

1. **association 问题**
   - split projection head 后明显改善。
   - association 在 Stage B3 中持续保持：

```text
unique = 2.95
fixed = 1
match = 0.4315
```

2. **q 分支问题**
   - q-direction loss 能让 q raw 动起来。
   - 但方向对齐弱，继续单纯加 step 的收益有限。

3. **输入信息问题**
   - 当前 prompt 只有较粗的通信摘要、感知摘要和 BEV 图描述。
   - 对于 15m 边界上的方向选择，模型可能缺少显式几何提示，例如最近用户方向、目标方向、用户热点、负载和候选移动方向。

因此，老师提出的“文字 prompt 和图片信息量增加，并且二者相辅相成”是合理下一步。

## 已完成几何增强代码

今日已修改：

```text
src/data/prompt_builder.py
src/env/bev_renderer.py
scripts/generate_mm_smoke.py
```

### 文字 prompt 增强

新增 `[Geometry Guidance g(t)]` 段落，包含：

```text
- 15m mobility radius 的方向学习提示
- 用户几何质心
- 用户需求加权质心
- 每架 UAV 的当前位置、高度、当前负载
- 每架 UAV 的候选用户：距离、方向、用户权重
- 每架 UAV 的候选感知目标：距离、方向、感知 SINR
```

设计目的：

```text
把 delta_q 从“黑箱坐标回归”改成更明确的方向决策问题。
```

### BEV 图片增强

BEV renderer 新增：

```text
- 15m 单槽移动圈
- UAV 到候选用户的绿色引导线
- UAV 到候选感知目标的橙色虚线
- 用户点大小按 user weight 调整
- 保留 UAV、用户、目标、覆盖圆、当前关联线
```

设计目的：

```text
让图像和文字表达同一组几何候选信息：
prompt 说“UAV m 到 u/t 的方向和距离”，图像中画出对应候选线。
```

### 数据标记

新生成的多模态数据会写入：

```text
prompt_type = multimodal_bev_image_v2_geometry_guided
```

用于和旧版数据区分。

## 下一步计划

### 2026-07-16 追加：compact geom v3 已落地

geom v2 结果显示：

```text
q direction cosine 未提升；
association 从 0.4315 回退到 0.357；
control_states 方差变大但 projection head 读出更弱。
```

因此已将几何增强从 v2 降噪为 compact v3：

```text
prompt_type = multimodal_bev_image_v3_compact_geometry
```

v3 prompt 仅保留：

```text
- 15m movement radius
- 每架 UAV 当前 load
- 每架 UAV 指向 weighted user center 的方向/距离
- 每架 UAV 最近 1 个用户的方向/距离/权重
- 每架 UAV 最近 1 个 sensing target 的方向/距离
```

v3 BEV 仅保留：

```text
- 15m movement circle
- 用户权重大小
- weighted user center 星标
- UAV -> weighted user center 紫色虚线
- UAV -> nearest user 绿色线
- UAV -> nearest target 橙色虚线
```

并关闭当前 association 灰线，避免线条过密干扰视觉特征。

服务器生成增强版 100 条数据，不覆盖旧数据：

```bash
python scripts/generate_mm_smoke.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --output_dir /root/autodl-tmp/data/mm_smoke_100_geom_v3 \
  --num_samples 100 \
  --num_restarts 3 \
  --overwrite
```

检查：

```bash
wc -l /root/autodl-tmp/data/mm_smoke_100_geom_v3/sft_dataset.jsonl
python - <<'PY'
import json
p="/root/autodl-tmp/data/mm_smoke_100_geom_v3/sft_dataset.jsonl"
x=json.loads(open(p).readline())
print(x["prompt_type"])
print("[Geometry Guidance g(t)]" in x["prompt"])
print(x["bev_image_path"])
PY
```

之后复现同一条实验路线：

```text
1. Stage A2 split association warmup
2. Stage B3 q-direction
3. 对比旧数据的 q direction cosine = 0.2742
```

验收目标：

```text
association 保持不退；
delta_q_raw_dir_cosine_mean 明显高于 0.2742；
若能接近或超过 0.4，说明增强 prompt+BEV 对 q 方向学习有实质帮助。
```

## Git 状态备注

截至记录时，本地已有几何增强提交：

```text
91c931d Add geometry-guided multimodal prompt and BEV rendering
```

但由于 GitHub 网络连接中断，本地仍显示：

```text
main...origin/main [ahead 1]
```

下一次继续工作时，优先执行：

```powershell
cd C:\Users\Shardeom-PC\Desktop\Projects\UAV
git status -sb
git push origin main
```

确认该提交推送成功后，服务器再 `git pull --ff-only origin main`。
