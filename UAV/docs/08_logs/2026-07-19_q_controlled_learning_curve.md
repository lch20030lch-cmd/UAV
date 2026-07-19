---
type: log
status: q_closed_selected_checkpoint_verified
stage: q_only_controlled_learning_curve
last_updated: 2026-07-19
---

# 2026-07-19 Q-only 受控学习曲线

## 主线与边界

当前立即处理 Q，不进入 A、P 或联合训练。Q3 的 50-step 结果只证明 residual adapter
能够获得梯度并改善三维方向，尚未完成效果验收：

```text
Q3 val100 3D cosine:       0.615960
fixed geometry 3D cosine:  0.582721
Q3 val100 XY cosine:       0.683122
fixed geometry XY cosine:  0.693712
```

因此下一步从同一个 Q2 基线重新启动一条 200-step Q-only 轨迹，在 50/100/150/200
step 保存并验证。这样不会把多次重启、不同 optimizer 状态或 A/P/LoRA 更新混入比较。

## 本次最小代码调整

`train_sft_mm.py` 新增两个不改变默认行为的参数：

```text
--checkpoint_dir PATH
--save_steps N
```

用途：

1. 中间 checkpoint 写入本实验独立目录，避免不同 smoke 的同名 step checkpoint 相互覆盖；
2. 本轮每 50 步保存一次，直接得到同一轨迹的 50/100/150/200-step 模型；
3. `--load_lora` 且 LoRA 冻结时，中间 checkpoint 不再重复复制 LoRA；分析时显式加载
   Q2 基线的同一份 LoRA。最终 checkpoint 仍保持自包含。

未修改 Q 网络、损失或投影逻辑。

## 服务器训练命令

```bash
cd ~/Projects/UAV/UAV
git pull --ff-only origin main

Q_BASE=/root/autodl-tmp/outputs/mm_geom_v3_stage_q2_fixed_residual_lora_preflight200/mm_sft_lora_smoke_final
Q4_OUT=/root/autodl-tmp/outputs/mm_geom_v3_stage_q4_residual_curve200
Q4_CKPT="$Q4_OUT/checkpoints"

mkdir -p "$Q4_OUT" "$Q4_CKPT"
set -o pipefail

python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_geom_v3_train500_seed42 \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --init_checkpoint "$Q_BASE" \
  --load_lora \
  --max_steps 200 \
  --max_length 3072 \
  --output_dir "$Q4_OUT" \
  --checkpoint_dir "$Q4_CKPT" \
  --save_steps 50 \
  --projection_head_type split \
  --q_projection_mode direction \
  --q_geometry_mode fixed_residual_xy \
  --freeze_all_except_q \
  --projection_lr 0.001 \
  --lambda_q 0 \
  --lambda_a 0 \
  --lambda_p 0 \
  --lambda_q_dir 0 \
  --lambda_q_projected_dir 1.0 \
  --lambda_q_cue_ce 0 \
  --lambda_assoc_ce 0 \
  --lambda_assoc_raw_ce 0 \
  --lambda_p_raw_kl 0 \
  2>&1 | tee "$Q4_OUT/train.log"
```

本轮从 Q2 而不是 Q3 开始。当前实现加载 Q2 时会忽略已删除的旧 global gate，并以
zero-init residual adapter 开始，因此 step 50 应当能够复现 Q3 的学习阶段，同时
100/150/200 属于同一条连续轨迹。

## 训练日志先验检查

训练完成后先确认隔离边界与数值稳定性：

```bash
tr '\r' '\n' < "$Q4_OUT/train.log" \
  | grep -E 'step=(1|50|100|150|200) '

grep -Ei 'nan|inf|error|out of memory' "$Q4_OUT/train.log"
```

必须满足：

```text
grad_norm_q_residual > 0
q_residual_adapter_norm 持续离开 0
grad_norm_lora = 0
loss_q_projected_dir 有限
```

## 独立 val100 验证

中间 checkpoint 没有重复保存冻结 LoRA，所以显式使用 `$Q_BASE/lora`：

```bash
for STEP in 50 100 150 200; do
  CKPT="$Q4_CKPT/mm_sft_lora_smoke_step_${STEP}"
  python scripts/analyze_mm_delta_outputs.py \
    --config configs/rtx5090_multimodal_smoke.yaml \
    --data_dir /root/autodl-tmp/data/mm_geom_v3_val100_seed2026 \
    --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
    --checkpoint "$CKPT" \
    --lora_checkpoint "$Q_BASE/lora" \
    --projection_head_type split \
    --q_projection_mode direction \
    --q_geometry_mode fixed_residual_xy \
    --num_samples 100 \
    --name "q4_val100_step${STEP}" \
    --output "$Q4_OUT/delta_diag_val100_step${STEP}.json" \
    2>&1 | tee "$Q4_OUT/delta_diag_val100_step${STEP}.log"
done
```

## 决策规则

每个 step 同时检查：

```text
delta_q_vs_target_3d_cosine_mean
delta_q_vs_target_xy_cosine_mean
q_fixed_geometry_vs_target_3d_cosine_mean
q_fixed_geometry_vs_target_xy_cosine_mean
delta_q_direction_per_dim_std_mean
delta_q_target_direction_per_dim_std_mean
delta_q_mobility_violation_ratio
warnings
```

保留 residual 的必要条件：

1. 独立 val100 的 3D cosine 稳定高于 fixed 3D；
2. XY cosine 不低于 fixed XY 超过现有 0.01 容差；
3. 方向方差不随训练继续坍缩；
4. mobility violation 保持 0；
5. 选择实际验证最优 checkpoint，不默认使用 step 200。

如果 50/100/150/200 均不能同时满足 3D 增益与 XY 守门条件，则停止继续堆步数，Q 主线
回退到 fixed geometry；residual 只保留为消融。只有完成这一二选一结论后才进入 A。

## 本地验证状态

```text
python -m py_compile src/training/train_sft_mm.py: PASS
```

本机 Python 环境没有安装 torch/numpy，相关 17 个单元测试无法在本机导入；服务器拉取后
仍需在 `uavmllm` 环境执行现有测试集，本次没有通过降低或删除测试绕过该限制。

## Q4 200-step 服务器训练结果

训练已完成，最终 checkpoint：

```text
/root/autodl-tmp/outputs/mm_geom_v3_stage_q4_residual_curve200/mm_sft_lora_smoke_final
```

关键轨迹：

```text
step   projected loss   residual grad norm   adapter norm   LoRA grad norm
1      0.274353         4.335295             0.003464       0.0
50     0.133906         3.758512             0.032536       0.0
100    0.072724         5.474355             0.042406       0.0
150    0.467487         2.670131             0.045474       0.0
200    0.070081         0.543273             0.052957       0.0
```

阶段判定：

1. step 1/50 与 Q3 轨迹一致，zero-init 重跑具备可复现性；
2. residual adapter 梯度始终非零，参数范数持续增长，梯度链路没有再次堵塞；
3. LoRA 梯度始终为 0，A/P/LoRA 隔离边界保持正确；
4. 没有 NaN/Inf/OOM；
5. step 150 是单个 batch 的损失尖峰，不能由此判断 checkpoint 退化；是否过拟合或发生
   3D/XY 权衡必须由四个独立 val100 结果决定。

当前不选择 step 200，也不转入 A；下一步严格执行 50/100/150/200 的独立验证比较。

## 四节点独立 val100 结果

```text
step   3D cosine   XY cosine   direction std   Q warning   violation
50     0.615960    0.683122    0.442655        yes         0.0
100    0.623934    0.692386    0.432132        no          0.0
150    0.624318    0.693525    0.434704        no          0.0
200    0.624162    0.692636    0.431530        no          0.0

fixed  0.582721    0.693712
target direction std: 0.556831
```

step 150 相对 fixed geometry：

```text
3D cosine gain:  +0.041597
XY cosine change: -0.000187
mobility violation ratio: 0.0
```

最终判定：

1. 选择 step 150，不默认选择最后的 step 200；
2. step 150 在独立验证集上获得明确 3D 增益，同时基本保持 fixed XY；
3. step 100–200 的 direction std 在 0.4315–0.4347 范围内进入平台，没有继续训练导致的
   渐进坍缩，但仍低于 target 0.5568，作为后续联合评估的监控项保留；
4. step 100 起 Q warning 消失，所有 checkpoint 的物理违规率均为 0；
5. Q residual 通过本阶段单分支效果门槛，Q 主线闭环，不再增加步数或新增 Q 模式；
6. 后续使用 step 150 的 projection/control states，并配套 Q2 中已冻结的同一份 LoRA；
7. 只有把 step 150 提升成自包含 checkpoint 后，主线才转入 A。

用于后续阶段的自包含目录命名为：

```text
/root/autodl-tmp/outputs/mm_geom_v3_stage_q4_residual_curve200/mm_sft_lora_selected_step150
```

## Selected checkpoint 提升与核验

服务器已将 step 150 的 projection/control/processor 文件复制到 selected 目录，并只补充
一份 Q2 中保持冻结的 LoRA：

```text
selected checkpoint:
/root/autodl-tmp/outputs/mm_geom_v3_stage_q4_residual_curve200/mm_sft_lora_selected_step150

projection byte comparison: PASS
lora/adapter_config.json:    present
selected directory size:    129M
```

`cmp` 已确认 selected 的 `projection_head.pt` 与 step 150 完全一致。该目录现在可以作为
后续 A 阶段的自包含 `--init_checkpoint`。Q 本阶段至此闭环，后续不再继续 Q-only 训练。
