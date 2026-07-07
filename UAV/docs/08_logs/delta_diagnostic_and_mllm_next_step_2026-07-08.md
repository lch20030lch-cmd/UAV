# Delta 诊断结果与 MLLM 下一步决策记录

> 日期：2026-07-08  
> 背景：当前 text-grid baseline 已完成 SFT、DPO-2k、200 样本评估。今天补充了 delta 输出诊断，用于确认 SFT/DPO 输出是否存在严重常数坍塌，并判断是否可以进入真正 BEV-image MLLM 分支。

---

## 1. 今日核心问题

今天主要确认两个问题：

1. 之前跑完的 SFT-only 和 DPO-2k 200 样本评估是否仍然有效。
2. 之前担心的 `delta_q / delta_a / delta_p` 输出中有两个趋于常数、且 DPO 与 SFT 数值相同的问题，是否在当前模型中仍然严重存在。

结论：

```text
SFT-only 200 与 DPO-2k 200 的评估仍然有效。
当前模型没有出现严重 delta constant collapse。
DPO-2k 并不是简单复制 SFT 输出。
当前 text-grid baseline 可以保留，下一步可以进入 BEV-image MLLM 分支。
```

---

## 2. 已有 200 样本评估结果

### 2.1 SFT-only 200

模型：

```text
/root/autodl-tmp/outputs/stage1_sft_final
```

输出：

```text
/root/autodl-tmp/outputs/eval_sft_only_200.json
```

结果：

| Metric | Mean | Std |
|---|---:|---:|
| sum_rate | 40.3508 | 16.9632 |
| mean_sensing_sinr_db | 14.4712 | 1.0801 |
| mean_crb | 0.0000 | 0.0000 |
| joint_satisfaction | 0.5099 | 0.0236 |
| sca_fp_iterations_warm | 2.2200 | 0.4142 |
| sca_fp_iterations_cold | 2.6950 | 0.4604 |
| sca_fp_speedup | 1.2492 | 0.2863 |
| inference_latency_ms | 212.5551 | 37.0303 |

Valid samples:

```text
200
```

### 2.2 DPO-2k 200

模型：

```text
/root/autodl-tmp/outputs/stage2_dpo_2k_final
```

输出：

```text
/root/autodl-tmp/outputs/eval_dpo_2k_200.json
```

结果：

| Metric | Mean | Std |
|---|---:|---:|
| sum_rate | 40.3467 | 16.1153 |
| mean_sensing_sinr_db | 14.4760 | 1.0788 |
| mean_crb | 0.0000 | 0.0000 |
| joint_satisfaction | 0.5134 | 0.0238 |
| sca_fp_iterations_warm | 2.0050 | 0.0705 |
| sca_fp_iterations_cold | 2.6950 | 0.4604 |
| sca_fp_speedup | 1.3450 | 0.2312 |
| inference_latency_ms | 212.5476 | 37.6417 |

Valid samples:

```text
200
```

### 2.3 评估结论

DPO-2k 相比 SFT-only：

```text
sum_rate 基本持平
sensing SINR 略升
joint_satisfaction 略升
warm iterations 从 2.2200 降到 2.0050
speedup 从 1.2492 提升到 1.3450
inference latency 基本不变
```

主要收益：

```text
DPO-2k 主要改善 solver convergence，而不是显著提高最终 sum-rate。
```

可以用于论文/汇报的表述：

```text
On the textualized-BEV baseline, lightweight DPO preference refinement mainly improves solver convergence. Compared with SFT-only, DPO-2k reduces warm-start SCA-FP iterations from 2.22 to 2.01 and increases the warm/cold speedup from 1.25x to 1.35x, while maintaining comparable sum-rate and sensing SINR.
```

---

## 3. 今日新增 Delta 输出诊断

### 3.1 新增脚本

已新增：

```text
scripts/analyze_delta_outputs.py
```

用途：

```text
只运行模型 forward / warm-start generation，不运行 SCA-FP。
统计 delta_q / delta_a / delta_p 的跨样本多样性。
比较 SFT 与 DPO-2k 的 delta 输出差异。
检测 delta_a / delta_p 是否趋于常数。
```

运行命令：

```bash
cd /root/Projects/UAV/UAV
conda activate uavmllm

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/analyze_delta_outputs.py \
  --config configs/rtx5090_eval_local.yaml \
  --models \
    sft=/root/autodl-tmp/outputs/stage1_sft_final \
    dpo2k=/root/autodl-tmp/outputs/stage2_dpo_2k_final \
  --num_samples 200 \
  --output /root/autodl-tmp/outputs/delta_diag_sft_vs_dpo2k_200.json \
  --save_raw
```

输出文件：

```text
/root/autodl-tmp/outputs/delta_diag_sft_vs_dpo2k_200.json
/root/autodl-tmp/outputs/delta_diag_sft_vs_dpo2k_200.npz
```

### 3.2 诊断结果

SFT:

| Diagnostic | Value |
|---|---:|
| delta_q_per_dim_std_mean | 0.3956 |
| delta_a_per_dim_std_mean | 0.0169 |
| delta_p_per_dim_std_mean | 0.0067 |
| delta_a_argmax_unique_per_user_mean | 1.5 |
| delta_a_entropy_mean | 0.9425 |
| delta_p_entropy_mean | 1.3454 |
| warnings | [] |

DPO-2k:

| Diagnostic | Value |
|---|---:|
| delta_q_per_dim_std_mean | 0.3784 |
| delta_a_per_dim_std_mean | 0.0151 |
| delta_p_per_dim_std_mean | 0.0090 |
| delta_a_argmax_unique_per_user_mean | 1.3 |
| delta_a_entropy_mean | 0.8697 |
| delta_p_entropy_mean | 1.5148 |
| warnings | [] |

SFT vs DPO-2k:

| Difference | Value |
|---|---:|
| delta_q_l2_mean | 1.5030 |
| delta_q_mean_abs_diff | 0.3226 |
| delta_a_l2_mean | 0.8153 |
| delta_a_mean_abs_diff | 0.0604 |
| delta_p_l2_mean | 0.3065 |
| delta_p_mean_abs_diff | 0.0097 |

---

## 4. Delta 诊断结论

当前结果说明：

```text
delta_q 明显随环境变化。
delta_a / delta_p 方差较小，但不是 0。
脚本未触发 constant-head warning。
DPO-2k 与 SFT 的输出不是完全相同。
DPO-2k 确实改变了三个 head，尤其是 delta_q 和 delta_a。
```

因此，之前担心的严重问题：

```text
两个 delta head 完全趋于常数；
DPO 与 SFT 输出完全相同；
模型只输出固定模板；
```

在当前 SFT/DPO-2k 模型上没有严重复现。

但仍需注意：

```text
delta_a_argmax_unique_per_user_mean:
SFT = 1.5
DPO = 1.3
```

这个值偏低，说明每个用户的推荐 UAV 变化范围仍然不大，association prior 仍然偏保守。

因此，当前诊断不是说 `delta_a / delta_p` 已经非常强，而是说：

```text
没有严重崩坏；
baseline 有效；
association/power 的环境适应性仍是后续改进重点。
```

---

## 5. 对已有 SFT/DPO 结果的影响

今日诊断不会推翻已有 200 样本评估。

已有评估有效证明：

```text
SFT-only 和 DPO-2k 都能产生可用 warm-start。
DPO-2k 相比 SFT-only 改善了 solver convergence。
当前 text-grid baseline 可用于实验对照。
```

今日诊断补充证明：

```text
DPO-2k 不是简单复制 SFT 输出。
当前没有严重 delta constant collapse。
delta_a / delta_p 虽然变化较小，但仍有非零多样性。
```

所以当前结论是：

```text
不需要重跑 SFT。
不需要重跑 DPO。
不需要重跑 200-sample eval。
可以保留当前 text-grid baseline，并进入 MLLM 分支。
```

---

## 6. 对 MLLM 下一步的影响

现在可以继续做真正 BEV-image MLLM。

当前框架可复用：

```text
SCA-FP solver
projection head
loss framework
control token readout
warm-start evaluation
SFT/DPO baseline 结果
delta diagnostic 脚本
```

下一阶段新增：

```text
BEV image renderer
multimodal prompt
multimodal dataset
multimodal model wrapper
multimodal SFT smoke
multimodal evaluate
```

MLLM 分支除了看最终指标，还必须继续看 delta 诊断：

```text
BEV-image MLLM 不只要提升 speedup；
还应提升 delta_a / delta_p 的环境相关性；
尤其要关注 delta_a_argmax_unique_per_user_mean 是否提升。
```

后续目标可以写成：

```text
Use BEV-image multimodal context to improve spatially grounded association and power priors, especially reducing the conservative behavior observed in textualized-BEV delta_a.
```

---

## 7. 下一步执行建议

推荐下一步顺序：

```text
1. 将 scripts/analyze_delta_outputs.py 和 docs/09_code_modification_plans/ 同步到 GitHub/服务器。
2. 保留当前 SFT-only 200 与 DPO-2k 200 作为 text-grid baseline。
3. 不再继续扩大 text-grid DPO 到 10k。
4. 开始实现 BEV image renderer。
5. 生成 20 条 mm_smoke 数据。
6. 做 processor smoke。
7. 做 multimodal model forward smoke。
8. 再做 multimodal SFT smoke。
```

当前是否可以进入 MLLM：

```text
可以。
```

一句话总结：

```text
当前 baseline 已经足够稳，delta 诊断也没有暴露致命坍塌；下一步应把算力和工程时间投入真正 BEV-image MLLM，而不是继续扩展 text-grid DPO。
```
