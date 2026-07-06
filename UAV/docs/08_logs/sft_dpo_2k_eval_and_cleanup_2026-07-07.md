# SFT vs DPO-2k 评估结果与清理建议

> 日期：2026-07-07  
> 硬件：RTX 5090 32GB  
> 模型：Gemma3-4B + text-grid BEV + control tokens + projection head  
> 定位：当前结果作为 textualized-BEV baseline；真正 BEV-image MLLM 仍作为下一阶段主线。

---

## 1. 当前实验状态

截至 2026-07-07，当前 text-grid baseline 已形成完整闭环：

```text
Stage I SFT        完成
DPO smoke          完成
DPO-2k training    完成
SFT-only eval 200  完成
DPO-2k eval 200    完成
```

关键模型目录：

```text
SFT-only:
/root/autodl-tmp/outputs/stage1_sft_final

DPO-2k:
/root/autodl-tmp/outputs/stage2_dpo_2k_final
```

DPO-2k 训练设置：

```text
data_dir: /root/autodl-tmp/data/dpo_2k
DPO pairs: 2000
epochs: 1
batch: 1
grad_accum: 16
max_seq_length: 1536
mu: 0.0
```

DPO-2k 训练结束信号：

```text
DPO E1: 100% 2000/2000
Stage II complete! Model saved to /root/autodl-tmp/outputs/stage2_dpo_final
[Step 125] loss_dpo=0.2283, dpo_accuracy=1.0000, loss_ctl=0.9399
```

训练完成后已重命名为：

```text
/root/autodl-tmp/outputs/stage2_dpo_2k_final
```

---

## 2. 200 样本评估结果

### 2.1 SFT-only 200

输出文件：

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

输出文件：

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

---

## 3. 对比结论

公平比较条件：

```text
SFT-only 和 DPO-2k 都使用 200 samples。
二者使用相同 eval config、相同 cold-start baseline、相同 solver 流程。
```

主要变化：

| Metric | SFT-only 200 | DPO-2k 200 | 变化 |
|---|---:|---:|---:|
| sum_rate | 40.3508 | 40.3467 | 基本持平 |
| mean_sensing_sinr_db | 14.4712 | 14.4760 | 略升 |
| joint_satisfaction | 0.5099 | 0.5134 | 略升 |
| sca_fp_iterations_warm | 2.2200 | 2.0050 | 降低 0.215 |
| sca_fp_speedup | 1.2492 | 1.3450 | 提升 0.0958 |
| inference_latency_ms | 212.5551 | 212.5476 | 基本持平 |

相对变化：

```text
warm iterations 降低约 9.7%
speedup 提升约 7.7%
```

解释：

- DPO-2k 没有显著提升最终 `sum_rate`。
- DPO-2k 对 `joint_satisfaction` 有轻微正向影响。
- DPO-2k 的主要收益体现在优化器收敛效率：warm-start 后 SCA-FP 平均迭代次数更低，warm/cold speedup 更高。
- inference latency 基本不变，说明 DPO 没有增加推理成本。

可以写入论文/汇报的表述：

```text
On the textualized-BEV baseline, lightweight DPO preference refinement mainly improves solver convergence. Compared with SFT-only, DPO-2k reduces warm-start SCA-FP iterations from 2.22 to 2.01 and increases the warm/cold speedup from 1.25x to 1.35x, while maintaining comparable sum-rate and sensing SINR.
```

当前判断：

```text
DPO-2k 结果可以作为 text-grid baseline / training-stage ablation 使用。
不建议继续花费算力跑 10k DPO，除非后续需要更强的 text-grid baseline。
下一阶段应把主要经费和时间转向 BEV-image MLLM。
```

---

## 4. 当前结果的可靠性边界

可认为可靠的部分：

1. 训练链路完整跑通：SFT -> DPO -> eval。
2. 200 样本评估比 20 样本健康检查更可信。
3. SFT-only 与 DPO-2k 使用同样 200 样本设置，可公平对比。
4. DPO 对 solver convergence 的改善信号清楚。

仍需谨慎的部分：

1. 当前仍是 text-grid / textualized-BEV，不是严格 BEV-image MLLM。
2. DPO 只使用 2k preference pairs，不能表述成 full DPO。
3. `sum_rate` 基本持平，不能宣称 DPO 显著提升所有性能指标。
4. 最终论文主方法若强调 MLLM，仍需要新增视觉 BEV image 分支。

建议实验定位：

| 结果 | 定位 |
|---|---|
| SFT-only 200 | textualized-BEV SFT baseline |
| DPO-2k 200 | textualized-BEV SFT+DPO baseline / ablation |
| future BEV-image MLLM | proposed method |

---

## 5. 可清理或合并的文件建议

以下只是建议，尚未执行删除。

### 5.1 服务器可清理项

如果服务器磁盘紧张，可清理烟测和中间数据：

```bash
rm -rf /root/autodl-tmp/outputs/dpo_smoke
rm -rf /root/autodl-tmp/checkpoints/dpo_smoke
rm -rf /root/autodl-tmp/data/dpo_smoke
```

DPO-10k 未完整训练，若只是临时尝试，可清理：

```bash
rm -rf /root/autodl-tmp/data/dpo_10k
rm -f /root/Projects/UAV/UAV/dpo_10k.log
```

保留项：

```text
/root/autodl-tmp/outputs/stage1_sft_final
/root/autodl-tmp/outputs/stage2_dpo_2k_final
/root/autodl-tmp/outputs/eval_sft_only_200.json
/root/autodl-tmp/outputs/eval_dpo_2k_200.json
/root/autodl-tmp/data/full5000
/root/autodl-tmp/data/dpo_2k
```

如果确认不再从 DPO checkpoint 续训，可考虑清理中间 checkpoint：

```bash
rm -rf /root/autodl-tmp/checkpoints/stage2_step_100
```

但如果还想保留可恢复点，则暂时不要删。

### 5.2 本地仓库可清理项

当前 Git 状态显示：

```text
M  src/model/projection_head.py
?? ../论文.txt
```

建议：

- `src/model/projection_head.py`：先确认是不是你或 DeepSeek 的有效代码改动；不建议盲目提交或回退。
- `../论文.txt`：位于 `Projects` 根目录，不在 `UAV` 仓库内；若需要版本管理，建议移动到 `UAV/docs/` 下并重命名，否则继续保持仓库外。

本地 `UAV/data/` 已被 `.gitignore` 忽略，里面的 `pilot_local/dpo_dataset.jsonl` 较大，但不会被提交。若本地空间紧张且不再使用 pilot 数据，可删除：

```powershell
Remove-Item -Recurse -Force C:\Users\Shardeom-PC\Desktop\Projects\UAV\data\pilot_local
```

不建议提交本地 data 目录。

### 5.3 本地 Projects 根目录可整理项

根目录下存在：

```text
UAV.7z
smoke_result/
smoke_v3_data/
论文.txt
```

建议：

- `UAV.7z`：若只是旧压缩备份，确认无用后可删除。
- `smoke_result/`、`smoke_v3_data/`：若已迁移到 AutoDL 或不再复现实验，可归档或删除。
- `论文.txt`：建议不要放在 `Projects` 根目录长期散放；可移动到 `UAV/docs/paper/`，或保持仓库外作为私人草稿。

### 5.4 文档合并建议

当前 `docs/08_logs/` 里文档较多：

```text
rtx5090_smoke_test_2026-07-04.md
project_feasibility_review_2026-07-05.md
hardware_method_decision_record_2026-07-05.md
current_training_status_2026-07-05.md
sft_dpo_2k_eval_and_cleanup_2026-07-07.md
```

建议保留原始日志，不急着物理合并。后续可以新增一个总索引：

```text
docs/08_logs/README.md
```

索引中按时间和主题链接各文档。这样既不破坏历史记录，也方便查找。

如果一定要合并，建议合并方向：

```text
hardware_method_decision_record_2026-07-05.md
  + current_training_status_2026-07-05.md
  + sft_dpo_2k_eval_and_cleanup_2026-07-07.md
  -> training_and_method_decision_record.md
```

但由于已有文档较长且部分终端显示存在编码问题，当前更推荐“保留原文 + 新建 README 索引”，不要直接大合并。

---

## 6. 下一步行动

建议下一步：

```text
1. 保留当前 SFT-only 200 与 DPO-2k 200 作为 text-grid baseline。
2. 不再继续跑 10k DPO。
3. 开始设计 BEV-image MLLM 分支。
4. 首先实现 BEV image 生成与 dataset 字段。
5. 在 RTX PRO 6000 96GB 上做 multimodal SFT smoke。
```

一句话结论：

```text
当前 text-grid baseline 已经足够可用；下一笔算力应该投向真正 MLLM，而不是继续扩大 text-grid DPO。
```
