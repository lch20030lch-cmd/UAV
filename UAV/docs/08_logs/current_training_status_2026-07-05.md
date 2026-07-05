# 当前训练状态更新：SFT 完成与 DPO 决策

> 日期：2026-07-05 晚间  
> 背景：当前项目已从 Stage I SFT 推进到 DPO smoke test 通过。后续仍可能转向真正 BEV-image MLLM，但当前 text-grid 路线建议先形成完整 baseline。

---

## 1. 当前方法与硬件状态

当前正在推进的实际方法是：

```text
Gemma3-4B + textualized BEV grid + control tokens + constraint projection head + SFT/DPO
```

当前方法不是严格意义上的视觉多模态 MLLM，而是 text-grid / textualized-BEV 版本。它仍然可以作为后续 MLLM 方法的 baseline 或消融实验。

当前硬件与训练配置：

```text
GPU: RTX 5090
VRAM: 32GB
Backbone: google/gemma-3-4b-it
Precision: bitsandbytes NF4 4-bit QLoRA
```

`configs/rtx5090.yaml` 当前安全配置：

```yaml
training:
  sft:
    per_device_batch_size: 1
    gradient_accumulation_steps: 16
    max_seq_length: 3456

  dpo:
    per_device_batch_size: 1
    gradient_accumulation_steps: 16
    mu: 0.0
    max_seq_length: 1536
```

---

## 2. Stage I SFT 当前状态

Stage I SFT 已完成，最终输出目录：

```text
/root/autodl-tmp/outputs/stage1_sft_final
```

主要文件：

```text
ctrl_embed.pt
ctrl_lm_head.pt
lora/adapter_model.safetensors
projection_head.pt
tokenizer/
```

Phase 1 关键日志：

```text
step 50:  loss_ctl=35.75, sens=0.0011
step 100: loss_ctl=28.13, sens=0.0067
step 150: loss_ctl=14.69, sens=0.2681
```

解释：

- `loss_ctl` 明显下降。
- `sensitivity=0.2681 > 0.1`，说明 control token 已经对环境变化敏感。
- Phase 1 在 step 150 自动切换到 Phase 2 是合理的。
- 当前 SFT 不需要重跑。

---

## 3. SFT-only 小规模评估

已完成 20 样本 SFT-only evaluation：

```text
sum_rate.mean                 = 49.6351
mean_sensing_sinr_db.mean     = 14.8163
joint_satisfaction.mean       = 0.5200
sca_fp_iterations_warm.mean   = 2.2000
sca_fp_iterations_cold.mean   = 2.5500
sca_fp_speedup.mean           = 1.1917
inference_latency_ms.mean     = 238.6460
valid samples                 = 20
```

判断：

- SFT-only 已经出现正向 warm-start 信号。
- 20 样本只能作为健康检查，不能作为论文最终实验。
- 后续需要跑 100/200 样本正式评估。

---

## 4. DPO Smoke Test 当前状态

已新增并使用 RTX 5090 专用 DPO smoke test：

```text
UAV/smoke_test/dpo_smoke_5090.yaml
UAV/smoke_test/run_dpo_smoke.sh
UAV/smoke_test/README.md
```

服务器启动命令：

```bash
cd /root/Projects/UAV/UAV
conda activate uavmllm
bash smoke_test/run_dpo_smoke.sh
tail -f dpo_smoke.log
```

烟测设置：

```text
source data: /root/autodl-tmp/data/full5000
smoke data: /root/autodl-tmp/data/dpo_smoke
DPO pairs: 160
optimizer steps: 10
```

烟测已完成，输出目录：

```text
/root/autodl-tmp/outputs/dpo_smoke/stage2_dpo_final
```

关键结果：

```text
Stage II complete! Model saved to /root/autodl-tmp/outputs/dpo_smoke/stage2_dpo_final

[Step 1]  loss_dpo=0.6931, dpo_accuracy=0.0000, loss_ctl=2.1671, grad_norm_lora_total=71.1878
[Step 5]  loss_dpo=0.6823, dpo_accuracy=1.0000, loss_ctl=1.5243, grad_norm_lora_total=63.3609
[Step 9]  loss_dpo=0.4876, dpo_accuracy=1.0000, loss_ctl=1.0228, grad_norm_lora_total=15.9540
[Step 10] loss_dpo=0.6218, dpo_accuracy=1.0000, loss_ctl=0.9345, grad_norm_lora_total=19.7688
```

判断：

- policy model 加载成功。
- reference model 加载成功。
- DPO dataset 加载成功。
- RTX 5090 32GB 下未出现 OOM。
- `loss_dpo` 有正常信号，未出现 NaN。
- `loss_ctl` 没有爆炸。
- `grad_norm_lora_total` 有正常数值，LoRA 梯度路径有效。

结论：

```text
当前 RTX 5090 配置可以进入正式 DPO。
```

---

## 5. 如果后续做真正 MLLM，当前 DPO 还要不要跑

结论：

```text
建议继续跑当前正式 DPO。
```

原因：

1. 当前 SFT 已完成，DPO smoke 已通过，正式 DPO 风险较低。
2. 当前 DPO 可以形成完整 baseline：`text-grid SFT+DPO`。
3. DPO 验证的是 preference pair、chosen/rejected 优化、control head 联合训练、SCA-FP warm-start 这一整套机制，未来 MLLM 版仍可复用。
4. 如果后续真正 MLLM 改造遇到时间、显存或代码风险，当前 SFT+DPO 至少能保留一个完整实验闭环。

论文定位建议：

| 当前路线 | 建议论文定位 |
|---|---|
| SFT-only text-grid Gemma3-4B | lightweight / textualized-BEV baseline |
| SFT+DPO text-grid Gemma3-4B | preference-optimized textualized-BEV baseline |
| future BEV-image multimodal Gemma3-4B | proposed MLLM method |

因此，当前 DPO 不应被视为浪费，而应作为后续 MLLM 的 baseline 和消融支撑。

---

## 6. 下一步建议

短期执行顺序：

```text
1. 启动正式 DPO。
2. 观察前 20-30 分钟，确认无 OOM、无 NaN、梯度正常。
3. DPO 完成后跑 SFT+DPO evaluation。
4. 用 SFT-only vs SFT+DPO 对比验证 DPO 是否带来额外收益。
5. 再决定是否切到 RTX PRO 6000 96GB 做真正 BEV-image MLLM。
```

正式 DPO 命令：

```bash
cd /root/Projects/UAV/UAV
conda activate uavmllm

nohup python src/training/train_dpo.py \
  --config configs/rtx5090.yaml \
  --stage1_ckpt /root/autodl-tmp/outputs/stage1_sft_final \
  --data_dir /root/autodl-tmp/data/full5000 \
  > dpo.log 2>&1 &

echo "PID: $!"
tail -f dpo.log
```

正式 DPO 观察指标：

```text
loss_dpo: not NaN, preferably slowly decreasing
dpo_accuracy: should not remain all 0 for a long time
loss_ctl: should not explode to tens or hundreds continuously
grad_norm_lora_total: should be nonzero and finite
GPU memory: should remain stable without CUDA OOM
```

---

## 7. 仍需保留的问题

即使 DPO 跑通，以下问题仍未完全解决：

1. 当前方法仍不是严格视觉多模态 MLLM。
2. 如果论文标题继续强调 MLLM，后续仍建议补 BEV image modality。
3. SFT-only 20 样本评估不能替代正式评估。
4. 数据生成和 evaluation solver config 仍应在最终大评估前统一。
5. DPO 为适配 RTX 5090 显存关闭了 SFT anchor (`mu=0.0`) 并缩短到 `max_seq_length=1536`，论文中需要如实说明硬件约束和工程设置。

当前总判断：

```text
5090 text-grid SFT+DPO 路线已经具备完整跑通条件，建议作为 baseline 继续完成。
真正 MLLM 路线仍建议作为下一阶段，在 96GB GPU 上新增分支实现，而不是推翻当前结果。
```
