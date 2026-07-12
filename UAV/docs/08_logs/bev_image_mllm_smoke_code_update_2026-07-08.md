# BEV-image MLLM 烟雾测试代码更新与服务器验证日志

> 日期：2026-07-08 起
> 最近更新：2026-07-12
> 范围：根据 `docs/09_code_modification_plans` 推进 RTX 5090 32GB 上的 BEV-image MLLM 最小闭环。
> 原则：新增多模态分支，不破坏已经跑通的 text-grid SFT/DPO baseline。

---

## 1. 背景

论文需要真正的多模态 MLLM 路径：

```text
communication summary + sensing summary + BEV image
  -> multimodal backbone / processor
  -> control-token hidden states
  -> projection head
  -> delta_q / delta_a / delta_p
  -> SCA-FP warm-start
```

此前已经完成的是 text-grid baseline：

```text
Gemma3 text-only input
BEV grid 以文本形式拼进 prompt
control tokens
projection head
SFT / DPO
```

09 计划文档建议保留 text-grid baseline，同时新增独立的 BEV-image MLLM 分支。本日志记录了 BEV-image 分支从数据生成、处理器、前向传播、训练烟雾测试到 delta 诊断的阶段性结果。

---

## 2. 本轮新增与修改文件

### 2.1 BEV 图像渲染

新增：

```text
src/env/bev_renderer.py
```

用途：

```text
将 UAV、用户、目标的位置关系渲染为 BEV PNG。
UAV 使用蓝色三角形。
用户使用绿色点。
目标使用红色 X。
可选绘制关联线和 UAV 覆盖圆。
```

设计原则：

```text
图像只表达空间几何，不做复杂展示。
不使用复杂背景。
不使用大段文字 legend。
坐标轴固定在服务区域范围内。
```

### 2.2 多模态烟雾测试数据生成

新增：

```text
scripts/generate_mm_smoke.py
```

用途：

```text
生成小规模 BEV-image 多模态烟雾测试数据。
复用现有 scenario generator、SCA-FP solver、oracle prior 提取逻辑和 JSON response 格式。
每条样本额外写入 prompt_type="multimodal_bev_image" 和 bev_image_path。
```

输出目录：

```text
/root/autodl-tmp/data/mm_smoke/
  sft_dataset.jsonl
  dpo_dataset.jsonl
  checkpoint.txt
  images/
    env_000000.png
    env_000001.png
```

### 2.3 多模态处理器烟雾测试

新增：

```text
scripts/smoke_mm_processor.py
```

用途：

```text
读取一条 multimodal JSONL 样本。
打开对应 BEV 图片。
加载 Gemma3 AutoProcessor。
验证文本 + 图像能被处理器编码。
追加 control tokens 并按 token id 定位。
```

关键检查：

```text
control_token_count 必须等于 8。
```

### 2.4 多模态前向传播烟雾测试

新增：

```text
src/data/multimodal_dataset.py
src/model/gemma_multimodal_isac.py
scripts/smoke_mm_forward.py
```

用途：

```text
验证 prompt + BEV image 能进入 Gemma3 多模态模型。
从 control token hidden states 读取控制表示。
送入现有 ConstraintProjectionHead。
输出 delta_q / delta_a / delta_p。
```

### 2.5 多模态 SFT 烟雾测试

新增：

```text
src/training/train_sft_mm.py
```

当前训练模式：

```text
默认只训练投影头的 CTL 烟雾测试。
默认冻结 Gemma3 多模态 backbone。
默认冻结视觉塔。
默认不计算 token-level CE。
显式传入 --train_lora 后，训练 projection head + LoRA。
```

### 2.6 多模态 delta 输出诊断

新增：

```text
scripts/analyze_mm_delta_outputs.py
```

用途：

```text
读取 BEV-image 烟雾测试数据。
加载 Gemma3 多模态模型。
可选加载 projection_head.pt。
只运行前向传播 / 投影头，不运行 SCA-FP。
统计 delta_q / delta_a / delta_p 的跨样本多样性。
```

---

## 3. 配置调整

修改：

```text
configs/rtx5090_multimodal_smoke.yaml
```

关键设置：

```text
use_4bit: true
freeze_vision_tower: true
image_size: 224
use_bev_text_grid: false
use_bev_image: true
max_seq_length: 3072
num_environments: 20
num_restarts: 3
```

说明：

```text
处理器烟雾测试中，单样本 input_ids 实测约 2025，因此 1024 不够。
当前烟雾测试默认将 max_seq_length 调整为 3072。
```

---

## 4. 本地静态检查

已通过：

```bash
python -m py_compile \
  scripts/smoke_mm_processor.py \
  scripts/generate_mm_smoke.py \
  scripts/smoke_mm_forward.py \
  scripts/analyze_mm_delta_outputs.py \
  src/env/bev_renderer.py \
  src/data/multimodal_dataset.py \
  src/model/gemma_multimodal_isac.py \
  src/training/train_sft_mm.py
```

新增 Python 文件均已检查，当前没有非预期编码字符问题。

---

## 5. 服务器验证结果

服务器路径：

```text
/root/Projects/UAV/UAV
```

数据路径：

```text
/root/autodl-tmp/data/mm_smoke
```

模型路径：

```text
/root/autodl-tmp/huggingface/models/gemma-3-4b-it
```

## 5.1 Step 1：生成 BEV-image 烟雾测试数据

命令：

```bash
python scripts/generate_mm_smoke.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --output_dir /root/autodl-tmp/data/mm_smoke \
  --num_samples 20 \
  --num_restarts 3 \
  --overwrite
```

结果：

```text
sft_dataset.jsonl 已生成。
dpo_dataset.jsonl 已生成。
images/env_000000.png 等 BEV 图片已生成。
checkpoint.txt 已生成。
```

样本包含：

```text
prompt
response
bev_image_path
prompt_type="multimodal_bev_image"
q_current
delta_q / delta_a / delta_p
```

结论：

```text
Step 1 PASS。
BEV-image 多模态烟雾测试数据生成成功。
```

## 5.2 Step 2：处理器烟雾测试

首次问题：

```text
ValueError: Prompt contained 0 image tokens but received 1 images.
```

原因：

```text
Gemma3 处理器要求 prompt 中包含模型专用图像 token。
脚本已改为在 [Bird's-Eye-View Image] 前自动插入 Gemma BOI image token。
```

第二个问题：

```text
Mismatch in image token count between text and input_ids.
Likely due to truncation='max_length'.
```

原因：

```text
max_seq_length=1024 太短。
Gemma3 处理器会将图片展开为大量 image tokens。
```

成功命令：

```bash
python scripts/smoke_mm_processor.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_length 4096
```

成功输出：

```text
OK: multimodal processor smoke
  data: /root/autodl-tmp/data/mm_smoke/sft_dataset.jsonl
  image: /root/autodl-tmp/data/mm_smoke/images/env_000000.png size=(224, 224)
  input_ids: (1, 2025)
  attention_mask: (1, 2025)
  token_type_ids: (1, 2017)
  pixel_values: (1, 3, 896, 896)
  control_token_count: 8
```

结论：

```text
Step 2 PASS。
prompt + BEV image 能被本地 Gemma3 处理器正常编码。
control tokens 能正确追加和定位。
Gemma3 会将 224 x 224 图片处理为 pixel_values=(1, 3, 896, 896)，多模态训练显存压力会明显高于 text-only。
```

## 5.3 Step 3：多模态前向传播烟雾测试

命令：

```bash
python scripts/smoke_mm_forward.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_length 3072
```

成功输出：

```text
OK: multimodal model forward smoke
  data: /root/autodl-tmp/data/mm_smoke/sft_dataset.jsonl
  max_length: 3072
  input_ids: (1, 3072)
  attention_mask: (1, 3072)
  pixel_values: (1, 3, 896, 896)
  control_token_count: 8
  control_states: (1, 8, 2560)
  delta_q: (1, 4, 3)
  delta_a: (1, 4, 20)
  delta_p: (1, 4, 21)
```

结论：

```text
Step 3 PASS。
BEV image + text prompt
  -> Gemma3 多模态处理器
  -> Gemma3 multimodal model
  -> control-token hidden states
  -> projection head
  -> delta_q / delta_a / delta_p

最小多模态前向传播闭环已经成立。
```

## 5.4 Step 4a：10-step 只训练投影头的 SFT 烟雾测试

命令：

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_steps 10 \
  --max_length 3072
```

训练模式：

```text
只训练投影头的 CTL 烟雾测试。
Gemma3 多模态 backbone 冻结。
视觉塔冻结。
不启用 LoRA。
不计算 token-level CE。
```

结果：

```text
10 step 完成。
无 OOM。
无 NaN。
loss_ctl 有数值。
grad_norm_proj 有数值。
final_checkpoint 已保存。
```

checkpoint：

```text
/root/autodl-tmp/outputs/mm_smoke/mm_sft_smoke_final
/root/autodl-tmp/checkpoints/mm_smoke/mm_sft_smoke_step_10
```

结论：

```text
Step 4a PASS。
训练外壳、control loss、backward、optimizer step、checkpoint 保存均可用。
```

## 5.5 Step 4b：30-step projection-head-only 稳定性检查

命令：

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_steps 30 \
  --max_length 3072
```

代表性指标：

```text
step=1  loss_ctl=72.270164   grad_norm_proj=348.923516
step=10 loss_ctl=76.983398   grad_norm_proj=244.714702
step=20 loss_ctl=65.143181   grad_norm_proj=201.112136
step=30 loss_ctl=69.149162   grad_norm_proj=199.785992
```

结果：

```text
30 / 30 step 完成。
运行约 26 秒。
速度约 1.12 it/s。
无 OOM。
无 NaN。
grad_norm_proj 全程有限，整体从约 350 降到约 200。
```

结论：

```text
Step 4b PASS。
只训练投影头的 BEV-image MLLM 训练烟雾测试在 RTX 5090 上 30 step 稳定。
```

## 5.6 Step 5：multimodal delta 输出诊断

命令：

```bash
python scripts/analyze_mm_delta_outputs.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --checkpoint /root/autodl-tmp/outputs/mm_smoke/mm_sft_smoke_final \
  --name mm_sft_smoke_30step \
  --num_samples 20 \
  --max_length 3072 \
  --output /root/autodl-tmp/outputs/mm_smoke/delta_diag_mm_sft_smoke_20.json \
  --save_raw
```

输出摘要：

```text
delta_q_per_dim_std_mean: 0.2534600794315338
delta_a_per_dim_std_mean: 0.028002941980957985
delta_p_per_dim_std_mean: 0.008766541257500648
delta_a_argmax_unique_per_user_mean: 1.15
delta_a_entropy_mean: 0.8596800911881917
delta_p_entropy_mean: 1.9475480959227327
warnings: ['delta_a_argmax_nearly_constant']
```

解读：

```text
delta_q 有明显跨样本变化。
delta_p 有非零跨样本变化，功率分配较平滑。
delta_a 的 soft value 有变化，但 argmax UAV 选择几乎固定。
```

结论：

```text
只训练投影头得到的 checkpoint 没有全局 delta collapse。
但 association argmax 仍然偏保守，触发 delta_a_argmax_nearly_constant warning。
这符合预期：当前 backbone 冻结、未启用 LoRA，且只有 20 条烟雾测试数据与 30 step 投影头训练。
```

---

## 6. 当前里程碑状态

```text
Step 1: generate_mm_smoke.py                    PASS
Step 2: smoke_mm_processor.py                   PASS
Step 3: smoke_mm_forward.py                     PASS
Step 4a: train_sft_mm.py, 10-step 烟雾测试      PASS
Step 4b: train_sft_mm.py, 30-step stability     PASS
Step 5: analyze_mm_delta_outputs.py             PASS
```

当前结论：

```text
BEV-image MLLM 最小闭环已经完整跑通：

数据生成
  -> 处理器
  -> 模型前向传播
  -> projection head
  -> 仅 CTL 训练烟雾测试
  -> delta 输出诊断

下一步应测试启用 LoRA 的多模态 SFT 烟雾测试。
```

---

## 7. 下一步建议

优先级 1：

```text
运行 3-step 启用 LoRA 的多模态 CTL 烟雾测试。
目标不是效果，而是验证显存、LoRA 梯度、projection head 梯度和 checkpoint 保存。
```

建议命令：

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_steps 3 \
  --max_length 3072 \
  --train_lora
```

关注输出：

```text
trainable: projection_head + LoRA
trainable LoRA tensors > 0
loss_ctl 有数值
grad_norm_proj 有数值
grad_norm_lora 有数值
无 OOM
无 NaN
```

如果 3 step 成功，再跑：

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_steps 10 \
  --max_length 3072 \
  --train_lora
```

如果 OOM：

```text
优先将 max_length 从 3072 降到 2304。
其次考虑减小 image_size 或进一步缩短 prompt。
```

---

## 8. 2026-07-12：LoRA checkpoint 诊断链路补强

背景：

```text
train_sft_mm.py 已经支持 --train_lora，并会保存 lora/ adapter。
但此前 smoke_mm_forward.py 和 analyze_mm_delta_outputs.py 只加载 projection_head.pt。
如果 LoRA 烟雾测试后直接做 delta 诊断，可能没有真正评估 LoRA adapter 的影响。
```

本轮代码更新：

```text
src/model/gemma_multimodal_isac.py
  - 新增 lora_checkpoint 参数。
  - 支持从 checkpoint/lora 加载 PEFT adapter。
  - 新增控制 token embedding 的保存与恢复函数。

src/training/train_sft_mm.py
  - 保存 projection_head.pt 时同步保存 ctrl_embed.pt。
  - --train_lora 时强制检查可训练 LoRA 参数数量。
  - LoRA 学习率优先使用 training.sft.phase1.lr_lora。

scripts/smoke_mm_forward.py
  - 新增 --checkpoint 和 --lora_checkpoint。
  - 可自动加载 checkpoint/lora、projection_head.pt、ctrl_embed.pt。

scripts/analyze_mm_delta_outputs.py
  - 新增 --lora_checkpoint。
  - 可自动从 --checkpoint/lora 发现并加载 LoRA adapter。
  - delta 诊断输出会打印 loaded_projection、loaded_control_embeddings、loaded_lora_checkpoint。
```

服务器下一步建议命令：

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_steps 3 \
  --max_length 3072 \
  --train_lora
```

3-step 成功后，先做带 checkpoint 的单 batch 前向传播验证：

```bash
python scripts/smoke_mm_forward.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --checkpoint /root/autodl-tmp/outputs/mm_smoke/mm_sft_lora_smoke_final \
  --max_length 3072
```

再做 LoRA checkpoint 的 delta 诊断：

```bash
python scripts/analyze_mm_delta_outputs.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --checkpoint /root/autodl-tmp/outputs/mm_smoke/mm_sft_lora_smoke_final \
  --name mm_sft_lora_smoke_3step \
  --num_samples 20 \
  --max_length 3072 \
  --output /root/autodl-tmp/outputs/mm_smoke/delta_diag_mm_sft_lora_smoke_3step.json \
  --save_raw
```

验收关注点：

```text
trainable LoRA tensors > 0
grad_norm_lora 有数值
loaded_lora_checkpoint 指向 checkpoint/lora
loaded_control_embeddings 包含 ctrl_embed.pt
delta_q / delta_a / delta_p shape 正确
无 OOM
无 NaN
```

---

## 9. 2026-07-12：LoRA 3-step 烟雾测试服务器结果

训练命令：

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_steps 3 \
  --max_length 3072 \
  --train_lora
```

训练输出摘要：

```text
trainable projection tensors: 17
trainable LoRA tensors:       434
projection lr:                0.001
LoRA lr:                      0.0005

step=1 loss_ctl=72.921425 grad_norm_proj=344.476310 grad_norm_lora=59.234765
step=2 loss_ctl=86.507225 grad_norm_proj=256.141179 grad_norm_lora=57.440685
step=3 loss_ctl=69.128632 grad_norm_proj=227.342428 grad_norm_lora=23.161257

final_checkpoint: /root/autodl-tmp/outputs/mm_smoke/mm_sft_lora_smoke_final
```

结论：

```text
LoRA 3-step 训练烟雾测试 PASS。
LoRA 参数数量非零，LoRA 梯度全程有数值。
无 OOM。
无 NaN。
checkpoint 成功保存。
```

带 checkpoint 的单 batch 前向传播验证：

```text
loaded_projection: /root/autodl-tmp/outputs/mm_smoke/mm_sft_lora_smoke_final/projection_head.pt
loaded_control_embeddings: {'ctrl_embed': '/root/autodl-tmp/outputs/mm_smoke/mm_sft_lora_smoke_final/ctrl_embed.pt'}
loaded_lora_checkpoint: /root/autodl-tmp/outputs/mm_smoke/mm_sft_lora_smoke_final/lora
control_token_count: 8
control_states: (1, 8, 2560)
delta_q: (1, 4, 3)
delta_a: (1, 4, 20)
delta_p: (1, 4, 21)
```

结论：

```text
LoRA checkpoint 加载链路 PASS。
projection_head.pt、ctrl_embed.pt、lora adapter 均被成功加载。
模型输出 shape 正确。
```

LoRA 3-step delta 诊断摘要：

```text
delta_q_per_dim_std_mean: 0.0464276485145092
delta_a_per_dim_std_mean: 0.00784333422780037
delta_p_per_dim_std_mean: 0.001915224944241345
delta_a_argmax_unique_per_user_mean: 1.05
delta_a_entropy_mean: 0.648697207038731
delta_p_entropy_mean: 1.7171620636059495
warnings: ['delta_a_argmax_nearly_constant']
```

与上一轮 30-step projection-head-only 诊断对比：

```text
projection-only 30-step:
  delta_q_per_dim_std_mean: 0.2534600794315338
  delta_a_per_dim_std_mean: 0.028002941980957985
  delta_p_per_dim_std_mean: 0.008766541257500648
  delta_a_argmax_unique_per_user_mean: 1.15

LoRA 3-step:
  delta_q_per_dim_std_mean: 0.0464276485145092
  delta_a_per_dim_std_mean: 0.00784333422780037
  delta_p_per_dim_std_mean: 0.001915224944241345
  delta_a_argmax_unique_per_user_mean: 1.05
```

解释：

```text
LoRA 3-step 的目标是链路验证，不是效果验证。
当前 LoRA 链路已经通过，但训练步数太短，delta 多样性没有改善，association argmax 仍然几乎固定。
这不构成失败；它说明可以进入更长 LoRA 烟雾训练。
```

下一步：

```text
建议运行 LoRA 10-step。
如果 10-step 无 OOM / NaN，再做同样的 checkpoint forward smoke 和 delta 诊断。
重点观察 delta_a_argmax_unique_per_user_mean 是否高于 1.05，以及 delta_a_per_dim_std_mean 是否回升。
```

---

## 10. 2026-07-12：LoRA 10-step delta 诊断结果

LoRA 10-step 训练完成，并使用同一 final checkpoint 跑完 delta 诊断。

checkpoint 加载确认：

```text
loaded_projection: /root/autodl-tmp/outputs/mm_smoke/mm_sft_lora_smoke_final/projection_head.pt
loaded_control_embeddings: {'ctrl_embed': '/root/autodl-tmp/outputs/mm_smoke/mm_sft_lora_smoke_final/ctrl_embed.pt'}
loaded_lora_checkpoint: /root/autodl-tmp/outputs/mm_smoke/mm_sft_lora_smoke_final/lora
```

delta 诊断摘要：

```text
delta_q_per_dim_std_mean: 0.04045112803578377
delta_a_per_dim_std_mean: 0.006317050661891699
delta_p_per_dim_std_mean: 0.001469331793487072
delta_a_argmax_unique_per_user_mean: 1.15
delta_a_entropy_mean: 0.5970468373315758
delta_p_entropy_mean: 1.6385927804566645
warnings: ['delta_a_argmax_nearly_constant']
```

与 LoRA 3-step 对比：

```text
LoRA 3-step:
  delta_q_per_dim_std_mean: 0.0464276485145092
  delta_a_per_dim_std_mean: 0.00784333422780037
  delta_p_per_dim_std_mean: 0.001915224944241345
  delta_a_argmax_unique_per_user_mean: 1.05

LoRA 10-step:
  delta_q_per_dim_std_mean: 0.04045112803578377
  delta_a_per_dim_std_mean: 0.006317050661891699
  delta_p_per_dim_std_mean: 0.001469331793487072
  delta_a_argmax_unique_per_user_mean: 1.15
```

判断：

```text
LoRA 10-step 训练与 checkpoint 诊断链路 PASS。
association argmax 唯一性从 1.05 回到 1.15，与 projection-head-only 30-step 持平。
但 delta_q / delta_a / delta_p 的跨样本 soft 方差仍低于 projection-head-only 30-step。
warnings 仍包含 delta_a_argmax_nearly_constant。
```

解释：

```text
10-step LoRA 已经不只是链路验证；它开始恢复少量 association argmax 多样性。
但样本数只有 20，训练步数仍短，且当前只做 CTL-only 监督，没有 DPO 或 SCA-FP 闭环评估。
因此不应据此判断 LoRA 效果不好，只能判断：LoRA 10-step 仍未解决 association argmax 保守问题。
```

下一步建议：

```text
建议运行 LoRA 30-step，和此前 projection-head-only 30-step 做同步步数对照。
为了避免覆盖 10-step checkpoint，建议给 30-step 单独 output_dir。
如果 LoRA 30-step 仍然低方差，则下一轮重点转向数据量和监督信号，而不是继续盲目加步数。
```
