# RTX 5090 32GB MLLM 最小闭环 Smoke 计划

> 日期：2026-07-08  
> 目标：在暂不更换 RTX PRO 6000 96GB 的情况下，用 RTX 5090 32GB 先跑通真正 BEV-image MLLM 的最小代码闭环。  
> 定位：省钱路线；目标是代码链路验证和小规模 smoke，不是正式多模态大训练。

---

## 1. 决策背景

当前 text-grid baseline 已经完成：

```text
SFT-only 200 eval
DPO-2k 200 eval
delta diagnostic
```

结论：

```text
当前 baseline 有效；
没有严重 delta constant collapse；
可以进入真正 BEV-image MLLM 分支。
```

但考虑经费，暂时不切换 RTX PRO 6000 96GB，而是在 RTX 5090 32GB 上先做最小 MLLM smoke。

核心策略：

```text
32GB 用于跑通代码闭环；
96GB 后续再用于正式多模态训练。
```

---

## 2. 32GB 能做什么，不能做什么

### 2.1 可以做

RTX 5090 32GB 可以承担：

```text
BEV image renderer
multimodal smoke 数据生成
processor smoke
single-batch multimodal forward smoke
极小 multimodal SFT smoke
delta output diagnostic
20-sample quick eval
```

这些任务主要验证：

```text
text + image 是否能进模型
control token 是否仍能定位
projection head 是否输出正确 shape
loss 是否非 NaN
显存是否能撑住最小 batch
```

### 2.2 暂不建议做

RTX 5090 32GB 暂不建议承担：

```text
multimodal DPO
大规模 multimodal SFT
多 epoch 训练
大图像尺寸
长序列输入
解冻 vision tower
Gemma3-12B
```

原因：

```text
多模态模型会引入 vision tower / image tokens / pixel_values；
activation memory 明显高于 text-only；
当前 text-grid DPO 在 32GB 上已经需要 batch=1、seq=1536、mu=0.0；
multimodal DPO 还要加载 reference model，显存风险更高。
```

---

## 3. 32GB Smoke 的最小目标

本阶段不追求正式论文结果，只追求：

```text
能生成 BEV image；
能构造 text + image prompt；
processor 能正常编码；
multimodal model 能 forward；
projection head 能输出 delta_q / delta_a / delta_p；
小步数 SFT 能 backward；
checkpoint 能保存；
delta diagnostic 能跑。
```

最低验收：

```text
无 OOM
无 NaN
delta shape 正确
control token 数量正确
loss_ctl 有数值
grad_norm_lora_total 有值
```

增强验收：

```text
10-30 step 内 loss_ctl 有下降趋势
delta_q/a/p 跨样本不全为常数
20-sample quick eval 能完成
```

---

## 4. 初始配置建议

32GB smoke 初始配置：

```yaml
hardware:
  gpu: "RTX 5090"
  vram_gb: 32
  use_4bit: true
  gradient_checkpointing: true
  max_grad_norm: 1.0

model:
  backbone: "google/gemma-3-4b-it"
  use_multimodal: true
  freeze_vision_tower: true
  torch_dtype: "bfloat16"
  attn_implementation: "sdpa"

training:
  sft:
    epochs: 1
    per_device_batch_size: 1
    gradient_accumulation_steps: 8
    learning_rate: 2.0e-4
    max_seq_length: 1024
    save_steps: 10
    logging_steps: 1
    phase1:
      enabled: true
      max_steps: 30
      sensitivity_check_steps: 10
      sensitivity_threshold: 0.1

data:
  num_environments: 20
  image_size: 224
  modalities:
    use_comm_summary: true
    use_sensing_summary: true
    use_bev_text_grid: false
    use_bev_image: true
```

如果 OOM，按顺序降低：

```text
max_seq_length: 1024 -> 768
image_size: 224 -> 168/160
gradient_accumulation_steps: 8 -> 4
只跑 forward smoke，不跑 backward
```

如果显存富余，再考虑：

```text
max_seq_length: 1024 -> 1536
num_samples: 20 -> 50
steps: 30 -> 100
```

---

## 5. 代码新增清单

本阶段建议只新增，不覆盖旧 baseline。

### 5.1 第一批：数据与图像

优先实现：

```text
src/env/bev_renderer.py
scripts/generate_mm_smoke.py
configs/rtx5090_multimodal_smoke.yaml
```

用途：

```text
bev_renderer.py:
  把 UAV / user / target 位置渲染成 BEV PNG。

generate_mm_smoke.py:
  生成 20-50 条 mm_smoke 数据和 images/。

rtx5090_multimodal_smoke.yaml:
  32GB 专用最小多模态 smoke 配置。
```

输出目录：

```text
/root/autodl-tmp/data/mm_smoke/
  sft_dataset.jsonl
  images/
    env_000000.png
    env_000001.png
```

JSONL 新增字段：

```json
{
  "bev_image_path": "images/env_000000.png",
  "prompt_type": "multimodal_bev_image"
}
```

### 5.2 第二批：Processor Smoke

新增：

```text
scripts/smoke_mm_processor.py
```

检查：

```text
能读取一条 JSONL；
能打开 BEV image；
能构造 multimodal prompt；
processor(text + image) 能输出 input_ids / pixel_values；
control tokens 能在 input_ids 中定位。
```

### 5.3 第三批：Model Forward Smoke

新增：

```text
src/data/multimodal_dataset.py
src/model/gemma_multimodal_isac.py
scripts/smoke_mm_forward.py
```

检查：

```text
单 batch forward 成功；
输出 logits；
输出 delta_q / delta_a / delta_p；
shape 正确；
无 OOM；
无 NaN。
```

### 5.4 第四批：Multimodal SFT Smoke

新增：

```text
src/training/train_sft_mm.py
```

最小训练：

```text
20 samples
10-30 steps
batch=1
freeze vision tower
4-bit
```

不做 DPO。

---

## 6. Prompt Builder 修改策略

当前 `build_full_prompt()` 保留给 text-grid baseline。

新增：

```python
def build_multimodal_prompt(env_sample, config: dict) -> str:
    ...
```

多模态 prompt 不再拼接完整 `bev_grid_text`，而是添加短描述：

```text
The attached bird's-eye-view image encodes the spatial geometry of UAVs,
ground users, and sensing targets. Use it together with the communication
and sensing summaries to infer coverage holes, load imbalance, target
proximity, and movement directions.
```

注意：

```text
不要直接删除 build_full_prompt。
不要破坏已有 SFT/DPO baseline。
是否需要显式 <image> token 取决于具体 processor。
```

---

## 7. BEV Image 设计

图像原则：

```text
简单
稳定
低噪声
空间关系清楚
不要复杂装饰
```

建议元素：

```text
UAV: blue triangle
users: green dots
targets: red crosses/stars
optional UAV coverage circle: translucent blue
optional association line: light gray
axis fixed to 0-1000m
square aspect ratio
image_size default 224
```

不建议：

```text
大段文字标签
复杂 legend
渐变背景
过多颜色
过密网格
```

首批 smoke 要人工查看 5 张图片，确认图像确实表达了：

```text
UAV / user / target 的相对位置
目标密集区
用户密集区
可能的覆盖空洞
```

---

## 8. 32GB Smoke 执行顺序

### Step 1: 生成 mm_smoke 数据

```bash
cd /root/Projects/UAV/UAV
conda activate uavmllm

python scripts/generate_mm_smoke.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --output_dir /root/autodl-tmp/data/mm_smoke \
  --num_samples 20
```

验收：

```bash
ls -lh /root/autodl-tmp/data/mm_smoke
ls -lh /root/autodl-tmp/data/mm_smoke/images | head
head -1 /root/autodl-tmp/data/mm_smoke/sft_dataset.jsonl
```

### Step 2: Processor smoke

```bash
python scripts/smoke_mm_processor.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke
```

验收：

```text
input_ids shape 正常
pixel_values shape 正常
control token count = 8
```

### Step 3: Model forward smoke

```bash
python scripts/smoke_mm_forward.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke
```

验收：

```text
delta_q: [1, M, 3]
delta_a: [1, M, K]
delta_p: [1, M, K+1]
无 OOM
无 NaN
```

### Step 4: Multimodal SFT smoke

```bash
nohup python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  > mm_sft_smoke.log 2>&1 &

tail -f mm_sft_smoke.log
```

验收：

```text
训练至少跑 10-30 steps
loss_ctl 有数值
grad_norm_lora_total 有值
checkpoint 保存成功
```

### Step 5: Delta diagnostic

如果 SFT smoke 保存模型：

```bash
python scripts/analyze_delta_outputs.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --models mm_sft=/root/autodl-tmp/outputs/mm_sft_smoke_final \
  --num_samples 20 \
  --output /root/autodl-tmp/outputs/delta_diag_mm_sft_smoke_20.json
```

---

## 9. 成功/失败判断

### 成功

认为 32GB 最小 MLLM 跑通，如果满足：

```text
mm_smoke 数据生成成功；
processor smoke 成功；
model forward smoke 成功；
multimodal SFT smoke 能跑 10-30 steps；
模型能保存；
delta diagnostic 能输出。
```

### 部分成功

如果 forward 成功但 backward OOM：

```text
说明代码链路基本成立；
32GB 可用于开发；
正式训练仍建议 96GB。
```

### 失败

如果 processor 或 model loading 阶段失败：

```text
先不要碰训练；
优先解决 Gemma3 multimodal API / transformers 版本 / processor 格式。
```

---

## 10. 对论文的意义

即使 32GB 只跑 smoke，也有价值：

```text
证明项目已从 textualized-BEV baseline 扩展到 BEV-image multimodal pipeline；
证明当前 optimizer-aware framework 可兼容图像输入；
为后续 96GB 正式训练降低风险。
```

但论文最终主结果仍建议：

```text
至少跑一个 multimodal SFT 的 100/200 样本 evaluation；
最好在 96GB 上完成更稳定训练。
```

如果经费不足，32GB smoke 可以作为阶段性开发证明，但不宜包装成最终充分实验。

---

## 11. 当前结论

推荐路线：

```text
先在 RTX 5090 32GB 上实现 BEV-image MLLM 最小闭环。
只做 smoke，不做大训练。
确认链路可行后，再根据经费决定是否上 96GB 做正式 multimodal SFT。
```

一句话：

```text
32GB 不是完全不能做 MLLM；它适合开发和 smoke。正式效果验证是否足够，则取决于后续是否能在更大显存上跑稳定训练。
```
