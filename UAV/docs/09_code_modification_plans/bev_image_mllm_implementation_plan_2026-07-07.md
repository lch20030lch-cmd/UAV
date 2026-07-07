# BEV-Image MLLM 实现方案与代码修改计划

> 日期：2026-07-07  
> 目标：在保留当前 text-grid baseline 的基础上，新增真正 BEV-image 多模态 MLLM 分支，使论文标题中的 `MLLM` 名副其实。  
> 当前 baseline：Gemma3-4B + textualized BEV grid + control tokens + projection head + SFT/DPO。  
> 建议硬件：RTX PRO 6000 96GB 优先；RTX 5090 32GB 仅适合极小 smoke，不建议承担主训练。

---

## 1. 总体判断

当前项目已经跑通：

```text
SFT-only text-grid baseline
DPO-2k text-grid baseline
200-sample evaluation
```

当前结果可作为：

```text
textualized-BEV baseline
training-stage ablation
SFT vs SFT+DPO 对比
```

但当前实现不是严格视觉多模态 MLLM，因为：

```text
model.use_multimodal = false
data.modalities.use_bev_image = false
模型入口是 AutoModelForCausalLM
BEV map 以 env_sample.bev_grid_text 拼进 prompt
没有 image processor / pixel_values / vision tower
```

如果论文标题继续使用：

```text
Constraint-Aware MLLM Adaptation for UAV-Enabled ISAC in Low-Altitude IoT Networks
```

则建议新增真正多模态分支：

```text
text prompt + BEV image
  -> Gemma3 vision-capable / multimodal processor
  -> control token hidden states
  -> projection head
  -> delta_q / delta_a / delta_p
  -> SCA-FP warm-start
```

---

## 2. 设计原则

### 2.1 不推翻当前 baseline

当前 text-grid 结果已经有实验价值，不能为了多模态重构而破坏。

保留：

```text
src/model/gemma_isac.py
src/data/dataset.py
src/training/train_sft.py
src/training/train_dpo.py
src/eval/evaluate.py
configs/rtx5090.yaml
```

新增：

```text
src/model/gemma_multimodal_isac.py
src/data/multimodal_dataset.py
src/training/train_sft_mm.py
src/eval/evaluate_mm.py
configs/pro6000_multimodal.yaml
```

这样做的好处：

1. 当前 baseline 可复现。
2. 多模态分支可独立 smoke 和 debug。
3. 多模态失败不会破坏已经可用的 text-grid 结果。
4. 论文中可以自然形成 text-grid vs BEV-image ablation。

### 2.2 保持 solver-facing 接口不变

无论输入是 text-grid 还是 BEV image，最终都应输出同样格式：

```text
delta_q: (M, 3)
delta_a: (M, K)
delta_p: (M, K+1)
```

因此以下模块不需要重写：

```text
src/solver/sca_fp.py
src/model/projection_head.py
src/model/losses.py
oracle extraction / Xi mapping
SCA-FP warm-start evaluation
```

MLLM 只改变“如何理解环境状态”，不改变“如何接入优化器”。

---

## 3. 多模态输入定义

论文中的三种 task-relevant modalities：

```text
c(t): communication summary
r(t): sensing summary
V(t): bird's-eye-view map
```

当前 text-grid 版：

```text
c(t) -> text
r(t) -> text
V(t) -> text grid
```

目标 MLLM 版：

```text
c(t) -> text
r(t) -> text
V(t) -> rendered BEV image
```

推荐 prompt 结构：

```text
System instruction
Communication summary
Sensing summary
Short description that the attached BEV image encodes UAV/user/target geometry
Final instruction to produce warm-start prior
```

不要在 multimodal prompt 中继续塞完整 BEV text grid，否则图像分支贡献会被文本替代。

---

## 4. 文件级修改计划

## 4.1 `src/env/isac_scenario.py`

### 当前状态

`EnvironmentSample` 当前字段：

```python
bev_grid_text: str
```

`ISACScenarioGenerator._build_bev_text_grid()` 当前生成 10x10 文本网格。

注释中已有 `use_bev_image=True` 的方向，但还没有真正保存图像路径。

### 需要新增

新增字段：

```python
bev_image_path: Optional[str] = None
```

或为了避免影响旧逻辑，新增多模态样本包装类也可以：

```python
MultimodalEnvironmentSample(EnvironmentSample)
```

但更简单的是给 `EnvironmentSample` 加可选字段。

新增函数：

```python
def render_bev_image(
    self,
    network: UAVNetwork,
    state: dict,
    save_path: str,
    image_size: int = 512,
) -> str:
    ...
```

图像应包含：

```text
UAV positions: triangle / blue marker
user positions: small dots / green
target positions: cross or star / red
optional association lines: thin gray lines
optional UAV coverage circles: translucent radius
axis range: [0, 1000] x [0, 1000]
legend: preferably off or minimal, because model sees image not human caption
```

建议输出：

```text
/root/autodl-tmp/data/full_mm/images/env_000001.png
```

### 注意

BEV image 的设计要稳定、简洁、信息密度高。

不要做太花的图：

```text
不要渐变背景
不要复杂装饰
不要过多文字标签
不要把表格数字塞进图像
```

图像主要表达空间几何。

---

## 4.2 `src/data/prompt_builder.py`

### 当前状态

当前 `build_full_prompt()` 拼接：

```python
parts.append(build_system_prompt(config))
parts.append(build_communication_summary_str(env_sample.comm_summary))
parts.append(build_sensing_summary_str(env_sample.sensing_summary))
parts.append(env_sample.bev_grid_text)
parts.append("Now propose ...")
```

### 需要新增

保留旧函数：

```python
build_full_prompt()
```

新增：

```python
def build_multimodal_prompt(env_sample, config: dict) -> str:
    parts = []
    parts.append(build_system_prompt(config))
    parts.append(build_communication_summary_str(env_sample.comm_summary))
    parts.append(build_sensing_summary_str(env_sample.sensing_summary))
    parts.append(
        "[Bird's-Eye-View Image]\n"
        "The attached BEV image encodes the spatial geometry of UAVs, users, "
        "and sensing targets over the service area. Use the image together "
        "with the communication and sensing summaries to infer coverage holes, "
        "load imbalance, target proximity, and movement directions."
    )
    parts.append("\nNow propose the warm-start decision prior delta in JSON format.")
    return "\n\n".join(parts)
```

如果选用的 multimodal processor 要求显式 image placeholder，则由该模型规范决定，例如：

```text
<image>
```

或 chat template 中的：

```python
{"type": "image"}
{"type": "text", "text": prompt}
```

不要在 `build_multimodal_prompt()` 中硬编码不确定的 image token，除非已验证对应模型 API。

---

## 4.3 数据生成脚本

### 当前状态

`scripts/generate_data.py` 生成：

```text
sft_dataset.jsonl
dpo_dataset.jsonl
```

每条样本包含：

```json
{
  "id": "...",
  "prompt": "...",
  "response": "...",
  "utility": ...,
  "q_current": ...,
  "delta_q": ...,
  "delta_a": ...,
  "delta_p": ...
}
```

### 需要新增

新增多模态数据输出目录：

```text
/root/autodl-tmp/data/full_mm/
  sft_dataset.jsonl
  dpo_dataset.jsonl
  images/
    env_000000.png
    env_000001.png
```

JSONL 新增字段：

```json
{
  "bev_image_path": "images/env_000001.png",
  "prompt_type": "multimodal_bev_image"
}
```

推荐保留：

```json
"bev_grid_text": "..."
```

但在 multimodal training 中不使用它，只用于 debug 或 ablation。

### 建议脚本策略

不要覆盖原 `full5000` 数据。

新增：

```text
scripts/generate_multimodal_data.py
```

或在 `generate_data.py` 加配置开关：

```yaml
data:
  modalities:
    use_bev_text_grid: false
    use_bev_image: true
  bev_image_dir: "images"
```

更推荐一开始新建脚本，降低破坏旧数据流程的风险。

---

## 4.4 `src/data/multimodal_dataset.py`

### 目标

替代当前 text-only `SFTDataset` / `DPODataset` 的输入编码部分。

当前 text-only：

```text
prompt -> tokenizer -> input_ids / attention_mask / labels
```

多模态：

```text
prompt + image -> processor -> input_ids / attention_mask / pixel_values / image metadata / labels
```

### 新增类

```python
class MultimodalSFTDataset(Dataset):
    ...

class MultimodalDPODataset(Dataset):
    ...
```

### batch 字段

SFT batch 至少包含：

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...,
    "label_mask": ...,
    "control_mask": ...,
    "pixel_values": ...,
    "q_current": ...,
    "delta_q_target": ...,
    "delta_a_target": ...,
    "delta_p_target": ...,
}
```

DPO batch 至少包含：

```python
{
    "input_ids_chosen": ...,
    "attention_mask_chosen": ...,
    "labels_chosen": ...,
    "label_mask_chosen": ...,
    "control_mask_chosen": ...,

    "input_ids_rejected": ...,
    "attention_mask_rejected": ...,
    "labels_rejected": ...,
    "label_mask_rejected": ...,
    "control_mask_rejected": ...,

    "pixel_values": ...,
    "q_current": ...,
    "delta_q_target": ...,
    "delta_a_target": ...,
    "delta_p_target": ...,
}
```

具体 image metadata 字段取决于模型 processor，例如可能包括：

```text
pixel_values
image_grid_thw
```

### 关键风险

control token mask 必须对齐。

加入 image tokens 后：

```text
input_ids 中的 control token 位置可能不再等同于纯文本 token 位置。
```

必须通过 tokenizer/processor 输出后的 `input_ids` 查找 `<ctrl_i>` token id 来构造 `control_mask`，不要用字符偏移硬猜。

---

## 4.5 `src/model/gemma_multimodal_isac.py`

### 当前 text-grid model

当前 `Gemma3ISAC` 使用：

```python
AutoModelForCausalLM
AutoTokenizer
```

### 新增 multimodal model wrapper

新增：

```python
class Gemma3MultimodalISAC(nn.Module):
    ...
```

职责：

1. 加载 vision-capable Gemma3 / multimodal backbone。
2. 加载 processor。
3. 注入/扩展 control tokens。
4. 执行 text + image forward。
5. 从 hidden states 中读取 control token hidden states。
6. 输入 projection head。
7. 输出：

```python
{
    "logits": logits,
    "delta_q": delta_q,
    "delta_a": delta_a,
    "delta_p": delta_p,
    "control_hidden": control_hidden,
}
```

### 模型加载建议

候选接口可能是：

```python
AutoProcessor.from_pretrained(...)
AutoModelForImageTextToText.from_pretrained(...)
```

或对应 Gemma3 multimodal 类。

具体名称以服务器 transformers 版本实际支持为准。

### 参数建议

初始版本：

```yaml
use_4bit: true 或 false 视 96GB smoke 情况
freeze_vision_tower: true
lora_target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
projection_head.hidden_dim: 2560
num_control_tokens: 8
```

先冻结 vision tower，训练：

```text
LoRA
control token embedding
projection head
```

不要一开始全量解冻 vision tower。

### save/load

保持与 `Gemma3ISAC.save_pretrained()` 类似：

```text
lora/
processor/
tokenizer/ or processor/
projection_head.pt
ctrl_embed.pt
ctrl_lm_head.pt
```

---

## 4.6 `src/training/train_sft_mm.py`

### 目标

先只做 multimodal SFT，不急着 multimodal DPO。

原因：

```text
multimodal DPO = policy + reference + chosen/rejected + images
显存和调试难度显著高于 SFT
```

### 训练流程

复用当前 SFT 逻辑：

```text
Phase 1: CTL-only warmup
Phase 2: SFT + CTL + SEP
```

但 dataset/model 换成 multimodal：

```python
MultimodalSFTDataset
Gemma3MultimodalISAC
```

### 初始 smoke 配置

建议：

```text
samples: 20-50
steps: 10-30
batch: 1
grad_accum: 8 or 16
max_seq_length: 1536 or 2048
image_size: 224 or 336
freeze_vision_tower: true
```

观察：

```text
是否 OOM
loss_ctl 是否下降
sensitivity 是否从接近 0 上升
control token hidden 是否非空
projection head 输出 shape 是否正确
```

---

## 4.7 `src/eval/evaluate_mm.py`

### 目标

多模态模型评估流程：

```text
读取测试环境
构造 multimodal prompt
读取 BEV image
processor(text + image)
model generate / forward
projection head 输出 warm-start
SCA-FP solving
记录指标
```

### 复用

复用当前 `evaluate.py` 的：

```text
metric aggregation
SCA-FP solving
cold/warm iteration comparison
JSON output format
```

新增：

```text
image loading
multimodal processor
Gemma3MultimodalISAC loading
```

### 输出命名

建议：

```text
/root/autodl-tmp/outputs/eval_mm_sft_20.json
/root/autodl-tmp/outputs/eval_mm_sft_200.json
```

---

## 4.8 `configs/pro6000_multimodal.yaml`

新增配置。

建议初版：

```yaml
hardware:
  gpu: "RTX PRO 6000"
  vram_gb: 96
  use_4bit: true
  gradient_checkpointing: true
  max_grad_norm: 1.0

model:
  backbone: "google/gemma-3-4b-it"
  use_multimodal: true
  freeze_vision_tower: true
  torch_dtype: "bfloat16"
  attn_implementation: "sdpa"

  lora:
    rank: 16
    alpha: 32
    dropout: 0
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"

  control_token:
    num_tokens: 8
    hidden_dim: 2560

  projection_head:
    hidden_dim: 2560
    mlp_hidden: [256, 256]
    readout_out_dim: 128
    tau_power: 0.5
    tau_assoc: 0.5
    sinkhorn_iters: 20

data:
  output_dir: "/root/autodl-tmp/data/full_mm"
  sft_file: "sft_dataset.jsonl"
  dpo_file: "dpo_dataset.jsonl"
  bev_image_dir: "images"
  modalities:
    use_comm_summary: true
    use_sensing_summary: true
    use_bev_text_grid: false
    use_bev_image: true

training:
  sft:
    epochs: 1
    per_device_batch_size: 1
    gradient_accumulation_steps: 8
    learning_rate: 2.0e-4
    max_seq_length: 2048
    save_steps: 50
    phase1:
      enabled: true
      max_steps: 100
      sensitivity_check_steps: 20
      sensitivity_threshold: 0.1
```

实际 batch/seq/image size 要以 smoke 结果调整。

---

## 5. 实施顺序

推荐不要一次性大改，按下面阶段推进。

### Phase A: 数据图像化

目标：

```text
能为每个 sample 保存 BEV image
JSONL 中有 bev_image_path
旧 text-grid 数据不受影响
```

任务：

1. 新增 BEV image renderer。
2. 新增或扩展 generate script。
3. 生成 20 条 multimodal smoke 数据。
4. 人工抽查 5 张 BEV image。

验收：

```text
images/*.png 存在
JSONL bev_image_path 正确
图片能打开
UAV/user/target 空间位置可辨认
```

### Phase B: Multimodal Dataset Smoke

目标：

```text
processor 能吃 text + image
batch 字段齐全
control token mask 正确
```

任务：

1. 新增 `build_multimodal_prompt()`。
2. 新增 `MultimodalSFTDataset`。
3. 写一个小脚本或 notebook 检查单 batch。

验收：

```text
input_ids shape 正常
pixel_values shape 正常
labels shape 正常
control_mask 至少包含 8 个 control tokens
delta targets shape 正常
```

### Phase C: Model Forward Smoke

目标：

```text
Gemma3MultimodalISAC 单 batch forward 成功
projection head 输出 shape 正确
```

任务：

1. 新增 model wrapper。
2. 加载 processor/model。
3. 单 batch forward。
4. 检查 hidden states 和 control token readout。

验收：

```text
logits: [B, L, vocab]
delta_q: [B, M, 3]
delta_a: [B, M, K]
delta_p: [B, M, K+1]
无 OOM
无 NaN
```

### Phase D: Multimodal SFT Smoke

目标：

```text
训练能跑 10-30 steps
loss_ctl 有下降趋势
```

任务：

1. 新增 `train_sft_mm.py`。
2. 使用 20-50 样本 smoke 数据。
3. freeze vision tower。
4. batch=1。

验收：

```text
无 OOM
loss_ctl 不是 NaN
grad_norm_lora_total 有值
checkpoint 能保存
```

### Phase E: Multimodal SFT 小规模评估

目标：

```text
比较 multimodal SFT vs text-grid SFT-only
```

任务：

1. 新增 `evaluate_mm.py`。
2. 跑 20 样本 eval。
3. 若正常，跑 100/200 样本 eval。

验收：

```text
sca_fp_speedup 不低于 text-grid 太多
joint_satisfaction 有正向趋势
inference latency 可接受
```

### Phase F: 是否做 Multimodal DPO

只有在 multimodal SFT 有正向结果后再考虑。

不建议一开始做 multimodal DPO。

触发条件：

```text
multimodal SFT speedup >= text-grid SFT-only
或 joint_satisfaction / sensing SINR 有明显提升
显存仍有余量
```

---

## 6. 风险清单

### 风险 0: delta head 退化为常数

历史现象：

```text
SFT / DPO 输出的三个 delta 中，有两个趋于常数。
DPO 与 SFT 的部分 delta 数值几乎相同。
尤其是 delta_a / delta_p 曾出现退化为常数均值的情况。
```

旧文档中已有相关判断：

```text
Masked DPO 下 δ_q/δ_a/δ_p 共享 control token embedding。
δ_q 独占偏好梯度后，δ_a/δ_p 主要依赖 MSE/CTL 信号，容易退化为常数均值。
```

典型退化表现：

```text
delta_a mean 接近固定值，例如 0.25
delta_p mean 接近固定值，例如 0.05
不同环境输入下 delta_a / delta_p 方差很小
SFT 与 DPO 的 delta_a / delta_p 几乎一致
最终 SCA-FP speedup 有提升，但主要来自 delta_q 或 solver 自身修正
```

这说明当前框架虽然能提升 solver convergence，但模型可能没有充分学到：

```text
环境相关的 association policy
环境相关的 power allocation prior
```

对多模态 MLLM 的影响：

```text
如果 BEV image 分支只提升空间理解，但 delta_a / delta_p head 仍然常数化，
那么多模态收益会被限制在 UAV movement / delta_q 上，
无法充分体现 MLLM 对 association 和 power allocation 的贡献。
```

因此，多模态版本必须把 delta 多样性作为验收指标之一，而不能只看最终 speedup。

必须新增诊断：

```text
1. 对同一批 eval samples 保存 delta_q / delta_a / delta_p。
2. 统计每个 head 的跨样本 mean / std / min / max。
3. 比较 SFT vs DPO 的 delta 差异。
4. 比较 text-grid vs BEV-image MLLM 的 delta 差异。
5. 检查 delta_a 每个用户的 argmax UAV 是否随环境变化。
6. 检查 delta_p 的 per-UAV power split 是否随环境变化。
```

建议新增脚本：

```text
scripts/analyze_delta_outputs.py
```

当前已补充实现：

```text
scripts/analyze_delta_outputs.py
```

该脚本只运行模型 forward / warm-start generation，不运行 SCA-FP，因此成本远低于完整 evaluation。

推荐在服务器上运行：

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

输入：

```text
model checkpoint
eval data/config
num_samples
```

输出：

```text
delta_q_std
delta_a_std
delta_p_std
SFT_vs_DPO_delta_l2
per-user association entropy
per-UAV power allocation entropy
constant-head warning
```

验收阈值建议：

```text
delta_q: 应有明显跨样本方差
delta_a: 不应所有用户长期 argmax 到固定 UAV 或固定均匀分布
delta_p: 不应所有 UAV 长期输出同一 power split
SFT vs DPO: 至少 delta_q 或 delta_a/delta_p 中一项应有可解释差异
```

如果发现 multimodal SFT 后 `delta_a / delta_p` 仍常数化，优先排查：

```text
projection head 是否过强地投影到均匀解
delta_a / delta_p 标签本身是否方差不足
control token readout 是否只捕获 delta_q 信息
loss 权重 lambda_a / lambda_p 是否太弱
DPO pair 是否只扰动 delta_q，导致 preference signal 不覆盖 delta_a / delta_p
```

可能改进方向：

```text
1. 分离 control token block:
   <ctrl_q_*> 用于 delta_q
   <ctrl_a_*> 用于 delta_a
   <ctrl_p_*> 用于 delta_p

2. 分离 projection head:
   q_head / a_head / p_head 分别读取不同 control states。

3. 改造 DPO pair:
   不只构造 delta_q rejected，也构造 association/power rejected。

4. 增加 delta diversity regularization:
   鼓励同 batch 不同环境的 delta_a / delta_p 有非零方差。

5. 检查 oracle extraction:
   如果 oracle delta_a / delta_p 本身接近常数，模型学成常数是数据问题，不是模型问题。
```

优先级：

```text
高。该问题直接决定 MLLM 是否真的学到了环境相关控制策略。
```

### 风险 1: Gemma3 多模态 API 不稳定

可能问题：

```text
transformers 版本不支持对应 multimodal class
processor 字段与预期不一致
custom_generate 远程文件检查导致网络错误
```

缓解：

```text
优先使用服务器已验证 transformers 版本
模型下载后使用本地 snapshot + offline 模式评估
先写最小 processor smoke
```

### 风险 2: control token 位置错位

图像 tokens 会改变 sequence layout。

缓解：

```text
通过 token id 搜索 <ctrl_i>
不要用字符 offset 硬算 control token 位置
每个 batch assert control tokens 数量正确
```

### 风险 3: 图像没有提供额外有效信息

如果 BEV image 太粗糙，模型可能不如 text-grid。

缓解：

```text
设计稳定、清晰、低噪声 BEV image
保留 c(t), r(t) 文本摘要
加入 no-image ablation
```

### 风险 4: 显存压力高

multimodal SFT 比 text-only 更吃显存。

缓解：

```text
使用 RTX PRO 6000 96GB
batch=1
freeze vision tower
4-bit QLoRA
max_seq_length 从 1536/2048 起步
image_size 从 224/336 起步
```

### 风险 5: 评估时间长

200 样本 eval 中 SCA-FP solving 约 25 分钟，multimodal 推理可能更慢。

缓解：

```text
先 20 样本 smoke eval
再 100/200 样本正式 eval
保留 JSON 结果
```

---

## 7. 论文实验结构建议

最终论文可以组织为：

| Method | 输入 | 训练 | 定位 |
|---|---|---|---|
| Cold SCA-FP | none | none | optimization baseline |
| Text-grid SFT | c(t), r(t), text BEV | SFT | textualized-BEV baseline |
| Text-grid SFT+DPO | c(t), r(t), text BEV | SFT + DPO-2k | training-stage ablation |
| BEV-image MLLM SFT | c(t), r(t), BEV image | multimodal SFT | proposed method |
| BEV-image MLLM SFT+DPO | c(t), r(t), BEV image | optional | optional enhanced method |
| No projection head | same as proposed | SFT | feasibility ablation |

当前已完成结果：

```text
Text-grid SFT 200
Text-grid SFT+DPO-2k 200
```

下一阶段目标：

```text
BEV-image MLLM SFT 20-sample smoke
BEV-image MLLM SFT 200-sample evaluation
```

---

## 8. 是否需要重写关键模块

| 模块 | 是否重写 | 说明 |
|---|---|---|
| `src/solver/sca_fp.py` | 否 | 接收数值 warm-start，不关心来源 |
| `src/model/projection_head.py` | 否，最多小改 | 若 hidden size 不变可直接复用 |
| `src/model/losses.py` | 否 | loss 仍作用于 delta 与 log-prob |
| `src/data/prompt_builder.py` | 新增函数 | 保留 `build_full_prompt`，新增 `build_multimodal_prompt` |
| `src/data/dataset.py` | 不建议直接改 | 新增 `multimodal_dataset.py` |
| `src/model/gemma_isac.py` | 不建议直接改 | 新增 `gemma_multimodal_isac.py` |
| `src/training/train_sft.py` | 不建议直接改 | 新增 `train_sft_mm.py` |
| `src/training/train_dpo.py` | 暂不改 | 多模态 DPO 后置 |
| `src/eval/evaluate.py` | 不建议直接改 | 新增 `evaluate_mm.py` |

---

## 9. 推荐下一步具体任务

### Task 1: 新增 09 计划文档

当前文档即为 Task 1 输出。

### Task 2: BEV image renderer

新增：

```text
src/env/bev_renderer.py
```

或在 `isac_scenario.py` 中新增 `render_bev_image()`。

推荐单独文件：

```text
src/env/bev_renderer.py
```

原因：

```text
渲染逻辑独立，便于调试和复用
不污染 scenario generator
```

### Task 3: 生成 20 条 multimodal smoke 数据

新增：

```text
scripts/generate_mm_smoke.py
```

输出：

```text
/root/autodl-tmp/data/mm_smoke/sft_dataset.jsonl
/root/autodl-tmp/data/mm_smoke/images/*.png
```

### Task 4: Multimodal processor smoke

新增：

```text
scripts/smoke_mm_processor.py
```

检查：

```text
processor(text + image) 成功
input_ids / pixel_values shape 正常
control tokens 可定位
```

### Task 5: Model forward smoke

新增：

```text
scripts/smoke_mm_forward.py
```

检查：

```text
forward 无 OOM
delta_q/delta_a/delta_p shape 正确
```

### Task 6: Multimodal SFT smoke

新增：

```text
src/training/train_sft_mm.py
configs/pro6000_multimodal.yaml
```

先跑：

```text
10-30 steps
20-50 samples
```

---

## 10. 最终结论

当前框架仍然可用，而且是后续 MLLM 的核心骨架。

不需要重写：

```text
projection head
SCA-FP solver
loss framework
warm-start evaluation
```

必须新增：

```text
BEV image 数据
multimodal prompt
multimodal dataset
multimodal model wrapper
multimodal SFT/eval
```

建议路线：

```text
保留当前 text-grid SFT/DPO 作为 baseline。
新增 BEV-image MLLM 分支作为 proposed method。
先做 multimodal SFT，不急着 multimodal DPO。
```

一句话：

```text
下一阶段不是推翻当前项目，而是把输入编码器从 text-only 升级为 text+image，同时复用已经跑通的 optimizer-aware warm-start 框架。
```
