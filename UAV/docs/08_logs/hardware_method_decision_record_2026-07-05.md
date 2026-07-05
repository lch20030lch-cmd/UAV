# 硬件与方法路线决策记录

> 日期: 2026-07-05  
> 背景: 论文标题为 `Constraint-Aware MLLM Adaptation for UAV-Enabled ISAC in Low-Altitude IoT Networks`。  
> 当前训练: RTX 5090 32GB 上正在跑 Gemma3-4B text-grid SFT。  
> 目标: 记录当前路线与若切换到 RTX PRO 6000 96GB 后的多模态路线改动建议。

---

## 0. 当前共同前提

导师已确认将论文中的 Gemma 3-12B 调整为 Gemma 3-4B。

当前项目核心方法仍保持:

- Gemma3-4B backbone
- LoRA / QLoRA
- control token hidden states
- constraint projection head
- SFT + DPO 两阶段训练
- SCA-FP warm-start evaluation

已经确认的关键训练信号:

```text
Phase 1:
step 50:  loss_ctl=35.75, sens=0.0011
step 100: loss_ctl=28.13, sens=0.0067
step 150: loss_ctl=14.69, sens=0.2681
```

说明:

- control loss 明显下降。
- sensitivity 超过阈值 0.1。
- control token 表征已经从“近似恒定输出”进入“环境敏感”状态。
- 当前 SFT 不建议中断。

---

# 1. 维持现状: RTX 5090 32GB + Gemma3-4B Text-Grid 路线

## 1.1 当前方法

当前实际方法更准确地说是:

```text
Constraint-aware Gemma3-4B text-grid adaptation
```

而不是严格的视觉多模态 MLLM。

输入模态:

- communication summary: 文本
- sensing summary: 文本
- BEV map: 文本化 grid (`bev_grid_text`)
- task instruction: 文本

模型链路:

```text
text prompt
  -> Gemma3-4B CausalLM
  -> control token hidden states
  -> ConstraintProjectionHead
  -> delta_q / delta_a / delta_p
  -> SCA-FP warm-start
```

当前 BEV 是文本形式，不是真实图像输入。

## 1.2 当前硬件配置

当前硬件:

```text
GPU: RTX 5090
VRAM: 32GB
Precision: bitsandbytes NF4 4-bit QLoRA
Backbone: Gemma3-4B
```

当前 `configs/rtx5090.yaml` 已按 smoke test 修正:

```yaml
training:
  sft:
    per_device_batch_size: 1
    gradient_accumulation_steps: 16
    max_seq_length: 3456

  dpo:
    per_device_batch_size: 1
    gradient_accumulation_steps: 16
    max_seq_length: 1536
    mu: 0.0
```

实测结论:

- SFT `bs=2, seq=3456` 会 OOM。
- SFT `bs=1, seq=3456` 可跑。
- DPO `seq=3456` 会 OOM。
- DPO `seq=2048` 仍会 OOM。
- DPO `seq=1536, mu=0.0` 可跑。

## 1.3 维持现状的优点

1. 当前训练已经跑起来，不浪费已有 SFT 成本。
2. 32GB 成本较低，训练预算可控。
3. 代码主链路稳定性较高。
4. text-grid 版本可以作为 baseline 或 ablation。
5. 不需要马上重构 Dataset / Processor / Model forward。

## 1.4 维持现状的主要问题

### 问题 A: 与论文标题中的 MLLM 有出入

论文标题为:

```text
Constraint-Aware MLLM Adaptation for UAV-Enabled ISAC in Low-Altitude IoT Networks
```

但当前实现不是严格视觉多模态 MLLM，而是 text-grid / structured text prompt。

风险:

- 如果论文继续强调视觉多模态，而实验只有 text-grid，会被认为题文不一致。

### 问题 B: 数据生成和评估的 solver config 需要对齐

当前数据生成中 solver 参数和评估脚本中的 solver 参数不完全一致。

风险:

- teacher/evaluator 不在同一个物理世界中。
- 最终 speedup 和 utility 可能被系统性偏差污染。

### 问题 C: DPO 已经为显存做了工程妥协

当前 DPO 必须:

```yaml
max_seq_length: 1536
mu: 0.0
```

这与论文中通用公式的 `mu * L_SFT` anchor 有差异。

## 1.5 维持现状时的修改建议

### 建议 1: 不要中断当前 SFT

当前 Phase 1 结果健康，建议继续跑完 Stage I SFT。

是否需要重跑 SFT:

```text
否
```

### 建议 2: 将当前模型定位为 text-grid baseline

论文中不要把当前版本硬写成完整视觉 MLLM。

建议命名:

```text
Text-grid Gemma3-4B baseline
Textualized-BEV baseline
Single-modal structured-state baseline
```

是否需要重跑 SFT:

```text
否
```

### 建议 3: 修改论文中 Gemma3-12B 为 Gemma3-4B

需要修改:

- Abstract
- Introduction
- Simulation Setup
- Baseline description
- Experiment configuration

建议说明:

```text
Gemma3-4B is used to support memory-feasible QLoRA SFT and DPO on commodity GPU platforms while preserving the same optimizer-aware training interface.
```

是否需要重跑 SFT:

```text
否
```

### 建议 4: 修正 `requirements.txt`

当前 `requirements.txt` 中 `transformers==4.49.0` 已知不支持 `gemma3`。

建议改为服务器实测可用版本，例如:

```txt
transformers>=4.53.0
```

是否需要重跑 SFT:

```text
否
```

### 建议 5: SFT 完成后优先修评估 solver config

建议新增统一 YAML 配置:

```yaml
solver:
  max_iters: 100
  max_outer_iters: 30
  max_inner_iters: 50
  tol: 1.0e-4
  lambda_sensing: 0.5
  lambda_idle_penalty: 0.0
  ground_clutter_db: 6.0
  lambda_repel: 0.01
```

然后让以下脚本统一读取:

- `scripts/generate_data.py`
- `src/eval/evaluate.py`
- `scripts/calibrate_epsilon.py`

短期最稳做法:

```text
先把 evaluate.py 对齐到当前已生成数据的 solver 参数，不重新生成数据。
```

是否需要重跑 SFT:

```text
如果只改 evaluate.py: 否
如果重新生成数据: 是
```

### 建议 6: DPO 维持 smoke test 配置

DPO 前确认:

```yaml
dpo:
  per_device_batch_size: 1
  gradient_accumulation_steps: 16
  max_seq_length: 1536
  mu: 0.0
```

论文中解释:

```text
On the RTX 5090 32GB platform, the SFT anchor is disabled during DPO for memory feasibility, while the continuous control loss is retained.
```

是否需要重跑 SFT:

```text
否
```

## 1.6 维持现状路线的建议结论

如果经费紧、时间紧:

```text
维持 RTX 5090 text-grid 路线，跑完 SFT + DPO，将其作为 baseline 或 engineering-feasible variant。
```

但如果论文必须突出 MLLM:

```text
该路线不足以单独支撑标题中的 MLLM，需要后续补真实图像 BEV 分支。
```

---

# 2. 切换路线: RTX PRO 6000 96GB + 真多模态 MLLM

## 2.1 目标方法

切换到 RTX PRO 6000 96GB 后，可以尝试真正多模态方案。

目标方法:

```text
Constraint-aware Gemma3-4B multimodal adaptation
```

输入模态:

- communication summary: 文本
- sensing summary: 文本
- BEV spatial map: 图像
- task instruction: 文本

模型链路:

```text
text prompt + BEV image
  -> Gemma3 multimodal processor/model
  -> control token hidden states
  -> ConstraintProjectionHead
  -> delta_q / delta_a / delta_p
  -> SCA-FP warm-start
```

这条路线更符合论文标题中的:

```text
MLLM Adaptation
```

## 2.2 目标硬件配置

目标硬件:

```text
GPU: RTX PRO 6000
VRAM: 96GB
Backbone: Gemma3-4B multimodal / vision-capable variant
Precision: bf16 LoRA 或 4-bit QLoRA
```

推荐优先级:

1. 多模态 SFT 先试 bf16 LoRA。
2. 如果 DPO OOM，再切 4-bit QLoRA。
3. 一开始冻结 vision tower，只训练 LoRA、projector、projection head。

## 2.3 96GB 路线的可行性判断

| 阶段 | 可行性 | 说明 |
|---|---:|---|
| 多模态 SFT | 高 | 96GB 基本能支撑 Gemma3-4B + 图像输入 |
| 多模态 DPO | 中高 | 可尝试，但仍需控制 seq length 和 batch |
| 解冻 vision tower | 中 | 可以后续尝试，不建议一开始全解冻 |
| 完整 B1-B9 实验 | 中 | 工程量较大，但硬件不再是主要瓶颈 |

## 2.4 推荐训练配置

先做 smoke test:

```yaml
training:
  sft:
    per_device_batch_size: 1
    gradient_accumulation_steps: 16
    max_seq_length: 3456

  dpo:
    per_device_batch_size: 1
    gradient_accumulation_steps: 16
    max_seq_length: 2048
    mu: 0.05
```

如果 SFT 显存宽松，再尝试:

```yaml
sft:
  per_device_batch_size: 2
  gradient_accumulation_steps: 8
```

如果 DPO 显存宽松，再尝试:

```yaml
dpo:
  max_seq_length: 3456
```

## 2.5 需要修改的代码部分

### 修改 1: 数据生成增加 BEV 图像

当前数据只有文本 BEV grid。  
需要新增 BEV 图像渲染与保存。

建议新增字段:

```json
{
  "prompt": "...",
  "bev_image": "images/env_000123.png",
  "response": "...",
  "q_current": [...]
}
```

涉及文件:

- `src/env/isac_scenario.py`
- `src/data/oracle_generator.py`
- `scripts/generate_data.py`

建议:

- 保留原 `bev_grid_text`。
- 新增 `bev_image_path`。
- 图像尺寸固定，例如 `224x224` 或 `336x336`。
- image 文件放在数据目录下 `images/`。

### 修改 2: Dataset 支持图像输入

当前 Dataset 返回:

```python
input_ids
attention_mask
labels
control_mask
q_current
delta_q_target
delta_a_target
delta_p_target
```

多模态 Dataset 需要额外返回:

```python
pixel_values
image metadata
```

具体字段取决于 Gemma3 multimodal processor。

建议:

- 不要直接破坏现有 `SFTDataset` / `DPODataset`。
- 新增:

```text
src/data/multimodal_dataset.py
```

### 修改 3: 新增多模态模型类

当前:

```text
src/model/gemma_isac.py
```

用于 text-grid baseline。

建议新增:

```text
src/model/gemma_multimodal_isac.py
```

不要直接覆盖现有主线。

新模型需要处理:

- multimodal processor
- text + image inputs
- control token 添加与定位
- hidden states 提取
- projection head 输出
- checkpoint 保存与加载

关键风险:

```text
必须确认 image tokens 插入后 control_mask 仍然对齐 control token hidden states。
```

### 修改 4: 训练脚本支持多模态 batch

可以选择两种方式:

1. 新增训练脚本:

```text
src/training/train_sft_mm.py
src/training/train_dpo_mm.py
```

2. 或在原脚本中加 `use_multimodal` 分支。

推荐方案:

```text
新增脚本，避免破坏当前 5090 text-grid 训练主线。
```

模型调用从:

```python
model(input_ids, attention_mask, control_mask, q_current)
```

变成:

```python
model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    control_mask=control_mask,
    q_current=q_current,
)
```

### 修改 5: 评估脚本支持图像输入

当前 `evaluate.py` 只重建 prompt。

多模态后需要:

- 生成或读取 BEV image。
- processor 同时处理 text 和 image。
- `generate_warmstart` 支持 image input。

建议新增:

```text
src/eval/evaluate_mm.py
```

或在 `evaluate.py` 中加 `use_multimodal` 分支。

### 修改 6: 新增 96GB 多模态配置文件

建议新增:

```text
configs/pro6000_multimodal.yaml
```

示例:

```yaml
hardware:
  gpu: "RTX PRO 6000"
  vram_gb: 96
  use_4bit: false

model:
  use_multimodal: true
  backbone: "google/gemma-3-4b-it"
  freeze_vision_tower: true
  torch_dtype: "bfloat16"
  attn_implementation: "sdpa"

data:
  modalities:
    use_comm_summary: true
    use_sensing_summary: true
    use_bev_text_grid: true
    use_bev_image: true
  bev_image_dir: "/root/autodl-tmp/data/full_v2/images"
```

### 修改 7: checkpoint 保存 processor

当前保存 tokenizer。  
多模态需要保存:

- processor
- tokenizer
- LoRA adapter
- control token embedding
- projection head
- optional image projector

---

## 2.6 96GB 路线烟测必须重做

切换到真正多模态输入后，烟测必须重做。

原因:

- 输入结构变了。
- Dataset batch 变了。
- processor 变了。
- forward 路径变了。
- control token 位置可能变。
- 显存峰值变了。
- DPO 四次 forward 成本变了。

推荐 smoke test 顺序:

### Smoke 1: 数据烟测

- 生成 5-10 条样本。
- 确认 JSONL 有 `bev_image_path`。
- 图像能打开。
- 图像尺寸固定。

### Smoke 2: Processor 烟测

- 单条 prompt + image 能 processor。
- 输出包含 expected fields。
- 确认 `<ctrl_i>` 不被丢失。

### Smoke 3: 模型加载烟测

- 96GB 上加载多模态 Gemma3-4B。
- 测试 bf16 或 4-bit。
- 单条 forward 不 OOM。

### Smoke 4: Control token 位置烟测

必须确认:

- control token 在 input_ids 中存在。
- control_mask 和 hidden states 对齐。
- image tokens 不破坏 control token 读取。

### Smoke 5: SFT 单 batch

检查:

- forward
- loss_sft
- loss_ctl
- backward
- 显存峰值

### Smoke 6: 小样本 overfit

建议:

```text
10 samples, 50-100 steps
```

观察:

- `loss_ctl` 是否下降
- sensitivity 是否从 0 上升

### Smoke 7: DPO smoke

建议:

- 先 `seq=1024/1536`
- `bs=1`
- 再尝试 `seq=2048/3456`
- 先冻结 vision tower

---

## 2.7 96GB 路线的改动建议

推荐不要推翻当前主线，而是并行新增多模态分支:

```text
src/model/gemma_isac.py              # 保留 text-grid baseline
src/model/gemma_multimodal_isac.py   # 新增 multimodal version
src/data/dataset.py                  # 保留 text-grid dataset
src/data/multimodal_dataset.py       # 新增 multimodal dataset
src/training/train_sft.py            # 保留
src/training/train_sft_mm.py         # 新增
src/eval/evaluate.py                 # 保留
src/eval/evaluate_mm.py              # 新增
configs/rtx5090.yaml                 # 保留
configs/pro6000_multimodal.yaml      # 新增
```

原因:

- 当前 5090 SFT 不受影响。
- 当前 text-grid 结果可作为 baseline。
- 多模态路线可以独立 smoke 和 debug。
- 如果多模态失败，不会破坏已可跑主线。

---

## 2.8 96GB 路线是否需要重新训练

| 项目 | 是否需要重新训练 |
|---|---|
| 当前 text-grid SFT | 不需要，保留为 baseline |
| 多模态 SFT | 需要 |
| 多模态 DPO | 建议 SFT 有正向结果后再跑 |
| 只改论文 12B → 4B | 不需要 |
| 只改 evaluate solver config | 不需要 |
| 只新增 CapAssign 推理后处理 | 不需要 |

---

## 2.9 96GB 路线的建议结论

如果经费允许，并且论文必须突出 MLLM:

```text
建议使用 RTX PRO 6000 96GB 新增真正多模态分支。
```

但不要废弃当前 5090 text-grid 训练:

```text
当前 text-grid 版本应保留为 baseline / ablation。
```

最终论文结构建议:

| 方法 | 定位 |
|---|---|
| cold-start SCA-FP | 传统优化 baseline |
| text-grid Gemma3-4B | single-modal / textualized-BEV baseline |
| multimodal Gemma3-4B + BEV image | proposed method |
| no projection head | ablation |
| SFT-only vs SFT+DPO | training-stage ablation |

---

# 3. 两条路线对比

| 项目 | RTX 5090 32GB text-grid | RTX PRO 6000 96GB multimodal |
|---|---|---|
| 成本 | 低 | 高 |
| 当前代码改动 | 小 | 中大型 |
| 当前 SFT 是否继续有效 | 是 | 是，作为 baseline |
| 是否支撑 MLLM 标题 | 不充分 | 充分 |
| SFT 可行性 | 已在跑 | 高，但需新 smoke |
| DPO 可行性 | 可跑但配置保守 | 可尝试更完整配置 |
| 是否需要重新 smoke | 已完成 text smoke | 必须重做 multimodal smoke |
| 是否需要新数据字段 | 否 | 是，BEV image path |
| 是否需要新模型类 | 否 | 建议新增 |
| 是否需要重跑多模态 SFT | 不适用 | 是 |

---

# 4. 最终建议

## 如果当前目标是省钱和尽快出结果

选择:

```text
维持 RTX 5090 text-grid 路线。
```

行动:

1. 当前 SFT 跑完。
2. 修 evaluation solver config。
3. 跑 SFT-only eval。
4. 跑 DPO。
5. 将结果作为 text-grid baseline。

## 如果当前目标是让论文标题中的 MLLM 名副其实

选择:

```text
切到 RTX PRO 6000 96GB，新增 multimodal 分支。
```

行动:

1. 保留当前 text-grid 训练结果。
2. 新增 BEV image 数据生成。
3. 新增 multimodal dataset/model/training/eval。
4. 重新做 multimodal smoke。
5. 跑 multimodal SFT。
6. 根据显存和效果决定是否跑 multimodal DPO。

## 推荐折中路线

最稳路线:

```text
当前 5090 SFT 继续跑完 → 作为 text-grid baseline。
若经费允许，上 RTX PRO 6000 96GB 补 multimodal SFT 主方法。
DPO 先不要强行多模态化，等 multimodal SFT 有正向评估后再决定。
```

一句话总结:

```text
5090 路线能保住工程结果，96GB 路线能保住 MLLM 论文叙事。两者不是互斥关系，当前 text-grid 训练应保留为 baseline。
```

