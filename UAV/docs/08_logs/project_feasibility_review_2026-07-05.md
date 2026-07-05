# 项目整体可行性审核与问题清单

> 日期: 2026-07-05  
> 背景: Gemma 3-12B 已与导师确认调整为 Gemma 3-4B；当前正在 RTX 5090 32GB 上跑 Stage I SFT。  
> 当前训练信号: Phase 1 已在 step 150 自动切换到 Phase 2，`loss_ctl=14.69`，`sensitivity=0.2681 > 0.1`。

---

## 1. 总体判断

当前代码主链路已经从“方案设计可行”进入“工程基本可跑、效果等待评估验证”的阶段。

分层判断:

| 模块 | 可行度 | 判断 |
|---|---:|---|
| Stage I SFT | 高，约 80-85% | Phase 1 loss 明显下降，sensitivity 达标，control token 表征已打通 |
| Stage II DPO | 中高，约 65-75% | smoke test 已证明可跑，但 RTX 5090 32GB 下必须 `seq=1536, mu=0` |
| 论文主方法复现 | 中等，约 55-65% | 训练链路可跑，但当前实现还不是严格视觉多模态 MLLM |
| 最终论文实验完整度 | 中等 | B1-B9、ablation、robustness sweep 尚需补齐 |

当前最重要的结论:

- Gemma 3-4B 替代 Gemma 3-12B 是合理且必要的。
- 当前正在跑的 SFT 不建议中断。
- 只要不重新生成 SFT 数据、不改变训练标签定义、不切换真正图像多模态输入，当前 SFT 不需要重跑。
- 如果论文重点必须是“多模态”，当前 text-grid 版本不能作为唯一主方法证据，建议后续补真正 BEV image modality。

---

## 2. 当前 Phase 1 训练结果解读

用户提供的 Phase 1 日志:

```text
step 50:  loss_ctl=35.75, sens=0.0011
step 100: loss_ctl=28.13, sens=0.0067
step 150: loss_ctl=14.69, sens=0.2681
```

解读:

1. `loss_ctl` 从 35.75 降到 14.69，说明 projection head 和 LoRA 控制表征在学习连续 warm-start 变量。
2. `sensitivity` 从接近 0 跳到 0.2681，说明模型不再对不同环境输出同一个模板。
3. 这直接反证了旧版 mode collapse 问题，即 control token 完全不携带环境信息。
4. step 150 自动切换 Phase 2 是合理的。

建议:

- 当前 SFT 继续跑完。
- 重点观察 Phase 2 中 `loss_sft`、`loss_ctl`、`grad_norm_lora_total`。
- 如果 Phase 2 没有 NaN/OOM，先不要干预。

---

## 3. 已经修正的问题

### 3.1 RTX 5090 配置已对齐 smoke test

`configs/rtx5090.yaml` 已经从旧配置:

```yaml
sft:
  per_device_batch_size: 2
  gradient_accumulation_steps: 8
dpo:
  mu: 0.05
  max_seq_length: 3456
```

修正为 smoke test 实测可跑配置:

```yaml
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

影响:

- 当前 SFT 不需要重跑。
- DPO 前应再次确认服务器使用的是这份配置。

### 3.2 4-bit 路径已从 Unsloth 切到 bitsandbytes NF4

当前 `Gemma3ISAC` 已使用:

- `BitsAndBytesConfig(load_in_4bit=True)`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_compute_dtype=torch.bfloat16`
- `prepare_model_for_kbit_training`
- PEFT LoRA

影响:

- 与 RTX 5090 32GB 训练路线一致。
- 旧的 Unsloth + SDPA 冲突风险已基本移除。

### 3.3 q_current 数据链路已修复

Dataset 中已经有:

- `q_current`
- `has_q_current`

训练时 projection head 会收到当前 UAV 位置，从而预测相对位移 `delta_q`。

影响:

- 旧数据缺失 `q_current` 导致的 mode collapse 根因已修复。
- 当前 SFT 数据若来自新 `full_v2`，不需要因为这个问题重跑。

---

## 4. 主要问题与建议解决办法

### 问题 1: 论文重点是多模态，但当前实现主要是 text-grid

#### 现状

论文中核心叙事是 MLLM:

- communication summary
- sensing summary
- visual bird's-eye-view map
- multimodal backbone

但当前代码实际是:

- `AutoModelForCausalLM`
- `use_multimodal: false`
- BEV 使用 `env_sample.bev_grid_text`
- prompt 中拼接文本化地图

这更准确地说是:

```text
textualized BEV / structured text prompt / text-grid surrogate
```

而不是严格的视觉多模态 MLLM。

#### 风险

如果论文继续强写“视觉多模态 MLLM”，但实验实现只有 text-grid，导师或审稿人很容易发现口径不一致。

#### 建议方案

推荐方案: **保留当前 SFT 作为 text-only/text-grid baseline，后续补一个真正 BEV image modality 分支作为 proposed method。**

执行顺序:

1. 当前 SFT 继续跑完。
2. 将当前模型定位为 text-grid baseline。
3. 数据生成阶段额外保存 BEV 图像或 `bev_image_path`。
4. 增加轻量 image encoder，将 BEV image feature 融合进 projection head。
5. 训练 multimodal 版本。
6. 论文中报告:
   - text-only / text-grid baseline
   - image BEV multimodal proposed
   - no-image ablation
   - no-projection-head ablation

是否需要重跑当前 SFT:

- 当前这轮 SFT 不需要重跑，可作为 baseline。
- 如果新增真正图像多模态主方法，则多模态版本需要单独训练一轮 SFT。

不推荐现在完全切到 Gemma3 vision/multimodal processor，原因:

- 需要改 Dataset、processor、forward、模型加载。
- 32GB 显存压力显著增加。
- 当前训练会被打断，deadline 风险高。

---

### 问题 2: 数据生成和评估的 solver 配置不一致

#### 现状

数据生成中 `scripts/generate_data.py` 使用:

```python
ground_clutter_db=6.0
lambda_idle_penalty=0.0
lambda_repel=0.01
```

评估中 `src/eval/evaluate.py` 使用:

```python
lambda_idle_penalty=5.0
```

且没有显式传入 `ground_clutter_db`，因此会使用 `SCAFPConfig` 默认值:

```python
ground_clutter_db=12.0
```

#### 风险

Teacher、training data、evaluator 不是同一个物理世界。  
这会污染最终 speedup、sum-rate、satisfaction 等指标。

论文和项目文档中也明确强调:

```text
数据生成、训练评估、最终测试必须使用完全一致的 SCAFPConfig。
```

#### 建议方案

SFT 跑完后立即修。

建议不要重新生成数据，而是先让评估对齐当前已生成数据使用的 solver 配置:

```python
ground_clutter_db=6.0
lambda_idle_penalty=0.0
lambda_repel=0.01
```

更稳的长期方案:

1. 在 YAML 中新增统一 solver 配置:

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

2. `generate_data.py`、`evaluate.py`、`calibrate_epsilon.py` 都从同一段 YAML 读取。
3. 禁止脚本里硬编码 solver 物理参数。

是否需要重跑当前 SFT:

- 如果只是修改评估代码以对齐当前数据生成配置，不需要重跑 SFT。
- 如果决定修改数据生成物理参数并重新生成 `full_v2` 数据，则需要重跑 SFT。

建议:

```text
不要现在重生数据。先让 evaluate.py 对齐当前数据，然后用当前 SFT checkpoint 做 SFT-only eval。
```

---

### 问题 3: requirements.txt 中 transformers 版本过旧

#### 现状

`requirements.txt` 仍写:

```txt
transformers==4.49.0
```

但 RTX 5090 smoke test 已记录该版本不支持 `gemma3`:

```text
KeyError: 'gemma3'
```

服务器实际是升级 transformers 后才跑通。

#### 风险

新环境按 `requirements.txt` 重装会复现失败。

#### 建议方案

现在即可修改:

```txt
transformers>=4.53.0
```

或者钉住服务器实际成功版本。

是否需要重跑当前 SFT:

- 不需要。

---

### 问题 4: DPO 配置与论文公式存在工程偏差

#### 现状

论文写:

```text
L_II = L_DPO + μ L_SFT + λ_ctl L_ctl
```

但 RTX 5090 32GB smoke test 证明:

```yaml
mu: 0.0
max_seq_length: 1536
```

才可稳定跑 DPO。

#### 风险

如果论文中仍说 DPO 使用 `μ=0.05`，但实际实验用 `μ=0.0`，公式和实验不一致。

#### 建议方案

论文中保留通用公式，但实验设置写清楚:

```text
On the RTX 5090 32GB platform, we set μ=0 for memory-feasible DPO and retain the continuous control anchor L_ctl.
```

中文解释:

```text
受 32GB 显存限制，DPO 阶段关闭 token-level SFT anchor，但保留连续控制损失作为格式和控制变量的稳定项。
```

是否需要重跑当前 SFT:

- 不需要。

是否需要重跑 DPO:

- DPO 尚未正式跑全量时，直接用当前 `rtx5090.yaml` 即可。

---

### 问题 5: 当前关联离散化不是论文中的 CapAssign

#### 现状

Projection head 使用 Sinkhorn 产生软关联 `delta_a`。  
但 solver warm-start 中离散化为:

```python
best_m_per_k = np.argmax(delta_a, axis=0)
```

这不是论文 Algorithm 中的 capacitated minimum-cost flow / CapAssign。

#### 风险

argmax 可以保证每个用户只选一个 UAV，但不一定保证每个 UAV 的 load cap。  
后续 solver 会重新优化 association，但 warm-start 初始点可能违反容量约束。

#### 建议方案

放到 SFT 完成、DPO 前后均可处理。优先级低于 solver config 一致性。

建议实现:

- 输入: soft association score `delta_a`，shape `(M, K)`
- 目标: maximize score 或 minimize `-delta_a`
- 约束:
  - 每个 user 恰好分配给一个 UAV
  - 每个 UAV 最多 `K_max` 个 user
- 简单可用方案:
  - greedy with capacity
  - 或 scipy min-cost flow / linear_sum_assignment 扩展行复制

是否需要重跑当前 SFT:

- 不需要。

是否需要重跑 DPO:

- 如果只改 inference/evaluation 的离散化，不需要。
- 如果改训练 loss 或 projection target，才需要重跑。

---

### 问题 6: 评估脚本还不是完整论文 B1-B9 protocol

#### 现状

当前 `evaluate.py` 主要评估:

- model warm-start
- cold-start
- speedup
- sum-rate
- sensing SINR
- joint satisfaction

但论文中要求:

- B1 CSI-only
- B2 ISAC-SCA alternating
- B3 DRL-assisted
- B4 single-modal frozen
- B5 MoE baseline
- B6 frozen multimodal prompting
- B7 SFT-LoRA only
- B8 DPO directly on G0
- B9 SFT+DPO no projection head
- projection-head ablation
- data-pipeline ablation
- robustness sweeps

#### 风险

最终论文实验表不完整。

#### 建议方案

当前训练阶段不处理。  
SFT/DPO 结果出来后，再决定补哪些 baseline。

优先补最关键的:

1. cold-start / random restart baseline
2. SFT-only
3. SFT+DPO
4. no-projection-head
5. text-grid vs image-BEV

是否需要重跑当前 SFT:

- 不需要。

---

### 问题 7: 论文中 Gemma 3-12B 需要全局替换为 Gemma 3-4B

#### 现状

论文中仍多处写:

```text
Gemma 3 (12B)
```

但当前已与导师确认改为:

```text
Gemma 3 (4B)
```

#### 建议方案

现在即可改论文:

- Abstract
- Introduction
- Simulation Setup
- Baselines
- Conclusion 或实验讨论

建议增加一句合理性说明:

```text
We instantiate the backbone with Gemma 3 (4B) to enable memory-feasible QLoRA SFT and DPO on a 32GB RTX 5090 platform while preserving the same optimizer-aware training interface.
```

是否需要重跑当前 SFT:

- 不需要。

---

## 5. 建议处理顺序

### 当前 SFT 正在跑时

不要中断训练。

可以并行做:

1. 修改论文 12B → 4B。
2. 修改 `requirements.txt` 的 transformers 版本。
3. 更新过时文档。

不要现在做:

1. 重新生成数据。
2. 大改模型结构。
3. 切换到真正 vision Gemma。

### SFT 跑完后

优先:

1. 修 `evaluate.py` 的 solver config，使其与数据生成一致。
2. 跑 SFT-only 小规模 eval。
3. 确认 warm-start speedup 是否大于 1。
4. 若 SFT-only 有正向信号，再进入 DPO。

### DPO 前

确认:

```yaml
dpo:
  per_device_batch_size: 1
  gradient_accumulation_steps: 16
  max_seq_length: 1536
  mu: 0.0
```

### DPO 后

补:

1. SFT-only vs SFT+DPO 对比。
2. no projection head ablation。
3. text-grid vs image-BEV 分支。
4. 论文 baseline 表。

---

## 6. 是否需要重新跑 SFT

| 修改项 | 是否需要重跑 SFT |
|---|---|
| 改论文 12B → 4B | 否 |
| 改 `requirements.txt` | 否 |
| 改 `evaluate.py` 对齐 solver config | 否 |
| 改 DPO `seq=1536, mu=0` | 否 |
| 增加 CapAssign 仅用于推理/评估 | 否 |
| 补 B1-B9 baseline | 否 |
| 将当前 text-grid 写成 baseline | 否 |
| 新增 image-BEV multimodal proposed | 需要为 multimodal 版本单独训练 |
| 重新生成 SFT 数据 | 是 |
| 改训练标签定义或 projection target | 是 |
| 切换真正 vision Gemma processor | 基本需要 |

---

## 7. 推荐论文叙事调整

如果论文重点必须是多模态，推荐叙事:

```text
The framework accepts heterogeneous state inputs, including communication summaries,
sensing summaries, and a BEV spatial map. We first validate the optimizer-aware
training interface using a textualized BEV map, and then instantiate the proposed
multimodal variant by incorporating rendered BEV image features.
```

中文理解:

```text
当前 text-grid 版本作为单模态/文本化 BEV baseline；真正主方法应补 BEV 图像特征融合。
```

不要把当前 text-grid 版本硬说成完整视觉 MLLM。

---

## 8. 最终建议

当前最稳路线:

1. 继续跑完当前 SFT。
2. 不重新生成数据。
3. SFT 后先修 evaluation solver config。
4. 跑 SFT-only eval。
5. 若有 speedup，再跑 DPO。
6. DPO 后补 image-BEV multimodal 分支，让论文多模态重点站得住。

一句话总结:

```text
当前训练链路已经活了，不要打断；真正需要补的是论文多模态证据和 teacher/evaluator 物理配置一致性。
```

