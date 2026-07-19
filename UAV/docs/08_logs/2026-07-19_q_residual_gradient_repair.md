---
type: log
status: implementation_complete_server_smoke_pending
stage: q_residual_gradient_repair
last_updated: 2026-07-19
---

# 2026-07-19 Q 残差梯度瓶颈修复

## 为什么 Q2 200-step 几乎没有进一步学习

Q2 无训练和 200-step 后的独立 val100：

```text
                             init       200-step
projected 3D cosine:         0.593184   0.593601
raw direction cosine:       0.214046   0.221283
residual gate:              0.050000   0.051953
```

raw 分支有微小变化，但投影后只提高约 0.00042。根因是旧组合式：

```text
q_dir = normalize(fixed_dir + g * normalize(raw_q))
g = sigmoid(gate_logit) ~= 0.05
```

这里有三个结构性瓶颈：

1. 投影输出对 `raw_q` 的梯度被全局 gate 乘约 0.05；
2. gate logit 本身的梯度还要乘 `g(1-g) ~= 0.0475`；
3. `normalize(raw_q)` 删除了样本级残差幅度，只剩一个所有样本共享的全局 gate。

因此问题不是简单的 step 太少，而是残差参数化不适合优化。继续扩步只会浪费训练时间。

## 替换而非叠加

没有新增 Q4/Q5 模式；原 `fixed_residual_xy` 内部直接替换为：

```text
residual = tanh(W * raw_q + b)
q_dir = normalize(fixed_dir + 0.5 * residual)
```

其中 `W,b` 是共享的 3x3 residual adapter：

```text
parameters = 12
initial W = 0
initial b = 0
```

性质：

1. 初始化严格等于 fixed geometry，不会先破坏 XY；
2. residual adapter 的首步梯度直接乘固定 scale=0.5，不再经过 0.05 sigmoid gate；
3. `tanh` 保留每个样本的残差方向和幅度，同时把最大修正限制在安全范围；
4. 旧的 global gate 被删除，不与新 adapter 并存。

## Smoke 可观测性

训练日志新增：

```text
grad_norm_q_residual
q_residual_adapter_norm
```

单元测试覆盖：

1. zero-init 输出严格等于 fixed geometry；
2. projected direction loss 首步能到达 adapter weight/bias；
3. synthetic 多步优化确实降低 projected direction loss；
4. `q_geometry_mode=none` 不向旧 checkpoint 增加 adapter key；
5. Q-only 冻结边界只打开 readout_q/q_mlp/q_residual_adapter。

## LoRA 加载与训练职责拆分

旧 `train_sft_mm.py` 只有 `--train_lora`，同时控制 LoRA 加载和更新，无法在固定 Q2
backbone 的情况下隔离测试 projection。新增：

```text
--load_lora   # 加载 init checkpoint/lora，但冻结它
--train_lora  # 加载/创建并训练 LoRA
```

下一轮先用 `--load_lora` 做 50-step projection-only smoke。这样如果 adapter 不学习，
可以直接归因于 Q 投影路径；不会再被 LoRA 更新或 A/P retention loss 混淆。

## 下一轮门槛

50-step 只验证代码闭环，不宣称模型收敛：

```text
grad_norm_q_residual > 0
q_residual_adapter_norm 从 0 增长
loss_q_projected_dir / total loss 无 NaN/Inf
grad_norm_lora = 0
A/P 与 Q2 基线完全保持（LoRA 和 A/P projection 均冻结）
```

通过后再做独立 val100 前向，必须看到 projected 3D cosine 相对纯 fixed baseline 有实际
增益，才允许进入更长预检。
