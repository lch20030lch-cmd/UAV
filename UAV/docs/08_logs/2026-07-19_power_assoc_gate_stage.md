---
type: log
status: soft_gate_implemented_validation_pending
stage: power_association_soft_gate
last_updated: 2026-07-19
---

# 2026-07-19 P/A 软门控修复

## 当前问题

Q2 独立 val100 中 P 分支保持：

```text
P MSE:                    0.007581
P active MSE:             0.011688
P inactive MSE:           0.000773
P sensing MSE:            0.089171
P inactive leakage:       0.025418
P budget MAE:             0.0000355
```

预算和总体回归已经稳定，但普通 softmax 必须给所有条目正概率，20 个通信条目累计后
仍形成高于 0.01 门槛的未关联泄漏。

## 约束

A 的独立 val100 accuracy 约 0.436，不能把预测 association 直接做硬 mask；否则一旦
A 选错 UAV，P 会把 oracle active 条目直接清零。修复必须：

1. 保持可微；
2. 默认关闭，旧 checkpoint 行为完全不变；
3. 只先做不训练验证；
4. 同时检查 active、inactive、sensing 与总预算，不能只追 leakage。

## 实现

`PowerProjection` 新增 `association_gate_strength`。启用时：

```text
scaled_comm_logits = p_comm_logits / tau + strength * log(clamp(association))
p_hat = P_max * softmax([scaled_comm_logits, sensing_logits / tau])
```

等价于在原功率概率上乘以 `association ** strength` 后重新归一化。感知项不门控，
功率预算仍严格等于 `P_max`。`strength=0` 完全走旧路径。

接口新增：

```text
--power_assoc_gate_strength
model.projection_head.power_assoc_gate_strength
checkpoint metadata: power_assoc_gate_strength
```

配置默认值为 0，不会静默改变已有实验。

## 第一轮验收

使用 Q2 checkpoint、独立 val100、不训练，先测试 `strength=1.0`。通过条件：

```text
inactive leakage < 0.02（理想 < 0.01）
overall P MSE <= 0.009
active MSE 不显著高于 0.0117
sensing MSE 不显著高于 0.0892
budget MAE 保持约 1e-5
A/Q 指标不变
```

若 active 或 sensing 明显回退，则降低 strength；若 leakage 几乎不变，则不把该耦合
接入主线，并保留当前 P 作为 SCA-FP 可行化前的软 warm-start。
