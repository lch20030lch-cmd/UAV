---
type: reference
status: current
stage: data_regeneration
last_updated: 2026-07-02
---

# Data Schema — 标准数据格式

## SFT Dataset (`sft_dataset.jsonl`)

每行一个 JSON 对象：

```json
{
  "env_id": 0,
  "prompt": "Environment: 1000x1000m area, 4 UAVs, 20 users...\n<ctrl_0><ctrl_1>...<ctrl_7>\nOutput:",
  "response": "{\"delta_q\": [[x,y,z], ...], \"delta_a\": [[...], ...], \"delta_p\": [[...], ...]}",
  "q_current": [[x,y,z], [x,y,z], [x,y,z], [x,y,z]],
  "q_star": [[x,y,z], [x,y,z], [x,y,z], [x,y,z]],
  "a_star": [[...], [...]],
  "p_star": [[...], [...]],
  "has_q_current": true,
  "utility_gap": 66.9,
  "metadata": {
    "num_users": 20,
    "area_size": [1000, 1000],
    "ground_clutter_db": 12.0
  }
}
```

### 字段说明

| 字段 | 类型 | 维度 | 说明 |
|------|------|------|------|
| `env_id` | int | — | 环境唯一 ID |
| `prompt` | str | — | 完整输入 prompt (含 control tokens) |
| `response` | str | — | JSON 格式的 ground truth 输出 |
| `q_current` | float[][] | [4, 3] | **当前 UAV 位置 (x, y, z)** — 必须存在 |
| `q_star` | float[][] | [4, 3] | SCA-FP 最优位置 |
| `a_star` | float[][] | [4, K] | SCA-FP 最优用户关联 |
| `p_star` | float[][] | [4, K] | SCA-FP 最优功率分配 |
| `has_q_current` | bool | — | **关键 flag** — 标记 q_current 是否存在 |
| `utility_gap` | float | — | chosen vs rejected 的效用差距 (DPO 质量闸门) |
| `metadata` | object | — | 环境生成参数 |

### 关键约束

1. **`q_current` 必须存在** — `has_q_current` 必须为 `true`。缺失时分离惩罚 `L_sep` 永远为 0，导致 mode collapse。
2. **`response` 使用 compact JSON** — 无缩进、无空格，控制 token 数量在 824 以内。
3. **所有数值使用 3 位小数精度** (`np.round(x, 3)`) — 平衡精度和 token 效率。

## DPO Dataset (`dpo_dataset.jsonl`)

每行一个 JSON 对象，在 SFT 格式基础上增加偏好对：

```json
{
  "env_id": 0,
  "prompt": "...",
  "chosen": "{\"delta_q\": ..., \"delta_a\": ..., \"delta_p\": ...}",
  "rejected": "{\"delta_q\": ..., \"delta_a\": ..., \"delta_p\": ...}",
  "q_current": [[...], [...]],
  "has_q_current": true,
  "utility_gap": 66.9,
  "mask_delta_a": true,
  "mask_delta_p": true,
  "metadata": {...}
}
```

### DPO 特有字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `chosen` | str | 较优的响应 (compact JSON) |
| `rejected` | str | 较劣的响应 (compact JSON) |
| `mask_delta_a` | bool | 是否在 DPO loss 中遮蔽 δ_a 的 token (Masked DPO) |
| `mask_delta_p` | bool | 是否在 DPO loss 中遮蔽 δ_p 的 token (Masked DPO) |

### Masked DPO 策略

当前策略 (ADR-007): δ_a 和 δ_p 的 chosen/rejected 相同 (刻意设计)，只对 δ_q 做偏好学习。在 token 级别将 δ_a/δ_p 的 label 设为 `-100`，使其不参与 DPO loss。

→ 详见 [adr_007_dpo_masking_strategy.md](../06_decisions/adr_007_dpo_masking_strategy.md)

## 生成参数

```bash
python scripts/generate_data.py \
    --config configs/default.yaml \
    --num-env 5000 \
    --num-restarts 3 \
    --ground-clutter-db 12.0 \
    --lambda-repel 0.01 \
    --output-dir /root/autodl-tmp/data/full_v2 \
    --workers 30
```

关键参数：
- `ground_clutter_db=12.0`: 地面杂波强度 (修复数据退化)
- `lambda_repel=0.01`: 空间互斥力 (防止 UAV 碰撞)
- `num_restarts=3`: SCA-FP 重启次数 (从 10 优化到 3，节省 70% 算力)
