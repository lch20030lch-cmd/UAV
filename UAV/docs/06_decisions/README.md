---
type: reference
status: current
stage: all
last_updated: 2026-07-02
---

# Architecture Decision Records (ADR)

## 索引

| ADR | 标题 | 状态 | 日期 |
|-----|------|------|------|
| [001](adr_001_unsloth_removal.md) | Unsloth Removal — Pure PyTorch Pipeline | Accepted | 2026-06-26 |
| [002](adr_002_dpo_independent_ref.md) | DPO Reference Model Independent Loading | Accepted | 2026-06-23 |
| [003](adr_003_sdpa_canonical.md) | SDPA as Canonical Attention | Accepted | 2026-06-26 |
| [004](adr_004_4bit_qlora_blackwell.md) | bf16 Full Precision on RTX PRO 6000 (No Quantization) | Accepted | 2026-06-23 |
| [005](adr_005_control_token_mechanism.md) | Control Token Design | Accepted | 2026-06-23 |
| [006](adr_006_data_regeneration.md) | Data Regeneration + DPO Strategy | Accepted | 2026-06-29 |
| [007](adr_007_dpo_masking_strategy.md) | DPO Masking Strategy — Masked vs Unmasked | Decision Point | 2026-07-01 |
| [008](adr_008_performance_planA.md) | Performance Optimization — Plan A (Pure PyTorch) | Accepted | 2026-06-26 |

## 决策分类

### 平台/基础设施
- **ADR-001**: 移除 Unsloth — 全局 monkey-patch 不可共存
- **ADR-003**: SDPA 作为标准 attention — Blackwell 上唯一可用选项
- **ADR-004**: bf16 全精度 — 96GB 无需量化
- **ADR-008**: 速度优化从 21s→2.5s/step — Plan A 决策过程

### 训练策略
- **ADR-002**: DPO reference model 独立加载 — 避免 deepcopy OOM
- **ADR-006**: 数据重生完整路线图 — 5 轮 Grilling 终稿
- **ADR-007**: Masked DPO 策略抉择 — 量纲冲突与表征饥饿分析

### 模型架构
- **ADR-005**: Control Token 机制设计 — 8 tokens, 投影头映射

## 格式

所有 ADR 使用标准格式:
- **Title**: 决策简述
- **Status**: Proposed / Accepted / Deprecated / Superseded
- **Context**: 为什么需要这个决策
- **Decision**: 我们选择了什么
- **Consequences**: 正面和负面的后果

## 如何提出新 ADR

1. 分配下一个序号 (当前最大: 008)
2. 使用已有 ADR 的格式作为模板
3. 填写所有章节
4. 更新本 README 的索引表
5. 在 PR 中引用 ADR 编号

决策被取代时: 将状态改为 `Superseded by ADR-NNN`，保留原文件。
