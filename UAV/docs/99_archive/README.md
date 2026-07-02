---
type: reference
status: deprecated
stage: all
last_updated: 2026-07-02
---

# Archive

本目录包含已废弃、被取代或仅作历史参考的文档。

**⚠️ 此处信息可能已过时。** 当前信息请参考:
- [../00_system_state/](../00_system_state/) — 项目状态
- [../01_architecture/](../01_architecture/) — 技术参考
- [../03_bugs/](../03_bugs/) — Bug registry
- [../06_decisions/](../06_decisions/) — 架构决策

## 子目录

| 目录 | 内容 | 归档原因 |
|------|------|----------|
| `deprecated_experiments/` | Plan B (Unsloth chunked CE), EDA postmortem raw | 被 Plan A 取代；清理版在 bugs/ |
| `old_results/` | 数据验证 Run 1-3 | 被 smoke v3 + full v2 取代；且旧数据全部作废 |
| `old_handoffs/` | 历史交接文档 #13-#26 | 有效信息已提取到 canonical 文档 |
| `old_setup_docs/` | 旧版项目文档 (01/08/09) | 被 `01_architecture/` 取代 |

## 独立归档文件

| 文件 | 内容 | 归档原因 |
|------|------|----------|
| [data_validation_v1.md](data_validation_v1.md) | Run 4 数据验证 (5000 SFT + 186,896 DPO) | 旧数据 — q_current 缺失 + 数据退化 |
| [refactor_instructions.md](refactor_instructions.md) | 2026-06-26 文档重构规范 | 基础设施 artifact |

## 重构历史

- **2026-06-26**: 第一次重构 — 从编号文档系统迁移到功能分层结构
- **2026-07-02**: 第二次重构 — 统一 bug registry、清理 training_log、重组 data/decisions
