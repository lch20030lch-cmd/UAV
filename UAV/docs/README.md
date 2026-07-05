---
type: reference
status: current
stage: code_complete
last_updated: 2026-07-02
---

# UAV-ISAC-MLLM — Documentation

**目标**: 用 Gemma 3 4B (LoRA + 约束投影头) 为 UAV-ISAC 的 SCA-FP 数值优化器提供智能热启动。
**硬件**: RTX PRO 6000 96GB (bf16) / RTX 5090 32GB (4-bit QLoRA)。

## 快速导航

### 🚀 新成员？从这里开始

| 顺序 | 文档 | 时间 | 内容 |
|------|------|------|------|
| **1** | [00_system_state/status.md](00_system_state/status.md) | **5 min** | 项目当前状态：双重根因、数据资产、下一步 |
| **2** | [00_system_state/quickstart.md](00_system_state/quickstart.md) | **10 min** | 从零开始在服务器上跑起来 |
| **3** | [00_system_state/canonical_config.md](00_system_state/canonical_config.md) | **5 min** | 当前 blessed 配置和 pipeline 命令 |
| **4** | [01_architecture/problem_formulation.md](01_architecture/problem_formulation.md) | **10 min** | 我们到底在解决什么问题？数学框架 |
| **5** | [01_architecture/system_design.md](01_architecture/system_design.md) | **10 min** | 模块拓扑、数据流、接口契约 |

**总计: ~35 分钟**理解整个项目。

### 📂 目录地图

```
docs/
├── README.md                              ← 你在这里
│
├── 00_system_state/                       ★ 系统状态 — 第一站
│   ├── status.md                          当前状态、blocker、下一步
│   ├── onboarding.md                      新成员接手文档
│   ├── quickstart.md                      Zero-to-running 操作指南
│   ├── canonical_config.md                Blessed 配置 + server 命令
│   ├── server_ops.md                      服务器运维速查
│   └── training_monitoring.md             训练监控指南 — 如何判断训练是否正常
│
├── 01_architecture/                       稳定技术参考
│   ├── problem_formulation.md             UAV-ISAC 数学、系统模型
│   ├── system_design.md                   模块拓扑、数据流
│   ├── training_pipeline.md               Stage I SFT + II DPO 设计
│   └── hardware_adaptation.md             Blackwell RTX PRO 6000 特定方案
│
├── 02_training_log/                       训练纪实 (session log + experiment report)
│   ├── session_2026-07-01.md              数据重生执行全纪录 (20K gen + 5 bugs)
│   ├── smoke_v3_full_report.md            全链路闭环验证 v3 ⭐
│   ├── implementation_2026-06-29.md       代码落地全纪录 (commit 7cedb02)
│   ├── phase1_warmup_diagnostic.md        Phase 1 控制表示学习调试
│   └── sft_phase2_historical.md           Phase 2 旧数据训练快照 (已废弃)
│
├── 03_bugs/                               ★ Bug 注册中心 (所有 bug 统一收敛)
│   ├── README.md                          完整 registry (Bug #1-#20 + 严重度定义)
│   └── resolved/                          13 个已修复 bug postmortem
│       ├── oom_chain.md                   OOM 1-7 诊断全链 ⭐
│       ├── data_degeneracy.md             数据退化根因分析 ⭐
│       ├── q_current_missing.md           q_current 缺失→Mode Collapse ⭐
│       ├── verification_gaps_audit.md     20 项验证缺口审计 (已闭合)
│       └── ...                            (物理约束、RNG、Token、运行时等)
│
├── 04_reviews/                            代码审查历史
│   ├── README.md                          7 轮审查总结 + 累计修复表
│   ├── pre_launch/                        Rounds 1-6
│   └── multiprocessing_branch/            Round 7
│
├── 05_data/                               数据层 (格式定义 + 历史)
│   ├── README.md                          数据状态 (⚠️ 旧数据全部作废)
│   └── data_schema.md                     标准数据格式规范
│
├── 06_decisions/                          架构决策记录 (ADR)
│   ├── README.md                          ADR 索引 (8 条)
│   ├── adr_001_unsloth_removal.md         ★ 最重要的决策
│   ├── adr_002_dpo_independent_ref.md
│   ├── adr_003_sdpa_canonical.md
│   ├── adr_004_4bit_qlora_blackwell.md
│   ├── adr_005_control_token_mechanism.md
│   ├── adr_006_data_regeneration.md       ★ 数据重生 + DPO 路线
│   ├── adr_007_dpo_masking_strategy.md    ★ DPO 困境：Masked vs Unmasked
│   └── adr_008_performance_planA.md       速度优化：Plan A 决策
│
├── 07_conventions/                        文档维护规范
│   ├── naming_conventions.md
│   ├── handoff_template.md
│   ├── bug_postmortem_template.md
│   └── archive_rules.md
│
├── 08_logs/                                ★ 运行日志 (smoke test / deployment log)
│   └── rtx5090_smoke_test_2026-07-04.md   RTX 5090 32GB Smoke Test 完整记录
│
└── 99_archive/                            已废弃 / 历史参考
    ├── README.md
    ├── data_validation_v1.md              旧版数据验证 (已作废)
    ├── deprecated_experiments/            失败方案 (Plan B 等)
    ├── old_results/                       早期数据验证结果
    ├── old_handoffs/                      历史交接文档 (#13-#26)
    └── old_setup_docs/                    旧版项目文档
```

### 🔗 关键外部资源

| 资源 | 路径/URL |
|------|----------|
| GitHub | `Lampotaku/UAV-ISAC-MLLM` (private) |
| 服务器 | AutoDL RTX PRO 6000 96GB, `/root/UAV-ISAC-MLLM` |
| 数据盘 | `/root/autodl-tmp/` (系统盘仅 30GB) |
| 配置文件 | `configs/default.yaml` |
| Conda env | `uavmllm` (Python 3.12) |

### 📖 阅读建议

- **接手的工程师**: 按快速导航 1→2→3→4→5 顺序，然后读 [onboarding.md](00_system_state/onboarding.md)
- **排查 bug**: 先去 [03_bugs/README.md](03_bugs/README.md) 查 registry
- **理解架构决策**: 去 [06_decisions/](06_decisions/) 看 ADR
- **查当前状态**: 看 [00_system_state/status.md](00_system_state/status.md)
- **找废弃方案**: 去 [99_archive/](99_archive/)

### 🗺️ 文档分层逻辑

| 层 | 目录 | 回答的问题 |
|----|------|-----------|
| **System State** | `00_system_state/` | "现在系统在什么状态？下一步做什么？" |
| **Architecture** | `01_architecture/` | "系统怎么设计的？为什么这样设计？" |
| **Training Log** | `02_training_log/` | "训练过程中发生了什么？" |
| **Bugs** | `03_bugs/` | "出过什么问题？怎么修的？" |
| **Reviews** | `04_reviews/` | "代码审查发现了什么？" |
| **Data** | `05_data/` | "数据长什么样？怎么生成？" |
| **Decisions** | `06_decisions/` | "关键决策是什么？为什么？" |
| **Conventions** | `07_conventions/` | "怎么维护这些文档？" |
| **Logs** | `08_logs/` | "某次部署/测试发生了什么？" |
| **Archive** | `99_archive/` | "历史上还有什么？" |
