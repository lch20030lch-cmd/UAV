---
type: reference
status: current
stage: all
last_updated: 2026-07-02
---

# Bug Registry

所有 bug 统一收敛于此。不再分散在 training_log、handoff 或其他目录。

## 严重度定义

| 级别 | 定义 | 示例 |
|------|------|------|
| **P0** | 训练/数据无效，或结果不可用 | 物理约束违反、mode collapse、OOM |
| **P1** | 训练运行但严重降级 | DPO validation 被绕过、TensorBoard 静默丢失 |
| **P2** | 次优但可运行 | 硬编码常数、性能浪费 |
| **P3** | 装饰性 / 未来增强 | 文档改进 |

## 完整 Bug 清单 (全部已解决)

### 数据生成阶段

| # | Bug | 严重度 | 文件 |
|---|-----|--------|------|
| 1 | 物理约束违反 (SCA-FP random_init 忽略 q_current) | P0 | [physical_constraint.md](resolved/physical_constraint.md) |
| 2 | 环境多样性崩溃 (ProcessPoolExecutor RNG pickle) | P0 | [rng_diversity_collapse.md](resolved/rng_diversity_collapse.md) |
| 3 | 响应 JSON 截断 (BPE 碎片化, 512→824 tokens) | P0 | [response_token_overflow.md](resolved/response_token_overflow.md) |
| 11 | 数据退化 — 地面杂波缺失导致退化解 | P0 | [data_degeneracy.md](resolved/data_degeneracy.md) |
| 14 | q_current 缺失 → Mode Collapse (0.893x) | P0 | [q_current_missing.md](resolved/q_current_missing.md) |

### SFT 训练阶段

| # | Bug | 严重度 | 文件 |
|---|-----|--------|------|
| 4 | 服务器运行时错误 (Blackwell 8 连击) | P0 | [server_runtime_errors.md](resolved/server_runtime_errors.md) |
| 5 | 训练代码 Bug (scheduler/zero_grad/LR 时序) | P0 | [training_code_bugs.md](resolved/training_code_bugs.md) |
| 6 | OOM #1-#5 (HF wrapper→CE→CheckpointError) | P0 | [oom_1_through_5.md](resolved/oom_1_through_5.md) |
| 7 | TensorBoard 日志静默丢失 (缺失 init_trackers) | P1 | [tensorboard_init_trackers.md](resolved/tensorboard_init_trackers.md) |
| 10 | OOM #6-#7 (Phase 2 切换泄漏 + Grad diag retain_graph) | P0 | [oom_chain.md](resolved/oom_chain.md) |

### 评估阶段

| # | Bug | 严重度 | 文件 |
|---|-----|--------|------|
| 8 | Eval Pipeline 7 处缺陷 (tqdm/CPU/加速比) | P0 | [eval_pipeline_7_bugs.md](resolved/eval_pipeline_7_bugs.md) |
| 9 | Checkpoint 4GB→100MB (modules_to_save 完整权重) | P0 | [checkpoint_modules_to_save_4gb.md](resolved/checkpoint_modules_to_save_4gb.md) |

### 代码审查阶段

| # | Bug | 严重度 | 文件 |
|---|-----|--------|------|
| 12a | `converged` 引用旧 `max_outer_iters` | P1 | 详见 [implementation_2026-06-29.md](../02_training_log/implementation_2026-06-29.md) |
| 12b | `_compute_utility_of_delta_q` 额外 SCA-FP 调用 (×20,000) | P1 | 同上 |
| 12c | `calibrate_epsilon._pareto_filter` baseline 用随机重启 | P1 | 同上 |

### 数据重生阶段 (2026-07-01 Session)

| # | Bug | 严重度 | 文件 |
|---|-----|--------|------|
| 13 | 负数 utility Pareto 灾难 (39/50 envs 误杀) | P0 | 详见 [session_2026-07-01.md](../02_training_log/session_2026-07-01.md) |
| 14 | `delta_q_perturbed` 未定义 (NameError) | P1 | 同上 |
| 15 | Baseline 检查误杀 (80% envs 丢弃) | P0 | 同上 |
| 16 | DPO Chosen≈Rejected (偏好信号为零) | P1 | 同上 |
| 17 | Snapback 无区分度 (0 variance, 浪费 3 calls/env) | P2 | 同上 |

### Smoke v3 阶段 (2026-07-02)

| # | Bug | 严重度 | 文件 |
|---|-----|--------|------|
| 18 | Phase 1 31.9s/it (max_steps 过高) | P2 | 详见 [smoke_v3_full_report.md](../02_training_log/smoke_v3_full_report.md) |
| 19 | `save_state` 磁盘爆满 (中间 checkpoint 存 optimizer state) | P1 | 同上 |
| 20 | Eval+DPO GPU 争抢 OOM | P1 | 同上 |

### 审计

| # | 审计 | 严重度 | 文件 |
|---|------|--------|------|
| — | 验证缺口审计 (20 项, 已闭合) | P0-P2 | [verification_gaps_audit.md](resolved/verification_gaps_audit.md) |

## 统计

- **总 Bug 数**: 20 + 1 审计
- **P0**: 13 个
- **P1**: 5 个
- **P2**: 3 个
- **P3**: 0 个
- **全部已解决** ✅

## OOM 链 (Bug #6 + #10)

OOM 系列是最复杂的 bug 链，7 次连续 OOM 从 94GB 压到 48GB。完整叙事见 [oom_chain.md](resolved/oom_chain.md)。

## 如何登记新 Bug

1. 在 `resolved/` 中创建新文件
2. 使用 [bug postmortem 模板](../07_conventions/bug_postmortem_template.md)
3. 添加 metadata header (YAML frontmatter)
4. 更新本文件的对应表格
5. 更新 [status.md](../00_system_state/status.md) 如果该 bug 是新的 blocker

## Bug 文件命名

```
{short_kebab_description}.md
```

例: `physical_constraint.md`, `q_current_missing.md`, `oom_chain.md`
