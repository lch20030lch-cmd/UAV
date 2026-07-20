---
type: status
status: current
stage: v5_train20_val20_preflight
last_updated: 2026-07-20
related: [onboarding, quickstart, canonical_config]
---

# 项目当前状态

## 一句话结论

多模态 v5 的 Solver/Oracle、数据契约、SFT 与 DPO 运行链路已经闭环；下一步只生成
相互独立的 v5 train20/val20 做质量预检。当前不能复用任何 v3/v4 数据或 checkpoint，
也不能直接进入 train500、5000 条数据或正式联合训练。

## 当前主线架构

```text
BEV PNG + constraint-aware text prompt
                 │
                 ▼
Gemma 3 4B multimodal backbone + 8 control tokens
                 │  control states [B, 8, 2560]
                 ▼
Split projection head
    ├── Q branch: displacement projection; v5 当前使用 direction/none 预检
    ├── A branch: capacity-aware association projection
    └── P branch: association-aware simplex power projection
                 │
                 ▼
δq [B,4,3], δa [B,4,20], δp [B,4,21]
```

- Stage I：图像条件 multimodal SFT，可训练 projection 与语言 LoRA；视觉塔与视觉 LoRA
  默认冻结。
- Stage II：图像条件 chosen/rejected DPO，保留 response token log-prob、SFT anchor 与
  control anchor。
- 评估：模型给出 Q/A/P warm start，再进入统一物理配置的 downstream solver。

## 已完成门禁

### P0：Solver / Oracle 与多模态主线

- 新 solver 会在第一次 deployment update 使用模型 Q/A/P warm start。
- Q 移动后重算几何相关 channel；association 满足容量，power 满足结构零与预算。
- 统一检查边界、高度、移动半径、UAV 间距、SINR、association 和 power 约束。
- 修复 LoS/path-loss 物理参数传播；生成器只接受 feasible Oracle solution。
- v5 SFT/DPO 一对一配对，rejected tuple 使用其真实 Q/A/P 重新评分。
- multimodal SFT、DPO 与 downstream evaluation 入口均已建立。

注意：当前 solver 的实际算法标记为
`constraint_aware_alternating_optimization`。除非以后实现真正的凸 SCA/FP，论文中不能
把当前实现描述成形式化 SCA/FP 收敛算法。

### P1：数据来源与训练运行一致性

- schema、prompt、solver/channel revision、完整物理配置 fingerprint 已进入数据契约。
- checkpoint 保存数据 provenance；主线拒绝旧数据、错物理配置或错 checkpoint。
- SFT/DPO 的 scheduler、warmup、epoch/step 解析、梯度累积、学习率日志和 checkpoint
  轮转均按配置真实执行。
- DPO 中间与最终 metadata 明确覆盖 Stage-I 的 progress 字段，不再继承旧步数。
- 无有效 DPO pair 的环境不再渲染孤儿 BEV 图片。

### 服务器验证

- P0/P1 综合回归：57/57 PASS。
- DPO progress metadata 与 runtime 回归：8/8 PASS。
- v5 runtime 数据：2 SFT / 2 DPO / 2 BEV，schema=5，prompt type 正确。
- 实际多模态 control-only 长度：2703--2714；3072 下无截断。
- v5 2-step SFT：语言 LoRA 272、视觉 LoRA 0；cosine、独立裁剪与保留上限生效。
- v5 3-step DPO：loss 有限，无 NaN/Inf/OOM；只保留 step 3；最终 metadata 记录
  DPO 自身的 global/micro step 3/3。

## 兼容性边界

- v3/v4 数据由旧 solver/channel 语义生成，只能作为历史诊断，不得用于 v5 主线训练。
- v3/v4 Q fixed-mixture 权重不得迁移到 v5；配置目前保持
  `q_fixed_cue_weights: null`。
- tiny2 SFT/DPO checkpoint 只证明代码能跑通，不代表 Q/A/P 精度、泛化或论文方法达标。
- 旧日志中的 v3 association/power/Q 数值不能作为 v5 验收基线。

## 当前有效资产

```text
代码主线：main（P0/P1 已合入）

runtime 数据：
/root/autodl-tmp/data/mm_oracle_v5_runtime_smoke2b_seed42

runtime SFT：
/root/autodl-tmp/outputs/mm_oracle_v5_runtime_sft2/mm_sft_lora_smoke_final

runtime DPO：
/root/autodl-tmp/outputs/mm_oracle_v5_runtime_dpo3_metadata_fix
```

以上三个路径都不是正式模型资产。

## 未完成问题

运行闭环通过不等于模型质量通过。v5 语义下仍需重新回答：

1. train/validation Oracle target 分布、可行率和 prompt 长度是否稳定一致；
2. Q direct direction 或重新标定的 geometry baseline 能否在独立验证集超过固定基线；
3. A 是否在 v5 control states 上高于 per-user majority，并在独立环境泛化；
4. P 是否降低 active/sensing MSE 与 inactive leakage，而不是退化成常数分配；
5. 联合 SFT 是否在不破坏已通过分支的情况下获得净增益；
6. DPO 和 downstream solver 是否最终改善 feasibility、utility、迭代数或时间。

## 下一步（严格顺序）

1. 生成独立 v5 `train20_seed42` 与 `val20_seed2026`，两者使用相同物理配置。
2. 分别验证 schema/provenance、SFT/DPO/BEV 数量、Oracle feasibility、target 分布与
   实际多模态序列长度。
3. 先做小规模 Stage-I 质量预检并分别诊断 Q/A/P；runtime checkpoint 不作为初始化。
4. 只有 train20/val20 的数据和质量门通过，才生成 v5 train500/val100。
5. 单分支有效后再做联合 SFT；联合 SFT 通过后才运行正式 multimodal DPO 和 solver
   评估。

## 禁止事项

- 不从旧 v3/v4 checkpoint 继续训练 v5。
- 不把 tiny runtime loss 下降当作模型效果。
- 不在 train20 上通过增加步数替代独立 validation。
- 不跳过 train20/val20 门禁直接生成 500/5000 条数据。
- 不在未重新标定前启用 `fixed_residual_xy`。
