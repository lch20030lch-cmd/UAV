# 2026-07-20 P1 训练运行一致性修复

## 前置验收

P1 第一组数据契约、物理配置与 checkpoint provenance 修复已在服务器完成 52 项回归，
全部通过。本轮只收口训练运行层，不修改 Q/A/P 投影或损失的数学定义。

## 根因

代码审阅确认有三类“配置存在但训练没有按配置运行”的问题：

1. SFT 配置声明了 `lr_scheduler` 和 `warmup_ratio`，训练循环却从未建立或执行
   scheduler，实际始终使用常数学习率。
2. SFT 配置声明了 `save_total_limit`，但中间 checkpoint 只保存、不轮转，长实验会
   持续占用磁盘；DPO 也没有中间 checkpoint 与保留上限。
3. DPO 命令行把 `max_steps` 默认硬编码为 50，所以未显式传参时配置中的 `epochs`
   完全失效；同时 DPO 日志只记录累积窗口最后一个 micro-batch，不能代表本次
   optimizer update 的平均损失。

## 修复内容

- 新增 `src/training/runtime_utils.py`，集中维护三个无框架依赖的运行规则：
  - 根据 dataloader batch 数、梯度累积步数和 epoch 数计算 optimizer updates；
  - 根据总步数和 warmup ratio 计算 warmup steps；
  - 仅轮转指定根目录下、名称严格匹配指定前缀的旧 step checkpoint。
- SFT：
  - 用 Transformers `get_scheduler` 按配置建立 scheduler，并在每次 optimizer update
    后执行 `scheduler.step()`；
  - 日志新增实际用于该 update 的 `lr_proj` 与 `lr_lora`；
  - 新增 `--save_total_limit`，默认读取 SFT 配置，保存后仅删除同一前缀最旧的
    step checkpoint；
  - 显式传入 `--output_dir` 且未传 `--checkpoint_dir` 时，中间 checkpoint 默认放在
    当前实验的 `output_dir/checkpoints`，避免不同实验共用全局目录并互相覆盖；
  - metadata 保存 scheduler、warmup、保存间隔和保留上限；
  - 显式拒绝非正的 `max_steps/save_steps/save_total_limit`。
- DPO：
  - `--max_steps` 默认改为 `None`。未传该参数时，按配置 `epochs`、dataloader 长度
    与梯度累积计算更新次数；传入时仍作为明确的 optimizer-step override；
  - epoch 模式严格消费配置的 micro-batch 总数，最后不足一个累积窗口时按实际
    窗口大小缩放 loss，不额外重复样本凑满窗口；
  - 学习率和 `beta` 默认读取配置，scheduler 类型不再写死为 cosine；
  - 日志改为整个梯度累积窗口的平均 loss，并记录 LoRA/projection 实际学习率；
  - 新增中间 checkpoint、`--checkpoint_dir`、`--save_steps` 与
    `--save_total_limit`，默认最多保留两个同前缀 checkpoint；
  - 最终和中间 metadata 记录 epoch、累积、学习率、scheduler 与保存策略。
- 配置中的 DPO 段新增 `save_total_limit: 2`。

## 安全边界

checkpoint 轮转不会扫描或删除任意路径：它只检查指定 checkpoint 根目录的直接
子目录，只接受“固定前缀 + 整数 step”的完整名称匹配，忽略符号链接，并在删除前
再次确认解析后的父目录就是目标根目录。最终 checkpoint 和其他实验目录不会匹配。

## 本地验证

- `python -m compileall -q src scripts tests`：PASS。
- `python -m unittest tests.test_training_runtime -v`：5/5 PASS。
- `git diff --check`：PASS。
- 本机完整测试发现受环境限制：当前 Windows Python 没有 NumPy/Torch，因此依赖
  这些包的测试模块无法导入；这是本机依赖缺失，不是测试断言失败。完整回归必须在
  `uavmllm` 服务器环境执行。

## 服务器验收命令

```bash
python -m unittest \
  tests.test_training_runtime \
  tests.test_oracle_contract \
  tests.test_multimodal_sequence_budget \
  tests.test_multimodal_dpo \
  tests.test_association_solver \
  tests.test_channel_model \
  tests.test_q_geometry_branch \
  tests.test_power_branch \
  tests.test_training_branch_freeze \
  tests.test_delta_diagnostics \
  -v
```

在上述回归通过前，不启动新的正式 SFT/DPO 训练。回归通过后先做 2-step SFT 和
2-step DPO runtime smoke，检查日志中的学习率、micro-step、checkpoint metadata 与
轮转结果，再进入下一优先级。

## 服务器验收结果

- 环境：AutoDL `uavmllm`。
- 结果：57/57 PASS，耗时 0.197 秒。
- 新增的 5 项训练运行测试与原有 52 项数据契约、Solver、Channel、Q/A/P、梯度累积
  和诊断回归全部通过。
- P1 训练运行代码测试门已关闭；下一步只生成全新 v5 极小数据并执行 2-step
  SFT/DPO runtime smoke，不使用任何 v3/v4 数据或 checkpoint。

## v5 极小数据预检发现与修复

- 首次生成 2 条有效配对记录时依次尝试了 `env_0/env_1/env_2`；其中 `env_1` 没有
  形成正 utility-gap DPO pair，因此有效 SFT/DPO ID 为 `env_0/env_2`，这是预期行为。
- 数据契约、有效记录配对和 4096-token 预算均通过，但目录中有 3 张图片：生成器在
  判断 DPO pair 是否有效之前就渲染了 BEV，给失败的 `env_1` 留下了孤儿图。
- 修复方式不是事后批量删除，而是把 BEV 渲染延后到有效 SFT/DPO pair 已确认之后；
  失败尝试从源头不再产生图片。该修复不改变 Solver、Oracle target 或 prompt 内容。
- 新增 `tests.test_multimodal_generation`，用无有效候选的环境验证渲染函数不会被调用。

## v5 2-step SFT runtime smoke

- 数据：2 条 v5 SFT/DPO 配对记录和 2 张有效 BEV 图片。
- 架构：`split + direction + q_geometry_mode=none`，从基础 Gemma checkpoint 初始化，
  不加载任何 v3/v4 projection 或 LoRA。
- 训练参数：2 optimizer steps、累积 1、projection LR `3e-4`、language LoRA LR
  `1e-5`、cosine、保存间隔 1、保留上限 1。
- 验收结果：
  - projection 58 个可训练 tensor；语言 LoRA 272；视觉 LoRA 0；
  - step 1/2 实际 projection LR 为 `3e-4/1.5e-4`，LoRA LR 为
    `1e-5/5e-6`；
  - 两组裁剪后梯度范数均为 1，未出现 NaN/Inf；
  - 中间目录只保留 `mm_sft_lora_smoke_step_2`；
  - 最终 metadata 的 schema=5、prompt type、架构模式、scheduler、步数与
    checkpoint 策略全部通过断言。
- `loss_ctl=97.26/135.46` 和很大的裁剪前梯度来自全新随机 split projection 只训练
  两条样本，不能作为质量指标。这个 checkpoint 仅用于 DPO runtime smoke，不作为
  正式 Stage-I checkpoint，也不据此判断 Q/A/P 收敛。

## DPO final metadata 进度污染修复

- 首次 2-step DPO 的最终 metadata 显示 `global_step=2`，但审查代码发现该字段来自
  展开的 Stage-I SFT metadata；SFT 与 DPO 本次恰好同为 2 步，所以形成了假通过。
- 新增统一 `_dpo_checkpoint_metadata` 构造器，中间与最终 checkpoint 共用同一份
  runtime metadata，并在最后强制用当前 DPO 的 `stage/global_step/micro_step` 覆盖
  Stage-I 字段，避免重复字典再次分叉。
- 新增回归：Stage-I 为 200/1600 步、DPO 为 17/136 步时，保存结果必须为 DPO 的
  17/136，同时保留数据 schema provenance。

## v5 DPO runtime 最终验收

- metadata 修复单测与 runtime 工具测试共 8/8 PASS。
- 使用 2-step v5 SFT checkpoint 运行独立 3-step DPO，以保证 Stage-I 与 DPO 步数
  不同并暴露任何残余字段继承。
- DPO 三步总 loss 为 `1.2375 -> 1.1362 -> 1.0443`，DPO loss 有限；未出现
  NaN、Inf、OOM 或运行时错误。这里只验证运行稳定性，不将三点下降解释为泛化证据。
- cosine 实际 LoRA LR 为 `1e-5 -> 7.5e-6 -> 2.5e-6`，projection LR 为
  `1e-4 -> 7.5e-5 -> 2.5e-5`。
- `save_steps=1/save_total_limit=1` 正常删除 step 1/2，最终只保留
  `mm_dpo_step_3`。
- 最终 metadata 明确记录 `stage=multimodal_dpo`、`global_step=3`、
  `micro_step=3`、`max_steps=3`、schema=5，并保留正确 Stage-I checkpoint 路径；
  DPO 进度不再继承 SFT 的 2/2。

## P1 关闭结论

v5 数据生成、数据契约、无孤儿图片、control-only multimodal SFT、response-conditioned
multimodal DPO、学习率调度、梯度累积、checkpoint 轮转和 checkpoint provenance 已形成
最小完整闭环。P1 状态为 `complete`。本节产生的 2 条数据、2-step SFT 与 3-step DPO
checkpoint 均为 runtime artifact，不是论文结果或正式训练初始化。

下一步按 P0 closure 的既定门禁生成相互独立的 v5 train20/val20，只做数据分布、长度、
solver feasibility 和小规模训练质量预检；这些门禁通过前不生成 train500/val100，
更不启动正式联合训练。
