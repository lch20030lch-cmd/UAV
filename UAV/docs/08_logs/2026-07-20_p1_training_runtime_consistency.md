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
