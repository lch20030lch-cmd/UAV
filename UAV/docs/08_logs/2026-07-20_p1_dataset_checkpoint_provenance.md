# 2026-07-20 P1 数据、物理配置与 checkpoint 来源一致性修复

## 前置验收

服务器 P0 回归测试共 43 项，全部通过。因此本轮在 P0 基线上进入 P1，
不回退 Solver/Oracle 主逻辑。

## 根因

P0 建立了 v5 数据契约，但仍有四个会污染后续实验的 P1 风险：

1. 断点续生成只比较 solver 名称与 seed；修改带宽、噪声、天线数、SINR
   门限等物理配置后，仍可能继续写入同一目录。
2. 进程若在追加 SFT/DPO 后、更新 `checkpoint.txt` 前硬中断，恢复时可能
   重复使用刚写入的环境 ID。
3. checkpoint 没有记录训练数据的物理来源，旧 v3/v4 checkpoint 可以静默
   加载到 v5 主线；其中旧的 Q fixed-mixture 权重也会被继续使用。
4. 场景摘要和端到端评估仍有 20 MHz 硬编码，且场景/solver 内部 Channel
   没有完整接收接收天线数与 noise figure。

## 修复内容

- 新增 `src/data/oracle_contract.py`，集中维护：
  - schema、prompt、solver revision 与 channel model；
  - 完整仿真配置的规范化 SHA-256 fingerprint；
  - 生成续跑兼容性检查；
  - SFT/DPO 成对 ID、重复 ID、实际行数检查；
  - checkpoint 与数据物理来源兼容性检查。
- v5 生成器在写入前校验不可变来源字段。恢复时同时读取 JSONL 中的最大
  环境 ID 与 `checkpoint.txt`，选择二者较大值，避免硬中断后重复样本。
- SFT checkpoint 保存 dataset provenance；从 checkpoint 继续训练时默认要求
  同一份训练数据来源。仅诊断/显式迁移可使用
  `--allow_checkpoint_dataset_mismatch`。
- 多模态 DPO 在加载两个大模型前先检查 Stage-I checkpoint 与训练数据来源；
  端到端独立验证允许 seed 不同，但要求物理配置、schema、solver/channel
  版本一致。
- projection head 配置统一从 checkpoint metadata 恢复。`fixed_residual_xy`
  不再内置旧 v3 固定权重；必须提供在当前数据契约上重新标定的三项权重。
- SFT/DPO response 默认长度改为 4096；移除 DPO 中写死的 compact-v4
  3584-token 判断，由真实样本预算检查负责报错。
- `ISACScenarioGenerator`、`SCAFPOptimizer` 与端到端评估完整传递 bandwidth、
  Tx/Rx antenna、noise figure 与 power；速率统一使用实际 `channel.B`。

## 行为变化

- 旧 v3/v4 checkpoint 不能作为 v5 正式训练的默认初始化。这是有意的保护，
  因为 P0 已改变 Oracle solver/channel 语义。
- 旧 checkpoint 仍可用于历史诊断，但必须显式使用 mismatch/legacy override；
  不能把这种结果当作 v5 主线结果。
- 新 v5 train/validation 可以使用不同 seed；checkpoint 在独立验证集上评估
  不会因为 seed 不同而被拒绝。
- 在新 v5 Q 几何统计完成之前，不启用 `fixed_residual_xy`，也不复用旧权重
  `[0.31186843, 0.09240539, 0.59572625]`。

## 本地验证

- `python -m compileall -q src scripts tests`: PASS。
- Ruff F/E9：PASS。
- `git diff --check`: PASS。
- 18 项 NumPy/SciPy 动态测试：PASS，包括：
  - 物理 fingerprint 稳定性与参数敏感性；
  - 不兼容配置拒绝续跑；
  - checkpoint/data provenance gate；
  - SFT/DPO ID 对齐与 crash-safe next ID；
  - bandwidth/Rx antenna/noise figure 传播；
  - P0 association/solver 回归。

本机没有完整 Torch 环境，因此新增 projection/checkpoint 路径仍需服务器回归。

## 服务器验收命令

```bash
python -m unittest \
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

上述测试通过前不生成 v5 数据。通过后继续检查剩余 P1 训练运行项，不直接
进入正式训练。
