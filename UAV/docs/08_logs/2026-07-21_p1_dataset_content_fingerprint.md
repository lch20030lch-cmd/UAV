# 2026-07-21 P1.1 数据内容 fingerprint 修复

## 发现背景

v5 train20/val20 数据 gate 已通过：两套数据物理 fingerprint 一致、seed 独立，40/40
Oracle feasible，SFT/DPO/BEV 均为 20/20/20，约束误差接近机器精度，所有 DPO utility
gap 为正，Q/A/P target 分布无坍缩，control-only 多模态长度在 3072 内无截断。

准备 Stage-I 质量预检时发现，checkpoint provenance 只包含 schema、solver/channel、
物理 fingerprint 和 seed，没有包含实际 JSONL 内容。tiny2 runtime 数据与 train20 都使用
seed42，因此旧门禁可能错误允许 tiny2 checkpoint 初始化 train20。

## 修复

- 新增 `dataset_content_fingerprint`：按固定顺序对完整 SFT/DPO JSONL 和两阶段引用的
  BEV 图片字节做 SHA-256，包括记录顺序、每个字段及实际多模态输入；不把无关孤儿
  文件计入数据身份。
- 完整生成的新数据把 `content_fingerprint` 写入 `dataset_metadata.json`。
- 已完成数据再次执行同一生成命令时，不启动 solver，而是校验已有 fingerprint 并为
  旧 v5 metadata 快速回填；若已保存的 fingerprint 与文件不一致则拒绝覆盖。
- `--overwrite` 同时清理生成器命名空间内的 `images/env_*.png`，避免旧图片残留造成
  SFT/DPO/BEV 数量不一致。
- 已生成的 v5 数据无需重生；`validate_dataset_metadata(..., data_dir=...)` 会从现有文件
  计算并返回 fingerprint。若 metadata 已保存 fingerprint，则同时校验文件未被修改。
- checkpoint 保存 `dataset_content_fingerprint`。
- Stage-I continuation 与 DPO 使用 `require_same_seed=True`，因此同时要求实际训练记录
  完全一致；tiny2 checkpoint 不能再静默加载到 train20。
- 独立 validation/evaluation 不要求相同 seed 或相同记录内容，但仍要求 schema、物理
  fingerprint、solver 与 channel revision 一致。

## 回归覆盖

- 相同 schema/seed/物理配置但 JSONL utility 不同，内容 fingerprint 必须不同。
- JSONL 完全相同但任一引用 BEV 图片字节不同，内容 fingerprint 必须不同。
- 已完成数据的快速返回路径必须回填 fingerprint，且不能掩盖已保存指纹的不一致。
- checkpoint 与相同数据内容兼容。
- checkpoint 与同 seed、不同内容的训练数据必须因
  `dataset_content_fingerprint` 不同而拒绝。
- 同一 checkpoint 在 held-out evaluation 模式下允许不同内容，但不能绕过物理契约。

## 下一步

本地 `py_compile`、`git diff --check`、直接契约闭环检查和 5 个 runtime 单测已通过；
完整 Oracle/generator 测试需在具备 numpy/torch 的服务器环境执行。服务器回归通过后，
先为现有 train20/val20 metadata 回填内容 fingerprint，再从基础 Gemma 模型启动 v5
train20 Stage-I 质量预检；不使用 tiny runtime checkpoint。
