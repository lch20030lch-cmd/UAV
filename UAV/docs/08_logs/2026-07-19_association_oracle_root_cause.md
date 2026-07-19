---
type: log
status: compact_v4_input_implemented_length_recheck_pending
stage: association_oracle_root_cause
last_updated: 2026-07-19
---

# 2026-07-19 Association Oracle 标签锁死根因

## 为什么 A3/A4 一直欠拟合

已有结果不是典型的过拟合：

```text
A3 train accuracy: 0.4372
A3 val accuracy:   0.4280
A4 train accuracy: 0.4445
A4 val accuracy:   0.4300
```

增加 projection-only 步数和小学习率 LoRA 都几乎没有改善，说明继续堆训练步数不是正确
方向。代码审计发现两个上游可辨识性问题。

## 根因 1：solver 的 association 被当前零功率锁死

旧交替优化顺序为：

```text
当前 A -> beamforming -> P_comm -> association cost
```

beamforming 只给当前 `A[m,k]=1` 的位置分配通信功率，未关联位置满足：

```text
P_comm[m,k] = 0
rate(m,k) = log2(1 + gain[m,k] * 0 / N0) = 0
```

旧 association cost 又直接使用该 `P_comm[m,k]`。当前关联位置的 cost 为负，所有候选
新 UAV 的 cost 为 0，因此用户几乎不可能切换 UAV。所谓 association 优化实际复用了
random restart 初始化出来的最近 UAV；容量溢出时还会通过顺序后处理产生任意重分配。

这会导致 best-of-N oracle 标签混入随机初始化，而不是稳定的候选链路优化结果。

## 最小替换

删除旧的“逐用户 argmin + 容量溢出后处理”，替换为：

1. 为每个 `(UAV,user)` 使用非零、统一的候选功率份额计算反事实可达速率；
2. 把每架 UAV 展开为 `K_max` 个容量 slot；
3. 对 slot-user cost 一次执行 Hungarian 指派；
4. 严格保证每个用户恰好关联一次、每架 UAV 不超过 `K_max`；
5. 当 `M*K_max < K` 时明确报错，不生成违反容量的标签。

新实现比旧后处理更短，不新增第二套 solver 路径。

## 测试覆盖

新增 `tests/test_association_solver.py`：

1. 当前 UAV 有功率、候选 UAV 当前功率为零时，仍能切换到更强候选链路；
2. 所有用户偏好同一 UAV 时，容量指派仍满足列和为 1、行和不超过上限；
3. 总容量不足时拒绝生成无效关联。

本地完成：

```text
python -m py_compile src/solver/sca_fp.py tests/test_association_solver.py: PASS
git diff --check: PASS
```

本机没有 numpy/scipy，数值单测需在服务器 `uavmllm` 环境运行。

服务器已执行包含新 solver 测试在内的完整回归：

```text
Ran 21 tests in 0.200s
OK
```

三个 association solver 测试、Q geometry、delta diagnostics、分支冻结与 power 测试
全部通过。Oracle 标签锁死修复验收完成。

## 根因 2：输入缺少输出矩阵的索引映射

当前 BEV 将所有用户画为没有编号的绿色点、所有 UAV 画为没有编号的蓝色三角；
`draw_association=False`。prompt 虽然列出 `UAV m` 的坐标，却没有给出完整的
`user k -> xy/channel candidates` 映射。

但 `delta_a[:,k]` 必须严格对应用户 `k`。模型无法从无编号点云确定某个图像点究竟是
输出矩阵的第几列。这是第二个独立的信息缺口。

该问题暂不和 solver 修复混在同一个提交中。solver 数值测试通过后，再用一个紧凑的
indexed association map 补齐 user/UAV 身份与候选链路信息，不修改 A 投影头结构。

## Indexed Association Map 实现

solver 的 21 项回归通过后，第二个提交只修改数据输入，不修改模型：

```text
[Indexed Association Map]
delta_a rows: m0..m3
delta_a columns: u0..u19
u{k}: xy, demand weight, best SINR, UAV channel rank, ranked relative gain dB
```

设计选择：

1. UAV 坐标继续复用已有 Geometry Guidance，不重复增加一套视觉分支；
2. 每个用户一行，明确矩阵列索引与 BEV 位置的对应关系；
3. 给出完整四 UAV 候选排名和相对增益，容量约束导致首选 UAV 满载时仍有次优信息；
4. BEV 保持无文字拥挤，精确 ID 由文本 map 提供；
5. 新数据写入 `prompt_type=multimodal_bev_image_v4_indexed_association`，防止和旧 v3
   数据静默混用。

新增 `tests/test_association_prompt.py`，覆盖用户列 ID、位置、权重、SINR、候选 UAV
排名、map 在图像说明前的顺序以及 shape 校验。

本地静态检查：

```text
python -m py_compile src/data/prompt_builder.py scripts/generate_mm_smoke.py
                     tests/test_association_prompt.py: PASS
git diff --check: PASS
```

服务器随后完成包含 indexed prompt 在内的完整回归：

```text
Ran 24 tests in 0.248s
OK
```

## Corrected-data 影响比较工具

在生成小规模 v4 数据前，扩展现有 `analyze_mm_target_distribution.py`，新增：

```text
--reference_data_dir
--reference_sft_file
```

脚本严格按样本 `id` 对齐新旧数据，不按文件行号猜测对应关系；输出：

```text
new-vs-old delta_q 3D / XY cosine
delta_q MSE / norm MAE
delta_a argmax match / switch rate
delta_p overall / sensing MSE
current / reference prompt_type histogram
```

新增 `tests/test_target_distribution_comparison.py`，覆盖乱序 ID 对齐、完全一致标签、Q
方向改变、A 切换、P 差异和重复 ID 拒绝。该工具只读取 JSONL 与 numpy，不加载大模型。

## Corrected train20 首轮结果

新旧同 ID 的 20 个 seed42 样本：

```text
delta_q 3D cosine:       0.953551
delta_q XY cosine:       0.972835
delta_q norm MAE:        0.002025
delta_a argmax match:    0.6975
delta_a switch rate:     0.3025
delta_p MSE:             0.013188
delta_p sensing MSE:     0.110250
```

判定：

1. solver 修复不是无效改动，约 30.25% 用户关联标签发生变化；旧 A 标签停止使用；
2. P 标签也发生实质变化，后续 P 必须建立在 corrected association 上；
3. Q 位移范数不变，方向变化相对较小但非零；保留 Q selected checkpoint，等待新 val
   复验，不直接宣称在 v4 上达标，也不立即重训。

## 首版 indexed map 长度失败与替换

首版 v4 train20 token 统计：

```text
min / mean / max: 4681 / 4746 / 4806
3072 内样本:      0 / 20
4096 内样本:      0 / 20
```

该表示会让 association map 或响应被 100% 截断，因此禁止继续生成或训练，也不通过把
`max_length` 硬加到 4800 掩盖问题。

替换方案仍保留全部必要信息，但删除每行重复字段：

```text
一次性表头: u|x,y|weight|UAV-rank|relative-gain-dB
每用户一行: 0|123,456|1.25|2,0,3,1|0,-3,-8,-12
```

同时把通信/感知摘要中的 Python 全精度浮点列表改为任务足够的固定精度紧凑列表。
新 prompt type：

```text
multimodal_bev_image_v4_compact_indexed_association
```

`analyze_seq_len.py` 也新增 prompt 与 response 各自的 mean/max，下一轮能明确长度主要来自
哪一部分。旧 verbose v4 train20 仅保留为失败审计样本，不用于训练。

## 对现有数据与 Q checkpoint 的影响边界

现有 train500/val100 是旧 solver 生成的，不能直接用于验证修复后的 A 标签质量。
但现在也不立刻重跑完整数据或推翻 Q：

1. 先通过 solver 单测；
2. 再补齐 indexed input；
3. 只生成小规模 corrected-data preflight；
4. 比较新旧 `delta_a/delta_q/delta_p` 标签变化；
5. 只有 `delta_q` 标签确实发生实质变化时，才重新验证 Q selected checkpoint。

在完成小规模影响评估前，保留并冻结现有 Q selected checkpoint，不删除、不重训。
