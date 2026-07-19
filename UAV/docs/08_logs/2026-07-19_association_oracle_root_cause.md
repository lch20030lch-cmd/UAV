---
type: log
status: corrected_val20_q_preflight_passed_a_overfit_pending
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

## Compact v4 长度复查与控制序列根因

紧凑 train20：

```text
full text min / mean / max: 3199 / 3219 / 3240
prompt mean / max:          2400 / 2425
response mean / max:         819 / 823
3072 内 full text:             0 / 20
```

继续审计发现 `train_sft_mm.py` 明确只计算 control loss，forward 也设置
`logits_to_keep=1`，不会使用语言 labels；但 `MultimodalSFTDataset` 仍把约 819 个 response
token 附加在 control token 后。这些 token 不提供当前训练目标，只增加计算与截断风险。

同时旧数据集先允许 image prompt 占满 `max_length`，再附加 control token，最后统一从尾部
截断；极端情况下 control token 会被静默裁掉，而模型用零向量补齐。这会隐藏数据过长问题。

修复：

1. `MultimodalSFTDataset` 新增 `include_response`，默认保持兼容；
2. 当前 control-loss 训练、delta 诊断和 forward smoke 显式使用
   `include_response=False`；
3. processor 编码 prompt 前先硬预留 8 个 control token 的预算；
4. 截断后若 control token 数量不是 8，立即报错，不再静默用零向量掩盖；
5. checkpoint metadata 记录 `include_response_tokens=false`；
6. `analyze_seq_len.py --control-only` 按真实主线统计 prompt + control tokens，而不是把
   未参与损失的 response 算入预算。

这不是删除监督标签：`delta_q/a/p` 仍由 JSONL 独立字段提供并参与 control loss；只是当前
训练不再把未使用的自然语言 JSON response 送入 backbone。未来若启用 token-level LM CE，
仍可使用默认 `include_response=True`，届时会同时为 response 与 control token 预留预算。

## 真实 processor / image / checkpoint forward 验收

使用 compact train20、Gemma-3-4B processor 和 Q selected checkpoint，在
`max_length=3072` 下完成真实前向：

```text
input_ids:          (1, 3072)
pixel_values:       (1, 3, 896, 896)
control_token_count: 8
control_states:     (1, 8, 2560)
delta_q:            (1, 4, 3)
delta_a:            (1, 4, 20)
delta_p:            (1, 4, 21)
loaded projection/control embeddings/LoRA: selected step150
```

脚本正常输出 `OK`，Q/A/P 无 NaN。序列预算、图像 processor、control token 提取与旧 Q
checkpoint 加载链路通过。下一步生成独立 seed2026 corrected val20，先复验 Q，不训练 A。

## Corrected val20 上的 Q selected 复验

使用独立 `seed=2026` corrected val20：

```text
Q selected 3D cosine:      0.570066
fixed geometry 3D cosine:  0.555483
Q selected XY cosine:      0.671826
fixed geometry XY cosine:  0.672722
direction std / target:    0.416991 / 0.553295
mobility violation ratio:  0.0
Q warning:                 none
```

相对 fixed：

```text
3D gain:   +0.014582
XY change: -0.000896
```

判定：

1. 满足既定的“3D 高于 fixed、XY 下降不超过 0.01、物理违规为 0”门槛；
2. corrected val20 只作为分布迁移预检，不能替代后续 corrected val100 最终报告；
3. 现阶段不重训 Q，继续保留 selected step150；
4. residual 增益从旧 val100 的约 +0.0416 缩小到 +0.0146，后续 v4 val100 必须复查；
5. A/P 当前输出来自旧标签与旧 prompt 训练参数，不能用本次 `A=0.24` 或 P 指标宣称
   corrected 分支失败；它们将在对应 corrected-data 阶段重新训练。

下一步只做 corrected train20 的 A projection-only overfit 预检，Q/P/LoRA 全部冻结。

## A-only 预检前的诊断与冻结补丁

在启动 A 训练前补齐两项必要保护：

1. `analyze_mm_delta_outputs.py` 同时报告投影后 A 与 raw association logits 的
   top-1 accuracy、top-2 accuracy、oracle probability、top-1 margin、逐用户准确率范围及
   预测/目标直方图。raw logits 使用稳定 softmax，避免把负 logits 当概率裁剪；
2. 修复 `--freeze_qp_branch` 未冻结 `q_residual_adapter` 的遗漏。该参数现在会冻结完整 Q
   （包括 fixed-residual adapter）和 P，只保留 `readout_a/a_mlp` 可训练。

这样 A-only 失败时可以区分：

- raw accuracy 也不上升：control states/A readout 没学到 corrected association；
- raw accuracy 上升但 projected accuracy 不升：问题位于 A 投影/Sinkhorn；
- train20 上升但 val20 不升：仅记忆，没有跨环境泛化。

本机完成 Python 语法检查与 `git diff --check`。本机默认 Python 缺少 NumPy，相关单元测试
必须在服务器 `uavmllm` 环境执行后，才能启动 A-only 训练。

## Corrected val20 的 A-only 训练前基线

使用 selected Q checkpoint 和新诊断器得到：

```text
projected top-1 / top-2:    0.2400 / 0.4975
raw top-1 / top-2:          0.2400 / 0.4975
fixed-user majority:        0.3675
raw oracle probability:     0.250850
raw top-1 margin:            0.037891
raw entropy:                 1.378901
raw prediction histogram:   58 / 48 / 9 / 285
target histogram:            91 / 107 / 101 / 101
fixed users / unique mean:   5 / 2.0
control-state std mean/max:  0.125314 / 2.190320
```

判定：

1. raw 与 projected 排序指标完全相同，当前失败不是 Sinkhorn 改坏了已有正确排序；
2. 四分类随机 oracle probability 为 `0.25`，当前 raw 值几乎等于随机；raw entropy 也接近
   `ln(4)=1.3863`，说明 logits 区分度很弱；
3. `285/400` 个预测落在 UAV 3，而目标在四架 UAV 间近似均衡，存在明显单类偏置；
4. 该旧 checkpoint 没有学过 compact corrected prompt。下一步仍是冻结 backbone/LoRA/Q/P，
   仅用 corrected train20 过拟合 A readout；此实验将判断现有 control states 是否仍包含
   可供 A 头读取的场景信息。

## 对现有数据与 Q checkpoint 的影响边界

现有 train500/val100 是旧 solver 生成的，不能直接用于验证修复后的 A 标签质量。
但现在也不立刻重跑完整数据或推翻 Q：

1. 先通过 solver 单测；
2. 再补齐 indexed input；
3. 只生成小规模 corrected-data preflight；
4. 比较新旧 `delta_a/delta_q/delta_p` 标签变化；
5. 只有 `delta_q` 标签确实发生实质变化时，才重新验证 Q selected checkpoint。

在完成小规模影响评估前，保留并冻结现有 Q selected checkpoint，不删除、不重训。
