---
type: log
status: solver_repair_implemented_server_tests_pending
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

## 根因 2：输入缺少输出矩阵的索引映射

当前 BEV 将所有用户画为没有编号的绿色点、所有 UAV 画为没有编号的蓝色三角；
`draw_association=False`。prompt 虽然列出 `UAV m` 的坐标，却没有给出完整的
`user k -> xy/channel candidates` 映射。

但 `delta_a[:,k]` 必须严格对应用户 `k`。模型无法从无编号点云确定某个图像点究竟是
输出矩阵的第几列。这是第二个独立的信息缺口。

该问题暂不和 solver 修复混在同一个提交中。solver 数值测试通过后，再用一个紧凑的
indexed association map 补齐 user/UAV 身份与候选链路信息，不修改 A 投影头结构。

## 对现有数据与 Q checkpoint 的影响边界

现有 train500/val100 是旧 solver 生成的，不能直接用于验证修复后的 A 标签质量。
但现在也不立刻重跑完整数据或推翻 Q：

1. 先通过 solver 单测；
2. 再补齐 indexed input；
3. 只生成小规模 corrected-data preflight；
4. 比较新旧 `delta_a/delta_q/delta_p` 标签变化；
5. 只有 `delta_q` 标签确实发生实质变化时，才重新验证 Q selected checkpoint。

在完成小规模影响评估前，保留并冻结现有 Q selected checkpoint，不删除、不重训。
