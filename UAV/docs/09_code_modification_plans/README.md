# Code Modification Plans

这个目录用于存放后续代码修改计划、实现路线和阶段性重构设计。

当前文档：

| 文件 | 内容 |
|---|---|
| `bev_image_mllm_implementation_plan_2026-07-07.md` | BEV-image MLLM 分支实现方案，包含需要保留/新增/改造的模块、训练烟测顺序、风险与验收标准 |
| `rtx5090_32g_mllm_smoke_plan_2026-07-08.md` | RTX 5090 32GB 省钱路线：先跑通 BEV-image MLLM 最小闭环，只做数据、processor、forward、SFT smoke，不做正式大训练 |

使用建议：

1. 先在这里写清楚设计和修改边界。
2. 再按阶段实现代码。
3. 每完成一个阶段，在对应文档中补充结果或新建阶段记录。
