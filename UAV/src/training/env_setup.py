"""
训练环境变量初始化 ("防爆盾" 0/1/2/3)

必须在 import numpy / torch 之前调用 setup_env()。
两个训练脚本 (train_sft.py / train_dpo.py) 原先各有一段 30 行的重复代码，
现提取为此模块。
"""

import os


def setup_env():
    """设置所有训练所需的环境变量和全局配置"""

    # ══ 线程控制: 防止 MKL/OpenBLAS 与 PyTorch DataLoader 多进程冲突 ══
    # 每个 worker 都试图开满全部核心 → CPU 100% 但进度卡死
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # PyTorch CUDA 内存分配器: 动态释放缓存段, 减少碎片化 OOM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # ── 防爆盾 0: 网络/遥测静默 ──
    # 0a: 禁止 Unsloth 连接 HuggingFace 上报统计 (国内超时 120s)
    # 虽然项目已移除 Unsloth, 但保留此设置以防未来误装
    os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
    # 0b: HuggingFace 镜像 (国内加速)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # ── 防爆盾 1: 核弹级环境变量 ──
    # Blackwell sm_120: 禁止 Inductor 使用 FlexAttention (共享内存不足)
    os.environ["TORCHINDUCTOR_FLEX_ATTENTION"] = "0"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"

    # ── 防爆盾 3: 代码级物理超度 FlexAttention ──
    # 即使环境变量未生效 (Python 已 import torch._inductor), 也强制禁用
    _disable_flex_attention()


def _disable_flex_attention():
    """强制禁用 PyTorch Inductor FlexAttention (Blackwell sm_120 兼容)"""
    try:
        import torch._inductor.config as inductor_config
        if hasattr(inductor_config, "flex_attention"):
            inductor_config.flex_attention = False
        if hasattr(inductor_config, "use_flex_attention"):
            inductor_config.use_flex_attention = False
    except ImportError:
        pass
