#!/usr/bin/env python
"""
分析 SFT 数据集的真实 Token 长度分布。

多模态 control-only 模式使用 AutoProcessor + 实际 BEV 图像展开 image tokens；
不能用纯 tokenizer 长度推断 Gemma3 多模态 max_length。

如果大部分样本 << 4096, 砍短 max_seq_length 可大幅提速
(注意力计算与序列长度呈超线性增长)

用法:
  python scripts/analyze_seq_len.py --data-dir /root/autodl-tmp/data/full5000
"""

import os
import sys
import json
import argparse

# 前置环境变量
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TORCHINDUCTOR_FLEX_ATTENTION"] = "0"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multimodal_dataset import (
    _encode_text_image,
    format_multimodal_user_prompt,
    resolve_multimodal_chat_template,
    validate_multimodal_oracle_contract,
)


def analyze(
    data_path: str,
    model_name: str = "/root/autodl-tmp/huggingface/models/gemma-3-12b-it",
    control_only: bool = False,
    num_control_tokens: int = 8,
    current_max_length: int = 4096,
    use_chat_template: bool = None,
    allow_legacy_dataset: bool = False,
):
    # ---- 加载数据 ----
    print(f"Loading dataset from {data_path}...")
    samples = []
    with open(data_path, "r") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Total samples: {len(samples)}")

    use_multimodal_processor = bool(
        control_only
        and samples
        and all(sample.get("bev_image_path") for sample in samples)
    )
    processor = None
    if use_multimodal_processor:
        print(f"Loading multimodal processor from {model_name}...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        tokenizer = processor.tokenizer
        print("Length backend: multimodal processor + actual BEV images")
    else:
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("Length backend: tokenizer-only text estimate")

    # ---- 统计 ----
    lengths = []
    prompt_lengths = []
    response_lengths = []

    data_root = Path(data_path).resolve().parent
    dataset_metadata = validate_multimodal_oracle_contract(
        data_root,
        allow_legacy=allow_legacy_dataset,
    )
    use_chat_template_value = resolve_multimodal_chat_template(
        dataset_metadata=dataset_metadata,
        override=use_chat_template,
    )
    if use_multimodal_processor:
        print(f"Chat template: {use_chat_template_value}")
    for s in samples:
        # prompt: instruction + control part
        prompt_text = s.get("prompt", "")
        # response: expected output (labels)
        response_text = s.get("response", s.get("completion", ""))

        full_text = prompt_text + response_text

        full_len = len(tokenizer.encode(full_text, add_special_tokens=True))
        if use_multimodal_processor:
            image_path = data_root / s["bev_image_path"]
            with Image.open(image_path) as image_handle:
                image = image_handle.convert("RGB")
                prompt = format_multimodal_user_prompt(
                    processor,
                    prompt_text,
                    use_chat_template=use_chat_template_value,
                )
                encoded = _encode_text_image(
                    processor,
                    prompt,
                    image,
                    max_length=None,
                )
            prompt_len = int(encoded["input_ids"].shape[-1])
        else:
            prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=True))
        resp_len = full_len - prompt_len
        if use_multimodal_processor:
            # Response is not passed to the current control-loss model. Keep its
            # text-only length as a separate informational field.
            resp_len = len(tokenizer.encode(response_text, add_special_tokens=False))

        lengths.append(
            prompt_len + num_control_tokens if control_only else full_len
        )
        prompt_lengths.append(prompt_len)
        response_lengths.append(resp_len)

    lengths = np.array(lengths)
    prompt_lengths = np.array(prompt_lengths)
    response_lengths = np.array(response_lengths)

    # ---- 报告 ----
    print("\n" + "=" * 60)
    mode = (
        "Multimodal prompt + control tokens"
        if use_multimodal_processor
        else ("Prompt + control tokens" if control_only else "Prompt + Response")
    )
    print(f"  Token 长度分布 ({mode})")
    print("=" * 60)
    print(f"  Samples:     {len(lengths)}")
    print(f"  Min:         {lengths.min()}")
    print(f"  Max:         {lengths.max()}")
    print(f"  Mean:        {lengths.mean():.0f}")
    print(f"  Median:      {np.median(lengths):.0f}")
    print(f"  Std:         {lengths.std():.0f}")
    print(f"  90th %ile:   {np.percentile(lengths, 90):.0f}")
    print(f"  95th %ile:   {np.percentile(lengths, 95):.0f}")
    print(f"  99th %ile:   {np.percentile(lengths, 99):.0f}")
    print(f"  99.9th %ile: {np.percentile(lengths, 99.9):.0f}")
    print(f"  Prompt mean/max:   {prompt_lengths.mean():.0f} / {prompt_lengths.max()}")
    print(f"  Response mean/max: {response_lengths.mean():.0f} / {response_lengths.max()}")

    # ---- 分桶 ----
    buckets = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]
    print(f"\n{'Bucket':>8}  {'Count':>6}  {'Cum%':>8}")
    print("-" * 30)
    for b in buckets:
        count = (lengths <= b).sum()
        pct = count / len(lengths) * 100
        print(f"  ≤{b:5d}  {count:6d}  {pct:7.1f}%")

    # ---- 推荐 ----
    print("\n" + "=" * 60)
    print("  推荐 max_seq_length")
    print("=" * 60)

    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)

    for percentile, val in [(95, p95), (99, p99)]:
        # 向上取整到 128 的倍数
        rounded = int(np.ceil(val / 128) * 128)
        # 裁剪到合理范围
        rounded = max(512, min(4096, rounded))
        truncated = (lengths > rounded).sum()
        trunc_pct = truncated / len(lengths) * 100
        print(f"  {percentile}th %ile → {rounded:4d} tokens "
              f"(截断 {truncated}/{len(lengths)} = {trunc_pct:.2f}% 样本)")

    # ---- 加速预估 ----
    current = current_max_length
    print(f"\n  当前 max_seq_length = {current}")
    for candidate in [2048, 2560, 3072]:
        if candidate < p95:
            continue
        speedup = (current ** 2) / (candidate ** 2)  # 近似: 注意力 O(n²)
        print(f"  若改为 {candidate} → 注意力加速约 {speedup:.1f}×"
              f" (截断 ~{(lengths > candidate).sum()} 样本)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to data directory containing sft_dataset.jsonl")
    parser.add_argument("--model", type=str,
                        default="/root/autodl-tmp/huggingface/models/gemma-3-12b-it")
    parser.add_argument("--control-only", action="store_true",
                        help="按当前 multimodal control-loss 管线统计 prompt + control tokens")
    parser.add_argument("--num-control-tokens", type=int, default=8)
    parser.add_argument("--current-max-length", type=int, default=4096)
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override formatting; schema-v5 data defaults to the Gemma chat template",
    )
    parser.add_argument("--allow-legacy-dataset", action="store_true")
    args = parser.parse_args()

    dataset_metadata = validate_multimodal_oracle_contract(
        args.data_dir,
        allow_legacy=args.allow_legacy_dataset,
    )
    data_path = os.path.join(
        args.data_dir,
        dataset_metadata.get("sft_file", "sft_dataset.jsonl"),
    )
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found!")
        sys.exit(1)

    analyze(
        data_path,
        args.model,
        control_only=args.control_only,
        num_control_tokens=args.num_control_tokens,
        current_max_length=args.current_max_length,
        use_chat_template=args.use_chat_template,
        allow_legacy_dataset=args.allow_legacy_dataset,
    )
