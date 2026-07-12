#!/usr/bin/env python
"""
Smoke test for multimodal processor inputs.

Checks:
  - one JSONL sample can be read
  - the BEV image path resolves and opens
  - AutoProcessor can encode text + image
  - control token ids can be appended and located by token id
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


def _load_jsonl_record(path: Path, sample_index: int) -> dict:
    current = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                if current == sample_index:
                    return json.loads(line)
                current += 1
    raise ValueError(f"No record index {sample_index} found in {path}")


def _get_tokenizer(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError("Processor does not expose a .tokenizer attribute")
    return tokenizer


def _encode_text_image(processor, prompt: str, image, max_length: int):
    kwargs = {
        "text": prompt,
        "images": image,
        "return_tensors": "pt",
        "truncation": True,
        "max_length": max_length,
    }
    try:
        return processor(**kwargs)
    except TypeError:
        kwargs.pop("truncation", None)
        kwargs.pop("max_length", None)
        return processor(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Smoke test multimodal processor")
    parser.add_argument("--config", type=str, default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="Override model.backbone from config")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    args = parser.parse_args()

    from PIL import Image
    from transformers import AutoProcessor
    import torch

    with (PROJECT_ROOT / args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]["sft"]
    data_cfg = cfg["data"]

    model_name = args.model or model_cfg["backbone"]
    data_dir = Path(args.data_dir or data_cfg["output_dir"])
    max_length = args.max_length or train_cfg["max_seq_length"]
    data_path = data_dir / data_cfg.get("sft_file", "sft_dataset.jsonl")

    item = _load_jsonl_record(data_path, args.sample_index)
    image_path = data_dir / item["bev_image_path"]
    image = Image.open(image_path).convert("RGB")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = _get_tokenizer(processor)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_control_tokens = int(model_cfg["control_token"]["num_tokens"])
    control_tokens = [f"<ctrl_{i}>" for i in range(num_control_tokens)]
    tokenizer.add_tokens(control_tokens, special_tokens=True)
    control_token_ids = tokenizer.convert_tokens_to_ids(control_tokens)
    if any(tid is None or tid == tokenizer.unk_token_id for tid in control_token_ids):
        raise ValueError("Control tokens were not registered correctly")

    encoded = _encode_text_image(processor, item["prompt"], image, max_length)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))

    ctrl_ids = torch.tensor([control_token_ids], dtype=input_ids.dtype, device=input_ids.device)
    input_ids = torch.cat([input_ids, ctrl_ids], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones_like(ctrl_ids)], dim=1)

    control_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in control_token_ids:
        control_mask |= input_ids == token_id

    pixel_keys = [
        key for key, value in encoded.items()
        if key not in {"input_ids", "attention_mask"} and hasattr(value, "shape")
    ]

    print("OK: multimodal processor smoke")
    print(f"  data: {data_path}")
    print(f"  image: {image_path} size={image.size}")
    print(f"  input_ids: {tuple(input_ids.shape)}")
    print(f"  attention_mask: {tuple(attention_mask.shape)}")
    for key in pixel_keys:
        print(f"  {key}: {tuple(encoded[key].shape)}")
    print(f"  control_token_count: {int(control_mask.sum().item())}")
    if int(control_mask.sum().item()) != num_control_tokens:
        raise RuntimeError("Control token count mismatch")


if __name__ == "__main__":
    main()
