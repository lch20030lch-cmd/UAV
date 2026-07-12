"""
Multimodal datasets for BEV-image UAV-ISAC smoke and training.

The first target is SFT/forward smoke: prompt + BEV image are encoded by the
model processor, then control tokens and response labels are appended in the
same solver-facing format as the text-grid baseline.
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset


def get_image_token(processor) -> str:
    tokenizer = getattr(processor, "tokenizer", None)
    token = getattr(tokenizer, "boi_token", None) if tokenizer is not None else None
    if token is None and tokenizer is not None:
        token = getattr(tokenizer, "image_token", None)
    if token is None:
        token = getattr(processor, "image_token", None)
    if token is None:
        token = "<start_of_image>"
    return str(token)


def ensure_one_image_token(processor, prompt: str) -> str:
    image_token = get_image_token(processor)
    if image_token in prompt:
        return prompt
    marker = "[Bird's-Eye-View Image]"
    if marker in prompt:
        return prompt.replace(marker, f"{image_token}\n{marker}", 1)
    return f"{image_token}\n{prompt}"


def _encode_text_image(processor, prompt: str, image, max_length: int) -> Dict:
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


def _squeeze_batch(encoded: Dict) -> Dict:
    result = {}
    for key, value in encoded.items():
        if hasattr(value, "shape") and value.shape[0] == 1:
            result[key] = value.squeeze(0)
        else:
            result[key] = value
    return result


class MultimodalSFTDataset(Dataset):
    """SFT-style dataset for prompt + BEV image + control-token forward."""

    def __init__(
        self,
        data_path: str,
        data_dir: str,
        processor,
        max_length: int = 4096,
        num_control_tokens: int = 8,
    ):
        self.data_path = Path(data_path)
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.num_control_tokens = num_control_tokens

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        control_tokens = [f"<ctrl_{i}>" for i in range(num_control_tokens)]
        self.tokenizer.add_tokens(control_tokens, special_tokens=True)
        self.control_token_ids = self.tokenizer.convert_tokens_to_ids(control_tokens)
        if any(tid is None or tid == self.tokenizer.unk_token_id for tid in self.control_token_ids):
            raise ValueError("Control tokens must be added before building MultimodalSFTDataset.")

        self.data: List[Dict] = []
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = self.data_dir / item["bev_image_path"]
        image = Image.open(image_path).convert("RGB")

        prompt = ensure_one_image_token(self.processor, item["prompt"])
        encoded = _encode_text_image(self.processor, prompt, image, self.max_length)
        encoded = _squeeze_batch(encoded)

        prompt_ids = encoded["input_ids"].tolist()
        prompt_attention = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).tolist()
        prompt_token_type = encoded.get("token_type_ids", torch.ones_like(encoded["input_ids"])).tolist()

        response_enc = self.tokenizer(
            item["response"],
            truncation=True,
            max_length=1024,
            add_special_tokens=False,
        )
        response_ids = response_enc["input_ids"] + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + self.control_token_ids + response_ids
        attention_mask = prompt_attention + [1] * self.num_control_tokens + [1] * len(response_ids)
        token_type_ids = prompt_token_type + [1] * self.num_control_tokens + [1] * len(response_ids)

        labels = [-100] * (len(prompt_ids) + self.num_control_tokens) + response_ids
        label_mask = [0] * (len(prompt_ids) + self.num_control_tokens) + [1] * len(response_ids)
        control_mask = [0] * len(prompt_ids) + [1] * self.num_control_tokens + [0] * len(response_ids)

        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len
            labels += [-100] * pad_len
            label_mask += [0] * pad_len
            control_mask += [0] * pad_len
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            token_type_ids = token_type_ids[:self.max_length]
            labels = labels[:self.max_length]
            label_mask = label_mask[:self.max_length]
            control_mask = control_mask[:self.max_length]

        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "label_mask": torch.tensor(label_mask, dtype=torch.float32),
            "control_mask": torch.tensor(control_mask, dtype=torch.bool),
            "q_current": torch.tensor(item["q_current"], dtype=torch.float32),
            "has_q_current": torch.tensor(True),
            "delta_q_target": torch.tensor(item["delta_q"], dtype=torch.float32),
            "delta_a_target": torch.tensor(item["delta_a"], dtype=torch.float32),
            "delta_p_target": torch.tensor(item["delta_p"], dtype=torch.float32),
        }

        for key, value in encoded.items():
            if key in {"input_ids", "attention_mask", "token_type_ids"}:
                continue
            if hasattr(value, "shape"):
                result[key] = value

        return result
