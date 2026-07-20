"""
BEV 图像多模态数据集。

当前主要服务于 SFT / 前向传播烟雾测试：
先用模型 processor 编码 prompt + BEV image，再追加 control tokens；仅在显式要求时
追加 response labels。最终 batch 仍保持与 text-grid baseline 一致的
solver-facing 字段，便于复用 projection head 与控制损失。
"""

import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.geometry_cues import parse_q_geometry_cues
from src.data.oracle_contract import validate_dataset_metadata


def validate_multimodal_oracle_contract(
    data_dir: str,
    *,
    allow_legacy: bool = False,
    expected_simulation: Dict = None,
    expected_seed: int = None,
) -> Dict:
    """Reject stale Oracle data after solver/channel semantics change."""
    metadata_path = Path(data_dir) / "dataset_metadata.json"
    if not metadata_path.exists():
        if allow_legacy:
            return {}
        raise FileNotFoundError(
            f"missing {metadata_path}; regenerate v5 Oracle data or explicitly "
            "pass the legacy-data override for diagnostics only"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    try:
        return validate_dataset_metadata(
            metadata,
            data_dir=Path(data_dir),
            expected_simulation=expected_simulation,
            expected_seed=expected_seed,
        )
    except (KeyError, TypeError, ValueError):
        if allow_legacy:
            return metadata
        raise


def get_image_token(processor) -> str:
    # 不同 transformers 版本暴露的图像 token 名称不完全一致，这里集中做兼容。
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


def format_multimodal_user_prompt(
    processor, prompt: str, *, use_chat_template: bool
) -> str:
    """Format one image/text user turn for an instruction-tuned checkpoint.

    Legacy control-only checkpoints were trained from raw prompt text, so the
    behavior remains opt-in.  New response-SFT and DPO stages must enable the
    chat template to match Gemma instruction-tuning semantics.
    """
    if not use_chat_template:
        return ensure_one_image_token(processor, prompt)
    apply_template = getattr(processor, "apply_chat_template", None)
    if apply_template is None:
        raise ValueError(
            "use_chat_template=True but the multimodal processor does not "
            "provide apply_chat_template"
        )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    formatted = apply_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return ensure_one_image_token(processor, str(formatted))


def _encode_text_image(
    processor,
    prompt: str,
    image,
    max_length: int = None,
) -> Dict:
    """Encode one multimodal prompt without truncating through image tokens."""
    kwargs = {
        "text": prompt,
        "images": image,
        "return_tensors": "pt",
    }
    encoded = processor(**kwargs)
    input_length = int(encoded["input_ids"].shape[-1])
    if max_length is not None and input_length > int(max_length):
        raise ValueError(
            "Encoded multimodal prompt exceeds the reserved prompt budget: "
            f"encoded_length={input_length}, prompt_budget={max_length}. "
            "Increase max_length or compact the prompt; multimodal image tokens "
            "must not be truncated."
        )
    return encoded


def _squeeze_batch(encoded: Dict) -> Dict:
    result = {}
    for key, value in encoded.items():
        if hasattr(value, "shape") and value.shape[0] == 1:
            result[key] = value.squeeze(0)
        else:
            result[key] = value
    return result


def _compute_prompt_budget(
    max_length: int,
    num_control_tokens: int,
    response_length: int,
) -> int:
    budget = int(max_length) - int(num_control_tokens) - int(response_length)
    if budget <= 0:
        raise ValueError(
            "max_length cannot fit reserved control/response tokens: "
            f"max_length={max_length}, controls={num_control_tokens}, "
            f"response={response_length}"
        )
    return budget


class MultimodalSFTDataset(Dataset):
    """用于 prompt + BEV image + control-token forward 的 SFT 风格数据集。"""

    def __init__(
        self,
        data_path: str,
        data_dir: str,
        processor,
        max_length: int = 4096,
        num_control_tokens: int = 8,
        include_response: bool = True,
        use_chat_template: bool = False,
    ):
        self.data_path = Path(data_path)
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.num_control_tokens = num_control_tokens
        self.include_response = include_response
        self.use_chat_template = use_chat_template

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

        response_ids = []
        if self.include_response:
            response_enc = self.tokenizer(
                item["response"],
                truncation=True,
                max_length=1024,
                add_special_tokens=False,
            )
            response_ids = response_enc["input_ids"] + [self.tokenizer.eos_token_id]

        # 先为 control/response 预留硬预算，防止 prompt 截断后把控制 token 静默裁掉。
        prompt_budget = _compute_prompt_budget(
            self.max_length,
            self.num_control_tokens,
            len(response_ids),
        )
        prompt = format_multimodal_user_prompt(
            self.processor,
            item["prompt"],
            use_chat_template=self.use_chat_template,
        )
        encoded = _encode_text_image(self.processor, prompt, image, prompt_budget)
        encoded = _squeeze_batch(encoded)

        prompt_ids = encoded["input_ids"].tolist()
        prompt_attention = encoded.get("attention_mask", torch.ones_like(encoded["input_ids"])).tolist()
        prompt_token_type = encoded.get("token_type_ids", torch.ones_like(encoded["input_ids"])).tolist()
        if len(prompt_token_type) < len(prompt_ids):
            fill_value = prompt_token_type[-1] if prompt_token_type else 1
            prompt_token_type += [fill_value] * (len(prompt_ids) - len(prompt_token_type))
        elif len(prompt_token_type) > len(prompt_ids):
            prompt_token_type = prompt_token_type[:len(prompt_ids)]

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

        if sum(control_mask) != self.num_control_tokens:
            raise RuntimeError(
                "Control tokens were truncated from the multimodal sequence: "
                f"expected {self.num_control_tokens}, got {sum(control_mask)}"
            )

        if not (len(input_ids) == len(attention_mask) == len(token_type_ids) == len(labels) == len(control_mask)):
            raise RuntimeError("Multimodal token fields have inconsistent lengths after padding/truncation.")

        num_uavs = len(item["q_current"])
        q_geometry_cues, q_geometry_mask = parse_q_geometry_cues(item["prompt"], num_uavs)

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
            "q_geometry_cues": torch.tensor(q_geometry_cues, dtype=torch.float32),
            "q_geometry_mask": torch.tensor(q_geometry_mask, dtype=torch.float32),
        }

        for key, value in encoded.items():
            if key in {"input_ids", "attention_mask", "token_type_ids"}:
                continue
            if hasattr(value, "shape"):
                result[key] = value

        return result


def _masked_response_tokens(tokenizer, response: str, mask_fields: List[str]):
    encoded = tokenizer(
        response,
        truncation=True,
        max_length=1024,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    response_ids = list(encoded["input_ids"])
    response_mask = [1] * len(response_ids)
    offsets = encoded.get("offset_mapping", [])
    for field in mask_fields:
        match = re.search(rf'"{re.escape(field)}"\s*:', response)
        if match is None:
            continue
        next_match = re.search(r'"delta_[qap]"\s*:', response[match.end():])
        end = match.end() + next_match.start() if next_match else len(response)
        for token_idx, (char_start, char_end) in enumerate(offsets):
            midpoint = (char_start + char_end) / 2.0
            if char_end > char_start and match.start() <= midpoint < end:
                response_mask[token_idx] = 0
    return response_ids, response_mask


class MultimodalDPODataset(Dataset):
    """Image-conditioned chosen/rejected pairs for multimodal DPO."""

    def __init__(
        self,
        data_path: str,
        data_dir: str,
        processor,
        max_length: int = 4096,
        num_control_tokens: int = 8,
        mask_fields: List[str] = None,
        use_chat_template: bool = True,
    ):
        self.data_path = Path(data_path)
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = int(max_length)
        self.num_control_tokens = int(num_control_tokens)
        self.mask_fields = list(mask_fields or ["delta_a", "delta_p"])
        self.use_chat_template = bool(use_chat_template)
        control_tokens = [f"<ctrl_{i}>" for i in range(num_control_tokens)]
        self.tokenizer.add_tokens(control_tokens, special_tokens=True)
        self.control_token_ids = self.tokenizer.convert_tokens_to_ids(control_tokens)
        self.data = []
        with self.data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def _encode(self, prompt: str, response: str, image) -> Dict:
        response_ids, response_mask = _masked_response_tokens(
            self.tokenizer, response, self.mask_fields
        )
        response_ids.append(self.tokenizer.eos_token_id)
        response_mask.append(1)
        prompt_budget = _compute_prompt_budget(
            self.max_length, self.num_control_tokens, len(response_ids)
        )
        formatted_prompt = format_multimodal_user_prompt(
            self.processor,
            prompt,
            use_chat_template=self.use_chat_template,
        )
        encoded = _squeeze_batch(
            _encode_text_image(
                self.processor, formatted_prompt, image, prompt_budget
            )
        )
        prompt_ids = encoded["input_ids"].tolist()
        prompt_attention = encoded.get(
            "attention_mask", torch.ones_like(encoded["input_ids"])
        ).tolist()
        prompt_types = encoded.get(
            "token_type_ids", torch.ones_like(encoded["input_ids"])
        ).tolist()
        if len(prompt_types) < len(prompt_ids):
            fill_value = prompt_types[-1] if prompt_types else 1
            prompt_types += [fill_value] * (len(prompt_ids) - len(prompt_types))
        elif len(prompt_types) > len(prompt_ids):
            prompt_types = prompt_types[: len(prompt_ids)]
        input_ids = prompt_ids + self.control_token_ids + response_ids
        attention_mask = prompt_attention + [1] * (
            self.num_control_tokens + len(response_ids)
        )
        token_type_ids = prompt_types + [1] * (
            self.num_control_tokens + len(response_ids)
        )
        labels = [-100] * (len(prompt_ids) + self.num_control_tokens) + [
            token if keep else -100
            for token, keep in zip(response_ids, response_mask)
        ]
        label_mask = [0] * (len(prompt_ids) + self.num_control_tokens) + response_mask
        control_mask = (
            [0] * len(prompt_ids)
            + [1] * self.num_control_tokens
            + [0] * len(response_ids)
        )
        pad_len = self.max_length - len(input_ids)
        if pad_len < 0:
            raise RuntimeError("multimodal DPO sequence exceeded max_length")
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        token_type_ids += [0] * pad_len
        labels += [-100] * pad_len
        label_mask += [0] * pad_len
        control_mask += [0] * pad_len
        if sum(control_mask) != self.num_control_tokens:
            raise RuntimeError(
                "multimodal DPO sequence lost control tokens: "
                f"expected {self.num_control_tokens}, got {sum(control_mask)}"
            )
        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "label_mask": torch.tensor(label_mask, dtype=torch.float32),
            "control_mask": torch.tensor(control_mask, dtype=torch.bool),
        }
        for key, value in encoded.items():
            if key not in {"input_ids", "attention_mask", "token_type_ids"}:
                result[key] = value
        return result

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(self.data_dir / item["bev_image_path"]).convert("RGB")
        chosen = self._encode(item["prompt"], item["chosen"], image)
        rejected = self._encode(item["prompt"], item["rejected"], image)
        result = {}
        shared_image_fields = {
            key: value
            for key, value in chosen.items()
            if key not in {
                "input_ids", "attention_mask", "token_type_ids", "labels",
                "label_mask", "control_mask",
            }
        }
        for prefix, encoded in (("chosen", chosen), ("rejected", rejected)):
            for key in (
                "input_ids", "attention_mask", "token_type_ids", "labels",
                "label_mask", "control_mask",
            ):
                result[f"{key}_{prefix}"] = encoded[key]
        result.update(shared_image_fields)
        result["q_current"] = torch.tensor(item["q_current"], dtype=torch.float32)
        result["has_q_current"] = torch.tensor(True)
        result["delta_q_target"] = torch.tensor(item["delta_q"], dtype=torch.float32)
        result["delta_a_target"] = torch.tensor(item["delta_a"], dtype=torch.float32)
        result["delta_p_target"] = torch.tensor(item["delta_p"], dtype=torch.float32)
        q_cues, q_mask = parse_q_geometry_cues(
            item["prompt"], len(item["q_current"])
        )
        result["q_geometry_cues"] = torch.tensor(q_cues, dtype=torch.float32)
        result["q_geometry_mask"] = torch.tensor(q_mask, dtype=torch.float32)
        return result
