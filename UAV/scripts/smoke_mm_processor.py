#!/usr/bin/env python
"""Validate the exact multimodal processor path used by SFT/evaluation."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data.multimodal_dataset import (
    MultimodalSFTDataset,
    resolve_multimodal_chat_template,
    validate_multimodal_oracle_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/rtx5090_multimodal_smoke.yaml",
    )
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--allow_legacy_dataset", action="store_true")
    args = parser.parse_args()

    from transformers import AutoProcessor

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]["sft"]
    data_cfg = cfg["data"]

    data_dir = Path(args.data_dir or data_cfg["output_dir"])
    dataset_metadata = validate_multimodal_oracle_contract(
        data_dir,
        allow_legacy=args.allow_legacy_dataset,
        expected_simulation=cfg["simulation"],
    )
    data_path = data_dir / dataset_metadata.get(
        "sft_file", data_cfg.get("sft_file", "sft_dataset.jsonl")
    )
    model_name = args.model or model_cfg["backbone"]
    max_length = int(args.max_length or train_cfg["max_seq_length"])
    num_control_tokens = int(model_cfg["control_token"]["num_tokens"])
    use_chat_template = resolve_multimodal_chat_template(
        dataset_metadata=dataset_metadata,
        configured_value=train_cfg.get("use_chat_template"),
        override=args.use_chat_template,
    )

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    dataset = MultimodalSFTDataset(
        data_path=str(data_path),
        data_dir=str(data_dir),
        processor=processor,
        max_length=max_length,
        num_control_tokens=num_control_tokens,
        include_response=False,
        use_chat_template=use_chat_template,
    )
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(
            f"sample_index {args.sample_index} outside dataset size {len(dataset)}"
        )

    record = dataset.data[args.sample_index]
    batch = dataset[args.sample_index]
    excluded = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "labels",
        "label_mask",
        "control_mask",
        "q_current",
        "has_q_current",
        "delta_q_target",
        "delta_a_target",
        "delta_p_target",
        "q_geometry_cues",
        "q_geometry_mask",
    }
    multimodal_keys = [
        key
        for key, value in batch.items()
        if key not in excluded and hasattr(value, "shape")
    ]
    control_count = int(batch["control_mask"].sum().item())
    if control_count != num_control_tokens:
        raise RuntimeError(
            "control token count mismatch: "
            f"{control_count} != {num_control_tokens}"
        )

    print("OK: multimodal processor smoke")
    print(f"  data: {data_path}")
    print(f"  image: {data_dir / record['bev_image_path']}")
    print(f"  chat template: {use_chat_template}")
    print(f"  input_ids: {tuple(batch['input_ids'].shape)}")
    print(f"  attention_mask: {tuple(batch['attention_mask'].shape)}")
    for key in multimodal_keys:
        print(f"  {key}: {tuple(batch[key].shape)}")
    print(f"  control_token_count: {control_count}")


if __name__ == "__main__":
    main()
