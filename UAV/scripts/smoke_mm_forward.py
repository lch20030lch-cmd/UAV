#!/usr/bin/env python
"""
单 batch BEV-image Gemma3 前向传播烟雾测试。

用于验证最小模型侧多模态闭环：
  data + image -> processor/dataset -> Gemma3 multimodal forward
  -> control-token states -> projection head -> delta_q/a/p
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.multimodal_dataset import (
    MultimodalSFTDataset,
    resolve_multimodal_chat_template,
    validate_multimodal_oracle_contract,
)
from src.data.oracle_contract import (
    validate_checkpoint_dataset_compatibility,
)
from src.model import Gemma3MultimodalISAC, build_proj_head_config


def _resolve_lora_checkpoint(checkpoint: str, lora_checkpoint: str):
    if lora_checkpoint:
        return str(Path(lora_checkpoint))
    if not checkpoint:
        return None
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_file():
        return None
    candidate = ckpt_path / "lora"
    if (candidate / "adapter_config.json").exists():
        return str(candidate)
    return None


def _load_projection_head(model: Gemma3MultimodalISAC, checkpoint: str):
    if not checkpoint:
        return None
    ckpt_path = Path(checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / "projection_head.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"projection_head checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.projection_head.load_state_dict(state, strict=True)
    return str(ckpt_path)


def _load_control_token_embeddings(
    model: Gemma3MultimodalISAC,
    checkpoint: str,
    *,
    require_complete: bool = False,
):
    if not checkpoint:
        return {}
    ckpt_path = Path(checkpoint)
    ckpt_root = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    loaded = model.load_control_token_embeddings(ckpt_root)
    if require_complete:
        missing = {"ctrl_embed", "ctrl_offset"} - set(loaded)
        if missing:
            raise FileNotFoundError(
                "checkpoint is missing required control-token state: "
                f"{sorted(missing)}"
            )
    return loaded


def _read_checkpoint_metadata(checkpoint: str) -> dict:
    if not checkpoint:
        return {}
    checkpoint_path = Path(checkpoint)
    root = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _move_batch(batch, device):
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser(description="运行一次多模态模型前向传播烟雾测试")
    parser.add_argument("--config", type=str, default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="覆盖配置文件中的 model.backbone")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="包含 projection_head.pt 的目录，或 projection_head.pt 文件本身")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="LoRA adapter 目录；不填时会尝试从 --checkpoint/lora 自动发现")
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument(
        "--allow_legacy_dataset",
        action="store_true",
        help="diagnostic-only override for pre-v5 data",
    )
    parser.add_argument(
        "--allow_checkpoint_dataset_mismatch",
        action="store_true",
        help="diagnostic-only checkpoint provenance override",
    )
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="diagnostic override; otherwise use checkpoint metadata or v5 default",
    )
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--projection_head_type", type=str, choices=["shared", "split"], default=None,
                        help="可选 projection head 类型；加载 split checkpoint 时需要传 split")
    parser.add_argument("--q_projection_mode", type=str, choices=["clip", "direction"], default=None,
                        help="可选 q 投影模式；加载 direction checkpoint 时需要传 direction")
    parser.add_argument(
        "--q_geometry_mode",
        type=str,
        choices=["none", "cue_xy", "fixed_residual_xy"],
        default=None,
        help="可选：加载 cue_xy 或 fixed_residual_xy checkpoint 时传入对应模式",
    )
    args = parser.parse_args()

    with (PROJECT_ROOT / args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    sim_cfg = cfg["simulation"]
    train_cfg = cfg["training"]["sft"]
    data_cfg = cfg["data"]

    model_name = args.model or model_cfg["backbone"]
    data_dir = Path(args.data_dir or data_cfg["output_dir"])
    dataset_metadata = validate_multimodal_oracle_contract(
        data_dir,
        allow_legacy=args.allow_legacy_dataset,
        expected_simulation=sim_cfg,
    )
    data_path = data_dir / dataset_metadata.get(
        "sft_file", data_cfg.get("sft_file", "sft_dataset.jsonl")
    )
    max_length = args.max_length or train_cfg["max_seq_length"]
    lora_checkpoint = _resolve_lora_checkpoint(args.checkpoint, args.lora_checkpoint)
    checkpoint_metadata = _read_checkpoint_metadata(args.checkpoint)
    if args.checkpoint and dataset_metadata and not checkpoint_metadata:
        raise FileNotFoundError(
            "current-schema checkpoint diagnostics require metadata.json"
        )
    if checkpoint_metadata and dataset_metadata:
        validate_checkpoint_dataset_compatibility(
            checkpoint_metadata,
            dataset_metadata,
            allow_mismatch=args.allow_checkpoint_dataset_mismatch,
        )
    use_chat_template = resolve_multimodal_chat_template(
        dataset_metadata=dataset_metadata,
        checkpoint_metadata=checkpoint_metadata,
        configured_value=train_cfg.get("use_chat_template"),
        override=args.use_chat_template,
    )
    proj_head_config = build_proj_head_config(
        model_cfg, sim_cfg, checkpoint_metadata=checkpoint_metadata
    )
    mode_fields = {
        "projection_head_type": ("head_type", args.projection_head_type),
        "q_projection_mode": ("q_projection_mode", args.q_projection_mode),
        "q_geometry_mode": ("q_geometry_mode", args.q_geometry_mode),
    }
    for metadata_key, (config_key, cli_value) in mode_fields.items():
        saved_value = checkpoint_metadata.get(metadata_key)
        if cli_value is not None and saved_value is not None and cli_value != saved_value:
            raise ValueError(
                f"{metadata_key} CLI/checkpoint mismatch: "
                f"{cli_value!r} != {saved_value!r}"
            )
        if cli_value is not None:
            proj_head_config[config_key] = cli_value

    model = Gemma3MultimodalISAC(
        model_name_or_path=model_name,
        use_4bit=cfg["hardware"].get("use_4bit", True) and not args.no_4bit,
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        proj_head_config=proj_head_config,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        freeze_vision_tower=model_cfg.get("freeze_vision_tower", True),
        lora_rank=model_cfg["lora"]["rank"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"].get("dropout", 0.0),
        lora_target_modules=model_cfg["lora"]["target_modules"],
        lora_checkpoint=lora_checkpoint,
    )
    loaded_projection = _load_projection_head(model, args.checkpoint)
    loaded_control_embeddings = _load_control_token_embeddings(
        model,
        args.checkpoint,
        require_complete=bool(dataset_metadata),
    )
    model.eval()

    dataset = MultimodalSFTDataset(
        data_path=str(data_path),
        data_dir=str(data_dir),
        processor=model.processor,
        max_length=max_length,
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        include_response=False,
        use_chat_template=use_chat_template,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    batch = _move_batch(batch, model.device)

    forward_keys = {
        key: value for key, value in batch.items()
        if key not in {
            "labels",
            "label_mask",
            "has_q_current",
            "delta_q_target",
            "delta_a_target",
            "delta_p_target",
        }
    }

    with torch.no_grad():
        outputs = model(**forward_keys)

    print("OK: multimodal model forward smoke")
    print(f"  data: {data_path}")
    print(f"  loaded_projection: {loaded_projection}")
    print(f"  loaded_control_embeddings: {loaded_control_embeddings}")
    print(f"  loaded_lora_checkpoint: {model.loaded_lora_checkpoint}")
    print(f"  projection_head_type: {proj_head_config.get('head_type', 'shared')}")
    print(f"  q_projection_mode: {proj_head_config.get('q_projection_mode', 'clip')}")
    print(f"  q_geometry_mode: {proj_head_config.get('q_geometry_mode', 'none')}")
    print(f"  max_length: {max_length}")
    print(f"  use_chat_template: {use_chat_template}")
    print(f"  input_ids: {tuple(batch['input_ids'].shape)}")
    print(f"  attention_mask: {tuple(batch['attention_mask'].shape)}")
    if "pixel_values" in batch:
        print(f"  pixel_values: {tuple(batch['pixel_values'].shape)}")
    print(f"  control_token_count: {int(batch['control_mask'].sum().item())}")
    print(f"  control_states: {tuple(outputs['control_states'].shape)}")
    print(f"  delta_q: {tuple(outputs['delta_q'].shape)}")
    print(f"  delta_a: {tuple(outputs['delta_a'].shape)}")
    print(f"  delta_p: {tuple(outputs['delta_p'].shape)}")
    if "q_cue_logits" in outputs:
        print(f"  q_cue_logits: {tuple(outputs['q_cue_logits'].shape)}")

    for name in ["delta_q", "delta_a", "delta_p"]:
        tensor = outputs[name]
        if torch.isnan(tensor).any():
            raise RuntimeError(f"{name} contains NaN")


if __name__ == "__main__":
    main()
