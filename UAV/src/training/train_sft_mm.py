#!/usr/bin/env python
"""
BEV-image Gemma3 多模态 SFT 烟雾测试。

首版训练烟雾测试对 RTX 5090 32GB 保持保守：
  - 默认冻结 Gemma3 多模态 backbone
  - 默认只训练 projection head
  - 默认只优化 CTL loss

它用于在前向传播烟雾测试已通过后验证训练闭环：
  dataset -> multimodal forward -> projection head -> control loss
  -> accumulated backward -> optimizer step -> checkpoint

如需测试 LoRA 链路，可显式传入 --train_lora。
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.env_setup import setup_env

setup_env()

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler, set_seed

from src.data.multimodal_dataset import (
    MultimodalSFTDataset,
    resolve_multimodal_chat_template,
    validate_multimodal_oracle_contract,
)
from src.data.oracle_contract import (
    checkpoint_dataset_fields,
    validate_checkpoint_dataset_compatibility,
)
from src.model import Gemma3MultimodalISAC, UAVISACLosses, build_proj_head_config
from src.model.gemma_multimodal_isac import (
    is_vision_parameter_name,
    keep_vision_modules_in_eval_mode,
)
from src.training.runtime_utils import resolve_warmup_steps, rotate_step_checkpoints


def _move_batch(batch, device):
    result = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value
    return result


def _grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += param.grad.detach().float().norm().item() ** 2
    return total ** 0.5


def _parameter_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        total += param.detach().float().norm().item() ** 2
    return total ** 0.5


def _resolve_gradient_accumulation_steps(train_cfg: dict, override=None) -> int:
    steps = int(
        override
        if override is not None
        else train_cfg.get("gradient_accumulation_steps", 1)
    )
    if steps <= 0:
        raise ValueError("gradient_accumulation_steps must be a positive integer")
    return steps


def _is_accumulation_boundary(micro_step: int, accumulation_steps: int) -> bool:
    return micro_step > 0 and micro_step % accumulation_steps == 0


def _backward_accumulated_loss(loss: torch.Tensor, accumulation_steps: int):
    """Scale one micro-batch loss so accumulated gradients form a batch mean."""
    (loss / accumulation_steps).backward()


def _clip_projection_and_lora_gradients(
    projection_parameters,
    lora_parameters,
    max_norm: float,
):
    """Clip projection and LoRA gradients independently.

    The two optimizer groups have different parameter counts and gradient scales.
    Joint clipping lets a large LoRA norm suppress an otherwise safe projection
    gradient, preventing the task head from tracking representation updates.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    projection_parameters = [
        parameter for parameter in projection_parameters if parameter.grad is not None
    ]
    lora_parameters = [
        parameter for parameter in lora_parameters if parameter.grad is not None
    ]
    if projection_parameters:
        torch.nn.utils.clip_grad_norm_(projection_parameters, max_norm)
    if lora_parameters:
        torch.nn.utils.clip_grad_norm_(lora_parameters, max_norm)
    return (
        _grad_norm(projection_parameters),
        _grad_norm(lora_parameters),
    )


def _save_mm_smoke(model, save_dir: Path, metadata: dict, save_lora: bool = False):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.projection_head.state_dict(), save_dir / "projection_head.pt")
    metadata["control_token_embeddings"] = model.save_control_token_embeddings(save_dir)
    model.processor.save_pretrained(save_dir / "processor")
    if save_lora and hasattr(model.base_model, "save_pretrained"):
        model.base_model.save_pretrained(save_dir / "lora")
    with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _resolve_lora_checkpoint(init_checkpoint: str, enable_lora: bool):
    if not init_checkpoint or not enable_lora:
        return None
    candidate = Path(init_checkpoint) / "lora"
    if (candidate / "adapter_config.json").exists():
        return str(candidate)
    return None


def _load_mm_smoke_checkpoint(
    model,
    init_checkpoint: str,
    *,
    expected_projection_config: dict,
    expected_dataset_metadata: dict = None,
    allow_partial_projection_load: bool = False,
    allow_checkpoint_dataset_mismatch: bool = False,
) -> dict:
    if not init_checkpoint:
        return {}
    ckpt_dir = Path(init_checkpoint)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"init checkpoint not found: {ckpt_dir}")

    loaded = {"init_checkpoint": str(ckpt_dir)}
    metadata_path = ckpt_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        loaded["metadata"] = str(metadata_path)
        checkpoint_modes = {
            "projection_head_type": metadata.get("projection_head_type"),
            "q_projection_mode": metadata.get("q_projection_mode"),
            "q_geometry_mode": metadata.get("q_geometry_mode"),
        }
        mismatches = {
            key: (checkpoint_modes[key], expected_projection_config[key])
            for key in checkpoint_modes
            if checkpoint_modes[key] is not None
            and checkpoint_modes[key] != expected_projection_config[key]
        }
        if mismatches and not allow_partial_projection_load:
            raise ValueError(
                "checkpoint projection metadata does not match requested model: "
                f"{mismatches}. Use --allow_partial_projection_load only for an "
                "intentional architecture migration."
            )
    if expected_dataset_metadata:
        validate_checkpoint_dataset_compatibility(
            metadata,
            expected_dataset_metadata,
            allow_mismatch=allow_checkpoint_dataset_mismatch,
            require_same_seed=True,
        )
    proj_path = ckpt_dir / "projection_head.pt"
    if not proj_path.exists():
        raise FileNotFoundError(
            f"init checkpoint is missing projection_head.pt: {ckpt_dir}"
        )
    state = torch.load(proj_path, map_location="cpu")
    load_result = model.projection_head.load_state_dict(
        state, strict=not allow_partial_projection_load
    )
    loaded["projection_head"] = str(proj_path)
    loaded["projection_missing_keys"] = list(load_result.missing_keys)
    loaded["projection_unexpected_keys"] = list(load_result.unexpected_keys)

    loaded_ctrl = model.load_control_token_embeddings(ckpt_dir)
    required_control_files = {"ctrl_embed", "ctrl_offset"}
    missing_control = required_control_files - set(loaded_ctrl)
    if missing_control:
        raise FileNotFoundError(
            "init checkpoint is missing control-token state: "
            f"{sorted(missing_control)}"
        )
    loaded["control_token_embeddings"] = loaded_ctrl
    return loaded


def _set_projection_branch_trainable(model, branch_prefixes, trainable: bool):
    """按名称冻结/解冻 split projection head 的指定分支。"""
    changed = []
    for name, param in model.projection_head.named_parameters():
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in branch_prefixes):
            param.requires_grad = trainable
            changed.append(name)
    return changed


def _freeze_projection_except(model, trainable_prefixes):
    """只保留指定 projection head 前缀可训练，其余全部冻结。"""
    frozen = []
    trainable = []
    for name, param in model.projection_head.named_parameters():
        keep_trainable = any(
            name == prefix or name.startswith(f"{prefix}.")
            for prefix in trainable_prefixes
        )
        param.requires_grad = keep_trainable
        if keep_trainable:
            trainable.append(name)
        else:
            frozen.append(name)
    return frozen, trainable


def _isolate_loss_weights(branch: str, weights: dict) -> dict:
    """Make a branch-isolation flag authoritative for every shared gradient.

    Freezing projection tensors alone is insufficient when LoRA or the control
    offsets are trainable: unrelated losses still update those shared
    representation parameters.  This helper ensures that a Q/A/P isolation
    experiment is optimized by only its own objective.
    """
    result = {key: float(value) for key, value in weights.items()}
    allowed = {
        None: set(result),
        "q": {
            "lambda_q",
            "lambda_q_dir",
            "lambda_q_projected_dir",
            "lambda_sep",
        },
        "q_cue": {
            "lambda_q_projected_dir",
            "lambda_q_cue_ce",
        },
        "q_power": {
            "lambda_q",
            "lambda_p",
            "lambda_sep",
            "lambda_q_dir",
            "lambda_q_projected_dir",
            "lambda_q_cue_ce",
            "lambda_p_raw_kl",
        },
        "association": {
            "lambda_a",
            "lambda_assoc_ce",
            "lambda_assoc_raw_ce",
        },
        "power": {
            "lambda_p",
            "lambda_p_raw_kl",
        },
    }
    if branch not in allowed:
        raise ValueError(f"unsupported isolated projection branch: {branch}")
    for key in result:
        if key not in allowed[branch]:
            result[key] = 0.0
    return result


def train_mm_sft_smoke(
    config_path: str,
    data_dir: str = None,
    model_path: str = None,
    max_steps: int = None,
    max_length: int = None,
    output_dir: str = None,
    checkpoint_dir: str = None,
    save_steps: int = None,
    save_total_limit: int = None,
    train_lora: bool = False,
    load_lora: bool = False,
    lambda_assoc_ce: float = None,
    lambda_q: float = None,
    lambda_a: float = None,
    lambda_p: float = None,
    lambda_assoc_raw_ce: float = None,
    lambda_q_dir: float = None,
    lambda_q_projected_dir: float = None,
    lambda_q_cue_ce: float = None,
    lambda_p_raw_kl: float = None,
    projection_lr: float = None,
    lora_lr_override: float = None,
    init_checkpoint: str = None,
    projection_head_type: str = None,
    q_projection_mode: str = None,
    q_geometry_mode: str = None,
    freeze_assoc_branch: bool = False,
    freeze_qp_branch: bool = False,
    freeze_all_except_q: bool = False,
    freeze_all_except_q_cue: bool = False,
    freeze_all_except_p: bool = False,
    gradient_accumulation_steps: int = None,
    lambda_lm_ce: float = None,
    use_chat_template: bool = None,
    train_control_offsets: bool = None,
    allow_partial_projection_load: bool = False,
    allow_legacy_oracle_data: bool = False,
    allow_checkpoint_dataset_mismatch: bool = False,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    model_cfg = cfg["model"]
    sim_cfg = cfg["simulation"]
    train_cfg = cfg["training"]["sft"]
    data_cfg = cfg["data"]

    data_root = Path(data_dir or data_cfg["output_dir"])
    dataset_metadata = validate_multimodal_oracle_contract(
        data_root,
        allow_legacy=allow_legacy_oracle_data,
        expected_simulation=sim_cfg,
    )
    checkpoint_provenance = checkpoint_dataset_fields(dataset_metadata)
    sft_path = data_root / dataset_metadata.get(
        "sft_file", data_cfg.get("sft_file", "sft_dataset.jsonl")
    )
    model_name = model_path or model_cfg["backbone"]
    max_seq_length = int(max_length or train_cfg["max_seq_length"])
    steps_limit = int(
        max_steps
        if max_steps is not None
        else train_cfg.get("phase1", {}).get("max_steps", 30)
    )
    if steps_limit <= 0:
        raise ValueError("max_steps must be a positive integer")
    grad_accum_steps = _resolve_gradient_accumulation_steps(
        train_cfg,
        gradient_accumulation_steps,
    )
    lambda_lm_ce_value = float(
        lambda_lm_ce
        if lambda_lm_ce is not None
        else train_cfg.get("phase1", {}).get("lambda_lm_ce", 0.0)
    )
    if lambda_lm_ce_value < 0.0:
        raise ValueError("lambda_lm_ce must be non-negative")
    phase1_lambda_ctl_value = float(
        train_cfg.get("phase1", {}).get("lambda_ctl", 1.0)
    )
    if phase1_lambda_ctl_value < 0.0:
        raise ValueError("phase1.lambda_ctl must be non-negative")
    include_response_tokens = lambda_lm_ce_value > 0.0
    if include_response_tokens and not train_lora:
        raise ValueError(
            "token-level multimodal SFT requires --train_lora; a frozen "
            "backbone cannot optimize response CE"
        )
    out_root = Path(output_dir or cfg.get("output_dir", "/root/autodl-tmp/outputs/mm_smoke"))
    if checkpoint_dir is not None:
        ckpt_root = Path(checkpoint_dir)
    elif output_dir is not None:
        ckpt_root = out_root / "checkpoints"
    else:
        ckpt_root = Path(
            cfg.get("checkpoint_dir", "/root/autodl-tmp/checkpoints/mm_smoke")
        )
    checkpoint_interval = int(
        save_steps if save_steps is not None else train_cfg.get("save_steps", 10)
    )
    if checkpoint_interval <= 0:
        raise ValueError("save_steps must be a positive integer")
    checkpoint_limit = int(
        save_total_limit
        if save_total_limit is not None
        else train_cfg.get("save_total_limit", 2)
    )
    if checkpoint_limit <= 0:
        raise ValueError("save_total_limit must be a positive integer")
    ckpt_root.mkdir(parents=True, exist_ok=True)
    lora_enabled = bool(train_lora or load_lora)
    checkpoint_prefix = f"mm_sft_{'lora_' if lora_enabled else ''}smoke_step_"
    init_lora_checkpoint = _resolve_lora_checkpoint(init_checkpoint, lora_enabled)
    if init_checkpoint and lora_enabled and init_lora_checkpoint is None:
        raise FileNotFoundError(
            "LoRA-enabled continuation requires --init_checkpoint containing "
            "lora/adapter_config.json"
        )

    init_metadata = {}
    if init_checkpoint:
        init_metadata_path = Path(init_checkpoint) / "metadata.json"
        if init_metadata_path.exists():
            with init_metadata_path.open("r", encoding="utf-8") as handle:
                init_metadata = json.load(handle)
    use_chat_template_value = resolve_multimodal_chat_template(
        dataset_metadata=dataset_metadata,
        checkpoint_metadata=init_metadata,
        configured_value=train_cfg.get("use_chat_template"),
        override=use_chat_template,
    )

    print("=" * 60)
    print("BEV-image multimodal SFT smoke")
    print("=" * 60)
    print(f"  data:       {sft_path}")
    print(f"  data contract: {dataset_metadata or 'legacy override'}")
    print(f"  model:      {model_name}")
    print(f"  max_length: {max_seq_length}")
    print(f"  optimizer steps: {steps_limit}")
    print(f"  gradient accumulation: {grad_accum_steps}")
    print(
        "  effective batch size: "
        f"{int(train_cfg['per_device_batch_size']) * grad_accum_steps}"
    )
    print(
        f"  checkpoints:{ckpt_root} "
        f"(every {checkpoint_interval} steps, keep {checkpoint_limit})"
    )
    print(
        "  input mode: prompt + image + control tokens"
        + (" + response" if include_response_tokens else " (response omitted)")
    )
    print(f"  chat template: {use_chat_template_value}")
    control_offsets_trainable = bool(
        train_control_offsets
        if train_control_offsets is not None
        else train_cfg.get("phase1", {}).get(
            "train_control_offsets", True
        )
    )
    if train_lora:
        trainable_label = "projection_head + LoRA"
    elif load_lora:
        trainable_label = "projection_head (loaded LoRA frozen)"
    else:
        trainable_label = "projection_head only"
    if control_offsets_trainable:
        trainable_label += " + control offsets"
    print(f"  trainable:  {trainable_label}")
    print()

    proj_head_config = build_proj_head_config(
        model_cfg, sim_cfg, checkpoint_metadata=init_metadata
    )
    if projection_head_type is not None:
        proj_head_config["head_type"] = projection_head_type
    if q_projection_mode is not None:
        proj_head_config["q_projection_mode"] = q_projection_mode
    if q_geometry_mode is not None:
        proj_head_config["q_geometry_mode"] = q_geometry_mode
    head_type = proj_head_config.get("head_type", "shared")
    q_mode = proj_head_config.get("q_projection_mode", "clip")
    q_geom_mode = proj_head_config.get("q_geometry_mode", "none")
    freeze_modes = (
        freeze_assoc_branch,
        freeze_qp_branch,
        freeze_all_except_q,
        freeze_all_except_q_cue,
        freeze_all_except_p,
    )
    if any(freeze_modes) and head_type != "split":
        raise ValueError("分支冻结参数只适用于 --projection_head_type split。")
    if sum(bool(mode) for mode in freeze_modes) > 1:
        raise ValueError("Projection freeze options are mutually exclusive.")

    model = Gemma3MultimodalISAC(
        model_name_or_path=model_name,
        use_4bit=cfg["hardware"].get("use_4bit", True),
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        proj_head_config=proj_head_config,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        freeze_vision_tower=model_cfg.get("freeze_vision_tower", True),
        enable_lora=lora_enabled,
        lora_rank=model_cfg["lora"]["rank"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"].get("dropout", 0.0),
        lora_target_modules=model_cfg["lora"]["target_modules"],
        lora_checkpoint=init_lora_checkpoint,
    )
    loaded_init = _load_mm_smoke_checkpoint(
        model,
        init_checkpoint,
        expected_projection_config={
            "projection_head_type": head_type,
            "q_projection_mode": q_mode,
            "q_geometry_mode": q_geom_mode,
        },
        expected_dataset_metadata=dataset_metadata,
        allow_partial_projection_load=allow_partial_projection_load,
        allow_checkpoint_dataset_mismatch=allow_checkpoint_dataset_mismatch,
    )
    if loaded_init:
        print(f"  init_checkpoint: {loaded_init}")

    if train_lora:
        model.base_model.train()
        frozen_vision_modules = keep_vision_modules_in_eval_mode(
            model.base_model
        )
    else:
        for param in model.base_model.parameters():
            param.requires_grad = False
        model.base_model.eval()
        frozen_vision_modules = []
    # This compact adapter can train without opening the 4-bit backbone.  It
    # provides a controlled way to repair collapsed control states before a
    # larger LoRA stage is justified.
    model.control_token_offsets.requires_grad_(control_offsets_trainable)
    model.projection_head.train()

    frozen_projection_branches = []
    trainable_projection_branches = []
    isolated_projection_branch = None
    if freeze_all_except_p:
        frozen_projection_branches, trainable_projection_branches = _freeze_projection_except(
            model,
            trainable_prefixes=("readout_p", "p_mlp"),
        )
        isolated_projection_branch = "power"
    elif freeze_all_except_q:
        frozen_projection_branches, trainable_projection_branches = _freeze_projection_except(
            model,
            trainable_prefixes=("readout_q", "q_mlp", "q_residual_adapter"),
        )
        isolated_projection_branch = "q"
    elif freeze_all_except_q_cue:
        frozen_projection_branches, trainable_projection_branches = _freeze_projection_except(
            model,
            trainable_prefixes=("readout_q_cue",),
        )
        isolated_projection_branch = "q_cue"
    elif freeze_assoc_branch:
        frozen_projection_branches = _set_projection_branch_trainable(
            model,
            branch_prefixes=("readout_a", "a_mlp"),
            trainable=False,
        )
        isolated_projection_branch = "q_power"
    elif freeze_qp_branch:
        (
            frozen_projection_branches,
            trainable_projection_branches,
        ) = _freeze_projection_except(
            model, trainable_prefixes=("readout_a", "a_mlp")
        )
        isolated_projection_branch = "association"

    dataset = MultimodalSFTDataset(
        data_path=str(sft_path),
        data_dir=str(data_root),
        processor=model.processor,
        max_length=max_seq_length,
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        include_response=include_response_tokens,
        use_chat_template=use_chat_template_value,
    )
    if len(dataset) == 0:
        raise ValueError(f"multimodal SFT dataset is empty: {sft_path}")
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["per_device_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    assoc_ce_weight = (
        float(lambda_assoc_ce)
        if lambda_assoc_ce is not None
        else float(train_cfg.get("phase1", {}).get("lambda_assoc_ce", 0.0))
    )
    assoc_raw_ce_weight = (
        float(lambda_assoc_raw_ce)
        if lambda_assoc_raw_ce is not None
        else float(train_cfg.get("phase1", {}).get("lambda_assoc_raw_ce", 0.0))
    )
    lambda_q_value = float(lambda_q) if lambda_q is not None else float(model_cfg["loss"]["lambda_q"])
    lambda_a_value = float(lambda_a) if lambda_a is not None else float(model_cfg["loss"]["lambda_a"])
    lambda_p_value = float(lambda_p) if lambda_p is not None else float(model_cfg["loss"]["lambda_p"])
    lambda_q_dir_value = (
        float(lambda_q_dir)
        if lambda_q_dir is not None
        else float(train_cfg.get("phase1", {}).get("lambda_q_dir", 0.0))
    )
    lambda_q_projected_dir_value = (
        float(lambda_q_projected_dir)
        if lambda_q_projected_dir is not None
        else float(train_cfg.get("phase1", {}).get("lambda_q_projected_dir", 0.0))
    )
    lambda_q_cue_ce_value = (
        float(lambda_q_cue_ce)
        if lambda_q_cue_ce is not None
        else float(train_cfg.get("phase1", {}).get("lambda_q_cue_ce", 0.0))
    )
    lambda_p_raw_kl_value = (
        float(lambda_p_raw_kl)
        if lambda_p_raw_kl is not None
        else float(
            train_cfg.get("phase1", {}).get(
                "lambda_p_raw_kl",
                model_cfg["loss"].get("lambda_p_raw_kl", 0.0),
            )
        )
    )
    effective_loss_weights = _isolate_loss_weights(
        isolated_projection_branch,
        {
            "lambda_q": lambda_q_value,
            "lambda_a": lambda_a_value,
            "lambda_p": lambda_p_value,
            "lambda_sep": model_cfg["loss"]["lambda_sep"],
            "lambda_assoc_ce": assoc_ce_weight,
            "lambda_assoc_raw_ce": assoc_raw_ce_weight,
            "lambda_q_dir": lambda_q_dir_value,
            "lambda_q_projected_dir": lambda_q_projected_dir_value,
            "lambda_q_cue_ce": lambda_q_cue_ce_value,
            "lambda_p_raw_kl": lambda_p_raw_kl_value,
        },
    )
    negative_loss_weights = {
        key: value
        for key, value in effective_loss_weights.items()
        if value < 0.0
    }
    if negative_loss_weights:
        raise ValueError(
            f"control loss weights must be non-negative: {negative_loss_weights}"
        )
    lambda_q_value = effective_loss_weights["lambda_q"]
    lambda_a_value = effective_loss_weights["lambda_a"]
    lambda_p_value = effective_loss_weights["lambda_p"]
    assoc_ce_weight = effective_loss_weights["lambda_assoc_ce"]
    assoc_raw_ce_weight = effective_loss_weights["lambda_assoc_raw_ce"]
    lambda_q_dir_value = effective_loss_weights["lambda_q_dir"]
    lambda_q_projected_dir_value = effective_loss_weights[
        "lambda_q_projected_dir"
    ]
    lambda_q_cue_ce_value = effective_loss_weights["lambda_q_cue_ce"]
    lambda_p_raw_kl_value = effective_loss_weights["lambda_p_raw_kl"]
    lambda_sep_value = effective_loss_weights["lambda_sep"]
    loss_fn = UAVISACLosses(
        lambda_ctl=model_cfg["loss"]["lambda_ctl"],
        lambda_q=lambda_q_value,
        lambda_a=lambda_a_value,
        lambda_p=lambda_p_value,
        lambda_sep=lambda_sep_value,
        lambda_assoc_ce=assoc_ce_weight,
        lambda_assoc_raw_ce=assoc_raw_ce_weight,
        lambda_q_dir=lambda_q_dir_value,
        lambda_q_projected_dir=lambda_q_projected_dir_value,
        lambda_q_cue_ce=lambda_q_cue_ce_value,
        lambda_p_raw_kl=lambda_p_raw_kl_value,
        power_temperature=float(model_cfg["projection_head"]["tau_power"]),
    )
    # 默认只训练投影头；--load_lora 只加载并冻结 adapter，--train_lora 才更新它。
    proj_params = [
        p for p in model.projection_head.parameters() if p.requires_grad
    ]
    control_offset_params = (
        [model.control_token_offsets]
        if model.control_token_offsets.requires_grad
        else []
    )
    projection_optimizer_params = proj_params + control_offset_params
    lora_named_params = [
        (name, parameter)
        for name, parameter in model.base_model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    lora_params = [parameter for _, parameter in lora_named_params]
    trainable_vision_lora_names = [
        name for name, _ in lora_named_params if is_vision_parameter_name(name)
    ]
    trainable_language_lora_names = [
        name for name, _ in lora_named_params if not is_vision_parameter_name(name)
    ]
    freeze_vision_tower = bool(model_cfg.get("freeze_vision_tower", True))
    if freeze_vision_tower and trainable_vision_lora_names:
        raise RuntimeError(
            "freeze_vision_tower=True but trainable vision LoRA tensors remain: "
            f"{trainable_vision_lora_names[:5]}"
        )
    proj_lr = (
        float(projection_lr)
        if projection_lr is not None
        else float(
            train_cfg.get("phase1", {}).get("projection_lr", 1e-3)
        )
    )
    lora_lr = (
        float(lora_lr_override)
        if lora_lr_override is not None
        else train_cfg.get("phase1", {}).get("lr_lora", train_cfg.get("learning_rate", 2e-4))
    )
    if proj_lr <= 0.0:
        raise ValueError("projection_lr must be positive")
    if train_lora and float(lora_lr) <= 0.0:
        raise ValueError("lora_lr must be positive when --train_lora is used")
    if train_lora and not lora_params:
        raise RuntimeError("已传入 --train_lora，但没有发现可训练的 LoRA 参数。")

    param_groups = [
        {"params": projection_optimizer_params, "lr": proj_lr}
    ]
    if train_lora and lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr})
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler_name = str(train_cfg.get("lr_scheduler", "constant")).lower()
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    warmup_steps = resolve_warmup_steps(steps_limit, warmup_ratio)
    scheduler = get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=steps_limit,
    )
    print(f"  trainable projection tensors: {len(proj_params)}")
    print(
        "  trainable control offsets:   "
        f"{len(control_offset_params)}"
    )
    print(f"  trainable LoRA tensors:       {len(lora_params)}")
    print(f"  trainable language LoRA:      {len(trainable_language_lora_names)}")
    print(f"  trainable vision LoRA:        {len(trainable_vision_lora_names)}")
    print(
        "  frozen vision parameters:    "
        f"{len(getattr(model, 'frozen_vision_parameter_names', []))}"
    )
    print(f"  vision modules kept in eval:  {len(frozen_vision_modules)}")
    print(f"  projection lr:                {proj_lr}")
    print(f"  LoRA lr:                      {lora_lr if train_lora else 0.0}")
    print(f"  lr scheduler:                 {scheduler_name}")
    print(f"  warmup steps:                 {warmup_steps}")
    print(f"  projection head type:         {head_type}")
    print(f"  q projection mode:            {q_mode}")
    print(f"  q geometry mode:              {q_geom_mode}")
    print(f"  frozen projection tensors:    {len(frozen_projection_branches)}")
    print(f"  isolated projection branch:  {isolated_projection_branch or 'none'}")
    print(f"  isolated trainable tensors:   {len(trainable_projection_branches)}")
    print(f"  lambda_q/a/p:                 {lambda_q_value} / {lambda_a_value} / {lambda_p_value}")
    print(f"  q direction weight:           {lambda_q_dir_value}")
    print(f"  projected q direction weight: {lambda_q_projected_dir_value}")
    print(f"  q cue CE weight:              {lambda_q_cue_ce_value}")
    print(f"  power raw KL weight:          {lambda_p_raw_kl_value}")
    print(f"  association CE weight:        {assoc_ce_weight}")
    print(f"  association raw CE weight:    {assoc_raw_ce_weight}")
    print(f"  language-model CE weight:     {lambda_lm_ce_value}")
    print(f"  phase-I control weight:       {phase1_lambda_ctl_value}")
    print(f"  separation weight:            {lambda_sep_value}")

    device = model.device
    global_step = 0
    micro_step = 0
    epoch = 0
    metric_sums = {}
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=steps_limit, desc="MM SFT smoke")

    while global_step < steps_limit:
        epoch += 1
        for batch in dataloader:
            if global_step >= steps_limit:
                break
            batch = _move_batch(batch, device)

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
            forward_keys["compute_full_logits"] = include_response_tokens
            response_logit_positions = None
            if include_response_tokens:
                response_logit_positions = (
                    batch["label_mask"][:, 1:].bool().any(dim=0).nonzero(as_tuple=True)[0]
                )
                if response_logit_positions.numel() == 0:
                    raise RuntimeError("response SFT batch contains no supervised tokens")
                forward_keys["logits_to_keep"] = response_logit_positions

            # 多模态 smoke 阶段只算控制损失，先确认 delta_q/a/p 的可训练闭环。
            outputs = model(**forward_keys)
            delta_hat = {
                "delta_q": outputs["delta_q"],
                "delta_a": outputs["delta_a"],
                "delta_p": outputs["delta_p"],
            }
            if "delta_a_raw" in outputs:
                delta_hat["delta_a_raw"] = outputs["delta_a_raw"]
            if "delta_q_raw" in outputs:
                delta_hat["delta_q_raw"] = outputs["delta_q_raw"]
            if "delta_p_raw" in outputs:
                delta_hat["delta_p_raw"] = outputs["delta_p_raw"]
            if "q_cue_logits" in outputs:
                delta_hat["q_cue_logits"] = outputs["q_cue_logits"]
            delta_target = {
                "delta_q": batch["delta_q_target"],
                "delta_a": batch["delta_a_target"],
                "delta_p": batch["delta_p_target"],
            }
            if isolated_projection_branch in {None, "q"}:
                delta_target["q_current"] = batch["q_current"]
            if "q_geometry_cues" in batch:
                delta_target["q_geometry_cues"] = batch["q_geometry_cues"]
            if "q_geometry_mask" in batch:
                delta_target["q_geometry_mask"] = batch["q_geometry_mask"]
            total_loss, metrics = loss_fn.compute_phase1_total(
                delta_hat=delta_hat,
                delta_target=delta_target,
                phase1_lambda_ctl=phase1_lambda_ctl_value,
            )
            loss_lm_ce = total_loss.new_zeros(())
            if include_response_tokens:
                logits = outputs.get("logits")
                if logits is None or logits.shape[1] != response_logit_positions.numel():
                    raise RuntimeError(
                        "selected response logits are required for multimodal response SFT"
                    )
                selected_labels = batch["labels"][:, 1:][
                    :, response_logit_positions
                ]
                loss_lm_ce = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    selected_labels.reshape(-1),
                    ignore_index=-100,
                )
                total_loss = total_loss + lambda_lm_ce_value * loss_lm_ce
            metrics["loss_lm_ce"] = float(loss_lm_ce.detach().item())
            metrics["loss_total"] = float(total_loss.detach().item())

            with torch.no_grad():
                p_prob = outputs["delta_p"].float().clamp_min(1e-12)
                p_prob = p_prob / p_prob.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                metrics["delta_p_entropy"] = float(
                    (-(p_prob * torch.log(p_prob)).sum(dim=-1)).mean().item()
                )
                inactive_mask = batch["delta_a_target"] <= 0.5
                inactive_power = outputs["delta_p"][..., :-1].float()[inactive_mask]
                metrics["delta_p_inactive_leakage"] = float(
                    inactive_power.mean().item() if inactive_power.numel() else 0.0
                )

            if not torch.isfinite(total_loss):
                raise RuntimeError("Non-finite loss detected in multimodal SFT smoke.")

            _backward_accumulated_loss(total_loss, grad_accum_steps)
            micro_step += 1
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)

            if not _is_accumulation_boundary(micro_step, grad_accum_steps):
                continue

            metrics = {
                key: value / grad_accum_steps
                for key, value in metric_sums.items()
            }
            metric_sums = {}
            grad_norm = _grad_norm(proj_params)
            grad_norm_control = _grad_norm(control_offset_params)
            grad_norm_lora = _grad_norm(lora_params) if train_lora else 0.0
            q_residual_adapter = getattr(model.projection_head, "q_residual_adapter", None)
            grad_norm_q_residual = (
                _grad_norm(q_residual_adapter.parameters())
                if q_residual_adapter is not None
                else 0.0
            )
            (
                grad_norm_proj_post_clip,
                grad_norm_lora_post_clip,
            ) = _clip_projection_and_lora_gradients(
                proj_params,
                lora_params,
                float(cfg["hardware"].get("max_grad_norm", 1.0)),
            )
            if control_offset_params:
                torch.nn.utils.clip_grad_norm_(
                    control_offset_params,
                    float(cfg["hardware"].get("max_grad_norm", 1.0)),
                )
            grad_norm_control_post_clip = _grad_norm(
                control_offset_params
            )
            step_proj_lr = float(optimizer.param_groups[0]["lr"])
            step_lora_lr = (
                float(optimizer.param_groups[1]["lr"])
                if train_lora and lora_params
                else 0.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            q_residual_adapter_norm = (
                _parameter_norm(q_residual_adapter.parameters())
                if q_residual_adapter is not None
                else 0.0
            )

            global_step += 1
            pbar.update(1)
            pbar.write(
                f"step={global_step} micro_step={micro_step} epoch={epoch} "
                f"loss_ctl={metrics['loss_ctl']:.6f} "
                f"loss_total={metrics['loss_total']:.6f} "
                f"loss_lm_ce={metrics['loss_lm_ce']:.6f} "
                f"loss_a_ce={metrics['loss_a_ce']:.6f} "
                f"loss_a_raw_ce={metrics['loss_a_raw_ce']:.6f} "
                f"loss_q_dir={metrics['loss_q_dir']:.6f} "
                f"loss_q_projected_dir={metrics['loss_q_projected_dir']:.6f} "
                f"loss_q_cue_ce={metrics['loss_q_cue_ce']:.6f} "
                f"loss_p={metrics['loss_p']:.6f} "
                f"loss_p_raw_kl={metrics['loss_p_raw_kl']:.6f} "
                f"loss_p_active={metrics['loss_p_active']:.6f} "
                f"loss_p_inactive={metrics['loss_p_inactive']:.6f} "
                f"loss_p_sensing={metrics['loss_p_sensing']:.6f} "
                f"loss_sep={metrics['loss_sep']:.6f} "
                f"delta_p_entropy={metrics['delta_p_entropy']:.6f} "
                f"delta_p_inactive_leakage={metrics['delta_p_inactive_leakage']:.6f} "
                f"grad_norm_proj={grad_norm:.6f} "
                f"grad_norm_control={grad_norm_control:.6f} "
                f"grad_norm_lora={grad_norm_lora:.6f} "
                f"grad_norm_proj_post_clip={grad_norm_proj_post_clip:.6f} "
                f"grad_norm_control_post_clip={grad_norm_control_post_clip:.6f} "
                f"grad_norm_lora_post_clip={grad_norm_lora_post_clip:.6f} "
                f"lr_proj={step_proj_lr:.9g} "
                f"lr_lora={step_lora_lr:.9g} "
                f"grad_norm_q_residual={grad_norm_q_residual:.6f} "
                f"q_residual_adapter_norm={q_residual_adapter_norm:.6f}"
            )

            if global_step % checkpoint_interval == 0:
                _save_mm_smoke(
                    model,
                    ckpt_root / f"{checkpoint_prefix}{global_step}",
                    {
                        "global_step": global_step,
                        "micro_step": micro_step,
                        "gradient_accumulation_steps": grad_accum_steps,
                        "effective_batch_size": (
                            int(train_cfg["per_device_batch_size"]) * grad_accum_steps
                        ),
                        "loss_ctl": metrics["loss_ctl"],
                        "loss_total": metrics["loss_total"],
                        "grad_norm_proj": grad_norm,
                        "grad_norm_control": grad_norm_control,
                        "grad_norm_lora": grad_norm_lora,
                        "grad_norm_proj_post_clip": grad_norm_proj_post_clip,
                        "grad_norm_control_post_clip": grad_norm_control_post_clip,
                        "grad_norm_lora_post_clip": grad_norm_lora_post_clip,
                        "trainable": trainable_label,
                        "train_lora": train_lora,
                        "load_lora": load_lora,
                        "train_control_offsets": control_offsets_trainable,
                        "trainable_control_offset_tensors": len(
                            control_offset_params
                        ),
                        "trainable_language_lora_tensors": len(trainable_language_lora_names),
                        "trainable_vision_lora_tensors": len(trainable_vision_lora_names),
                        "vision_modules_kept_in_eval": len(
                            frozen_vision_modules
                        ),
                        "frozen_vision_parameters": len(
                            getattr(model, "frozen_vision_parameter_names", [])
                        ),
                        "init_lora_checkpoint": init_lora_checkpoint,
                        "checkpoint_interval": checkpoint_interval,
                        "save_total_limit": checkpoint_limit,
                        "lr_scheduler": scheduler_name,
                        "warmup_ratio": warmup_ratio,
                        "warmup_steps": warmup_steps,
                        "include_response_tokens": include_response_tokens,
                        "use_chat_template": use_chat_template_value,
                        "lambda_lm_ce": lambda_lm_ce_value,
                        "phase1_lambda_ctl": phase1_lambda_ctl_value,
                        "projection_lr": proj_lr,
                        "lora_lr": lora_lr if train_lora else 0.0,
                        "lora_rank": model_cfg["lora"]["rank"] if lora_enabled else 0,
                        "lora_alpha": model_cfg["lora"]["alpha"] if lora_enabled else 0,
                        "projection_head_type": head_type,
                        "q_projection_mode": q_mode,
                        "q_geometry_mode": q_geom_mode,
                        "q_fixed_cue_weights": proj_head_config.get("q_fixed_cue_weights"),
                        "q_residual_max_scale": proj_head_config.get("q_residual_max_scale", 0.5),
                        "freeze_assoc_branch": freeze_assoc_branch,
                        "freeze_qp_branch": freeze_qp_branch,
                        "freeze_all_except_q": freeze_all_except_q,
                        "freeze_all_except_q_cue": freeze_all_except_q_cue,
                        "freeze_all_except_p": freeze_all_except_p,
                        "frozen_projection_tensors": len(frozen_projection_branches),
                        "isolated_projection_branch": isolated_projection_branch,
                        "isolated_trainable_projection_tensors": len(trainable_projection_branches),
                        "q_cue_only_trainable_tensors": (
                            len(trainable_projection_branches) if freeze_all_except_q_cue else 0
                        ),
                        "lambda_q": lambda_q_value,
                        "lambda_a": lambda_a_value,
                        "lambda_p": lambda_p_value,
                        "lambda_sep": lambda_sep_value,
                        "lambda_q_dir": lambda_q_dir_value,
                        "lambda_q_projected_dir": lambda_q_projected_dir_value,
                        "lambda_q_cue_ce": lambda_q_cue_ce_value,
                        "lambda_p_raw_kl": lambda_p_raw_kl_value,
                        "lambda_assoc_ce": assoc_ce_weight,
                        "lambda_assoc_raw_ce": assoc_raw_ce_weight,
                        "loaded_init": loaded_init,
                        **checkpoint_provenance,
                    },
                    # A frozen LoRA is unchanged and already stored in init_checkpoint.
                    # Do not duplicate it in every projection-only checkpoint.
                    save_lora=train_lora,
                )
                removed_checkpoints = rotate_step_checkpoints(
                    ckpt_root,
                    prefix=checkpoint_prefix,
                    save_total_limit=checkpoint_limit,
                )
                for removed_checkpoint in removed_checkpoints:
                    pbar.write(f"removed_old_checkpoint={removed_checkpoint}")

    pbar.close()

    final_dir = out_root / ("mm_sft_lora_smoke_final" if lora_enabled else "mm_sft_smoke_final")
    _save_mm_smoke(
        model,
        final_dir,
        {
            "global_step": global_step,
            "micro_step": micro_step,
            "gradient_accumulation_steps": grad_accum_steps,
            "effective_batch_size": (
                int(train_cfg["per_device_batch_size"]) * grad_accum_steps
            ),
            "max_steps": steps_limit,
            "max_seq_length": max_seq_length,
            "trainable": trainable_label,
            "train_lora": train_lora,
            "load_lora": load_lora,
            "train_control_offsets": control_offsets_trainable,
            "trainable_control_offset_tensors": len(
                control_offset_params
            ),
            "trainable_language_lora_tensors": len(trainable_language_lora_names),
            "trainable_vision_lora_tensors": len(trainable_vision_lora_names),
            "vision_modules_kept_in_eval": len(frozen_vision_modules),
            "frozen_vision_parameters": len(
                getattr(model, "frozen_vision_parameter_names", [])
            ),
            "init_lora_checkpoint": init_lora_checkpoint,
            "checkpoint_interval": checkpoint_interval,
            "save_total_limit": checkpoint_limit,
            "lr_scheduler": scheduler_name,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "include_response_tokens": include_response_tokens,
            "use_chat_template": use_chat_template_value,
            "lambda_lm_ce": lambda_lm_ce_value,
            "phase1_lambda_ctl": phase1_lambda_ctl_value,
            "projection_lr": proj_lr,
            "lora_lr": lora_lr if train_lora else 0.0,
            "lora_rank": model_cfg["lora"]["rank"] if lora_enabled else 0,
            "lora_alpha": model_cfg["lora"]["alpha"] if lora_enabled else 0,
            "projection_head_type": head_type,
            "q_projection_mode": q_mode,
            "q_geometry_mode": q_geom_mode,
            "q_fixed_cue_weights": proj_head_config.get("q_fixed_cue_weights"),
            "q_residual_max_scale": proj_head_config.get("q_residual_max_scale", 0.5),
            "freeze_assoc_branch": freeze_assoc_branch,
            "freeze_qp_branch": freeze_qp_branch,
            "freeze_all_except_q": freeze_all_except_q,
            "freeze_all_except_q_cue": freeze_all_except_q_cue,
            "freeze_all_except_p": freeze_all_except_p,
            "frozen_projection_tensors": len(frozen_projection_branches),
            "isolated_projection_branch": isolated_projection_branch,
            "isolated_trainable_projection_tensors": len(trainable_projection_branches),
            "q_cue_only_trainable_tensors": (
                len(trainable_projection_branches) if freeze_all_except_q_cue else 0
            ),
            "lambda_q": lambda_q_value,
            "lambda_a": lambda_a_value,
            "lambda_p": lambda_p_value,
            "lambda_sep": lambda_sep_value,
            "lambda_q_dir": lambda_q_dir_value,
            "lambda_q_projected_dir": lambda_q_projected_dir_value,
            "lambda_q_cue_ce": lambda_q_cue_ce_value,
            "lambda_p_raw_kl": lambda_p_raw_kl_value,
            "lambda_assoc_ce": assoc_ce_weight,
            "lambda_assoc_raw_ce": assoc_raw_ce_weight,
            "loaded_init": loaded_init,
            **checkpoint_provenance,
        },
        save_lora=lora_enabled,
    )
    print()
    print("OK: multimodal SFT smoke complete")
    print(f"  final_checkpoint: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行 BEV-image 多模态 SFT smoke")
    parser.add_argument("--config", type=str, default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="可选：中间 checkpoint 独立目录，避免不同 smoke 实验使用同名路径",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="可选：中间 checkpoint 保存间隔；默认读取配置文件",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="maximum number of matching intermediate step checkpoints to retain",
    )
    parser.add_argument("--train_lora", action="store_true")
    parser.add_argument(
        "--train_control_offsets",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "train the compact control-token representation adapter; "
            "defaults to training.sft.phase1.train_control_offsets"
        ),
    )
    parser.add_argument(
        "--load_lora",
        action="store_true",
        help="从 init checkpoint 加载 LoRA 但保持冻结，用于隔离 projection smoke",
    )
    parser.add_argument("--lambda_assoc_ce", type=float, default=None,
                        help="可选 association 分类辅助损失权重，默认使用配置或 0")
    parser.add_argument("--lambda_assoc_raw_ce", type=float, default=None,
                        help="可选 raw association logits 分类辅助损失权重，默认使用配置或 0")
    parser.add_argument("--lambda_q", type=float, default=None,
                        help="可选 delta_q 损失权重覆盖值")
    parser.add_argument("--lambda_a", type=float, default=None,
                        help="可选 delta_a BCE 损失权重覆盖值")
    parser.add_argument("--lambda_p", type=float, default=None,
                        help="可选 delta_p 损失权重覆盖值")
    parser.add_argument("--lambda_q_dir", type=float, default=None,
                        help="可选 delta_q raw 方向辅助损失权重，适用于 q target 贴移动边界的 smoke")
    parser.add_argument("--lambda_q_projected_dir", type=float, default=None,
                        help="可选：投影后 delta_q 方向损失权重，用于 fixed_residual_xy")
    parser.add_argument("--lambda_q_cue_ce", type=float, default=None,
                        help="可选：q 几何候选方向分类损失权重，用于 cue_xy 几何蒸馏")
    parser.add_argument("--lambda_p_raw_kl", type=float, default=None,
                        help="可选：PowerProjection 前 raw logits 的 soft-target KL 权重")
    parser.add_argument(
        "--lambda_lm_ce",
        type=float,
        default=None,
        help=(
            "multimodal response token CE weight; values >0 include the JSON "
            "response and require --train_lora"
        ),
    )
    parser.add_argument(
        "--use_chat_template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "override multimodal prompt formatting; schema-v5 fresh runs "
            "default to the Gemma chat template, while resumes preserve "
            "checkpoint metadata"
        ),
    )
    parser.add_argument("--projection_lr", type=float, default=None,
                        help="可选 projection head 学习率覆盖值")
    parser.add_argument("--lora_lr", type=float, default=None,
                        help="可选 LoRA 学习率覆盖值")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="可选：从已有 mm smoke checkpoint 加载 projection head / control token / LoRA")
    parser.add_argument(
        "--allow_partial_projection_load",
        action="store_true",
        help="explicitly allow missing/unexpected projection keys for architecture migration",
    )
    parser.add_argument(
        "--allow_legacy_oracle_data",
        action="store_true",
        help="diagnostic override only; permits pre-v5 Oracle data",
    )
    parser.add_argument(
        "--allow_checkpoint_dataset_mismatch",
        action="store_true",
        help=(
            "diagnostic/migration override only; permits an init checkpoint "
            "whose Oracle dataset provenance differs from the current data"
        ),
    )
    parser.add_argument("--projection_head_type", type=str, choices=["shared", "split"], default=None,
                        help="可选 projection head 类型；默认使用配置文件，split 用于 q/a/p 分支解耦实验")
    parser.add_argument("--q_projection_mode", type=str, choices=["clip", "direction"], default=None,
                        help="可选 q 投影模式；direction 用于 15m 边界饱和的 q 方向实验")
    parser.add_argument(
        "--q_geometry_mode",
        type=str,
        choices=["none", "cue_xy", "fixed_residual_xy"],
        default=None,
        help="可选：动态 cue_xy，或 train-only 固定几何先验加受限残差 fixed_residual_xy",
    )
    parser.add_argument("--freeze_assoc_branch", action="store_true",
                        help="split head 下冻结 association 分支，主要用于 Stage B2 训练 q/p")
    parser.add_argument("--freeze_qp_branch", action="store_true",
                        help="split head 下冻结 q/p 分支，主要用于 Stage A2 训练 association")
    parser.add_argument("--freeze_all_except_q", action="store_true",
                        help="split head 下只训练 readout_q / q_mlp，用于 direct Q 修复")
    parser.add_argument("--freeze_all_except_q_cue", action="store_true",
                        help="只训练 q 几何候选方向头 readout_q_cue，用于 B6")
    parser.add_argument("--freeze_all_except_p", action="store_true",
                        help="split head 下只训练 readout_p / p_mlp，用于 P-only 修复")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="覆盖配置中的梯度累积步数；max_steps 始终表示 optimizer updates",
    )
    args = parser.parse_args()

    train_mm_sft_smoke(
        config_path=args.config,
        data_dir=args.data_dir,
        model_path=args.model,
        max_steps=args.max_steps,
        max_length=args.max_length,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        train_lora=args.train_lora,
        train_control_offsets=args.train_control_offsets,
        load_lora=args.load_lora,
        lambda_assoc_ce=args.lambda_assoc_ce,
        lambda_q=args.lambda_q,
        lambda_a=args.lambda_a,
        lambda_p=args.lambda_p,
        lambda_assoc_raw_ce=args.lambda_assoc_raw_ce,
        lambda_q_dir=args.lambda_q_dir,
        lambda_q_projected_dir=args.lambda_q_projected_dir,
        lambda_q_cue_ce=args.lambda_q_cue_ce,
        lambda_p_raw_kl=args.lambda_p_raw_kl,
        projection_lr=args.projection_lr,
        lora_lr_override=args.lora_lr,
        init_checkpoint=args.init_checkpoint,
        projection_head_type=args.projection_head_type,
        q_projection_mode=args.q_projection_mode,
        q_geometry_mode=args.q_geometry_mode,
        freeze_assoc_branch=args.freeze_assoc_branch,
        freeze_qp_branch=args.freeze_qp_branch,
        freeze_all_except_q=args.freeze_all_except_q,
        freeze_all_except_q_cue=args.freeze_all_except_q_cue,
        freeze_all_except_p=args.freeze_all_except_p,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lambda_lm_ce=args.lambda_lm_ce,
        use_chat_template=args.use_chat_template,
        allow_partial_projection_load=args.allow_partial_projection_load,
        allow_legacy_oracle_data=args.allow_legacy_oracle_data,
        allow_checkpoint_dataset_mismatch=args.allow_checkpoint_dataset_mismatch,
    )
