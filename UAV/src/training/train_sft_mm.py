#!/usr/bin/env python
"""
BEV-image Gemma3 多模态 SFT 烟雾测试。

首版训练烟雾测试对 RTX 5090 32GB 保持保守：
  - 默认冻结 Gemma3 多模态 backbone
  - 默认只训练 projection head
  - 默认只优化 CTL loss

它用于在前向传播烟雾测试已通过后验证训练闭环：
  dataset -> multimodal forward -> projection head -> control loss
  -> backward -> optimizer step -> checkpoint

如需测试 LoRA 链路，可显式传入 --train_lora。
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.env_setup import setup_env

setup_env()

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

from src.data.multimodal_dataset import MultimodalSFTDataset
from src.model import Gemma3MultimodalISAC, UAVISACLosses, build_proj_head_config


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


def _save_mm_smoke(model, save_dir: Path, metadata: dict, save_lora: bool = False):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.projection_head.state_dict(), save_dir / "projection_head.pt")
    metadata["control_token_embeddings"] = model.save_control_token_embeddings(save_dir)
    model.processor.save_pretrained(save_dir / "processor")
    if save_lora and hasattr(model.base_model, "save_pretrained"):
        model.base_model.save_pretrained(save_dir / "lora")
    with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _resolve_lora_checkpoint(init_checkpoint: str, train_lora: bool):
    if not init_checkpoint or not train_lora:
        return None
    candidate = Path(init_checkpoint) / "lora"
    if (candidate / "adapter_config.json").exists():
        return str(candidate)
    return None


def _load_mm_smoke_checkpoint(model, init_checkpoint: str) -> dict:
    if not init_checkpoint:
        return {}
    ckpt_dir = Path(init_checkpoint)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"init checkpoint not found: {ckpt_dir}")

    loaded = {"init_checkpoint": str(ckpt_dir)}
    proj_path = ckpt_dir / "projection_head.pt"
    if proj_path.exists():
        state = torch.load(proj_path, map_location="cpu")
        load_result = model.projection_head.load_state_dict(state, strict=False)
        loaded["projection_head"] = str(proj_path)
        loaded["projection_missing_keys"] = list(load_result.missing_keys)
        loaded["projection_unexpected_keys"] = list(load_result.unexpected_keys)

    loaded_ctrl = model.load_control_token_embeddings(ckpt_dir)
    if loaded_ctrl:
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


def train_mm_sft_smoke(
    config_path: str,
    data_dir: str = None,
    model_path: str = None,
    max_steps: int = None,
    max_length: int = None,
    output_dir: str = None,
    train_lora: bool = False,
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
    power_assoc_gate_strength: float = None,
    freeze_assoc_branch: bool = False,
    freeze_qp_branch: bool = False,
    freeze_all_except_q: bool = False,
    freeze_all_except_q_cue: bool = False,
    freeze_all_except_p: bool = False,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])

    model_cfg = cfg["model"]
    sim_cfg = cfg["simulation"]
    train_cfg = cfg["training"]["sft"]
    data_cfg = cfg["data"]

    data_root = Path(data_dir or data_cfg["output_dir"])
    sft_path = data_root / data_cfg.get("sft_file", "sft_dataset.jsonl")
    model_name = model_path or model_cfg["backbone"]
    max_seq_length = int(max_length or train_cfg["max_seq_length"])
    steps_limit = int(max_steps or train_cfg.get("phase1", {}).get("max_steps", 30))
    out_root = Path(output_dir or cfg.get("output_dir", "/root/autodl-tmp/outputs/mm_smoke"))
    ckpt_root = Path(cfg.get("checkpoint_dir", "/root/autodl-tmp/checkpoints/mm_smoke"))
    ckpt_root.mkdir(parents=True, exist_ok=True)
    init_lora_checkpoint = _resolve_lora_checkpoint(init_checkpoint, train_lora)

    print("=" * 60)
    print("BEV-image multimodal SFT smoke")
    print("=" * 60)
    print(f"  data:       {sft_path}")
    print(f"  model:      {model_name}")
    print(f"  max_length: {max_seq_length}")
    print(f"  steps:      {steps_limit}")
    print(f"  trainable:  {'projection_head + LoRA' if train_lora else 'projection_head only'}")
    print()

    proj_head_config = build_proj_head_config(model_cfg, sim_cfg)
    if projection_head_type is not None:
        proj_head_config["head_type"] = projection_head_type
    if q_projection_mode is not None:
        proj_head_config["q_projection_mode"] = q_projection_mode
    if q_geometry_mode is not None:
        proj_head_config["q_geometry_mode"] = q_geometry_mode
    if power_assoc_gate_strength is not None:
        proj_head_config["power_assoc_gate_strength"] = float(power_assoc_gate_strength)
    head_type = proj_head_config.get("head_type", "shared")
    q_mode = proj_head_config.get("q_projection_mode", "clip")
    q_geom_mode = proj_head_config.get("q_geometry_mode", "none")
    p_assoc_gate_strength = float(proj_head_config.get("power_assoc_gate_strength", 0.0))
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
        enable_lora=train_lora,
        lora_rank=model_cfg["lora"]["rank"],
        lora_alpha=model_cfg["lora"]["alpha"],
        lora_dropout=model_cfg["lora"].get("dropout", 0.0),
        lora_target_modules=model_cfg["lora"]["target_modules"],
        lora_checkpoint=init_lora_checkpoint,
    )
    loaded_init = _load_mm_smoke_checkpoint(model, init_checkpoint)
    if loaded_init:
        print(f"  init_checkpoint: {loaded_init}")

    if train_lora:
        model.base_model.train()
    else:
        for param in model.base_model.parameters():
            param.requires_grad = False
        model.base_model.eval()
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
            trainable_prefixes=("readout_q", "q_mlp", "q_residual_gate_logit"),
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
    elif freeze_qp_branch:
        frozen_projection_branches = _set_projection_branch_trainable(
            model,
            branch_prefixes=("readout_q", "q_mlp", "readout_q_cue", "readout_p", "p_mlp"),
            trainable=False,
        )

    dataset = MultimodalSFTDataset(
        data_path=str(sft_path),
        data_dir=str(data_root),
        processor=model.processor,
        max_length=max_seq_length,
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
    )
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
    loss_fn = UAVISACLosses(
        lambda_ctl=model_cfg["loss"]["lambda_ctl"],
        lambda_q=lambda_q_value,
        lambda_a=lambda_a_value,
        lambda_p=lambda_p_value,
        lambda_sep=model_cfg["loss"]["lambda_sep"],
        lambda_assoc_ce=assoc_ce_weight,
        lambda_assoc_raw_ce=assoc_raw_ce_weight,
        lambda_q_dir=lambda_q_dir_value,
        lambda_q_projected_dir=lambda_q_projected_dir_value,
        lambda_q_cue_ce=lambda_q_cue_ce_value,
        lambda_p_raw_kl=lambda_p_raw_kl_value,
        power_temperature=float(model_cfg["projection_head"]["tau_power"]),
    )
    # 默认只训练投影头；传入 --train_lora 时，PEFT 会额外打开 LoRA 参数。
    proj_params = [p for p in model.projection_head.parameters() if p.requires_grad]
    lora_params = [
        p for n, p in model.base_model.named_parameters()
        if p.requires_grad and "lora_" in n
    ]
    proj_lr = float(projection_lr) if projection_lr is not None else 1e-3
    lora_lr = (
        float(lora_lr_override)
        if lora_lr_override is not None
        else train_cfg.get("phase1", {}).get("lr_lora", train_cfg.get("learning_rate", 2e-4))
    )
    if train_lora and not lora_params:
        raise RuntimeError("已传入 --train_lora，但没有发现可训练的 LoRA 参数。")

    param_groups = [{"params": proj_params, "lr": proj_lr}]
    if train_lora and lora_params:
        param_groups.append({"params": lora_params, "lr": lora_lr})
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    print(f"  trainable projection tensors: {len(proj_params)}")
    print(f"  trainable LoRA tensors:       {len(lora_params)}")
    print(f"  projection lr:                {proj_lr}")
    print(f"  LoRA lr:                      {lora_lr if train_lora else 0.0}")
    print(f"  projection head type:         {head_type}")
    print(f"  q projection mode:            {q_mode}")
    print(f"  q geometry mode:              {q_geom_mode}")
    print(f"  power association gate:       {p_assoc_gate_strength}")
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

    device = model.device
    global_step = 0
    epoch = 0
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
                    "q_geometry_mask",
                }
            }

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
            if "q_geometry_cues" in batch:
                delta_target["q_geometry_cues"] = batch["q_geometry_cues"]
            if "q_geometry_mask" in batch:
                delta_target["q_geometry_mask"] = batch["q_geometry_mask"]
            total_loss, metrics = loss_fn.compute_phase1_total(
                delta_hat=delta_hat,
                delta_target=delta_target,
                phase1_lambda_ctl=train_cfg.get("phase1", {}).get("lambda_ctl", 1.0),
            )

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
                metrics["q_residual_gate"] = float(
                    outputs["q_residual_gate"].float().item()
                    if "q_residual_gate" in outputs
                    else 0.0
                )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = _grad_norm(model.projection_head.parameters())
            grad_norm_lora = _grad_norm(lora_params) if train_lora else 0.0
            clip_params = list(model.projection_head.parameters()) + lora_params
            torch.nn.utils.clip_grad_norm_(clip_params, cfg["hardware"].get("max_grad_norm", 1.0))
            optimizer.step()

            global_step += 1
            pbar.update(1)
            pbar.write(
                f"step={global_step} epoch={epoch} "
                f"loss_ctl={metrics['loss_ctl']:.6f} "
                f"loss_total={metrics['loss_total']:.6f} "
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
                f"delta_p_entropy={metrics['delta_p_entropy']:.6f} "
                f"delta_p_inactive_leakage={metrics['delta_p_inactive_leakage']:.6f} "
                f"q_residual_gate={metrics['q_residual_gate']:.6f} "
                f"grad_norm_proj={grad_norm:.6f} "
                f"grad_norm_lora={grad_norm_lora:.6f}"
            )

            if torch.isnan(total_loss):
                raise RuntimeError("NaN loss detected in multimodal SFT smoke.")

            if global_step % train_cfg.get("save_steps", 10) == 0:
                _save_mm_smoke(
                    model,
                    ckpt_root / f"mm_sft_{'lora_' if train_lora else ''}smoke_step_{global_step}",
                    {
                        "global_step": global_step,
                        "loss_ctl": metrics["loss_ctl"],
                        "loss_total": metrics["loss_total"],
                        "grad_norm_proj": grad_norm,
                        "grad_norm_lora": grad_norm_lora,
                        "trainable": "projection_head_lora" if train_lora else "projection_head_only",
                        "projection_lr": proj_lr,
                        "lora_lr": lora_lr if train_lora else 0.0,
                        "lora_rank": model_cfg["lora"]["rank"] if train_lora else 0,
                        "lora_alpha": model_cfg["lora"]["alpha"] if train_lora else 0,
                        "projection_head_type": head_type,
                        "q_projection_mode": q_mode,
                        "q_geometry_mode": q_geom_mode,
                        "q_fixed_cue_weights": proj_head_config.get("q_fixed_cue_weights"),
                        "q_residual_max_scale": proj_head_config.get("q_residual_max_scale", 1.0),
                        "power_assoc_gate_strength": p_assoc_gate_strength,
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
                        "lambda_q_dir": lambda_q_dir_value,
                        "lambda_q_projected_dir": lambda_q_projected_dir_value,
                        "lambda_q_cue_ce": lambda_q_cue_ce_value,
                        "lambda_p_raw_kl": lambda_p_raw_kl_value,
                        "lambda_assoc_ce": assoc_ce_weight,
                        "lambda_assoc_raw_ce": assoc_raw_ce_weight,
                        "loaded_init": loaded_init,
                    },
                    save_lora=train_lora,
                )

    pbar.close()

    final_dir = out_root / ("mm_sft_lora_smoke_final" if train_lora else "mm_sft_smoke_final")
    _save_mm_smoke(
        model,
        final_dir,
        {
            "global_step": global_step,
            "max_steps": steps_limit,
            "max_seq_length": max_seq_length,
            "trainable": "projection_head_lora" if train_lora else "projection_head_only",
            "projection_lr": proj_lr,
            "lora_lr": lora_lr if train_lora else 0.0,
            "lora_rank": model_cfg["lora"]["rank"] if train_lora else 0,
            "lora_alpha": model_cfg["lora"]["alpha"] if train_lora else 0,
            "projection_head_type": head_type,
            "q_projection_mode": q_mode,
            "q_geometry_mode": q_geom_mode,
            "q_fixed_cue_weights": proj_head_config.get("q_fixed_cue_weights"),
            "q_residual_max_scale": proj_head_config.get("q_residual_max_scale", 1.0),
            "power_assoc_gate_strength": p_assoc_gate_strength,
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
            "lambda_q_dir": lambda_q_dir_value,
            "lambda_q_projected_dir": lambda_q_projected_dir_value,
            "lambda_q_cue_ce": lambda_q_cue_ce_value,
            "lambda_p_raw_kl": lambda_p_raw_kl_value,
            "lambda_assoc_ce": assoc_ce_weight,
            "lambda_assoc_raw_ce": assoc_raw_ce_weight,
            "loaded_init": loaded_init,
        },
        save_lora=train_lora,
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
    parser.add_argument("--train_lora", action="store_true")
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
    parser.add_argument("--projection_lr", type=float, default=None,
                        help="可选 projection head 学习率覆盖值")
    parser.add_argument("--lora_lr", type=float, default=None,
                        help="可选 LoRA 学习率覆盖值")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="可选：从已有 mm smoke checkpoint 加载 projection head / control token / LoRA")
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
    parser.add_argument(
        "--power_assoc_gate_strength",
        type=float,
        default=None,
        help="可选：PowerProjection 的可微 association log-gate 强度；0 表示关闭",
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
    args = parser.parse_args()

    train_mm_sft_smoke(
        config_path=args.config,
        data_dir=args.data_dir,
        model_path=args.model,
        max_steps=args.max_steps,
        max_length=args.max_length,
        output_dir=args.output_dir,
        train_lora=args.train_lora,
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
        power_assoc_gate_strength=args.power_assoc_gate_strength,
        freeze_assoc_branch=args.freeze_assoc_branch,
        freeze_qp_branch=args.freeze_qp_branch,
        freeze_all_except_q=args.freeze_all_except_q,
        freeze_all_except_q_cue=args.freeze_all_except_q_cue,
        freeze_all_except_p=args.freeze_all_except_p,
    )
