#!/usr/bin/env python
"""
Analyze delta_q / delta_a / delta_p diversity for one or more checkpoints.

This is a lightweight diagnostic: it runs model forward / projection-head
warm-start generation on deterministic evaluation environments, but does not
run SCA-FP. Use it to detect constant-head collapse and to compare SFT vs DPO.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.prompt_builder import build_full_prompt
from src.env import ISACScenarioGenerator
from src.model import Gemma3ISAC, build_proj_head_config


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _entropy_from_probs(probs: np.ndarray, axis: int = 0) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum(axis=axis, keepdims=True)
    return -(probs * np.log(probs)).sum(axis=axis)


def _summarize_tensor(name: str, values: np.ndarray) -> Dict:
    flat = values.reshape(values.shape[0], -1)
    per_dim_std = flat.std(axis=0)
    per_sample_norm = np.linalg.norm(flat, axis=1)
    return {
        f"{name}_shape": list(values.shape),
        f"{name}_mean": float(flat.mean()),
        f"{name}_std_all": float(flat.std()),
        f"{name}_mean_abs": float(np.abs(flat).mean()),
        f"{name}_per_dim_std_mean": float(per_dim_std.mean()),
        f"{name}_per_dim_std_max": float(per_dim_std.max()),
        f"{name}_per_sample_norm_mean": float(per_sample_norm.mean()),
        f"{name}_per_sample_norm_std": float(per_sample_norm.std()),
        f"{name}_min": float(flat.min()),
        f"{name}_max": float(flat.max()),
    }


def _summarize_deltas(delta_q: np.ndarray, delta_a: np.ndarray, delta_p: np.ndarray) -> Dict:
    summary = {}
    summary.update(_summarize_tensor("delta_q", delta_q))
    summary.update(_summarize_tensor("delta_a", delta_a))
    summary.update(_summarize_tensor("delta_p", delta_p))

    # Association diagnostics: does each user always select the same UAV?
    # delta_a shape: [N, M, K]
    assoc_choice = np.argmax(delta_a, axis=1)  # [N, K]
    assoc_unique_counts = [
        int(np.unique(assoc_choice[:, k]).size)
        for k in range(assoc_choice.shape[1])
    ]
    summary["delta_a_argmax_unique_per_user_mean"] = float(np.mean(assoc_unique_counts))
    summary["delta_a_argmax_unique_per_user_min"] = int(np.min(assoc_unique_counts))
    summary["delta_a_argmax_unique_per_user_max"] = int(np.max(assoc_unique_counts))

    # Entropy over UAVs for each user's soft association, averaged over samples/users.
    assoc_entropy = _entropy_from_probs(delta_a, axis=1)  # [N, K]
    summary["delta_a_entropy_mean"] = float(assoc_entropy.mean())
    summary["delta_a_entropy_std"] = float(assoc_entropy.std())

    # Power split diagnostics. Normalize each UAV row across K users + sensing.
    # delta_p shape: [N, M, K+1]
    p = np.clip(delta_p, 0.0, None)
    p_sum = p.sum(axis=2, keepdims=True)
    p_norm = p / np.maximum(p_sum, 1e-12)
    power_entropy = _entropy_from_probs(p_norm, axis=2)  # [N, M]
    summary["delta_p_entropy_mean"] = float(power_entropy.mean())
    summary["delta_p_entropy_std"] = float(power_entropy.std())

    # Lightweight warnings. Thresholds are intentionally conservative and should
    # be interpreted alongside eval metrics, not as hard pass/fail proof.
    warnings = []
    if summary["delta_q_per_dim_std_mean"] < 1e-3:
        warnings.append("delta_q_low_cross_sample_variance")
    if summary["delta_a_per_dim_std_mean"] < 1e-3:
        warnings.append("delta_a_low_cross_sample_variance")
    if summary["delta_p_per_dim_std_mean"] < 1e-4:
        warnings.append("delta_p_low_cross_sample_variance")
    if summary["delta_a_argmax_unique_per_user_mean"] <= 1.2:
        warnings.append("delta_a_argmax_nearly_constant")
    summary["warnings"] = warnings
    return summary


def _load_model(config: Dict, model_path: str) -> Gemma3ISAC:
    sim_cfg = config["simulation"]
    model_cfg = config["model"]
    model = Gemma3ISAC.from_pretrained(
        load_dir=model_path,
        base_model_name=model_cfg["backbone"],
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
        use_4bit=config["hardware"]["use_4bit"],
        lora_rank=model_cfg["lora"]["rank"],
        lora_alpha=model_cfg["lora"]["alpha"],
        num_control_tokens=model_cfg["control_token"]["num_tokens"],
        proj_head_config=build_proj_head_config(model_cfg, sim_cfg),
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model


def _collect_deltas(config: Dict, model_path: str, num_samples: int) -> Dict[str, np.ndarray]:
    sim_cfg = config["simulation"]
    scenario_gen = ISACScenarioGenerator(
        num_uavs=sim_cfg["num_uavs"],
        num_users=sim_cfg["num_users"],
        num_targets=sim_cfg["num_targets"],
        area_size=tuple(sim_cfg["area_size"]),
        carrier_freq_ghz=sim_cfg["carrier_freq_ghz"],
        bandwidth_mhz=sim_cfg["bandwidth_mhz"],
        num_antennas=sim_cfg["num_antennas_tx"],
        p_max_dbm=sim_cfg["p_max_dbm"],
        seed=42,
    )

    print(f"Loading model: {model_path}")
    model = _load_model(config, model_path)
    device = next(model.parameters()).device

    delta_qs: List[np.ndarray] = []
    delta_as: List[np.ndarray] = []
    delta_ps: List[np.ndarray] = []

    for i in tqdm(range(num_samples), desc=f"Delta inference {Path(model_path).name}"):
        env_sample = scenario_gen.sample(i)
        prompt = build_full_prompt(env_sample, sim_cfg)
        q_current = torch.tensor(
            env_sample.q_current, dtype=torch.float32, device=device
        ).unsqueeze(0)
        warm_start = model.generate_warmstart(prompt, q_current=q_current)
        delta_qs.append(_as_np(warm_start["delta_q"]))
        delta_as.append(_as_np(warm_start["delta_a"]))
        delta_ps.append(_as_np(warm_start["delta_p"]))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "delta_q": np.stack(delta_qs, axis=0),
        "delta_a": np.stack(delta_as, axis=0),
        "delta_p": np.stack(delta_ps, axis=0),
    }


def _compare_models(all_deltas: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    names = list(all_deltas.keys())
    comparisons = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            key = f"{a_name}_vs_{b_name}"
            comparisons[key] = {}
            for delta_name in ("delta_q", "delta_a", "delta_p"):
                a = all_deltas[a_name][delta_name].reshape(all_deltas[a_name][delta_name].shape[0], -1)
                b = all_deltas[b_name][delta_name].reshape(all_deltas[b_name][delta_name].shape[0], -1)
                diff = b - a
                comparisons[key][f"{delta_name}_l2_mean"] = float(np.linalg.norm(diff, axis=1).mean())
                comparisons[key][f"{delta_name}_l2_std"] = float(np.linalg.norm(diff, axis=1).std())
                comparisons[key][f"{delta_name}_mean_abs_diff"] = float(np.abs(diff).mean())
                comparisons[key][f"{delta_name}_max_abs_diff"] = float(np.abs(diff).max())
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Evaluation/training config YAML")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model specs as name=/path/to/checkpoint",
    )
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--save_raw",
        action="store_true",
        help="Also save raw delta arrays next to the JSON summary as .npz",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_specs = {}
    for spec in args.models:
        if "=" not in spec:
            raise ValueError(f"Invalid model spec {spec!r}; expected name=/path")
        name, path = spec.split("=", 1)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Model path not found for {name}: {path}")
        model_specs[name] = path

    all_deltas = {}
    summaries = {}
    for name, path in model_specs.items():
        deltas = _collect_deltas(config, path, args.num_samples)
        all_deltas[name] = deltas
        summaries[name] = _summarize_deltas(
            deltas["delta_q"], deltas["delta_a"], deltas["delta_p"]
        )

    result = {
        "config": args.config,
        "num_samples": args.num_samples,
        "models": model_specs,
        "summaries": summaries,
        "comparisons": _compare_models(all_deltas),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved summary to {output_path}")

    if args.save_raw:
        raw_path = output_path.with_suffix(".npz")
        npz_payload = {}
        for model_name, deltas in all_deltas.items():
            for delta_name, arr in deltas.items():
                npz_payload[f"{model_name}_{delta_name}"] = arr
        np.savez_compressed(raw_path, **npz_payload)
        print(f"Saved raw deltas to {raw_path}")

    print("\n=== Delta Diagnostic Summary ===")
    for model_name, summary in summaries.items():
        print(f"\n[{model_name}]")
        for k in (
            "delta_q_per_dim_std_mean",
            "delta_a_per_dim_std_mean",
            "delta_p_per_dim_std_mean",
            "delta_a_argmax_unique_per_user_mean",
            "delta_a_entropy_mean",
            "delta_p_entropy_mean",
            "warnings",
        ):
            print(f"  {k}: {summary[k]}")

    if result["comparisons"]:
        print("\n=== Model Comparisons ===")
        for pair_name, comp in result["comparisons"].items():
            print(f"\n[{pair_name}]")
            for k, v in comp.items():
                if k.endswith("_l2_mean") or k.endswith("_mean_abs_diff"):
                    print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
