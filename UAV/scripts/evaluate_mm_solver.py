#!/usr/bin/env python
"""End-to-end multimodal warm-start and downstream solver evaluation."""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.multimodal_dataset import (
    MultimodalSFTDataset,
    resolve_multimodal_chat_template,
    validate_multimodal_oracle_contract,
)
from src.data.oracle_contract import validate_checkpoint_dataset_compatibility
from src.data.oracle_runtime import (
    build_oracle_scenario,
    build_oracle_solver,
    environment_sample_to_solver_dict,
)
from src.training.train_dpo_mm import _checkpoint_metadata, _load_model


def _move_batch(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _solution_metrics(solver, solution, env):
    evaluated = solver.evaluate_solution(
        solution.Q,
        solution.A,
        solution.W_c_power,
        solution.W_s_power,
        env,
    )
    comm_sinr = (
        evaluated["communication_gains"]
        * solution.W_c_power
        / solver.N0
    )
    active = solution.A > 0.5
    sum_rate_mbps = float(
        solver.channel.B * np.log2(1.0 + comm_sinr[active]).sum() / 1e6
    )
    sensing_sinr = (
        solution.W_s_power[:, None]
        * evaluated["sensing_gains"]
        * solver.N_t
        * solver.N_r
        / solver.N0
    )
    detected_indices = np.flatnonzero(
        np.asarray(env["target_detected"], dtype=bool)
    )
    if detected_indices.size:
        visible_sinr = sensing_sinr[:, detected_indices]
        best_uav = np.argmax(visible_sinr, axis=0)
        best_sinr = visible_sinr[
            best_uav, np.arange(detected_indices.size)
        ]
        crbs = [
            solver.channel.compute_crb(
                solution.Q[best_uav[visible_index]],
                env["target_positions"][target_index],
                best_sinr[visible_index],
            )
            for visible_index, target_index in enumerate(detected_indices)
        ]
    else:
        best_sinr = np.empty(0, dtype=np.float64)
        crbs = []
    return {
        "utility": float(solution.utility),
        "raw_utility": float(solution.raw_utility),
        "initial_utility": float(solution.initial_utility),
        "iterations": int(solution.iterations),
        "solve_time": float(solution.solve_time),
        "feasible": float(solution.feasible),
        "sum_rate_mbps": sum_rate_mbps,
        "mean_sensing_sinr_db": (
            float(np.mean(10.0 * np.log10(best_sinr + 1e-12)))
            if best_sinr.size
            else 0.0
        ),
        "mean_crb": float(np.mean(crbs)) if crbs else 0.0,
        "communication_satisfaction": float(
            np.mean(comm_sinr[active] >= solver.comm_sinr_min)
        ),
        "sensing_satisfaction": (
            float(np.mean(best_sinr >= solver.cfg.sinr_s_min))
            if best_sinr.size
            else 1.0
        ),
        "constraint_violations": solution.constraint_violations,
    }


def _summarize(rows):
    summary = {"num_samples": len(rows)}
    scalar_keys = sorted(
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float))
    )
    for key in scalar_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--data_seed",
        type=int,
        default=None,
        help=(
            "optional assertion; the evaluator normally uses the sealed "
            "dataset seed from dataset_metadata.json"
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=3072)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow_legacy_dataset", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.model is not None:
        cfg["model"]["backbone"] = args.model
    checkpoint_dir = Path(args.checkpoint)
    data_root = Path(args.data_dir)
    dataset_metadata = validate_multimodal_oracle_contract(
        data_root,
        allow_legacy=args.allow_legacy_dataset,
        expected_simulation=cfg["simulation"],
        expected_seed=args.data_seed,
    )
    if not dataset_metadata:
        if args.data_seed is None:
            raise ValueError(
                "--data_seed is required only for an explicitly allowed "
                "legacy dataset without sealed metadata"
            )
        data_seed = int(args.data_seed)
    else:
        data_seed = int(dataset_metadata["seed"])
    if not args.allow_legacy_dataset:
        validate_checkpoint_dataset_compatibility(
            _checkpoint_metadata(checkpoint_dir), dataset_metadata
        )
    model, metadata = _load_model(cfg, checkpoint_dir, trainable=False)
    data_path = data_root / dataset_metadata.get(
        "sft_file", cfg["data"].get("sft_file", "sft_dataset.jsonl")
    )
    use_chat_template = resolve_multimodal_chat_template(
        dataset_metadata=dataset_metadata,
        checkpoint_metadata=metadata,
        configured_value=cfg["training"]["sft"].get("use_chat_template"),
    )
    dataset = MultimodalSFTDataset(
        data_path=str(data_path),
        data_dir=str(data_root),
        processor=model.processor,
        max_length=args.max_length,
        num_control_tokens=cfg["model"]["control_token"]["num_tokens"],
        include_response=False,
        use_chat_template=use_chat_template,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    sim = cfg["simulation"]
    scenario = build_oracle_scenario(sim, seed=data_seed)
    solver = build_oracle_solver(sim)

    warm_rows = []
    cold_rows = []
    comparisons = []
    model.eval()
    for index, batch in enumerate(tqdm(dataloader, desc="MM solver evaluation")):
        if index >= min(args.num_samples, len(dataset)):
            break
        record = dataset.data[index]
        match = re.fullmatch(r"env_(\d+)", record["id"])
        if match is None:
            raise ValueError(f"cannot recover sample id from {record['id']!r}")
        sample_id = int(match.group(1))
        env_sample = scenario.sample(sample_id)
        if not np.allclose(
            np.asarray(record["q_current"]), env_sample.q_current, atol=1e-5
        ):
            raise ValueError(
                "sealed dataset seed does not reproduce q_current for "
                f"sample {sample_id}"
            )
        env = environment_sample_to_solver_dict(env_sample)
        batch = _move_batch(batch, model.device)
        forward = {
            key: value
            for key, value in batch.items()
            if key not in {
                "labels", "label_mask", "has_q_current", "delta_q_target",
                "delta_a_target", "delta_p_target",
            }
        }
        with torch.no_grad():
            output = model(**forward)
        warm_start = {
            "delta_q": output["delta_q"].squeeze(0).float().cpu().numpy(),
            "delta_a": output["delta_a"].squeeze(0).float().cpu().numpy(),
            "delta_p": output["delta_p"].squeeze(0).float().cpu().numpy(),
        }
        warm_solution = solver.solve(env, warm_start=warm_start, seed=sample_id)
        cold_solution = solver.solve(env, warm_start=None, seed=sample_id)
        warm = _solution_metrics(solver, warm_solution, env)
        cold = _solution_metrics(solver, cold_solution, env)
        warm_rows.append(warm)
        cold_rows.append(cold)
        comparisons.append({
            "sample_id": sample_id,
            "iteration_speedup": cold["iterations"] / max(warm["iterations"], 1),
            "solve_time_speedup": cold["solve_time"] / max(warm["solve_time"], 1e-12),
            "utility_gain": warm["utility"] - cold["utility"],
            "warm_feasible": warm["feasible"],
            "cold_feasible": cold["feasible"],
        })

    if not warm_rows:
        raise RuntimeError("no samples were evaluated")
    result = {
        "checkpoint": str(checkpoint_dir),
        "data": str(data_path),
        "data_seed": data_seed,
        "solver_algorithm": "constraint_aware_alternating_optimization",
        "warm": _summarize(warm_rows),
        "cold": _summarize(cold_rows),
        "comparison": _summarize(comparisons),
        "per_sample": comparisons,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({
        "warm_feasible_rate": result["warm"]["feasible"]["mean"],
        "cold_feasible_rate": result["cold"]["feasible"]["mean"],
        "iteration_speedup": result["comparison"]["iteration_speedup"]["mean"],
        "utility_gain": result["comparison"]["utility_gain"]["mean"],
    }, indent=2))
    print(f"Saved end-to-end evaluation to {output_path}")


if __name__ == "__main__":
    main()
