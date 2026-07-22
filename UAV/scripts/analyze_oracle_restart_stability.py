#!/usr/bin/env python
"""Audit whether random-restart Oracle solutions provide stable Q targets.

The multimodal dataset stores the highest-utility feasible solution from a
small set of deterministic random restarts.  This script reconstructs those
same environments and restarts, then measures whether near-equal-utility
solutions disagree in Q direction or in the derived geometry-cue label.
"""

import argparse
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from scripts.generate_mm_smoke import _build_solver
from src.data.geometry_cues import CUE_NAMES, parse_q_geometry_cues
from src.data.multimodal_dataset import validate_multimodal_oracle_contract
from src.data.oracle_generator import (
    OracleDataGenerator,
    select_near_optimal_q_medoid,
)
from src.env import ISACScenarioGenerator


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def _cue_labels(cues: np.ndarray, mask: np.ndarray, delta_q: np.ndarray) -> np.ndarray:
    cosine = np.sum(_unit(cues) * _unit(delta_q[:, :2])[:, None, :], axis=-1)
    cosine = np.where(mask > 0.5, cosine, -1e9)
    return cosine.argmax(axis=-1)


def summarize_restart_set(
    utilities: np.ndarray,
    delta_q: np.ndarray,
    cues: np.ndarray,
    cue_mask: np.ndarray,
) -> Dict:
    """Summarize one environment after sorting restarts by utility."""
    utilities = np.asarray(utilities, dtype=np.float64)
    delta_q = np.asarray(delta_q, dtype=np.float64)
    if utilities.ndim != 1 or delta_q.ndim != 3:
        raise ValueError("utilities/delta_q must have shapes (R,) and (R,M,3)")
    if utilities.shape[0] != delta_q.shape[0]:
        raise ValueError("utilities and delta_q restart counts differ")
    if utilities.shape[0] < 2:
        return {
            "num_feasible_restarts": int(utilities.shape[0]),
            "top_second_relative_utility_gap": None,
            "top_second_q_3d_cosine": None,
            "restart_q_3d_cosine_mean": None,
            "restart_q_xy_cosine_mean": None,
            "restart_cue_agreement_mean": None,
            "near_equal_divergent": False,
        }

    order = np.argsort(-utilities)
    utilities = utilities[order]
    delta_q = delta_q[order]
    directions_3d = _unit(delta_q)
    directions_xy = _unit(delta_q[..., :2])
    labels = np.stack(
        [_cue_labels(cues, cue_mask, restart_delta) for restart_delta in delta_q]
    )

    pair_q_3d: List[float] = []
    pair_q_xy: List[float] = []
    pair_cue_agreement: List[float] = []
    for first, second in itertools.combinations(range(delta_q.shape[0]), 2):
        pair_q_3d.append(
            float(np.sum(directions_3d[first] * directions_3d[second], axis=-1).mean())
        )
        pair_q_xy.append(
            float(np.sum(directions_xy[first] * directions_xy[second], axis=-1).mean())
        )
        pair_cue_agreement.append(float((labels[first] == labels[second]).mean()))

    relative_gap = float(
        (utilities[0] - utilities[1]) / max(1.0, abs(float(utilities[0])))
    )
    top_second_q_3d = float(
        np.sum(directions_3d[0] * directions_3d[1], axis=-1).mean()
    )
    return {
        "num_feasible_restarts": int(utilities.shape[0]),
        "top_second_relative_utility_gap": relative_gap,
        "top_second_q_3d_cosine": top_second_q_3d,
        "restart_q_3d_cosine_mean": float(np.mean(pair_q_3d)),
        "restart_q_xy_cosine_mean": float(np.mean(pair_q_xy)),
        "restart_cue_agreement_mean": float(np.mean(pair_cue_agreement)),
        "near_equal_divergent": bool(relative_gap < 0.01 and top_second_q_3d < 0.5),
    }


def _mean(rows: List[Dict], key: str):
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _build_generator(cfg: Dict, seed: int) -> OracleDataGenerator:
    sim = cfg["simulation"]
    scenario = ISACScenarioGenerator(
        num_uavs=sim["num_uavs"],
        num_users=sim["num_users"],
        num_targets=sim["num_targets"],
        area_size=tuple(sim["area_size"]),
        carrier_freq_ghz=sim["carrier_freq_ghz"],
        bandwidth_mhz=sim["bandwidth_mhz"],
        num_antennas=sim["num_antennas_tx"],
        num_antennas_rx=sim.get("num_antennas_rx", sim["num_antennas_tx"]),
        p_max_dbm=sim["p_max_dbm"],
        noise_figure_db=sim["noise_figure_db"],
        seed=seed,
    )
    return OracleDataGenerator(
        scenario_gen=scenario,
        solver=_build_solver(sim),
        config=cfg["data"],
        sim_config=sim,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")

    config_path = PROJECT_ROOT / args.config
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    data_root = Path(args.data_dir)
    metadata = validate_multimodal_oracle_contract(
        data_root,
        expected_simulation=cfg["simulation"],
    )
    num_restarts = int(metadata["num_restarts"])
    generator = _build_generator(cfg, int(metadata["seed"]))
    records = []
    sft_path = data_root / metadata["sft_file"]
    with sft_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    records = records[: min(args.num_samples, len(records))]

    rows = []
    for record_index, record in enumerate(records, start=1):
        match = re.fullmatch(r"env_(\d+)", str(record.get("id", "")))
        if match is None:
            raise ValueError(f"invalid environment record id: {record.get('id')!r}")
        environment_id = int(match.group(1))
        environment = generator.scenario_gen.sample(environment_id)
        env_dict = generator._env_sample_to_dict(environment)
        cues, cue_mask = parse_q_geometry_cues(
            record["prompt"], cfg["simulation"]["num_uavs"]
        )

        feasible_solutions = []
        for restart in range(num_restarts):
            seed = environment_id * num_restarts + restart
            solution = generator.solver.solve(env_dict, warm_start=None, seed=seed)
            if solution.feasible and np.isfinite(solution.utility):
                feasible_solutions.append(solution)
        feasible_solutions.sort(key=lambda solution: solution.utility, reverse=True)
        restart_delta_q = np.stack(
            [generator._extract_prior(solution, environment)[0] for solution in feasible_solutions]
        ) if feasible_solutions else np.empty((0, cfg["simulation"]["num_uavs"], 3))
        restart_utilities = np.asarray(
            [solution.utility for solution in feasible_solutions], dtype=np.float64
        )
        row = {
            "environment_id": environment_id,
            **summarize_restart_set(
                restart_utilities, restart_delta_q, cues, cue_mask
            ),
        }
        if feasible_solutions:
            candidates = generator._pareto_filter(feasible_solutions.copy())
            selected_solution, selection_diagnostics = (
                select_near_optimal_q_medoid(
                    candidates,
                    env_dict["q_current"],
                    generator.oracle_selection_utility_tolerance,
                )
            )
            reproduced_delta_q = generator._extract_prior(
                selected_solution, environment
            )[0]
            row.update(selection_diagnostics)
            stored_delta_q = np.asarray(record["delta_q"], dtype=np.float64)
            row["stored_vs_reproduced_q_cosine"] = float(
                np.sum(_unit(stored_delta_q) * _unit(reproduced_delta_q), axis=-1).mean()
            )
            row["stored_vs_reproduced_q_mse"] = float(
                np.mean((stored_delta_q - reproduced_delta_q) ** 2)
            )
        rows.append(row)
        print(
            f"[{record_index}/{len(records)}] env={environment_id} "
            f"feasible={row['num_feasible_restarts']}/{num_restarts} "
            f"q_cos={row['restart_q_3d_cosine_mean']} "
            f"cue_agree={row['restart_cue_agreement_mean']}"
        )

    summary = {
        "num_environments": len(rows),
        "num_restarts": num_restarts,
        "feasible_restarts_mean": _mean(rows, "num_feasible_restarts"),
        "top_second_relative_utility_gap_mean": _mean(
            rows, "top_second_relative_utility_gap"
        ),
        "top_second_q_3d_cosine_mean": _mean(rows, "top_second_q_3d_cosine"),
        "restart_q_3d_cosine_mean": _mean(rows, "restart_q_3d_cosine_mean"),
        "restart_q_xy_cosine_mean": _mean(rows, "restart_q_xy_cosine_mean"),
        "restart_cue_agreement_mean": _mean(rows, "restart_cue_agreement_mean"),
        "oracle_near_optimal_candidate_count_mean": _mean(
            rows, "oracle_near_optimal_candidate_count"
        ),
        "oracle_chosen_candidate_rank_mean": _mean(
            rows, "oracle_chosen_candidate_rank"
        ),
        "oracle_chosen_relative_utility_gap_mean": _mean(
            rows, "oracle_chosen_relative_utility_gap"
        ),
        "oracle_chosen_q_consensus_cosine_mean": _mean(
            rows, "oracle_chosen_q_consensus_cosine"
        ),
        "oracle_best_q_consensus_cosine_mean": _mean(
            rows, "oracle_best_q_consensus_cosine"
        ),
        "oracle_q_consensus_gain_mean": _mean(
            rows, "oracle_q_consensus_gain"
        ),
        "near_equal_divergent_environment_ratio": float(
            np.mean([row["near_equal_divergent"] for row in rows])
        ) if rows else 0.0,
        "stored_vs_reproduced_q_cosine_mean": _mean(
            rows, "stored_vs_reproduced_q_cosine"
        ),
        "stored_vs_reproduced_q_mse_mean": _mean(
            rows, "stored_vs_reproduced_q_mse"
        ),
    }
    warnings = []
    if summary["stored_vs_reproduced_q_cosine_mean"] is not None and summary[
        "stored_vs_reproduced_q_cosine_mean"
    ] < 0.999:
        warnings.append("stored_oracle_target_not_reproducible")
    if summary["restart_q_3d_cosine_mean"] is not None and summary[
        "restart_q_3d_cosine_mean"
    ] < 0.75:
        warnings.append("oracle_q_direction_restart_instability")
    if summary["restart_cue_agreement_mean"] is not None and summary[
        "restart_cue_agreement_mean"
    ] < 0.75:
        warnings.append("oracle_q_cue_restart_instability")
    if summary["near_equal_divergent_environment_ratio"] > 0.2:
        warnings.append("near_equal_utility_has_divergent_q")
    summary["warnings"] = warnings

    result = {
        "config": args.config,
        "data_dir": str(data_root),
        "dataset_seed": int(metadata["seed"]),
        "summary": summary,
        "environments": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved restart stability report to {output_path}")
    print("\n=== Oracle Restart Stability Summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
