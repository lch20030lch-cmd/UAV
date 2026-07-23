#!/usr/bin/env python
"""Audit whether deterministic Oracle restarts provide stable Q/A/P targets.

The current dataset selects a real near-optimal Q-direction medoid instead of
blindly taking the numerically highest-utility restart.  This script
reconstructs the same environments and reports both all-restart diversity and
the label stability inside the near-optimal utility set.
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

from src.data.geometry_cues import CUE_NAMES, parse_q_geometry_cues
from src.data.multimodal_dataset import validate_multimodal_oracle_contract
from src.data.oracle_generator import (
    OracleDataGenerator,
    select_near_optimal_q_medoid,
)
from src.data.oracle_runtime import (
    build_oracle_scenario,
    build_oracle_solver,
)


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
    delta_a: np.ndarray,
    delta_p: np.ndarray,
    cues: np.ndarray,
    cue_mask: np.ndarray,
    utility_tolerance: float = 0.01,
) -> Dict:
    """Summarize one environment after sorting restarts by utility."""
    utilities = np.asarray(utilities, dtype=np.float64)
    delta_q = np.asarray(delta_q, dtype=np.float64)
    if (
        utilities.ndim != 1
        or delta_q.ndim != 3
        or delta_a.ndim != 3
        or delta_p.ndim != 3
    ):
        raise ValueError("restart utility/Q/A/P arrays have invalid ranks")
    restart_count = utilities.shape[0]
    if not (
        restart_count
        == delta_q.shape[0]
        == delta_a.shape[0]
        == delta_p.shape[0]
    ):
        raise ValueError("utility/Q/A/P restart counts differ")
    if utilities.shape[0] < 2:
        return {
            "num_feasible_restarts": int(utilities.shape[0]),
            "top_second_relative_utility_gap": None,
            "top_second_q_3d_cosine": None,
            "restart_q_3d_cosine_mean": None,
            "restart_q_xy_cosine_mean": None,
            "restart_cue_agreement_mean": None,
            "restart_a_user_agreement_mean": None,
            "restart_p_mse_mean": None,
            "near_optimal_q_3d_cosine_mean": None,
            "near_optimal_q_xy_cosine_mean": None,
            "near_optimal_cue_agreement_mean": None,
            "near_optimal_a_user_agreement_mean": None,
            "near_optimal_p_mse_mean": None,
            "near_equal_divergent": False,
        }

    order = np.argsort(-utilities)
    utilities = utilities[order]
    delta_q = delta_q[order]
    delta_a = delta_a[order]
    delta_p = delta_p[order]
    directions_3d = _unit(delta_q)
    directions_xy = _unit(delta_q[..., :2])
    labels = np.stack(
        [_cue_labels(cues, cue_mask, restart_delta) for restart_delta in delta_q]
    )

    pair_q_3d: List[float] = []
    pair_q_xy: List[float] = []
    pair_cue_agreement: List[float] = []
    pair_a_agreement: List[float] = []
    pair_p_mse: List[float] = []
    near_pair_q_3d: List[float] = []
    near_pair_q_xy: List[float] = []
    near_pair_cue_agreement: List[float] = []
    near_pair_a_agreement: List[float] = []
    near_pair_p_mse: List[float] = []
    utility_tolerance = float(utility_tolerance)
    if not 0.0 <= utility_tolerance < 1.0:
        raise ValueError("utility_tolerance must be in [0, 1)")
    near_threshold = float(utilities[0]) - utility_tolerance * max(
        1.0, abs(float(utilities[0]))
    )
    near_optimal = utilities >= near_threshold
    for first, second in itertools.combinations(range(delta_q.shape[0]), 2):
        q_3d = float(
            np.sum(
                directions_3d[first] * directions_3d[second], axis=-1
            ).mean()
        )
        q_xy = float(
            np.sum(
                directions_xy[first] * directions_xy[second], axis=-1
            ).mean()
        )
        cue_agreement = float(
            (labels[first] == labels[second]).mean()
        )
        a_agreement = float(
            (
                delta_a[first].argmax(axis=0)
                == delta_a[second].argmax(axis=0)
            ).mean()
        )
        p_mse = float(
            np.mean((delta_p[first] - delta_p[second]) ** 2)
        )
        pair_q_3d.append(q_3d)
        pair_q_xy.append(q_xy)
        pair_cue_agreement.append(cue_agreement)
        pair_a_agreement.append(a_agreement)
        pair_p_mse.append(p_mse)
        if near_optimal[first] and near_optimal[second]:
            near_pair_q_3d.append(q_3d)
            near_pair_q_xy.append(q_xy)
            near_pair_cue_agreement.append(cue_agreement)
            near_pair_a_agreement.append(a_agreement)
            near_pair_p_mse.append(p_mse)

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
        "restart_a_user_agreement_mean": float(np.mean(pair_a_agreement)),
        "restart_p_mse_mean": float(np.mean(pair_p_mse)),
        "near_optimal_q_3d_cosine_mean": (
            float(np.mean(near_pair_q_3d)) if near_pair_q_3d else None
        ),
        "near_optimal_q_xy_cosine_mean": (
            float(np.mean(near_pair_q_xy)) if near_pair_q_xy else None
        ),
        "near_optimal_cue_agreement_mean": (
            float(np.mean(near_pair_cue_agreement))
            if near_pair_cue_agreement
            else None
        ),
        "near_optimal_a_user_agreement_mean": (
            float(np.mean(near_pair_a_agreement))
            if near_pair_a_agreement
            else None
        ),
        "near_optimal_p_mse_mean": (
            float(np.mean(near_pair_p_mse))
            if near_pair_p_mse
            else None
        ),
        "near_equal_divergent": bool(
            relative_gap < utility_tolerance
            and top_second_q_3d < 0.5
        ),
    }


def _mean(rows: List[Dict], key: str):
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _build_generator(cfg: Dict, seed: int) -> OracleDataGenerator:
    sim = cfg["simulation"]
    scenario = build_oracle_scenario(sim, seed=seed)
    return OracleDataGenerator(
        scenario_gen=scenario,
        solver=build_oracle_solver(sim),
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
        restart_priors = [
            generator._extract_prior(solution, environment)
            for solution in feasible_solutions
        ]
        restart_delta_q = (
            np.stack([prior[0] for prior in restart_priors])
            if restart_priors
            else np.empty((0, cfg["simulation"]["num_uavs"], 3))
        )
        restart_delta_a = (
            np.stack([prior[1] for prior in restart_priors])
            if restart_priors
            else np.empty(
                (
                    0,
                    cfg["simulation"]["num_uavs"],
                    cfg["simulation"]["num_users"],
                )
            )
        )
        restart_delta_p = (
            np.stack([prior[2] for prior in restart_priors])
            if restart_priors
            else np.empty(
                (
                    0,
                    cfg["simulation"]["num_uavs"],
                    cfg["simulation"]["num_users"] + 1,
                )
            )
        )
        restart_utilities = np.asarray(
            [solution.utility for solution in feasible_solutions], dtype=np.float64
        )
        row = {
            "environment_id": environment_id,
            **summarize_restart_set(
                restart_utilities,
                restart_delta_q,
                restart_delta_a,
                restart_delta_p,
                cues,
                cue_mask,
                generator.oracle_selection_utility_tolerance,
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
            reproduced_delta_q, reproduced_delta_a, reproduced_delta_p = generator._extract_prior(
                selected_solution, environment
            )
            row.update(selection_diagnostics)
            stored_delta_q = np.asarray(record["delta_q"], dtype=np.float64)
            row["stored_vs_reproduced_q_cosine"] = float(
                np.sum(_unit(stored_delta_q) * _unit(reproduced_delta_q), axis=-1).mean()
            )
            row["stored_vs_reproduced_q_mse"] = float(
                np.mean((stored_delta_q - reproduced_delta_q) ** 2)
            )
            stored_delta_a = np.asarray(
                record["delta_a"], dtype=np.float64
            )
            stored_delta_p = np.asarray(
                record["delta_p"], dtype=np.float64
            )
            row["stored_vs_reproduced_a_user_agreement"] = float(
                (
                    stored_delta_a.argmax(axis=0)
                    == reproduced_delta_a.argmax(axis=0)
                ).mean()
            )
            row["stored_vs_reproduced_p_mse"] = float(
                np.mean((stored_delta_p - reproduced_delta_p) ** 2)
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
        "restart_a_user_agreement_mean": _mean(
            rows, "restart_a_user_agreement_mean"
        ),
        "restart_p_mse_mean": _mean(rows, "restart_p_mse_mean"),
        "near_optimal_q_3d_cosine_mean": _mean(
            rows, "near_optimal_q_3d_cosine_mean"
        ),
        "near_optimal_q_xy_cosine_mean": _mean(
            rows, "near_optimal_q_xy_cosine_mean"
        ),
        "near_optimal_cue_agreement_mean": _mean(
            rows, "near_optimal_cue_agreement_mean"
        ),
        "near_optimal_a_user_agreement_mean": _mean(
            rows, "near_optimal_a_user_agreement_mean"
        ),
        "near_optimal_p_mse_mean": _mean(
            rows, "near_optimal_p_mse_mean"
        ),
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
        "stored_vs_reproduced_a_user_agreement_mean": _mean(
            rows, "stored_vs_reproduced_a_user_agreement"
        ),
        "stored_vs_reproduced_p_mse_mean": _mean(
            rows, "stored_vs_reproduced_p_mse"
        ),
    }
    warnings = []
    if summary["stored_vs_reproduced_q_cosine_mean"] is not None and summary[
        "stored_vs_reproduced_q_cosine_mean"
    ] < 0.999:
        warnings.append("stored_oracle_target_not_reproducible")
    if (
        summary["stored_vs_reproduced_a_user_agreement_mean"] is not None
        and summary["stored_vs_reproduced_a_user_agreement_mean"] < 1.0
    ):
        warnings.append("stored_oracle_association_not_reproducible")
    if (
        summary["stored_vs_reproduced_p_mse_mean"] is not None
        and summary["stored_vs_reproduced_p_mse_mean"] > 1e-10
    ):
        warnings.append("stored_oracle_power_not_reproducible")
    if summary["near_optimal_q_3d_cosine_mean"] is not None and summary[
        "near_optimal_q_3d_cosine_mean"
    ] < 0.75:
        warnings.append("oracle_q_direction_restart_instability")
    if summary["near_optimal_cue_agreement_mean"] is not None and summary[
        "near_optimal_cue_agreement_mean"
    ] < 0.75:
        warnings.append("oracle_q_cue_restart_instability")
    if summary["near_optimal_a_user_agreement_mean"] is not None and summary[
        "near_optimal_a_user_agreement_mean"
    ] < 0.75:
        warnings.append("oracle_a_restart_instability")
    if summary["near_optimal_p_mse_mean"] is not None and summary[
        "near_optimal_p_mse_mean"
    ] > 0.01:
        warnings.append("oracle_p_restart_instability")
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
