"""
推理与评估脚本
论文 Section 6 — Evaluation Protocol

评估指标:
  1. Network sum rate (通信总速率)
  2. Mean sensing SINR
  3. Mean CRB
  4. Joint satisfaction rate
  5. SCA-FP convergence iterations
  6. Inference latency per slot
"""

import os
import sys
import yaml
import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model import Gemma3ISAC, build_proj_head_config
from src.solver import SCAFPOptimizer, SCAFPConfig
from src.env import ISACScenarioGenerator
from src.data.prompt_builder import build_full_prompt


# ==========================================================================
# Worker function (module-level — must be picklable for ProcessPoolExecutor)
# ==========================================================================

def _worker_solve_and_metric(
    sample_id: int,
    env_dict: dict,
    warm_start_dict: Optional[dict],
    solver_params: dict,
    sim_params: dict,
) -> dict:
    """CPU worker: recreate solver → SCA-FP warm + cold → compute metrics."""
    sp = solver_params

    # Recreate solver
    solver_cfg = SCAFPConfig(
        max_outer_iters=sp["max_outer_iters"],
        max_inner_iters=sp["max_inner_iters"],
        tol=sp["tol"],
        lambda_sensing=sp["lambda_sensing"],
        lambda_idle_penalty=sp["lambda_idle_penalty"],
        sinr_c_min=sp["sinr_c_min"],
        sinr_s_min=sp["sinr_s_min"],
        min_separation_m=sp["min_separation_m"],
        verbose=False,
    )

    solver = SCAFPOptimizer(
        config=solver_cfg,
        M=sp["M"],
        K=sp["K"],
        T=sp["T"],
        N_t=sp["N_t"],
        N_r=sp["N_r"],
        carrier_freq_ghz=sp["carrier_freq_ghz"],
        area_size=sp["area_size"],
        altitude_range=sp["altitude_range"],
        p_max=sp["p_max"],
        noise_power=sp["noise_power"],
        load_cap=sp["load_cap"],
        v_max=sp["v_max"],
        slot_duration=sp["slot_duration"],
    )

    # ---- SCA-FP warmstart ----
    sol_warm = solver.solve(env_dict, warm_start=warm_start_dict, seed=sample_id)

    # ---- SCA-FP cold-start baseline ----
    sol_cold = solver.solve(env_dict, warm_start=None, seed=sample_id)
    speedup = sol_cold.iterations / max(sol_warm.iterations, 1)

    # ---- Metrics: use the exact same channel/constraint model as the solver ----
    sol = sol_warm
    evaluated = solver.evaluate_solution(
        sol.Q, sol.A, sol.W_c_power, sol.W_s_power, env_dict
    )
    channel_gains = evaluated["communication_gains"]
    sensing_gains = evaluated["sensing_gains"]
    bw_hz = sim_params["bandwidth_mhz"] * 1e6

    # Sum rate
    sum_rate = 0.0
    for m in range(solver.M):
        for k in range(solver.K):
            if sol.A[m, k] > 0.5:
                sinr = channel_gains[m, k] * sol.W_c_power[m, k] / (solver.N0 + 1e-12)
                sum_rate += bw_hz * np.log2(1 + sinr)

    sensing_sinr = (
        sol.W_s_power[:, None]
        * sensing_gains
        * solver.N_t
        * solver.N_r
        / solver.N0
    )
    best_uav = np.argmax(sensing_sinr, axis=0)
    best_sensing_sinr = sensing_sinr[best_uav, np.arange(solver.T)]
    mean_sinr_db = float(np.mean(10 * np.log10(best_sensing_sinr + 1e-12)))
    crbs = [
        solver.channel.compute_crb(
            sol.Q[best_uav[t]],
            env_dict["target_positions"][t],
            best_sensing_sinr[t],
        )
        for t in range(solver.T)
    ]

    active_sinr = channel_gains * sol.W_c_power / solver.N0
    comm_sat = float(
        np.mean(active_sinr[sol.A > 0.5] >= solver.cfg.sinr_c_min)
    )
    sense_sat = float(np.mean(best_sensing_sinr >= solver.cfg.sinr_s_min))
    # Joint satisfaction is an actual all-constraints indicator for this
    # environment, not the arithmetic mean of two unrelated fractions.
    joint_sat = float(evaluated["feasible"])

    return {
        "sum_rate": float(sum_rate / 1e6),
        "mean_sensing_sinr_db": float(mean_sinr_db),
        "mean_crb": float(np.mean(crbs)),
        "communication_satisfaction": comm_sat,
        "sensing_satisfaction": sense_sat,
        "joint_satisfaction": float(joint_sat),
        "sca_fp_iterations_warm": float(sol_warm.iterations),
        "sca_fp_iterations_cold": float(sol_cold.iterations),
        "sca_fp_speedup": float(speedup),
    }


# ==========================================================================
# Main evaluation pipeline
# ==========================================================================

def run_evaluation(
    config_path: str,
    model_path: str,
    output_path: str = "./outputs/eval_results.json",
    n_workers: int = 0,
):
    """完整评估管线

    Args:
        n_workers: CPU solver 并行数 (0 = 串行, -1 = auto = cpu_count-2)
    """

    # ---- 加载配置 ----
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg["simulation"]
    eval_cfg = cfg["eval"]
    model_cfg = cfg["model"]

    if model_path and model_cfg.get("use_multimodal", False):
        raise RuntimeError(
            "src/eval/evaluate.py is the legacy text-only evaluator and cannot "
            "evaluate a multimodal checkpoint. Use scripts/evaluate_mm_solver.py."
        )

    num_test = eval_cfg.get("num_test_environments", 200)

    # Resolve n_workers
    if n_workers == -1:
        n_workers = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
    if n_workers > num_test:
        n_workers = num_test

    # ---- 初始化仿真环境 (main process only) ----
    scenario_gen = ISACScenarioGenerator(
        num_uavs=sim_cfg["num_uavs"],
        num_users=sim_cfg["num_users"],
        num_targets=sim_cfg["num_targets"],
        area_size=tuple(sim_cfg["area_size"]),
        carrier_freq_ghz=sim_cfg["carrier_freq_ghz"],
        bandwidth_mhz=sim_cfg["bandwidth_mhz"],
        num_antennas=sim_cfg["num_antennas_tx"],
        p_max_dbm=sim_cfg["p_max_dbm"],
        seed=eval_cfg.get("seed", 2026),
    )

    # ---- Solver params (pack for workers) ----
    solver_params = {
        "max_outer_iters": 30,
        "max_inner_iters": 5,
        "tol": 1e-4,
        "lambda_sensing": 0.5,
        "lambda_idle_penalty": 5.0,
        "sinr_c_min": 10 ** (sim_cfg["sinr_c_min_db"] / 10),
        "sinr_s_min": 10 ** (sim_cfg["sinr_s_min_db"] / 10),
        "min_separation_m": sim_cfg.get("uav_min_separation_m", 10.0),
        "M": sim_cfg["num_uavs"],
        "K": sim_cfg["num_users"],
        "T": sim_cfg["num_targets"],
        "N_t": sim_cfg["num_antennas_tx"],
        "N_r": sim_cfg.get("num_antennas_rx", sim_cfg["num_antennas_tx"]),
        "carrier_freq_ghz": sim_cfg["carrier_freq_ghz"],
        "area_size": tuple(sim_cfg["area_size"]),
        "altitude_range": (sim_cfg["altitude_min_m"], sim_cfg["altitude_max_m"]),
        "p_max": 10 ** ((sim_cfg["p_max_dbm"] - 30) / 10),
        "noise_power": 10 ** (
            (-174 + 10 * np.log10(sim_cfg["bandwidth_mhz"] * 1e6)
             + sim_cfg["noise_figure_db"] - 30) / 10
        ),
        "load_cap": sim_cfg["load_cap_per_uav"],
        "v_max": sim_cfg.get("uav_max_speed_ms", 15),
        "slot_duration": sim_cfg.get("slot_duration_s", 1.0),
    }

    # Sim params for metric computation
    sim_params = {
        "bandwidth_mhz": sim_cfg["bandwidth_mhz"],
        "carrier_freq_ghz": sim_cfg["carrier_freq_ghz"],
        "sinr_c_min_db": sim_cfg["sinr_c_min_db"],
        "sinr_s_min_db": sim_cfg["sinr_s_min_db"],
    }

    # ---- Load model ----
    model = None
    device = torch.device("cpu")
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = Gemma3ISAC.from_pretrained(
            load_dir=model_path,
            base_model_name=model_cfg["backbone"],
            attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
            use_4bit=cfg["hardware"]["use_4bit"],
            lora_rank=model_cfg["lora"]["rank"],
            lora_alpha=model_cfg["lora"]["alpha"],
            num_control_tokens=model_cfg["control_token"]["num_tokens"],
            proj_head_config=build_proj_head_config(model_cfg, sim_cfg),
        )
        model = model.to("cuda")
        model.eval()
        device = next(model.parameters()).device

    # ======================================================================
    # Phase 1: Model inference (GPU, serial) → collect warmstart_dicts
    # ======================================================================
    print(f"\nPhase 1: Model inference on {num_test} samples (GPU)...")
    tasks = []

    for i in tqdm(range(num_test), desc="Inference"):
        env_sample = scenario_gen.sample(i)
        env_dict = {
            "q_current": env_sample.q_current,
            "user_positions": env_sample.u_positions,
            "target_positions": env_sample.s_positions,
            "channel_gains": env_sample.channel_gains_users,
            "user_weights": env_sample.user_weights.copy(),
            "association": env_sample.association,
        }

        t0 = time.time()
        if model is not None:
            prompt = build_full_prompt(env_sample, sim_cfg)
            q_current_t = torch.tensor(
                env_sample.q_current, dtype=torch.float32, device=device
            ).unsqueeze(0)

            warm_start = model.generate_warmstart(prompt, q_current=q_current_t)
            warm_start_dict = {
                "delta_q": warm_start["delta_q"].cpu().numpy(),
                "delta_a": warm_start["delta_a"].cpu().numpy(),
                "delta_p": warm_start["delta_p"].cpu().numpy(),
            }
        else:
            warm_start_dict = None

        inference_time_ms = (time.time() - t0) * 1000
        tasks.append((i, env_dict, warm_start_dict, inference_time_ms))

    # ======================================================================
    # Phase 2: SCA-FP solving + metrics (CPU, parallel)
    # ======================================================================
    if n_workers > 1:
        print(f"\nPhase 2: SCA-FP solving with {n_workers} workers (CPU)...")
    else:
        print("\nPhase 2: constraint-aware solver (serial)...")

    results = {
        "sum_rate": [],
        "mean_sensing_sinr_db": [],
        "mean_crb": [],
        "communication_satisfaction": [],
        "sensing_satisfaction": [],
        "joint_satisfaction": [],
        "sca_fp_iterations_warm": [],
        "sca_fp_iterations_cold": [],
        "sca_fp_speedup": [],
        "inference_latency_ms": [],
    }

    def _process_one(task):
        """Process a single task (for serial mode)."""
        sample_id, env_dict, warm_start_dict, inf_time_ms = task
        try:
            metrics = _worker_solve_and_metric(
                sample_id, env_dict, warm_start_dict, solver_params, sim_params
            )
            metrics["inference_latency_ms"] = inf_time_ms
            return metrics
        except Exception as e:
            print(f"\n  Sample {sample_id} failed: {e}")
            return None

    if n_workers <= 1:
        # Serial mode
        for task in tqdm(tasks, desc="Solving"):
            m = _process_one(task)
            if m is not None:
                for k, v in m.items():
                    results[k].append(v)
    else:
        # Parallel mode
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    _worker_solve_and_metric,
                    sample_id, env_dict, warm_start_dict, solver_params, sim_params
                ): (sample_id, inf_time_ms)
                for sample_id, env_dict, warm_start_dict, inf_time_ms in tasks
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Solving"
            ):
                sample_id, inf_time_ms = futures[future]
                try:
                    metrics = future.result()
                    metrics["inference_latency_ms"] = inf_time_ms
                    for k, v in metrics.items():
                        results[k].append(v)
                except Exception as e:
                    print(f"\n  Sample {sample_id} failed: {e}")

    # ---- Summary ----
    summary = {}
    for k, vals in results.items():
        if vals:
            arr = np.array(vals)
            summary[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

    summary["num_samples"] = len(results["sum_rate"])

    # ---- Save ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Print ----
    print("\n" + "=" * 60)
    print("Evaluation Results Summary")
    print("=" * 60)
    for metric, stats in summary.items():
        if metric == "num_samples":
            print(f"\n  Total valid samples: {stats}")
        elif isinstance(stats, dict):
            print(f"\n  {metric}:")
            print(f"    mean = {stats['mean']:.4f}  ±  {stats['std']:.4f}")

    print(f"\nResults saved to {output_path}")
    return summary


if __name__ == "__main__":
    mp.freeze_support()  # Windows + PyInstaller compat
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output", type=str, default="./outputs/eval_results.json")
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU solver workers (0=serial, -1=auto, N=parallel)")
    args = parser.parse_args()

    run_evaluation(args.config, args.model, args.output, n_workers=args.workers)
