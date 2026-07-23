"""Shared construction of the versioned Oracle simulation and optimizer.

Every data-generation and evaluation entry point must use these builders so
that the dataset fingerprint describes the physics that actually run.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from src.env import ISACScenarioGenerator
from src.solver.sca_fp import SCAFPConfig, SCAFPOptimizer


def build_oracle_scenario(
    simulation: Mapping,
    *,
    seed: int,
) -> ISACScenarioGenerator:
    return ISACScenarioGenerator(
        num_uavs=simulation["num_uavs"],
        num_users=simulation["num_users"],
        num_targets=simulation["num_targets"],
        area_size=tuple(simulation["area_size"]),
        altitude_range=(
            simulation["altitude_min_m"],
            simulation["altitude_max_m"],
        ),
        rate_requirement_bps=simulation["rate_min_bps"],
        target_detection_probability=simulation[
            "target_detection_probability"
        ],
        carrier_freq_ghz=simulation["carrier_freq_ghz"],
        bandwidth_mhz=simulation["bandwidth_mhz"],
        num_antennas=simulation["num_antennas_tx"],
        num_antennas_rx=simulation.get(
            "num_antennas_rx", simulation["num_antennas_tx"]
        ),
        p_max_dbm=simulation["p_max_dbm"],
        noise_figure_db=simulation["noise_figure_db"],
        seed=int(seed),
    )


def build_oracle_solver(simulation: Mapping) -> SCAFPOptimizer:
    solver_config = SCAFPConfig(
        max_outer_iters=30,
        max_inner_iters=5,
        tol=1e-4,
        lambda_sensing=0.5,
        lambda_idle_penalty=0.0,
        sinr_c_min=10 ** (simulation["sinr_c_min_db"] / 10),
        sinr_s_min=10 ** (simulation["sinr_s_min_db"] / 10),
        rate_min_bps=simulation["rate_min_bps"],
        min_separation_m=simulation.get(
            "uav_min_separation_m", 10.0
        ),
        verbose=False,
    )
    noise_power = 10 ** (
        (
            -174
            + 10 * np.log10(simulation["bandwidth_mhz"] * 1e6)
            + simulation["noise_figure_db"]
            - 30
        )
        / 10
    )
    return SCAFPOptimizer(
        config=solver_config,
        M=simulation["num_uavs"],
        K=simulation["num_users"],
        T=simulation["num_targets"],
        N_t=simulation["num_antennas_tx"],
        N_r=simulation.get(
            "num_antennas_rx", simulation["num_antennas_tx"]
        ),
        carrier_freq_ghz=simulation["carrier_freq_ghz"],
        bandwidth_mhz=simulation["bandwidth_mhz"],
        noise_figure_db=simulation["noise_figure_db"],
        area_size=tuple(simulation["area_size"]),
        altitude_range=(
            simulation["altitude_min_m"],
            simulation["altitude_max_m"],
        ),
        p_max=10 ** ((simulation["p_max_dbm"] - 30) / 10),
        noise_power=noise_power,
        load_cap=simulation["load_cap_per_uav"],
        v_max=simulation.get("uav_max_speed_ms", 15),
        slot_duration=simulation.get("slot_duration_s", 1.0),
    )
