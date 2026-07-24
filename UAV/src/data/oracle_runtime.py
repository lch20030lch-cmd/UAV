"""Shared construction of the versioned Oracle simulation and optimizer.

Every data-generation and evaluation entry point must use these builders so
that the dataset fingerprint describes the physics that actually run.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from src.env import EnvironmentSample, ISACScenarioGenerator
from src.data.oracle_contract import canonical_simulation_config
from src.solver.sca_fp import SCAFPConfig, SCAFPOptimizer


def environment_sample_to_solver_dict(
    sample: EnvironmentSample,
) -> dict:
    """Build the one canonical solver environment for a sampled scenario.

    Scenario tensors are consumed by the Oracle at float32 source precision
    and promoted to float64 by ``SCAFPOptimizer._validate_environment``.
    Keeping this conversion in one place prevents generation, auditing, and
    downstream evaluation from assigning slightly different utilities to the
    same seeded environment.  In particular, ``UAVNetwork`` exposes user
    weights as float64 while the original Oracle generation path deliberately
    consumed their float32 representation.
    """

    def _float32_copy(value) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).copy()

    return {
        "q_current": _float32_copy(sample.q_current),
        "user_positions": _float32_copy(sample.u_positions),
        "target_positions": _float32_copy(sample.s_positions),
        "target_detected": np.asarray(
            sample.target_detected, dtype=bool
        ).copy(),
        "channel_gains": _float32_copy(sample.channel_gains_users),
        "user_weights": _float32_copy(sample.user_weights),
        "association": _float32_copy(sample.association),
    }


def build_oracle_scenario(
    simulation: Mapping,
    *,
    seed: int,
) -> ISACScenarioGenerator:
    simulation = canonical_simulation_config(simulation)
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
    simulation = canonical_simulation_config(simulation)
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
