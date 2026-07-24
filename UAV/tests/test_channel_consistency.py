import unittest
from types import SimpleNamespace

import numpy as np

from src.data.oracle_runtime import (
    build_oracle_solver,
    environment_sample_to_solver_dict,
)
from src.env.uav_channel import ISACChannel


class SensingChannelConsistencyTest(unittest.TestCase):
    def test_sample_to_solver_environment_has_one_canonical_precision(self):
        precise_weights = np.array(
            [1.123456789123, 0.876543210987],
            dtype=np.float64,
        )
        sample = SimpleNamespace(
            q_current=np.zeros((1, 3), dtype=np.float64),
            u_positions=np.zeros((2, 2), dtype=np.float64),
            s_positions=np.zeros((1, 2), dtype=np.float64),
            target_detected=np.array([True]),
            channel_gains_users=np.ones((1, 2), dtype=np.float64),
            user_weights=precise_weights,
            association=np.ones((1, 2), dtype=np.float64),
        )

        environment = environment_sample_to_solver_dict(sample)

        for key in (
            "q_current",
            "user_positions",
            "target_positions",
            "channel_gains",
            "user_weights",
            "association",
        ):
            self.assertEqual(environment[key].dtype, np.float32)
        np.testing.assert_array_equal(
            environment["user_weights"],
            precise_weights.astype(np.float32),
        )
        self.assertEqual(environment["target_detected"].dtype, np.bool_)

    def test_runtime_builder_normalizes_yaml_numeric_strings(self):
        solver = build_oracle_solver({
            "area_size": [1000, 1000],
            "num_uavs": 4,
            "num_users": 20,
            "num_targets": 6,
            "target_detection_probability": 0.8,
            "num_antennas_tx": 8,
            "num_antennas_rx": 8,
            "carrier_freq_ghz": 5.8,
            "bandwidth_mhz": 20,
            "p_max_dbm": 30,
            "noise_figure_db": 9,
            "altitude_min_m": 50,
            "altitude_max_m": 300,
            "uav_min_separation_m": 10,
            "uav_max_speed_ms": 15,
            "slot_duration_s": 1,
            "sinr_c_min_db": 0,
            "sinr_s_min_db": 10,
            "rate_min_bps": "1e6",
            "load_cap_per_uav": 10,
        })

        self.assertEqual(solver.cfg.rate_min_bps, 1_000_000.0)
        self.assertIsInstance(solver.cfg.rate_min_bps, float)

    def test_sensing_sinr_uses_configured_noise_floor_exactly(self):
        channel = ISACChannel(
            bandwidth_mhz=20.0,
            num_antennas_tx=8,
            num_antennas_rx=8,
            noise_figure_db=9.0,
        )
        uav = np.array([100.0, 200.0, 120.0])
        target = np.array([300.0, 400.0])
        sensing_power = 0.3

        actual = channel.compute_sensing_sinr(
            uav,
            target,
            sensing_power,
            np.eye(channel.N_t),
        )
        expected = (
            sensing_power
            * channel.sensing_path_gain(uav, target)
            * channel.N_t
            * channel.N_r
            / channel.noise_power
        )

        self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
