import unittest

import numpy as np

from src.data.oracle_runtime import build_oracle_solver
from src.env.uav_channel import ISACChannel


class SensingChannelConsistencyTest(unittest.TestCase):
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
