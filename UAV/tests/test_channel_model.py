import unittest

import numpy as np

from src.env.uav_channel import ISACChannel
from src.env.isac_scenario import ISACScenarioGenerator
from src.solver.sca_fp import SCAFPConfig, SCAFPOptimizer


class ChannelModelTest(unittest.TestCase):
    def test_los_probability_increases_with_elevation(self):
        channel = ISACChannel()
        low = channel.los_probability(50.0, 500.0)
        high = channel.los_probability(300.0, 100.0)

        self.assertGreater(high, low)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)

    def test_path_loss_uses_single_carrier_frequency_term(self):
        channel = ISACChannel(carrier_freq_ghz=5.8)
        los, nlos = channel.path_loss_db(100.0, 100.0, 0.0)
        expected_los = 28.0 + 22.0 * np.log10(100.0) + 20.0 * np.log10(5.8)
        expected_nlos = 32.4 + 30.0 * np.log10(100.0) + 20.0 * np.log10(5.8)

        self.assertAlmostEqual(los, expected_los)
        self.assertAlmostEqual(nlos, expected_nlos)

    def test_expected_gain_decreases_with_distance(self):
        channel = ISACChannel()
        user = np.array([0.0, 0.0])
        near = channel.expected_channel_gain(np.array([0.0, 0.0, 100.0]), user)
        far = channel.expected_channel_gain(np.array([500.0, 0.0, 100.0]), user)

        self.assertGreater(near, far)

    def test_scenario_propagates_bandwidth_rx_antennas_and_noise_figure(self):
        scenario = ISACScenarioGenerator(
            bandwidth_mhz=10.0,
            num_antennas=4,
            num_antennas_rx=6,
            noise_figure_db=7.0,
        )

        self.assertEqual(scenario.channel.B, 10e6)
        self.assertEqual(scenario.channel.N_t, 4)
        self.assertEqual(scenario.channel.N_r, 6)
        self.assertEqual(scenario.channel.NF, 7.0)

    def test_solver_channel_matches_physical_configuration(self):
        solver = SCAFPOptimizer(
            SCAFPConfig(),
            M=1,
            K=1,
            T=1,
            N_t=4,
            N_r=6,
            bandwidth_mhz=10.0,
            noise_figure_db=7.0,
            load_cap=1,
            p_max=0.5,
        )

        self.assertEqual(solver.channel.B, 10e6)
        self.assertEqual(solver.channel.N_t, 4)
        self.assertEqual(solver.channel.N_r, 6)
        self.assertEqual(solver.channel.NF, 7.0)
        self.assertAlmostEqual(solver.channel.P_max, 0.5)


if __name__ == "__main__":
    unittest.main()
