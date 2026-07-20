import unittest

import numpy as np

from src.env.uav_channel import ISACChannel


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


if __name__ == "__main__":
    unittest.main()
