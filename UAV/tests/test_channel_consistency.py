import unittest

import numpy as np

from src.env.uav_channel import ISACChannel


class SensingChannelConsistencyTest(unittest.TestCase):
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
