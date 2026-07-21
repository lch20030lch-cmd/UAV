import unittest

import numpy as np

from scripts.analyze_oracle_restart_stability import summarize_restart_set


class OracleRestartStabilityTest(unittest.TestCase):
    def setUp(self):
        self.cues = np.array(
            [
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        self.mask = np.ones((2, 3), dtype=np.float32)

    def test_identical_restart_directions_are_stable(self):
        delta = np.array([[15.0, 0.0, 0.0], [0.0, 15.0, 0.0]])
        result = summarize_restart_set(
            np.array([10.0, 9.0, 8.0]),
            np.stack([delta, delta, delta]),
            self.cues,
            self.mask,
        )

        self.assertAlmostEqual(result["restart_q_3d_cosine_mean"], 1.0)
        self.assertAlmostEqual(result["restart_cue_agreement_mean"], 1.0)
        self.assertFalse(result["near_equal_divergent"])

    def test_near_equal_orthogonal_solutions_are_flagged(self):
        first = np.array([[15.0, 0.0, 0.0], [15.0, 0.0, 0.0]])
        second = np.array([[0.0, 15.0, 0.0], [0.0, 15.0, 0.0]])
        result = summarize_restart_set(
            np.array([10.0, 9.95]),
            np.stack([first, second]),
            self.cues,
            self.mask,
        )

        self.assertAlmostEqual(result["top_second_q_3d_cosine"], 0.0)
        self.assertTrue(result["near_equal_divergent"])


if __name__ == "__main__":
    unittest.main()
