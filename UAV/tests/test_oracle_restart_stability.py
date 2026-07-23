import unittest
from types import SimpleNamespace

import numpy as np

from scripts.analyze_oracle_restart_stability import summarize_restart_set
from src.data.oracle_generator import select_near_optimal_q_medoid


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
            np.ones((3, 1, 2)),
            np.ones((3, 1, 3)),
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
            np.ones((2, 1, 2)),
            np.ones((2, 1, 3)),
            self.cues,
            self.mask,
        )

        self.assertAlmostEqual(result["top_second_q_3d_cosine"], 0.0)
        self.assertTrue(result["near_equal_divergent"])

    def test_far_suboptimal_outlier_does_not_poison_near_optimal_metrics(self):
        consensus = np.array(
            [[15.0, 0.0, 0.0], [15.0, 0.0, 0.0]]
        )
        outlier = np.array(
            [[0.0, 15.0, 0.0], [0.0, 15.0, 0.0]]
        )
        result = summarize_restart_set(
            np.array([10.0, 9.95, 1.0]),
            np.stack([consensus, consensus, outlier]),
            np.ones((3, 1, 2)),
            np.ones((3, 1, 3)),
            self.cues,
            self.mask,
            utility_tolerance=0.01,
        )

        self.assertLess(result["restart_q_3d_cosine_mean"], 1.0)
        self.assertAlmostEqual(
            result["near_optimal_q_3d_cosine_mean"], 1.0
        )
        self.assertAlmostEqual(
            result["near_optimal_cue_agreement_mean"], 1.0
        )


class NearOptimalOracleSelectionTest(unittest.TestCase):
    @staticmethod
    def _solution(direction, utility):
        q = np.asarray(direction, dtype=np.float64)[None, :]
        return SimpleNamespace(Q=q, utility=float(utility))

    def test_selects_real_consensus_candidate_within_utility_tolerance(self):
        highest_utility_outlier = self._solution([15.0, 0.0, 0.0], 100.0)
        consensus_first = self._solution([0.0, 15.0, 0.0], 99.8)
        consensus_medoid = self._solution([1.5, 14.9248, 0.0], 99.7)

        selected, diagnostics = select_near_optimal_q_medoid(
            [highest_utility_outlier, consensus_first, consensus_medoid],
            np.zeros((1, 3)),
            utility_tolerance=0.01,
        )

        self.assertIs(selected, consensus_medoid)
        self.assertEqual(diagnostics["oracle_chosen_candidate_rank"], 3)
        self.assertEqual(diagnostics["oracle_near_optimal_candidate_count"], 3)
        self.assertGreater(diagnostics["oracle_q_consensus_gain"], 0.0)
        self.assertLessEqual(
            diagnostics["oracle_chosen_relative_utility_gap"], 0.01
        )

    def test_excludes_candidates_outside_utility_tolerance(self):
        best = self._solution([15.0, 0.0, 0.0], 100.0)
        lower_consensus_first = self._solution([0.0, 15.0, 0.0], 98.0)
        lower_consensus_second = self._solution([1.5, 14.9248, 0.0], 97.9)

        selected, diagnostics = select_near_optimal_q_medoid(
            [best, lower_consensus_first, lower_consensus_second],
            np.zeros((1, 3)),
            utility_tolerance=0.01,
        )

        self.assertIs(selected, best)
        self.assertEqual(diagnostics["oracle_near_optimal_candidate_count"], 1)
        self.assertEqual(diagnostics["oracle_chosen_candidate_rank"], 1)


if __name__ == "__main__":
    unittest.main()
