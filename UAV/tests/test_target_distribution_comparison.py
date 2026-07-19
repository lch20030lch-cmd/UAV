import unittest

import numpy as np

from scripts.analyze_mm_target_distribution import _compare_target_sets


def _record(sample_id, delta_q, delta_a, delta_p, prompt_type="v4"):
    return {
        "id": sample_id,
        "prompt_type": prompt_type,
        "delta_q": delta_q,
        "delta_a": delta_a,
        "delta_p": delta_p,
    }


class TargetDistributionComparisonTest(unittest.TestCase):
    def setUp(self):
        self.first = _record(
            "env_0",
            [[1.0, 0.0, 0.0]],
            [[1.0], [0.0]],
            [[0.5, 0.5]],
        )
        self.second = _record(
            "env_1",
            [[0.0, 1.0, 0.0]],
            [[0.0], [1.0]],
            [[0.25, 0.75]],
        )

    def test_identical_targets_match_after_id_alignment(self):
        summary = _compare_target_sets(
            [self.first, self.second],
            [self.second, self.first],
        )

        self.assertEqual(summary["reference_common_samples"], 2)
        self.assertAlmostEqual(summary["reference_delta_q_3d_cosine_mean"], 1.0)
        self.assertAlmostEqual(summary["reference_delta_q_xy_cosine_mean"], 1.0)
        self.assertAlmostEqual(summary["reference_delta_a_argmax_match_rate"], 1.0)
        self.assertAlmostEqual(summary["reference_delta_p_mse"], 0.0)

    def test_reports_association_switch_and_q_direction_change(self):
        changed = _record(
            "env_0",
            [[0.0, 1.0, 0.0]],
            [[0.0], [1.0]],
            [[0.0, 1.0]],
        )

        summary = _compare_target_sets([changed], [self.first])

        self.assertAlmostEqual(summary["reference_delta_q_3d_cosine_mean"], 0.0)
        self.assertAlmostEqual(summary["reference_delta_q_xy_cosine_mean"], 0.0)
        self.assertAlmostEqual(summary["reference_delta_a_argmax_match_rate"], 0.0)
        self.assertAlmostEqual(summary["reference_delta_a_argmax_switch_rate"], 1.0)
        self.assertGreater(summary["reference_delta_p_mse"], 0.0)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate current id"):
            _compare_target_sets([self.first, self.first], [self.first])


if __name__ == "__main__":
    unittest.main()
