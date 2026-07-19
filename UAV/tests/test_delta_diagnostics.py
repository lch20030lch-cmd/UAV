import unittest

import numpy as np

from scripts.analyze_mm_delta_outputs import (
    _summarize_association_alignment,
    _summarize_fixed_q_geometry,
    _summarize_q_alignment,
)


class AssociationAlignmentDiagnosticTest(unittest.TestCase):
    def test_reports_accuracy_and_fixed_user_majority_baseline(self):
        target_idx = np.array(
            [
                [0, 1],
                [1, 0],
                [0, 1],
            ],
            dtype=np.int64,
        )
        target = np.eye(2, dtype=np.float32)[target_idx].transpose(0, 2, 1)
        prediction = target.copy()

        summary = _summarize_association_alignment(prediction, target)

        self.assertAlmostEqual(summary["delta_a_argmax_accuracy"], 1.0)
        self.assertAlmostEqual(
            summary["delta_a_fixed_user_majority_accuracy"],
            2.0 / 3.0,
        )
        self.assertAlmostEqual(
            summary["delta_a_accuracy_gain_over_fixed_user_majority"],
            1.0 / 3.0,
        )
        self.assertAlmostEqual(summary["delta_a_oracle_probability_mean"], 1.0)

    def test_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            _summarize_association_alignment(
                np.zeros((2, 4, 3), dtype=np.float32),
                np.zeros((2, 4, 2), dtype=np.float32),
            )


class QAlignmentDiagnosticTest(unittest.TestCase):
    def test_exact_boundary_direction_reports_perfect_alignment(self):
        prediction = np.array(
            [
                [[15.0, 0.0, 0.0], [0.0, 12.0, 9.0]],
                [[-15.0, 0.0, 0.0], [0.0, -12.0, -9.0]],
            ],
            dtype=np.float32,
        )

        summary = _summarize_q_alignment(prediction, prediction, q_max_norm=15.0)

        self.assertAlmostEqual(summary["delta_q_norm_mean"], 15.0)
        self.assertAlmostEqual(summary["delta_q_norm_mae"], 0.0)
        self.assertAlmostEqual(summary["delta_q_vs_target_3d_cosine_mean"], 1.0)
        self.assertAlmostEqual(summary["delta_q_near_max_radius_ratio"], 1.0)
        self.assertAlmostEqual(summary["delta_q_mobility_violation_ratio"], 0.0)

    def test_rejects_q_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            _summarize_q_alignment(
                np.zeros((2, 4, 3), dtype=np.float32),
                np.zeros((1, 4, 3), dtype=np.float32),
            )

    def test_fixed_geometry_summary_uses_configured_mixture(self):
        cues = np.array([[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]], dtype=np.float32)
        target = np.array([[[0.0, 15.0, 0.0]]], dtype=np.float32)

        summary = _summarize_fixed_q_geometry(cues, target, [0.0, 1.0, 0.0])

        self.assertAlmostEqual(
            summary["q_fixed_geometry_vs_target_xy_cosine_mean"],
            1.0,
        )
        self.assertAlmostEqual(
            summary["q_fixed_geometry_vs_target_3d_cosine_mean"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
