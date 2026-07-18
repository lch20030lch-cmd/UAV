import unittest

import numpy as np

from scripts.analyze_mm_delta_outputs import _summarize_association_alignment


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


if __name__ == "__main__":
    unittest.main()
