import unittest

import torch

from scripts.probe_association_control_states import (
    FlattenedLinearAssociationReadout,
    OnlineEquivalentAssociationReadout,
    _association_metrics,
    _normalize_cached_states,
)


class AssociationControlStateProbeTest(unittest.TestCase):
    def test_probe_readouts_return_association_logits(self):
        states = torch.randn(3, 8, 16)
        for model in (
            OnlineEquivalentAssociationReadout(8, 16, 4, 5),
            FlattenedLinearAssociationReadout(8, 16, 4, 5),
        ):
            self.assertEqual(tuple(model(states).shape), (3, 4, 5))

    def test_exact_logits_report_perfect_ranking(self):
        target_idx = torch.tensor([[0, 1], [1, 0]])
        targets = torch.nn.functional.one_hot(target_idx, num_classes=2).permute(0, 2, 1).float()
        logits = targets * 8.0 - 4.0

        metrics = _association_metrics(logits, targets)

        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["top2_accuracy"], 1.0)
        self.assertGreater(metrics["oracle_probability_mean"], 0.99)
        self.assertEqual(metrics["pred_hist"], {"0": 2, "1": 2})

    def test_hidden_feature_normalization_uses_training_statistics(self):
        train = torch.randn(7, 3, 5) * 4.0 + 9.0
        validation = torch.full((2, 3, 5), 9.0)

        normalized_train, normalized_validation, summary = _normalize_cached_states(
            train,
            validation,
            "hidden_feature",
        )

        self.assertEqual(summary["mode"], "hidden_feature")
        self.assertTrue(
            torch.allclose(
                normalized_train.mean(dim=(0, 1)),
                torch.zeros(5),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                normalized_train.std(dim=(0, 1), unbiased=False),
                torch.ones(5),
                atol=1e-5,
            )
        )
        self.assertEqual(tuple(normalized_validation.shape), (2, 3, 5))


if __name__ == "__main__":
    unittest.main()
