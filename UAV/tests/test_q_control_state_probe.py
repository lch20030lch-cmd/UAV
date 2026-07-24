import unittest

import torch

from scripts.probe_q_control_states import (
    FlattenedLinearQReadout,
    FlattenedMLPQReadout,
    MeanLinearQReadout,
    OnlineEquivalentQReadout,
    _classify_bottleneck,
    _direction_loss,
    _direction_metrics,
)


class QControlStateProbeTest(unittest.TestCase):
    def test_probe_readouts_return_q_vectors(self):
        states = torch.randn(3, 8, 16)
        for model in (
            OnlineEquivalentQReadout(8, 16, 4),
            MeanLinearQReadout(16, 4),
            FlattenedLinearQReadout(8, 16, 4),
            FlattenedMLPQReadout(8, 16, 4, 12),
        ):
            self.assertEqual(tuple(model(states).shape), (3, 4, 3))

    def test_exact_scaled_directions_report_perfect_alignment(self):
        target = torch.randn(5, 4, 3)
        prediction = target * 7.0

        metrics = _direction_metrics(prediction, target)

        self.assertAlmostEqual(metrics["direction_mse"], 0.0, places=6)
        self.assertAlmostEqual(metrics["cosine_mean"], 1.0, places=6)
        self.assertAlmostEqual(
            float(_direction_loss(prediction, target)),
            0.0,
            places=6,
        )

    def test_zero_target_directions_are_ignored(self):
        target = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        prediction = torch.tensor([[[2.0, 0.0, 0.0], [9.0, 9.0, 9.0]]])

        metrics = _direction_metrics(prediction, target)

        self.assertEqual(metrics["valid_direction_count"], 1)
        self.assertAlmostEqual(metrics["cosine_mean"], 1.0)

    def test_conclusion_prioritizes_online_fit(self):
        probes = {
            "online_equivalent": {"train": {"cosine_mean": 0.99}},
            "flat_linear": {"train": {"cosine_mean": 1.0}},
        }

        conclusion = _classify_bottleneck(probes, 0.95)

        self.assertIn("ONLINE OPTIMIZATION BOTTLENECK", conclusion)

    def test_conclusion_identifies_pooling_bottleneck(self):
        probes = {
            "online_equivalent": {"train": {"cosine_mean": 0.5}},
            "mean_linear": {"train": {"cosine_mean": 0.6}},
            "flat_linear": {"train": {"cosine_mean": 0.99}},
            "flat_mlp": {"train": {"cosine_mean": 1.0}},
        }

        conclusion = _classify_bottleneck(probes, 0.95)

        self.assertIn("CONTROL-TOKEN POOLING BOTTLENECK", conclusion)


if __name__ == "__main__":
    unittest.main()
