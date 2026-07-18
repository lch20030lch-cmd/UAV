import unittest

import torch

from src.model.losses import UAVISACLosses
from src.model.projection_head import PowerProjection


class PowerProjectionTest(unittest.TestCase):
    def test_default_projection_is_simplex_without_unconditional_floor(self):
        projection = PowerProjection(p_max=1.0, tau=1.0)
        logits = torch.tensor([[[20.0, -20.0, 0.0]]])

        power = projection(logits)

        self.assertTrue(torch.all(power >= 0))
        torch.testing.assert_close(power.sum(dim=-1), torch.ones(1, 1))
        self.assertLess(float(power[0, 0, 1]), 1e-8)

    def test_optional_floor_is_association_aware_and_budget_safe(self):
        projection = PowerProjection(p_max=1.0, tau=1.0, p_min_ratio=0.1)
        logits = torch.tensor([[[-20.0, -20.0, 0.0]]])
        association = torch.tensor([[[1.0, 0.0]]])

        power = projection(logits, association=association)

        self.assertGreaterEqual(float(power[0, 0, 0]), 0.1 - 1e-6)
        self.assertLess(float(power[0, 0, 1]), 1e-8)
        torch.testing.assert_close(power.sum(dim=-1), torch.ones(1, 1))

    def test_optional_floor_requires_association(self):
        projection = PowerProjection(p_max=1.0, tau=1.0, p_min_ratio=0.1)
        with self.assertRaisesRegex(ValueError, "requires association"):
            projection(torch.zeros(1, 1, 3))


class PowerLossTest(unittest.TestCase):
    def setUp(self):
        self.losses = UAVISACLosses(lambda_p=1.0)
        self.association = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.float32,
        )
        self.target = torch.tensor(
            [[[0.4, 0.0, 0.6], [0.0, 0.7, 0.3]]],
            dtype=torch.float32,
        )

    def test_exact_power_target_has_zero_grouped_loss(self):
        loss, parts = self.losses.compute_power_loss(
            self.target, self.target, self.association
        )

        torch.testing.assert_close(loss, torch.tensor(0.0))
        for value in parts.values():
            torch.testing.assert_close(value, torch.tensor(0.0))

    def test_inactive_entries_do_not_dominate_active_and_sensing_groups(self):
        prediction = self.target.clone().requires_grad_(True)
        with torch.no_grad():
            prediction[0, 0, 1] = 0.3
            prediction[0, 1, 0] = 0.3

        loss, parts = self.losses.compute_power_loss(
            prediction, self.target, self.association
        )

        torch.testing.assert_close(parts["loss_p_active"], torch.tensor(0.0))
        torch.testing.assert_close(parts["loss_p_sensing"], torch.tensor(0.0))
        torch.testing.assert_close(parts["loss_p_inactive"], torch.tensor(0.09))
        torch.testing.assert_close(loss, torch.tensor(0.03))
        loss.backward()
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)

    def test_raw_kl_keeps_gradient_when_softmax_is_wrong_and_saturated(self):
        raw_logits = torch.tensor(
            [[[20.0, -20.0, -20.0], [20.0, -20.0, -20.0]]],
            requires_grad=True,
        )

        loss = self.losses.compute_power_raw_kl_loss(raw_logits, self.target)

        self.assertGreater(float(loss), 1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(raw_logits.grad).all())
        self.assertGreater(float(raw_logits.grad.abs().sum()), 0.1)


if __name__ == "__main__":
    unittest.main()
