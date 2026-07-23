import unittest

import torch

from src.model.losses import UAVISACLosses
from src.model.projection_head import ConstraintProjectionHead


def _make_head(mode="fixed_residual_xy", weights=(1.0, 0.0, 0.0)):
    return ConstraintProjectionHead(
        hidden_dim=4,
        num_control_tokens=2,
        mlp_hidden=[4],
        M=1,
        K=1,
        head_type="split",
        q_projection_mode="direction",
        q_geometry_mode=mode,
        q_fixed_cue_weights=weights,
        q_residual_max_scale=0.5,
    )


class FixedResidualQGeometryTest(unittest.TestCase):
    def test_zero_residual_starts_from_fixed_geometry_direction(self):
        head = _make_head(weights=[1.0, 0.0, 0.0])
        raw = torch.tensor([[[3.0, -2.0, 1.0]]])
        cues = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]])

        composed = head._compose_q_from_geometry_cues(raw, cues, None)

        torch.testing.assert_close(composed, torch.tensor([[[15.0, 0.0, 0.0]]]))
        torch.testing.assert_close(
            head.q_residual_adapter.weight,
            torch.zeros_like(head.q_residual_adapter.weight),
        )

    def test_projected_direction_loss_reaches_zero_initialized_adapter(self):
        head = _make_head(weights=[1.0, 0.0, 0.0])
        raw = torch.tensor([[[0.0, 1.0, 0.0]]], requires_grad=True)
        cues = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]])
        target = torch.tensor([[[0.0, 15.0, 0.0]]])

        composed = head._compose_q_from_geometry_cues(raw, cues, None)
        loss = UAVISACLosses().compute_q_direction_loss(composed, target)
        loss.backward()

        self.assertIsNotNone(head.q_residual_adapter.weight.grad)
        self.assertGreater(float(head.q_residual_adapter.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(head.q_residual_adapter.bias.grad)
        self.assertGreater(float(head.q_residual_adapter.bias.grad.abs().sum()), 0.0)

    def test_adapter_optimization_reduces_projected_direction_loss(self):
        head = _make_head(weights=[1.0, 0.0, 0.0])
        raw = torch.tensor([[[0.0, 1.0, 0.0]]])
        cues = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]])
        target = torch.tensor([[[0.0, 15.0, 0.0]]])
        optimizer = torch.optim.SGD(head.q_residual_adapter.parameters(), lr=0.5)
        losses = UAVISACLosses()

        initial = losses.compute_q_direction_loss(
            head._compose_q_from_geometry_cues(raw, cues, None),
            target,
        ).item()
        for _ in range(20):
            optimizer.zero_grad()
            loss = losses.compute_q_direction_loss(
                head._compose_q_from_geometry_cues(raw, cues, None),
                target,
            )
            loss.backward()
            optimizer.step()
        final = losses.compute_q_direction_loss(
            head._compose_q_from_geometry_cues(raw, cues, None),
            target,
        ).item()

        self.assertLess(final, initial - 0.05)

    def test_geometry_mode_requires_cues(self):
        head = _make_head()
        with self.assertRaisesRegex(ValueError, "q_geometry_cues are required"):
            head._compose_q_from_geometry_cues(torch.zeros(1, 1, 3), None, None)

    def test_fixed_geometry_requires_dataset_calibrated_weights(self):
        with self.assertRaisesRegex(ValueError, "explicit q_fixed_cue_weights"):
            _make_head(weights=None)

    def test_fixed_geometry_renormalizes_over_valid_cues(self):
        head = _make_head(weights=[0.2, 0.3, 0.5])
        raw = torch.zeros(1, 1, 3)
        cues = torch.tensor(
            [[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]]
        )
        mask = torch.tensor([[[1.0, 0.0, 0.0]]])

        composed = head._compose_q_from_geometry_cues(
            raw, cues, None, mask
        )

        torch.testing.assert_close(
            composed, torch.tensor([[[15.0, 0.0, 0.0]]])
        )

    def test_old_mode_does_not_add_checkpoint_parameters(self):
        head = _make_head(mode="none")
        self.assertNotIn("q_residual_adapter.weight", head.state_dict())
        self.assertNotIn("q_fixed_cue_weights", head.state_dict())


class ProjectedQDirectionLossTest(unittest.TestCase):
    def test_exact_projected_direction_has_zero_loss(self):
        losses = UAVISACLosses(lambda_q_projected_dir=1.0)
        delta_q = torch.tensor([[[15.0, 0.0, 0.0]]])
        delta_a = torch.tensor([[[1.0]]])
        delta_p = torch.tensor([[[0.5, 0.5]]])

        total, parts = losses.compute_control_loss(
            {"delta_q": delta_q, "delta_a": delta_a, "delta_p": delta_p},
            {"delta_q": delta_q, "delta_a": delta_a, "delta_p": delta_p},
            return_components=True,
        )

        self.assertAlmostEqual(float(parts["loss_q_projected_dir"]), 0.0)
        self.assertAlmostEqual(float(total), 0.0)


if __name__ == "__main__":
    unittest.main()
