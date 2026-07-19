import unittest

import torch

from src.model.losses import UAVISACLosses
from src.model.projection_head import ConstraintProjectionHead


def _make_head(mode="fixed_residual_xy", weights=None):
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
        q_residual_max_scale=1.0,
        q_residual_gate_init=0.05,
    )


class FixedResidualQGeometryTest(unittest.TestCase):
    def test_zero_residual_starts_from_fixed_geometry_direction(self):
        head = _make_head(weights=[1.0, 0.0, 0.0])
        raw = torch.zeros(1, 1, 3)
        cues = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]])

        composed = head._compose_q_from_geometry_cues(raw, cues, None)

        torch.testing.assert_close(composed, torch.tensor([[[15.0, 0.0, 0.0]]]))
        self.assertAlmostEqual(
            float(torch.sigmoid(head.q_residual_gate_logit).detach()),
            0.05,
            places=6,
        )

    def test_projected_direction_loss_reaches_residual_and_gate(self):
        head = _make_head(weights=[1.0, 0.0, 0.0])
        raw = torch.tensor([[[0.0, 1.0, 0.0]]], requires_grad=True)
        cues = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]])
        target = torch.tensor([[[0.0, 15.0, 0.0]]])

        composed = head._compose_q_from_geometry_cues(raw, cues, None)
        loss = UAVISACLosses().compute_q_direction_loss(composed, target)
        loss.backward()

        self.assertIsNotNone(raw.grad)
        self.assertGreater(float(raw.grad.abs().sum()), 0.0)
        self.assertIsNotNone(head.q_residual_gate_logit.grad)
        self.assertGreater(float(head.q_residual_gate_logit.grad.abs()), 0.0)

    def test_geometry_mode_requires_cues(self):
        head = _make_head()
        with self.assertRaisesRegex(ValueError, "q_geometry_cues are required"):
            head._compose_q_from_geometry_cues(torch.zeros(1, 1, 3), None, None)

    def test_old_mode_does_not_add_checkpoint_parameters(self):
        head = _make_head(mode="none")
        self.assertNotIn("q_residual_gate_logit", head.state_dict())
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
