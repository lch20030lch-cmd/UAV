import unittest

import torch
import torch.nn as nn

from src.training.train_sft_mm import (
    _backward_accumulated_loss,
    _clip_projection_and_lora_gradients,
    _freeze_projection_except,
    _is_accumulation_boundary,
    _resolve_gradient_accumulation_steps,
)


class _DummyProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout_q = nn.Linear(2, 2)
        self.q_mlp = nn.Linear(2, 2)
        self.readout_q_cue = nn.Linear(2, 2)
        self.q_residual_adapter = nn.Linear(2, 2)
        self.readout_a = nn.Linear(2, 2)
        self.a_mlp = nn.Linear(2, 2)
        self.readout_p = nn.Linear(2, 2)
        self.p_mlp = nn.Linear(2, 2)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection_head = _DummyProjectionHead()


class ProjectionBranchFreezeTest(unittest.TestCase):
    def test_qp_freeze_includes_fixed_geometry_residual_adapter(self):
        model = _DummyModel()

        frozen, trainable = _freeze_projection_except(
            model,
            trainable_prefixes=("readout_a", "a_mlp"),
        )

        actual_trainable = {
            name
            for name, parameter in model.projection_head.named_parameters()
            if parameter.requires_grad
        }
        expected_trainable = {
            name
            for name, _ in model.projection_head.named_parameters()
            if name.startswith(("readout_a", "a_mlp"))
        }
        self.assertEqual(actual_trainable, expected_trainable)
        self.assertEqual(set(trainable), expected_trainable)
        self.assertTrue(set(frozen).isdisjoint(expected_trainable))

    def test_gradient_accumulation_uses_config_or_explicit_override(self):
        config = {"gradient_accumulation_steps": 8}

        self.assertEqual(_resolve_gradient_accumulation_steps(config), 8)
        self.assertEqual(_resolve_gradient_accumulation_steps(config, 20), 20)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _resolve_gradient_accumulation_steps(config, 0)

    def test_optimizer_boundary_occurs_only_after_full_accumulation_window(self):
        boundaries = [
            step
            for step in range(1, 17)
            if _is_accumulation_boundary(step, 8)
        ]
        self.assertEqual(boundaries, [8, 16])

    def test_accumulated_gradient_matches_full_batch_mean(self):
        accumulated = nn.Linear(1, 1, bias=False)
        full_batch = nn.Linear(1, 1, bias=False)
        full_batch.load_state_dict(accumulated.state_dict())
        inputs = torch.tensor([[1.0], [3.0]])
        targets = torch.tensor([[0.5], [-0.5]])

        for index in range(2):
            prediction = accumulated(inputs[index:index + 1])
            loss = (prediction - targets[index:index + 1]).square().mean()
            _backward_accumulated_loss(loss, accumulation_steps=2)

        full_loss = (full_batch(inputs) - targets).square().mean()
        full_loss.backward()

        self.assertTrue(
            torch.allclose(accumulated.weight.grad, full_batch.weight.grad)
        )

    def test_projection_and_lora_gradients_are_clipped_independently(self):
        projection = nn.Parameter(torch.zeros(2))
        lora = nn.Parameter(torch.zeros(2))
        projection.grad = torch.tensor([0.3, 0.4])
        lora.grad = torch.tensor([6.0, 8.0])

        projection_post, lora_post = _clip_projection_and_lora_gradients(
            [projection],
            [lora],
            max_norm=1.0,
        )

        self.assertAlmostEqual(projection_post, 0.5, places=5)
        self.assertAlmostEqual(lora_post, 1.0, places=5)
        self.assertTrue(torch.allclose(projection.grad, torch.tensor([0.3, 0.4])))

    def test_independent_gradient_clipping_rejects_nonpositive_limit(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            _clip_projection_and_lora_gradients([], [], max_norm=0.0)

    def test_direct_q_isolation_only_keeps_q_readout_and_mlp_trainable(self):
        model = _DummyModel()

        frozen, trainable = _freeze_projection_except(
            model,
            trainable_prefixes=("readout_q", "q_mlp", "q_residual_adapter"),
        )

        expected_trainable = {
            name
            for name, _ in model.projection_head.named_parameters()
            if name.startswith(("readout_q", "q_mlp"))
            and not name.startswith("readout_q_cue")
        }
        expected_trainable.update(
            name
            for name, _ in model.projection_head.named_parameters()
            if name.startswith("q_residual_adapter")
        )
        actual_trainable = {
            name
            for name, parameter in model.projection_head.named_parameters()
            if parameter.requires_grad
        }
        self.assertEqual(actual_trainable, expected_trainable)
        self.assertEqual(set(trainable), expected_trainable)
        self.assertTrue(set(frozen).isdisjoint(expected_trainable))


if __name__ == "__main__":
    unittest.main()
