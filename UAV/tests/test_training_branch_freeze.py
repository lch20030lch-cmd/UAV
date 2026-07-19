import unittest

import torch.nn as nn

from src.training.train_sft_mm import _freeze_projection_except


class _DummyProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout_q = nn.Linear(2, 2)
        self.q_mlp = nn.Linear(2, 2)
        self.readout_q_cue = nn.Linear(2, 2)
        self.readout_a = nn.Linear(2, 2)
        self.a_mlp = nn.Linear(2, 2)
        self.readout_p = nn.Linear(2, 2)
        self.p_mlp = nn.Linear(2, 2)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection_head = _DummyProjectionHead()


class ProjectionBranchFreezeTest(unittest.TestCase):
    def test_direct_q_isolation_only_keeps_q_readout_and_mlp_trainable(self):
        model = _DummyModel()

        frozen, trainable = _freeze_projection_except(
            model,
            trainable_prefixes=("readout_q", "q_mlp"),
        )

        expected_trainable = {
            name
            for name, _ in model.projection_head.named_parameters()
            if name.startswith(("readout_q", "q_mlp"))
            and not name.startswith("readout_q_cue")
        }
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
