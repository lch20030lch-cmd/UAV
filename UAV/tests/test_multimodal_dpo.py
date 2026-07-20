import unittest

import torch

from src.training.train_dpo_mm import (
    _dpo_checkpoint_metadata,
    _response_logit_positions,
    _sequence_log_prob,
)


class MultimodalDPOLogProbTest(unittest.TestCase):
    def test_dpo_progress_overrides_stage1_metadata(self):
        metadata = _dpo_checkpoint_metadata(
            {
                "stage": "multimodal_sft",
                "global_step": 200,
                "micro_step": 1600,
                "dataset_schema_version": 5,
            },
            {"max_steps": 50},
            global_step=17,
            micro_step=136,
        )

        self.assertEqual(metadata["stage"], "multimodal_dpo")
        self.assertEqual(metadata["global_step"], 17)
        self.assertEqual(metadata["micro_step"], 136)
        self.assertEqual(metadata["max_steps"], 50)
        self.assertEqual(metadata["dataset_schema_version"], 5)

    def test_only_response_prediction_positions_are_requested(self):
        batch = {
            "label_mask_chosen": torch.tensor(
                [[0.0, 0.0, 1.0, 1.0, 0.0]]
            )
        }

        positions = _response_logit_positions(batch, "chosen")

        torch.testing.assert_close(positions, torch.tensor([1, 2]))

    def test_selected_logits_match_shifted_response_labels(self):
        # Positions [1, 2] predict labels at sequence positions [2, 3].
        labels = torch.tensor([[-100, -100, 1, 2, -100]])
        positions = torch.tensor([1, 2])
        logits = torch.tensor(
            [[[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]],
            requires_grad=True,
        )

        actual = _sequence_log_prob(logits, labels, positions)
        expected = (
            torch.log_softmax(logits[0, 0], dim=-1)[1]
            + torch.log_softmax(logits[0, 1], dim=-1)[2]
        )

        torch.testing.assert_close(actual, expected.unsqueeze(0))
        actual.sum().backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
