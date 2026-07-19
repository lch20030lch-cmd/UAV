import unittest

from src.data.multimodal_dataset import _compute_prompt_budget


class MultimodalSequenceBudgetTest(unittest.TestCase):
    def test_control_only_reserves_every_control_token(self):
        self.assertEqual(_compute_prompt_budget(3072, 8, 0), 3064)

    def test_response_mode_reserves_control_and_response(self):
        self.assertEqual(_compute_prompt_budget(3072, 8, 819), 2245)

    def test_impossible_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot fit reserved"):
            _compute_prompt_budget(8, 8, 0)


if __name__ == "__main__":
    unittest.main()
