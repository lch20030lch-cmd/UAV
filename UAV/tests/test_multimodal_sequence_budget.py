import unittest

import torch

from src.data.multimodal_dataset import (
    _compute_prompt_budget,
    _encode_text_image,
    format_multimodal_user_prompt,
)


class _FakeProcessor:
    def __init__(self, encoded_length):
        self.encoded_length = encoded_length

    def __call__(self, **kwargs):
        return {
            "input_ids": torch.ones((1, self.encoded_length), dtype=torch.long),
        }

    image_token = "<image>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        self.add_generation_prompt = add_generation_prompt
        return "<image>CHAT_PROMPT"


class MultimodalSequenceBudgetTest(unittest.TestCase):
    def test_control_only_reserves_every_control_token(self):
        self.assertEqual(_compute_prompt_budget(3072, 8, 0), 3064)

    def test_response_mode_reserves_control_and_response(self):
        self.assertEqual(_compute_prompt_budget(3072, 8, 819), 2245)

    def test_impossible_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot fit reserved"):
            _compute_prompt_budget(8, 8, 0)

    def test_multimodal_encoding_rejects_image_unsafe_truncation(self):
        processor = _FakeProcessor(encoded_length=11)

        with self.assertRaisesRegex(ValueError, "must not be truncated"):
            _encode_text_image(processor, "prompt", object(), max_length=10)

    def test_multimodal_encoding_accepts_prompt_that_fits_budget(self):
        encoded = _encode_text_image(
            _FakeProcessor(encoded_length=10),
            "prompt",
            object(),
            max_length=10,
        )

        self.assertEqual(tuple(encoded["input_ids"].shape), (1, 10))

    def test_response_training_can_use_multimodal_chat_template(self):
        processor = _FakeProcessor(encoded_length=1)

        formatted = format_multimodal_user_prompt(
            processor, "hello", use_chat_template=True
        )

        self.assertEqual(formatted, "<image>CHAT_PROMPT")
        self.assertTrue(processor.add_generation_prompt)
        self.assertEqual(processor.messages[0]["role"], "user")


if __name__ == "__main__":
    unittest.main()
