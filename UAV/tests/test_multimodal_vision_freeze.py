import unittest

import torch.nn as nn

from src.model.gemma_multimodal_isac import (
    freeze_vision_parameter_tree,
    is_vision_parameter_name,
)


class _NestedMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Linear(2, 2)
        self.model.vision_tower = nn.Module()
        self.model.vision_tower.vision_model = nn.Linear(2, 2)
        self.model.vision_tower.q_proj_lora = nn.Linear(2, 2, bias=False)


class MultimodalVisionFreezeTest(unittest.TestCase):
    def test_nested_vision_parameter_names_are_detected(self):
        self.assertTrue(
            is_vision_parameter_name(
                "base_model.model.model.vision_tower.vision_model.encoder.q_proj.lora_A"
            )
        )
        self.assertFalse(
            is_vision_parameter_name(
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A"
            )
        )

    def test_recursive_freeze_includes_nested_vision_lora(self):
        model = _NestedMultimodalModel()

        frozen = freeze_vision_parameter_tree(model)

        self.assertTrue(frozen)
        for name, parameter in model.named_parameters():
            if is_vision_parameter_name(name):
                self.assertFalse(parameter.requires_grad, name)
            else:
                self.assertTrue(parameter.requires_grad, name)


if __name__ == "__main__":
    unittest.main()
