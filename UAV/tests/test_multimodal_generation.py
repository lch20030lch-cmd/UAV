import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.generate_mm_smoke import _process_one


class _RejectedEnvironmentGenerator:
    num_restarts = 1

    def __init__(self):
        self.scenario_gen = SimpleNamespace(sample=lambda _: SimpleNamespace())
        self.solver = SimpleNamespace(
            solve=lambda *args, **kwargs: SimpleNamespace(utility=0.0)
        )

    @staticmethod
    def _env_sample_to_dict(_sample):
        return {"q_current": SimpleNamespace(tolist=lambda: [])}

    @staticmethod
    def _pareto_filter(_solutions):
        return []


class MultimodalGenerationTest(unittest.TestCase):
    @patch("scripts.generate_mm_smoke.render_bev_sample")
    @patch("scripts.generate_mm_smoke.build_multimodal_prompt", return_value="prompt")
    def test_rejected_environment_does_not_render_orphan_image(
        self,
        _build_prompt,
        render_bev,
    ):
        simulation = {
            "area_size": [1000, 1000],
            "uav_max_speed_ms": 15,
            "slot_duration_s": 1.0,
        }
        with tempfile.TemporaryDirectory() as temporary_dir:
            sft_sample, dpo_samples = _process_one(
                7,
                _RejectedEnvironmentGenerator(),
                simulation,
                Path(temporary_dir),
                224,
            )

        self.assertIsNone(sft_sample)
        self.assertEqual(dpo_samples, [])
        render_bev.assert_not_called()


if __name__ == "__main__":
    unittest.main()
