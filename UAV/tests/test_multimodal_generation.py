import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.generate_mm_smoke import (
    _finalize_dataset_metadata,
    _process_one,
)


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
    def test_complete_dataset_metadata_backfills_content_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "images").mkdir()
            (root / "images" / "env_000000.png").write_bytes(b"image")
            sft_path = root / "sft_dataset.jsonl"
            dpo_path = root / "dpo_dataset.jsonl"
            sft_path.write_text(
                '{"id":"env_0","bev_image_path":'
                '"images/env_000000.png"}\n',
                encoding="utf-8",
            )
            dpo_path.write_text(
                '{"id":"env_0_dpo"}\n', encoding="utf-8"
            )

            finalized = _finalize_dataset_metadata(
                {"generation_complete": True},
                output_dir=root,
                sft_path=sft_path,
                dpo_path=dpo_path,
                num_sft_records=1,
                num_dpo_records=1,
                next_environment_id=1,
                generation_complete=True,
            )

            self.assertEqual(finalized["num_sft_records"], 1)
            self.assertEqual(len(finalized["content_fingerprint"]), 64)

            (root / "images" / "env_000000.png").write_bytes(
                b"modified-image"
            )
            with self.assertRaisesRegex(ValueError, "mismatched"):
                _finalize_dataset_metadata(
                    finalized,
                    output_dir=root,
                    sft_path=sft_path,
                    dpo_path=dpo_path,
                    num_sft_records=1,
                    num_dpo_records=1,
                    next_environment_id=1,
                    generation_complete=True,
                )

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
