import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.generate_mm_smoke import (
    _append_paired_jsonl,
    _finalize_dataset_metadata,
    _process_one,
    _recover_pending_pair,
)
from src.data.oracle_generator import serialize_oracle_prior_exact


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
    def test_utility_inputs_are_exactly_the_serialized_json_values(self):
        response, serialized = serialize_oracle_prior_exact(
            3,
            np.asarray([[1.23456789, -2.34567891, 0.00000049]]),
            np.asarray([[0.9999999, 0.0000001]]),
            np.asarray([[0.123456789, 0.876543211, 0.0]]),
        )
        payload = json.loads(response)

        for key, actual in zip(
            ("delta_q", "delta_a", "delta_p"), serialized
        ):
            np.testing.assert_array_equal(
                actual,
                np.asarray(payload[key], dtype=np.float64),
            )
        self.assertEqual(serialized[0][0, 0], 1.234568)

    def test_pending_pair_recovery_completes_interrupted_append(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            sft_path = root / "sft.jsonl"
            dpo_path = root / "dpo.jsonl"
            journal = root / ".pending_pair.json"
            sft = {"id": "env_3", "value": 1}
            dpo = {"id": "env_3_dpo", "value": 2}

            _append_paired_jsonl(
                journal, sft_path, dpo_path, sft, dpo
            )
            self.assertFalse(journal.exists())
            self.assertEqual(sft_path.read_text().count("\n"), 1)
            self.assertEqual(dpo_path.read_text().count("\n"), 1)

            journal.write_text(
                '{"sft":{"id":"env_4"},"dpo":{"id":"env_4_dpo"}}',
                encoding="utf-8",
            )
            with sft_path.open("a", encoding="utf-8") as handle:
                handle.write('{"id":"env_4"}\n')
            _recover_pending_pair(journal, sft_path, dpo_path)

            self.assertFalse(journal.exists())
            self.assertIn("env_4_dpo", dpo_path.read_text())

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
