import tempfile
import unittest
from pathlib import Path

from src.training.runtime_utils import (
    resolve_optimizer_controls,
    resolve_optimizer_steps,
    resolve_warmup_steps,
    rotate_step_checkpoints,
)


class OptimizerControlResolutionTest(unittest.TestCase):
    def test_defaults_are_read_without_mutating_config(self):
        config = {
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
        }

        result = resolve_optimizer_controls(
            config,
            configured_max_grad_norm=1.0,
        )

        self.assertEqual(result["lr_scheduler"], "cosine")
        self.assertEqual(result["warmup_ratio"], 0.03)
        self.assertEqual(result["weight_decay"], 0.01)
        self.assertEqual(result["max_grad_norm"], 1.0)
        self.assertEqual(config["lr_scheduler"], "cosine")

    def test_explicit_overrides_are_isolated_to_one_run(self):
        result = resolve_optimizer_controls(
            {
                "lr_scheduler": "cosine",
                "warmup_ratio": 0.03,
                "weight_decay": 0.01,
            },
            configured_max_grad_norm=1.0,
            lr_scheduler_override="constant",
            warmup_ratio_override=0.0,
            weight_decay_override=0.0,
            max_grad_norm_override=5.0,
        )

        self.assertEqual(
            result,
            {
                "lr_scheduler": "constant",
                "warmup_ratio": 0.0,
                "weight_decay": 0.0,
                "max_grad_norm": 5.0,
            },
        )

    def test_invalid_optimizer_controls_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_optimizer_controls(
                {},
                configured_max_grad_norm=1.0,
                warmup_ratio_override=1.1,
            )
        with self.assertRaises(ValueError):
            resolve_optimizer_controls(
                {},
                configured_max_grad_norm=1.0,
                weight_decay_override=-0.1,
            )
        with self.assertRaises(ValueError):
            resolve_optimizer_controls(
                {},
                configured_max_grad_norm=1.0,
                max_grad_norm_override=0.0,
            )


class OptimizerStepResolutionTest(unittest.TestCase):
    def test_epochs_are_converted_to_optimizer_updates(self):
        self.assertEqual(
            resolve_optimizer_steps(
                num_batches=500,
                gradient_accumulation_steps=8,
                epochs=2,
            ),
            125,
        )

    def test_explicit_max_steps_takes_precedence(self):
        self.assertEqual(
            resolve_optimizer_steps(
                num_batches=500,
                gradient_accumulation_steps=8,
                epochs=2,
                max_steps_override=17,
            ),
            17,
        )

    def test_invalid_runtime_values_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_optimizer_steps(
                num_batches=0,
                gradient_accumulation_steps=8,
                epochs=1,
            )
        with self.assertRaises(ValueError):
            resolve_warmup_steps(100, 1.1)


class CheckpointRotationTest(unittest.TestCase):
    def test_only_old_matching_step_directories_are_removed(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for step in (10, 20, 30):
                checkpoint = root / f"mm_sft_step_{step}"
                checkpoint.mkdir()
                (checkpoint / "marker.txt").write_text("keep", encoding="utf-8")
            unrelated = root / "mm_sft_final"
            unrelated.mkdir()

            removed = rotate_step_checkpoints(
                root,
                prefix="mm_sft_step_",
                save_total_limit=2,
            )

            self.assertEqual([Path(path).name for path in removed], ["mm_sft_step_10"])
            self.assertFalse((root / "mm_sft_step_10").exists())
            self.assertTrue((root / "mm_sft_step_20").exists())
            self.assertTrue((root / "mm_sft_step_30").exists())
            self.assertTrue(unrelated.exists())

    def test_none_limit_disables_rotation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "checkpoint_1").mkdir()
            self.assertEqual(
                rotate_step_checkpoints(
                    root,
                    prefix="checkpoint_",
                    save_total_limit=None,
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()
