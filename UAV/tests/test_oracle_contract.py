import tempfile
import unittest
from pathlib import Path

from src.data.oracle_contract import (
    assert_resume_compatible,
    build_dataset_metadata,
    checkpoint_dataset_fields,
    dataset_content_fingerprint,
    paired_record_state,
    simulation_fingerprint,
    validate_checkpoint_dataset_compatibility,
    validate_dataset_metadata,
)


def _simulation():
    return {
        "area_size": [1000, 1000],
        "num_uavs": 4,
        "num_users": 20,
        "num_targets": 6,
        "num_antennas_tx": 8,
        "num_antennas_rx": 8,
        "carrier_freq_ghz": 5.8,
        "bandwidth_mhz": 20,
        "p_max_dbm": 30,
        "noise_figure_db": 9,
        "altitude_min_m": 50,
        "altitude_max_m": 300,
        "uav_min_separation_m": 10,
        "uav_max_speed_ms": 15,
        "slot_duration_s": 1.0,
        "sinr_c_min_db": 0,
        "sinr_s_min_db": 10,
        "rate_min_bps": 1e6,
        "load_cap_per_uav": 10,
    }


def _metadata(simulation=None):
    return build_dataset_metadata(
        simulation or _simulation(),
        seed=42,
        num_environments_requested=2,
        num_restarts=3,
        image_size=224,
        sft_file="sft_dataset.jsonl",
        dpo_file="dpo_dataset.jsonl",
    )


class OracleContractTest(unittest.TestCase):
    def test_content_fingerprint_changes_with_actual_records(self):
        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            first = Path(first_dir)
            second = Path(second_dir)
            for root, utility in ((first, 1.0), (second, 2.0)):
                (root / "sft.jsonl").write_text(
                    f'{{"id":"env_0","utility":{utility}}}\n',
                    encoding="utf-8",
                )
                (root / "dpo.jsonl").write_text(
                    '{"id":"env_0_dpo"}\n',
                    encoding="utf-8",
                )

            self.assertNotEqual(
                dataset_content_fingerprint(first, "sft.jsonl", "dpo.jsonl"),
                dataset_content_fingerprint(second, "sft.jsonl", "dpo.jsonl"),
            )

    def test_content_fingerprint_includes_referenced_bev_images(self):
        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            for root_text, image_bytes in (
                (first_dir, b"first-image"),
                (second_dir, b"second-image"),
            ):
                root = Path(root_text)
                (root / "images").mkdir()
                (root / "images" / "env_000000.png").write_bytes(image_bytes)
                (root / "sft.jsonl").write_text(
                    '{"id":"env_0","bev_image_path":'
                    '"images/env_000000.png"}\n',
                    encoding="utf-8",
                )
                (root / "dpo.jsonl").write_text(
                    '{"id":"env_0_dpo"}\n',
                    encoding="utf-8",
                )

            self.assertNotEqual(
                dataset_content_fingerprint(
                    Path(first_dir), "sft.jsonl", "dpo.jsonl"
                ),
                dataset_content_fingerprint(
                    Path(second_dir), "sft.jsonl", "dpo.jsonl"
                ),
            )

    def test_fingerprint_is_stable_but_physics_sensitive(self):
        first = _simulation()
        reordered = dict(reversed(list(first.items())))
        changed = dict(first, bandwidth_mhz=40)

        self.assertEqual(
            simulation_fingerprint(first), simulation_fingerprint(reordered)
        )
        self.assertNotEqual(
            simulation_fingerprint(first), simulation_fingerprint(changed)
        )

    def test_resume_rejects_changed_physical_configuration(self):
        existing = _metadata()
        changed = _metadata(dict(_simulation(), noise_figure_db=7))

        with self.assertRaisesRegex(ValueError, "simulation_fingerprint"):
            assert_resume_compatible(existing, changed)

    def test_contract_records_oracle_consensus_selection(self):
        metadata = _metadata()

        self.assertEqual(metadata["solver_revision"], 2)
        self.assertEqual(
            metadata["oracle_selection_mode"], "near_optimal_q_medoid"
        )
        self.assertEqual(
            metadata["oracle_selection_utility_tolerance"], 0.01
        )

        changed = dict(
            metadata, oracle_selection_utility_tolerance=0.02
        )
        with self.assertRaisesRegex(
            ValueError, "oracle_selection_utility_tolerance"
        ):
            assert_resume_compatible(metadata, changed)

    def test_complete_contract_checks_actual_paired_record_counts(self):
        metadata = _metadata()
        metadata.update({
            "generation_complete": True,
            "num_sft_records": 2,
            "num_dpo_records": 2,
        })
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sft_dataset.jsonl").write_text(
                '{"id":"env_0"}\n{"id":"env_2"}\n', encoding="utf-8"
            )
            (root / "dpo_dataset.jsonl").write_text(
                '{"id":"env_0_dpo"}\n{"id":"env_2_dpo"}\n',
                encoding="utf-8",
            )

            validated = validate_dataset_metadata(
                metadata,
                data_dir=root,
                expected_simulation=_simulation(),
                expected_seed=42,
            )

            tampered_metadata = dict(metadata, content_fingerprint="0" * 64)
            with self.assertRaisesRegex(ValueError, "content fingerprint"):
                validate_dataset_metadata(
                    tampered_metadata,
                    data_dir=root,
                    expected_simulation=_simulation(),
                    expected_seed=42,
                )

        self.assertEqual(validated["num_sft_records"], 2)
        self.assertEqual(len(validated["content_fingerprint"]), 64)

    def test_checkpoint_must_share_dataset_provenance(self):
        dataset = _metadata()
        checkpoint = checkpoint_dataset_fields(dataset)
        validate_checkpoint_dataset_compatibility(checkpoint, dataset)

        checkpoint["dataset_seed"] = 2026
        with self.assertRaisesRegex(ValueError, "dataset_seed"):
            validate_checkpoint_dataset_compatibility(
                checkpoint, dataset, require_same_seed=True
            )

        validate_checkpoint_dataset_compatibility(checkpoint, dataset)

    def test_training_checkpoint_requires_exact_dataset_content(self):
        metadata = _metadata()
        metadata.update({
            "generation_complete": True,
            "num_sft_records": 1,
            "num_dpo_records": 1,
        })
        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            validated = []
            for root_text, utility in ((first_dir, 1.0), (second_dir, 2.0)):
                root = Path(root_text)
                (root / "sft_dataset.jsonl").write_text(
                    f'{{"id":"env_0","utility":{utility}}}\n',
                    encoding="utf-8",
                )
                (root / "dpo_dataset.jsonl").write_text(
                    '{"id":"env_0_dpo"}\n',
                    encoding="utf-8",
                )
                validated.append(validate_dataset_metadata(metadata, data_dir=root))

            checkpoint = checkpoint_dataset_fields(validated[0])
            validate_checkpoint_dataset_compatibility(
                checkpoint,
                validated[0],
                require_same_seed=True,
            )
            with self.assertRaisesRegex(ValueError, "dataset_content_fingerprint"):
                validate_checkpoint_dataset_compatibility(
                    checkpoint,
                    validated[1],
                    require_same_seed=True,
                )

            # Held-out evaluation intentionally shares physics, not records.
            validate_checkpoint_dataset_compatibility(checkpoint, validated[1])

    def test_paired_records_recover_next_id_after_checkpoint_lag(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sft.jsonl").write_text(
                '{"id":"env_3"}\n{"id":"env_8"}\n', encoding="utf-8"
            )
            (root / "dpo.jsonl").write_text(
                '{"id":"env_3_dpo"}\n{"id":"env_8_dpo"}\n',
                encoding="utf-8",
            )

            state = paired_record_state(root, "sft.jsonl", "dpo.jsonl")

        self.assertEqual(state["num_sft_records"], 2)
        self.assertEqual(state["next_environment_id"], 9)

    def test_paired_records_reject_misaligned_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sft.jsonl").write_text(
                '{"id":"env_3"}\n', encoding="utf-8"
            )
            (root / "dpo.jsonl").write_text(
                '{"id":"env_4_dpo"}\n', encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "one-to-one"):
                paired_record_state(root, "sft.jsonl", "dpo.jsonl")


if __name__ == "__main__":
    unittest.main()
