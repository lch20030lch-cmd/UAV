"""Versioned provenance contract for multimodal Oracle datasets.

The contract makes the physical simulation that produced a dataset explicit.
Training and evaluation can therefore reject a stale checkpoint or an unsafe
resume before loading a large model or appending any records.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Mapping, Optional


SCHEMA_VERSION = 5
PROMPT_TYPE = "multimodal_bev_image_v5_constraint_aware"
SOLVER_ALGORITHM = "constraint_aware_alternating_optimization"
SOLVER_REVISION = 1
CHANNEL_MODEL = "elevation_los_3gpp_pathloss_v2"

SIMULATION_KEYS = (
    "area_size",
    "num_uavs",
    "num_users",
    "num_targets",
    "num_antennas_tx",
    "num_antennas_rx",
    "carrier_freq_ghz",
    "bandwidth_mhz",
    "p_max_dbm",
    "noise_figure_db",
    "altitude_min_m",
    "altitude_max_m",
    "uav_min_separation_m",
    "uav_max_speed_ms",
    "slot_duration_s",
    "sinr_c_min_db",
    "sinr_s_min_db",
    "rate_min_bps",
    "load_cap_per_uav",
)

IMMUTABLE_DATASET_FIELDS = (
    "schema_version",
    "prompt_type",
    "solver_algorithm",
    "solver_revision",
    "channel_model",
    "simulation_fingerprint",
    "seed",
    "num_restarts",
    "image_size",
    "sft_file",
    "dpo_file",
)


def canonical_simulation_config(simulation: Mapping) -> Dict:
    """Return the physical configuration in a stable JSON representation."""
    missing = [key for key in SIMULATION_KEYS if key not in simulation]
    if missing:
        raise KeyError(f"simulation config is missing contract keys: {missing}")

    canonical = {}
    for key in SIMULATION_KEYS:
        value = simulation[key]
        if isinstance(value, tuple):
            value = list(value)
        canonical[key] = value
    return canonical


def simulation_fingerprint(simulation: Mapping) -> str:
    payload = json.dumps(
        canonical_simulation_config(simulation),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_dataset_metadata(
    simulation: Mapping,
    *,
    seed: int,
    num_environments_requested: int,
    num_restarts: int,
    image_size: int,
    sft_file: str,
    dpo_file: str,
) -> Dict:
    canonical = canonical_simulation_config(simulation)
    return {
        "schema_version": SCHEMA_VERSION,
        "prompt_type": PROMPT_TYPE,
        "seed": int(seed),
        "num_environments_requested": int(num_environments_requested),
        "num_restarts": int(num_restarts),
        "image_size": int(image_size),
        "sft_file": str(sft_file),
        "dpo_file": str(dpo_file),
        "solver_algorithm": SOLVER_ALGORITHM,
        "solver_revision": SOLVER_REVISION,
        "channel_model": CHANNEL_MODEL,
        "simulation": canonical,
        "simulation_fingerprint": simulation_fingerprint(canonical),
        "requires_oracle_feasible": True,
        "generation_complete": False,
    }


def assert_resume_compatible(existing: Mapping, expected: Mapping) -> None:
    mismatches = {
        key: (existing.get(key), expected.get(key))
        for key in IMMUTABLE_DATASET_FIELDS
        if existing.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError(
            "dataset resume metadata does not match the requested generation: "
            f"{mismatches}; use a new output directory"
        )


def _record_ids(path: Path, pattern: str) -> list[int]:
    if not path.exists():
        return []
    ids = []
    matcher = re.compile(pattern)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            match = matcher.fullmatch(str(record.get("id", "")))
            if match is None:
                raise ValueError(
                    f"invalid Oracle record id at {path}:{line_number}: "
                    f"{record.get('id')!r}"
                )
            ids.append(int(match.group(1)))
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate Oracle record ids in {path}")
    return ids


def paired_record_state(
    data_dir: Path,
    sft_file: str = "sft_dataset.jsonl",
    dpo_file: str = "dpo_dataset.jsonl",
) -> Dict[str, int]:
    """Validate paired IDs and recover a crash-safe next environment ID."""
    sft_ids = _record_ids(data_dir / sft_file, r"env_(\d+)")
    dpo_ids = _record_ids(data_dir / dpo_file, r"env_(\d+)_dpo")
    if sft_ids != dpo_ids:
        raise ValueError(
            "v5 generation requires ordered one-to-one SFT/DPO environment "
            f"ids, got SFT={sft_ids[:10]} DPO={dpo_ids[:10]}"
        )
    return {
        "num_sft_records": len(sft_ids),
        "num_dpo_records": len(dpo_ids),
        "next_environment_id": max(sft_ids, default=-1) + 1,
    }


def validate_dataset_metadata(
    metadata: Mapping,
    *,
    data_dir: Optional[Path] = None,
    expected_simulation: Optional[Mapping] = None,
    expected_seed: Optional[int] = None,
) -> Dict:
    required = {
        "schema_version": SCHEMA_VERSION,
        "prompt_type": PROMPT_TYPE,
        "solver_algorithm": SOLVER_ALGORITHM,
        "solver_revision": SOLVER_REVISION,
        "channel_model": CHANNEL_MODEL,
        "requires_oracle_feasible": True,
        "generation_complete": True,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if expected_simulation is not None:
        expected_fingerprint = simulation_fingerprint(expected_simulation)
        if metadata.get("simulation_fingerprint") != expected_fingerprint:
            mismatches["simulation_fingerprint"] = (
                metadata.get("simulation_fingerprint"),
                expected_fingerprint,
            )
    if expected_seed is not None and int(metadata.get("seed", -1)) != int(expected_seed):
        mismatches["seed"] = (metadata.get("seed"), int(expected_seed))
    if mismatches:
        raise ValueError(f"Oracle dataset contract mismatch: {mismatches}")

    result = dict(metadata)
    if data_dir is not None:
        sft_file = str(metadata.get("sft_file", "sft_dataset.jsonl"))
        dpo_file = str(metadata.get("dpo_file", "dpo_dataset.jsonl"))
        record_state = paired_record_state(data_dir, sft_file, dpo_file)
        actual_sft = record_state["num_sft_records"]
        actual_dpo = record_state["num_dpo_records"]
        expected_sft = int(metadata.get("num_sft_records", -1))
        expected_dpo = int(metadata.get("num_dpo_records", -1))
        if (
            actual_sft != expected_sft
            or actual_dpo != expected_dpo
            or actual_sft != actual_dpo
        ):
            raise ValueError(
                "Oracle dataset record counts do not match metadata: "
                f"actual={actual_sft}/{actual_dpo}, "
                f"metadata={expected_sft}/{expected_dpo}"
            )
    return result


def checkpoint_dataset_fields(dataset_metadata: Mapping) -> Dict:
    """Select the immutable provenance stored in every model checkpoint."""
    return {
        f"dataset_{key}": dataset_metadata.get(key)
        for key in (
            "schema_version",
            "prompt_type",
            "solver_algorithm",
            "solver_revision",
            "channel_model",
            "simulation_fingerprint",
            "seed",
        )
    }


def validate_checkpoint_dataset_compatibility(
    checkpoint_metadata: Mapping,
    dataset_metadata: Mapping,
    *,
    allow_mismatch: bool = False,
    require_same_seed: bool = False,
) -> None:
    expected = checkpoint_dataset_fields(dataset_metadata)
    if not require_same_seed:
        expected.pop("dataset_seed", None)
    mismatches = {
        key: (checkpoint_metadata.get(key), value)
        for key, value in expected.items()
        if checkpoint_metadata.get(key) != value
    }
    if mismatches and not allow_mismatch:
        raise ValueError(
            "checkpoint was not trained on the current Oracle dataset contract: "
            f"{mismatches}. Use an explicit checkpoint-dataset override only "
            "for diagnostics or intentional migration."
        )
