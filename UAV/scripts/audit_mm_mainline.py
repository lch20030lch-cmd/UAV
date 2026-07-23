#!/usr/bin/env python
"""Fail-fast preflight for the versioned multimodal Oracle training mainline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.data.geometry_cues import parse_q_geometry_cues
from src.data.multimodal_dataset import validate_multimodal_oracle_contract
from src.data.oracle_contract import (
    ORACLE_SELECTION_MODE,
    PROMPT_TYPE,
    SOLVER_ALGORITHM,
    validate_checkpoint_dataset_compatibility,
)
from src.data.oracle_runtime import (
    build_oracle_scenario,
    build_oracle_solver,
)
from src.data.prompt_builder import build_multimodal_prompt


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}"
                ) from exc
    return rows


def _array(record: dict, key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(record.get(key), dtype=np.float64)
    if value.shape != shape:
        raise ValueError(
            f"{record.get('id')}: {key} must have shape {shape}, "
            f"got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{record.get('id')}: {key} contains NaN/inf")
    return value


def _assert_close(name: str, first, second, atol: float = 5e-5) -> None:
    if not np.allclose(
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError(f"{name} differs between paired records")


def _audit_config(cfg: dict) -> dict:
    projection = cfg["model"]["projection_head"]
    phase1 = cfg["training"]["sft"]["phase1"]
    dpo = cfg["training"]["dpo"]
    loss = cfg["model"]["loss"]
    expected = {
        "projection_head.head_type": (
            projection.get("head_type"),
            "split",
        ),
        "projection_head.q_projection_mode": (
            projection.get("q_projection_mode"),
            "direction",
        ),
        "projection_head.q_geometry_mode": (
            projection.get("q_geometry_mode"),
            "none",
        ),
        "sft.use_chat_template": (
            cfg["training"]["sft"].get("use_chat_template"),
            True,
        ),
        "sft.phase1.train_control_offsets": (
            phase1.get("train_control_offsets"),
            True,
        ),
        "sft.phase1.lambda_lm_ce": (
            float(phase1.get("lambda_lm_ce", -1.0)),
            0.0,
        ),
        "dpo.train_control_offsets": (
            dpo.get("train_control_offsets"),
            True,
        ),
        "model.freeze_vision_tower": (
            cfg["model"].get("freeze_vision_tower"),
            True,
        ),
        "loss.lambda_q": (float(loss.get("lambda_q", -1.0)), 0.0),
    }
    mismatches = {
        key: values
        for key, values in expected.items()
        if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"unsafe multimodal mainline config: {mismatches}")
    required_positive = {
        "phase1.lambda_ctl": phase1.get("lambda_ctl", 0.0),
        "phase1.lambda_q_dir": phase1.get("lambda_q_dir", 0.0),
        "phase1.lambda_assoc_raw_ce": phase1.get(
            "lambda_assoc_raw_ce", 0.0
        ),
        "phase1.lambda_p_raw_kl": phase1.get("lambda_p_raw_kl", 0.0),
        "loss.lambda_p": loss.get("lambda_p", 0.0),
        "dpo.beta": dpo.get("beta", 0.0),
        "dpo.learning_rate": dpo.get("learning_rate", 0.0),
        "dpo.projection_lr": dpo.get("projection_lr", 0.0),
    }
    invalid_positive = {
        key: value
        for key, value in required_positive.items()
        if float(value) <= 0.0
    }
    if invalid_positive:
        raise ValueError(
            "required mainline weights/rates must be positive: "
            f"{invalid_positive}"
        )
    loss_weights = {
        **{
            key: float(value)
            for key, value in loss.items()
            if key.startswith("lambda_")
        },
        **{
            key: float(value)
            for key, value in phase1.items()
            if key.startswith("lambda_")
        },
    }
    negative = {
        key: value for key, value in loss_weights.items() if value < 0.0
    }
    if negative:
        raise ValueError(f"loss weights must be non-negative: {negative}")
    if float(phase1.get("projection_lr", 0.0)) <= 0.0:
        raise ValueError("phase1.projection_lr must be positive")
    if float(phase1.get("lr_lora", 0.0)) <= 0.0:
        raise ValueError("phase1.lr_lora must be positive")
    for key in ("sft_anchor", "control_anchor"):
        if float(dpo.get(key, -1.0)) < 0.0:
            raise ValueError(f"dpo.{key} must be non-negative")
    if int(cfg["model"]["control_token"].get("num_tokens", 0)) <= 0:
        raise ValueError("model.control_token.num_tokens must be positive")
    if int(cfg["training"]["sft"].get("max_seq_length", 0)) <= 0:
        raise ValueError("sft.max_seq_length must be positive")
    if int(dpo.get("max_seq_length", 0)) <= 0:
        raise ValueError("dpo.max_seq_length must be positive")
    return {
        "projection_head_type": "split",
        "q_projection_mode": "direction",
        "q_geometry_mode": "none",
        "use_chat_template": True,
    }


def _audit_dataset(data_root: Path, cfg: dict) -> tuple[dict, dict]:
    sim = cfg["simulation"]
    metadata = validate_multimodal_oracle_contract(
        data_root,
        expected_simulation=sim,
    )
    scenario = build_oracle_scenario(sim, seed=int(metadata["seed"]))
    solver = build_oracle_solver(sim)
    sft = _load_jsonl(data_root / metadata["sft_file"])
    dpo = _load_jsonl(data_root / metadata["dpo_file"])
    if len(sft) != len(dpo) or not sft:
        raise ValueError(
            f"dataset must contain non-empty paired SFT/DPO rows, got "
            f"{len(sft)}/{len(dpo)}"
        )

    m = int(sim["num_uavs"])
    k = int(sim["num_users"])
    t = int(sim["num_targets"])
    max_move = (
        float(sim["uav_max_speed_ms"])
        * float(sim["slot_duration_s"])
    )
    p_max = 10 ** ((float(sim["p_max_dbm"]) - 30.0) / 10.0)
    load_cap = int(sim["load_cap_per_uav"])
    max_violation = 0.0
    min_gap = float("inf")
    visible_targets = 0
    referenced_images = set()

    for row_index, (sft_row, dpo_row) in enumerate(zip(sft, dpo)):
        env_match = re.fullmatch(r"env_(\d+)", str(sft_row.get("id", "")))
        dpo_match = re.fullmatch(
            r"env_(\d+)_dpo", str(dpo_row.get("id", ""))
        )
        if env_match is None or dpo_match is None:
            raise ValueError(f"invalid paired ids at row {row_index}")
        if env_match.group(1) != dpo_match.group(1):
            raise ValueError(f"misaligned paired ids at row {row_index}")
        sample_id = sft_row["id"]
        environment_id = int(env_match.group(1))

        for row_name, row in (("SFT", sft_row), ("DPO", dpo_row)):
            if row.get("prompt_type") != PROMPT_TYPE:
                raise ValueError(
                    f"{sample_id}: {row_name} prompt_type mismatch"
                )
            if row.get("solver_algorithm") != SOLVER_ALGORITHM:
                raise ValueError(
                    f"{sample_id}: {row_name} solver_algorithm mismatch"
                )
            if row.get("oracle_selection_mode") != ORACLE_SELECTION_MODE:
                raise ValueError(
                    f"{sample_id}: {row_name} Oracle selection mode mismatch"
                )
            if row.get("oracle_feasible") is not True:
                raise ValueError(
                    f"{sample_id}: {row_name} Oracle is not feasible"
                )
        for key in (
            "prompt",
            "bev_image_path",
            "q_current",
            "target_detected",
            "delta_q",
            "delta_a",
            "delta_p",
        ):
            if sft_row.get(key) != dpo_row.get(key):
                raise ValueError(
                    f"{sample_id}: {key} differs between SFT and DPO"
                )
        if sft_row["response"] != dpo_row["chosen"]:
            raise ValueError(
                f"{sample_id}: SFT response is not the DPO chosen response"
            )
        relative_image = Path(str(sft_row.get("bev_image_path", "")))
        image_path = (data_root / relative_image).resolve()
        try:
            image_path.relative_to(data_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"{sample_id}: BEV image path escapes the dataset"
            ) from exc
        if not image_path.is_file():
            raise FileNotFoundError(
                f"{sample_id}: missing BEV image {image_path}"
            )
        if relative_image.as_posix() in referenced_images:
            raise ValueError(
                f"{sample_id}: duplicate BEV image path {relative_image}"
            )
        referenced_images.add(relative_image.as_posix())

        q_current = _array(sft_row, "q_current", (m, 3))
        delta_q = _array(sft_row, "delta_q", (m, 3))
        delta_a = _array(sft_row, "delta_a", (m, k))
        delta_p = _array(sft_row, "delta_p", (m, k + 1))
        detected = np.asarray(
            sft_row.get("target_detected"), dtype=bool
        )
        if detected.shape != (t,):
            raise ValueError(
                f"{sample_id}: target_detected must have shape {(t,)}"
            )
        visible_targets += int(detected.sum())

        reproduced = scenario.sample(environment_id)
        reproduced.bev_image_path = relative_image.as_posix()
        _assert_close(
            f"{sample_id}: reproduced q_current",
            reproduced.q_current,
            q_current,
            atol=1e-6,
        )
        if not np.array_equal(
            np.asarray(reproduced.target_detected, dtype=bool),
            detected,
        ):
            raise ValueError(
                f"{sample_id}: target_detected is not reproducible"
            )
        reproduced_prompt = build_multimodal_prompt(reproduced, sim)
        if sft_row["prompt"] != reproduced_prompt:
            raise ValueError(
                f"{sample_id}: prompt is not reproducible from the sealed "
                "environment and current prompt builder"
            )

        movement = np.linalg.norm(delta_q, axis=-1)
        if np.any(movement > max_move + 1e-3):
            raise ValueError(f"{sample_id}: delta_q violates mobility")
        q_next = q_current + delta_q
        if (
            np.any(q_next[:, 0] < -1e-3)
            or np.any(q_next[:, 0] > float(sim["area_size"][0]) + 1e-3)
            or np.any(q_next[:, 1] < -1e-3)
            or np.any(q_next[:, 1] > float(sim["area_size"][1]) + 1e-3)
            or np.any(q_next[:, 2] < float(sim["altitude_min_m"]) - 1e-3)
            or np.any(q_next[:, 2] > float(sim["altitude_max_m"]) + 1e-3)
        ):
            raise ValueError(f"{sample_id}: delta_q violates deployment bounds")

        if not np.allclose(delta_a, np.round(delta_a), atol=1e-4):
            raise ValueError(f"{sample_id}: delta_a is not binary")
        if not np.allclose(delta_a.sum(axis=0), 1.0, atol=1e-4):
            raise ValueError(f"{sample_id}: association columns do not sum to 1")
        if np.any(delta_a.sum(axis=1) > load_cap + 1e-4):
            raise ValueError(f"{sample_id}: association exceeds load cap")
        if np.any(delta_p < -1e-6):
            raise ValueError(f"{sample_id}: delta_p contains negative power")
        if np.any(delta_p.sum(axis=1) > p_max + 2e-4):
            raise ValueError(f"{sample_id}: delta_p exceeds power budget")
        if np.any(np.abs(delta_p[:, :k][delta_a < 0.5]) > 1e-4):
            raise ValueError(
                f"{sample_id}: inactive communication power is non-zero"
            )

        response = json.loads(sft_row["response"])
        for key, expected_value in (
            ("delta_q", delta_q),
            ("delta_a", delta_a),
            ("delta_p", delta_p),
        ):
            _assert_close(
                f"{sample_id}: response {key}",
                response.get(key),
                expected_value,
            )
        chosen = json.loads(dpo_row["chosen"])
        rejected = json.loads(dpo_row["rejected"])
        _assert_close(
            f"{sample_id}: rejected delta_a",
            rejected.get("delta_a"),
            chosen.get("delta_a"),
        )
        _assert_close(
            f"{sample_id}: rejected delta_p",
            rejected.get("delta_p"),
            chosen.get("delta_p"),
        )
        if np.allclose(
            np.asarray(chosen["delta_q"]),
            np.asarray(rejected["delta_q"]),
            atol=1e-4,
        ):
            raise ValueError(
                f"{sample_id}: DPO chosen/rejected Q are identical"
            )
        gap = float(dpo_row.get("utility_gap", 0.0))
        if not np.isfinite(gap) or gap <= 0.0:
            raise ValueError(f"{sample_id}: utility_gap must be positive")

        violations = sft_row.get("constraint_violations", {})
        if not violations:
            raise ValueError(
                f"{sample_id}: missing constraint violation audit"
            )
        violation_values = np.asarray(
            [float(v) for v in violations.values()], dtype=np.float64
        )
        if (
            not np.isfinite(violation_values).all()
            or np.any(violation_values < 0.0)
        ):
            raise ValueError(
                f"{sample_id}: invalid stored constraint violations"
            )
        sample_max_violation = float(violation_values.max())
        max_violation = max(max_violation, sample_max_violation)
        if sample_max_violation > 1e-5:
            raise ValueError(
                f"{sample_id}: stored Oracle violates constraints "
                f"({sample_max_violation})"
            )
        environment = {
            "q_current": reproduced.q_current,
            "user_positions": reproduced.u_positions,
            "target_positions": reproduced.s_positions,
            "target_detected": reproduced.target_detected,
            "channel_gains": reproduced.channel_gains_users,
            "user_weights": reproduced.user_weights,
        }
        q_eval = reproduced.q_current + delta_q
        a_eval = delta_a
        p_comm_eval = delta_p[:, :k]
        p_sense_eval = delta_p[:, k]
        recomputed = solver.evaluate_solution(
            q_eval,
            a_eval,
            p_comm_eval,
            p_sense_eval,
            environment,
        )
        if not recomputed["feasible"]:
            raise ValueError(
                f"{sample_id}: serialized Oracle target is not feasible: "
                f"{recomputed['constraint_violations']}"
            )
        recomputed_max_violation = max(
            float(value)
            for value in recomputed["constraint_violations"].values()
        )
        max_violation = max(max_violation, recomputed_max_violation)
        if not np.isclose(
            float(sft_row["utility"]),
            float(recomputed["utility"]),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(
                f"{sample_id}: stored utility does not match serialized "
                "Oracle target"
            )
        for key, value in recomputed["constraint_violations"].items():
            if key not in violations or not np.isclose(
                float(violations[key]),
                float(value),
                atol=1e-8,
                rtol=0.0,
            ):
                raise ValueError(
                    f"{sample_id}: stored constraint violation {key!r} "
                    "does not match the serialized Oracle target"
                )
        if not np.isclose(
            float(dpo_row.get("utility_chosen")),
            float(recomputed["utility"]),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(
                f"{sample_id}: DPO utility_chosen does not match the "
                "serialized chosen target"
            )

        rejected_delta_q = _array(
            {"id": dpo_row["id"], "value": rejected.get("delta_q")},
            "value",
            (m, 3),
        )
        rejected_movement = np.linalg.norm(rejected_delta_q, axis=-1)
        if np.any(rejected_movement > max_move + 1e-3):
            raise ValueError(
                f"{sample_id}: rejected delta_q violates mobility"
            )
        rejected_q = reproduced.q_current + rejected_delta_q
        if (
            np.any(rejected_q[:, 0] < -1e-3)
            or np.any(
                rejected_q[:, 0] > float(sim["area_size"][0]) + 1e-3
            )
            or np.any(rejected_q[:, 1] < -1e-3)
            or np.any(
                rejected_q[:, 1] > float(sim["area_size"][1]) + 1e-3
            )
            or np.any(
                rejected_q[:, 2]
                < float(sim["altitude_min_m"]) - 1e-3
            )
            or np.any(
                rejected_q[:, 2]
                > float(sim["altitude_max_m"]) + 1e-3
            )
        ):
            raise ValueError(
                f"{sample_id}: rejected delta_q violates deployment bounds"
            )
        rejected_eval = solver.evaluate_solution(
            rejected_q,
            delta_a,
            p_comm_eval,
            p_sense_eval,
            environment,
        )
        try:
            stored_rejected_utility = float(
                dpo_row["utility_rejected"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{sample_id}: missing or invalid DPO utility_rejected"
            ) from exc
        if not np.isclose(
            stored_rejected_utility,
            float(rejected_eval["utility"]),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(
                f"{sample_id}: DPO utility_rejected does not match the "
                "serialized rejected target"
            )
        recomputed_gap = float(
            recomputed["utility"] - rejected_eval["utility"]
        )
        if not np.isclose(gap, recomputed_gap, atol=1e-6, rtol=0.0):
            raise ValueError(
                f"{sample_id}: utility_gap does not match chosen/rejected "
                f"targets ({gap} != {recomputed_gap})"
            )
        min_gap = min(min_gap, gap)

        _, cue_mask = parse_q_geometry_cues(sft_row["prompt"], m)
        if not np.all(np.asarray(cue_mask).any(axis=-1)):
            raise ValueError(f"{sample_id}: a UAV has no valid geometry cue")
        nearest_target_ids = [
            int(value)
            for value in re.findall(
                r"nearest_target=t(\d+)", sft_row["prompt"]
            )
        ]
        if any(
            target_id < 0
            or target_id >= t
            or not detected[target_id]
            for target_id in nearest_target_ids
        ):
            raise ValueError(
                f"{sample_id}: prompt leaks an undetected target position"
            )

    image_dir = data_root / cfg["data"].get("bev_image_dir", "images")
    actual_images = {
        path.relative_to(data_root).as_posix()
        for path in image_dir.glob("*.png")
        if path.is_file()
    }
    if actual_images != referenced_images:
        raise ValueError(
            "BEV image directory does not exactly match dataset references: "
            f"unreferenced={sorted(actual_images - referenced_images)}, "
            f"missing={sorted(referenced_images - actual_images)}"
        )

    return metadata, {
        "num_records": len(sft),
        "num_images": len(referenced_images),
        "max_constraint_violation": max_violation,
        "min_utility_gap": min_gap,
        "visible_target_count": visible_targets,
    }


def _audit_checkpoint(
    checkpoint: Path,
    dataset_metadata: dict,
    expected_modes: dict,
) -> dict:
    metadata_path = checkpoint / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing checkpoint metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_checkpoint_dataset_compatibility(
        metadata, dataset_metadata, require_same_seed=True
    )
    mode_fields = {
        "projection_head_type": expected_modes["projection_head_type"],
        "q_projection_mode": expected_modes["q_projection_mode"],
        "q_geometry_mode": expected_modes["q_geometry_mode"],
        "use_chat_template": expected_modes["use_chat_template"],
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in mode_fields.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"checkpoint mode mismatch: {mismatches}")
    if not (checkpoint / "projection_head.pt").is_file():
        raise FileNotFoundError("checkpoint is missing projection_head.pt")
    if not (
        (checkpoint / "ctrl_embed.pt").is_file()
        or (checkpoint / "control_token_embeddings.pt").is_file()
    ):
        raise FileNotFoundError(
            "checkpoint is missing saved control-token embeddings"
        )
    if not (checkpoint / "ctrl_offset.pt").is_file():
        raise FileNotFoundError(
            "checkpoint is missing trainable control-token offsets"
        )
    requires_lora = (
        metadata.get("stage") == "multimodal_dpo"
        or bool(metadata.get("train_lora"))
        or bool(metadata.get("load_lora"))
    )
    if requires_lora and not (
        checkpoint / "lora" / "adapter_config.json"
    ).is_file():
        raise FileNotFoundError(
            "LoRA-enabled checkpoint is missing lora/adapter_config.json"
        )
    if int(metadata.get("trainable_vision_lora_tensors", 0)) != 0:
        raise ValueError(
            "checkpoint metadata reports trainable vision LoRA tensors"
        )
    if (
        int(metadata.get("trainable_language_lora_tensors", 0)) > 0
        and int(metadata.get("vision_modules_kept_in_eval", 0)) <= 0
    ):
        raise ValueError(
            "language-LoRA checkpoint did not record a frozen vision tower "
            "kept in eval mode"
        )
    return {
        "path": str(checkpoint),
        "global_step": metadata.get("global_step"),
        "stage": metadata.get("stage"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/rtx5090_multimodal_smoke.yaml"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    modes = _audit_config(cfg)
    metadata, dataset_summary = _audit_dataset(
        Path(args.data_dir), cfg
    )
    result = {
        "status": "PASS",
        "config": str(config_path),
        "data_dir": str(Path(args.data_dir)),
        "schema_version": metadata["schema_version"],
        "solver_revision": metadata["solver_revision"],
        "simulation_fingerprint": metadata["simulation_fingerprint"],
        "content_fingerprint": metadata["content_fingerprint"],
        "modes": modes,
        "dataset": dataset_summary,
    }
    if args.checkpoint:
        result["checkpoint"] = _audit_checkpoint(
            Path(args.checkpoint), metadata, modes
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    print(json.dumps(result, indent=2))
    print("MULTIMODAL MAINLINE PREFLIGHT: PASS")


if __name__ == "__main__":
    main()
