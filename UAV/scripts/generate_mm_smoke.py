#!/usr/bin/env python
"""
生成小规模 BEV-image 多模态烟雾测试数据集。

该脚本刻意独立于 scripts/generate_data.py，避免破坏既有 text-grid
baseline 的可复现性。它复用同一套 scenario、SCA-FP solver、oracle prior
提取逻辑和 JSON response 格式，但会额外为每条样本写入多模态 prompt
以及 BEV PNG 的相对路径。
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.data.oracle_generator import (
    OracleDataGenerator,
    serialize_oracle_prior_exact,
    select_near_optimal_q_medoid,
)
from src.data.oracle_contract import (
    DEFAULT_ORACLE_SELECTION_UTILITY_TOLERANCE,
    ORACLE_SELECTION_MODE,
    PROMPT_TYPE,
    assert_resume_compatible,
    build_dataset_metadata,
    dataset_content_fingerprint,
    paired_record_state,
    validate_dataset_metadata,
)
from src.data.oracle_runtime import (
    build_oracle_scenario,
    build_oracle_solver,
)
from src.data.prompt_builder import build_multimodal_prompt
from src.env import render_bev_sample


_stop_requested = False


def _on_interrupt(sig, frame):
    global _stop_requested
    _stop_requested = True
    print("\n[INTERRUPT] Stopping after current sample...")


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _read_record_by_id(path: Path, record_id: str):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id") == record_id:
                return record
    return None


def _recover_pending_pair(
    journal_path: Path,
    sft_path: Path,
    dpo_path: Path,
) -> None:
    """Complete an interrupted two-file append before validating paired IDs."""
    if not journal_path.exists():
        return
    pending = json.loads(journal_path.read_text(encoding="utf-8"))
    for label, path in (("sft", sft_path), ("dpo", dpo_path)):
        expected = pending[label]
        existing = _read_record_by_id(path, str(expected["id"]))
        if existing is None:
            _append_jsonl(path, expected)
        elif existing != expected:
            raise ValueError(
                f"pending {label} record conflicts with {path}: "
                f"{expected['id']}"
            )
    journal_path.unlink()


def _append_paired_jsonl(
    journal_path: Path,
    sft_path: Path,
    dpo_path: Path,
    sft_record: dict,
    dpo_record: dict,
) -> None:
    pending = {"sft": sft_record, "dpo": dpo_record}
    temporary = journal_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(pending, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(journal_path)
    _recover_pending_pair(journal_path, sft_path, dpo_path)


def _finalize_dataset_metadata(
    metadata: dict,
    *,
    output_dir: Path,
    sft_path: Path,
    dpo_path: Path,
    num_sft_records: int,
    num_dpo_records: int,
    next_environment_id: int,
    generation_complete: bool,
) -> dict:
    """Return count-consistent metadata and seal complete dataset content."""
    result = dict(metadata)
    result.update({
        "generation_complete": bool(generation_complete),
        "num_sft_records": int(num_sft_records),
        "num_dpo_records": int(num_dpo_records),
        "next_environment_id": int(next_environment_id),
    })
    if not generation_complete:
        result.pop("content_fingerprint", None)
        return result
    if num_sft_records != num_dpo_records:
        raise ValueError(
            "cannot finalize an unpaired Oracle dataset: "
            f"SFT={num_sft_records}, DPO={num_dpo_records}"
        )

    actual_fingerprint = dataset_content_fingerprint(
        output_dir,
        sft_path.name,
        dpo_path.name,
    )
    stored_fingerprint = metadata.get("content_fingerprint")
    if (
        stored_fingerprint is not None
        and stored_fingerprint != actual_fingerprint
    ):
        raise ValueError(
            "refusing to replace a mismatched Oracle dataset content "
            "fingerprint: "
            f"metadata={stored_fingerprint}, actual={actual_fingerprint}"
        )
    result["content_fingerprint"] = actual_fingerprint
    return result


def _process_one(sample_id: int, generator: OracleDataGenerator, sim_cfg: dict,
                 output_dir: Path, image_size: int):
    env_sample = generator.scenario_gen.sample(sample_id)

    rel_image_path = Path("images") / f"env_{sample_id:06d}.png"
    image_path = output_dir / rel_image_path
    env_sample.bev_image_path = rel_image_path.as_posix()

    prompt = build_multimodal_prompt(env_sample, sim_cfg)
    env_dict = generator._env_sample_to_dict(env_sample)
    q_current = env_dict["q_current"]

    solutions = []
    for j in range(generator.num_restarts):
        seed = sample_id * generator.num_restarts + j
        solutions.append(generator.solver.solve(env_dict, warm_start=None, seed=seed))
    solutions.sort(key=lambda s: s.utility, reverse=True)

    candidates = generator._pareto_filter(solutions)
    if not candidates:
        return None, []

    chosen_sol, selection_diagnostics = select_near_optimal_q_medoid(
        candidates,
        q_current,
        generator.oracle_selection_utility_tolerance,
    )
    delta_q, delta_a, delta_p = generator._extract_prior(
        chosen_sol, env_sample
    )
    response, serialized_chosen = serialize_oracle_prior_exact(
        sample_id, delta_q, delta_a, delta_p
    )
    delta_q, delta_a, delta_p = serialized_chosen
    chosen_eval = generator.evaluate_prior(
        env_dict,
        delta_q,
        delta_a,
        delta_p,
    )
    if not chosen_eval["feasible"]:
        return None, []
    chosen_utility = float(chosen_eval["utility"])

    common = {
        "bev_image_path": env_sample.bev_image_path,
        "prompt_type": PROMPT_TYPE,
        "bev_grid_text": env_sample.bev_grid_text,
        "q_current": q_current.tolist(),
        "target_detected": env_sample.target_detected.astype(bool).tolist(),
        "delta_q": delta_q.tolist(),
        "delta_a": delta_a.tolist(),
        "delta_p": delta_p.tolist(),
        "solver_algorithm": chosen_sol.algorithm,
        "oracle_feasible": bool(chosen_eval["feasible"]),
        "constraint_violations": chosen_eval["constraint_violations"],
        **selection_diagnostics,
    }

    sft_sample = {
        "id": f"env_{sample_id}",
        "prompt": prompt,
        "response": response,
        "utility": chosen_utility,
        **common,
    }

    rejected_delta_q, _ = generator._construct_rejected(
        env_dict, solutions, q_current, sample_id
    )
    rejected_response, serialized_rejected = serialize_oracle_prior_exact(
        sample_id, rejected_delta_q, delta_a, delta_p
    )
    (
        rejected_delta_q,
        rejected_delta_a,
        rejected_delta_p,
    ) = serialized_rejected
    rejected_eval = generator.evaluate_prior(
        env_dict,
        rejected_delta_q,
        rejected_delta_a,
        rejected_delta_p,
    )
    rejected_util = float(rejected_eval["utility"])
    if not np.isfinite(rejected_util):
        return None, []
    dpo_samples = []
    if not np.allclose(rejected_delta_q, delta_q, atol=1e-3):
        gap = (
            chosen_utility - rejected_util
            if rejected_util is not None
            else abs(chosen_utility) * 0.05
        )
        if gap > 0:
            dpo_samples.append({
                "id": f"env_{sample_id}_dpo",
                "prompt": prompt,
                "chosen": response,
                "rejected": rejected_response,
                "utility_chosen": chosen_utility,
                "utility_rejected": rejected_util,
                "utility_gap": gap,
                **common,
            })

    # Keep the two mainline datasets aligned one-to-one.  Environments without
    # a valid positive-gap preference pair are retried with a new sample id.
    if not dpo_samples:
        return None, []

    # Rendering is intentionally delayed until the environment has a valid
    # one-to-one SFT/DPO pair. Failed attempts must not leave orphan BEV files.
    render_bev_sample(
        env_sample,
        save_path=str(image_path),
        area_size=tuple(sim_cfg["area_size"]),
        image_size=image_size,
        movement_radius=(
            sim_cfg.get("uav_max_speed_ms", 15.0)
            * sim_cfg.get("slot_duration_s", 1.0)
        ),
    )
    return sft_sample, dpo_samples


def main():
    parser = argparse.ArgumentParser(description="生成 BEV-image MLLM smoke 数据")
    parser.add_argument("--config", type=str, default="configs/rtx5090_multimodal_smoke.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_restarts", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with (PROJECT_ROOT / args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sim_cfg = cfg["simulation"]
    data_cfg = cfg["data"]

    if args.num_samples is not None:
        data_cfg["num_environments"] = args.num_samples
    if args.num_restarts is not None:
        data_cfg["num_restarts"] = args.num_restarts
    if args.output_dir is not None:
        data_cfg["output_dir"] = args.output_dir
    if args.image_size is not None:
        data_cfg["image_size"] = args.image_size

    output_dir = Path(data_cfg["output_dir"])
    image_size = int(data_cfg.get("image_size", 224))
    num_samples = int(data_cfg["num_environments"])
    num_restarts = int(data_cfg["num_restarts"])
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if num_restarts <= 0:
        raise ValueError("num_restarts must be positive")
    if image_size <= 0:
        raise ValueError("image_size must be positive")

    sft_path = output_dir / data_cfg.get("sft_file", "sft_dataset.jsonl")
    dpo_path = output_dir / data_cfg.get("dpo_file", "dpo_dataset.jsonl")
    ckpt_path = output_dir / "checkpoint.txt"
    metadata_path = output_dir / "dataset_metadata.json"
    pair_journal_path = output_dir / ".pending_pair.json"

    expected_metadata = build_dataset_metadata(
        sim_cfg,
        seed=args.seed,
        num_environments_requested=num_samples,
        num_restarts=num_restarts,
        image_size=image_size,
        sft_file=sft_path.name,
        dpo_file=dpo_path.name,
        oracle_selection_mode=data_cfg.get(
            "oracle_selection_mode", ORACLE_SELECTION_MODE
        ),
        oracle_selection_utility_tolerance=data_cfg.get(
            "oracle_selection_utility_tolerance",
            DEFAULT_ORACLE_SELECTION_UTILITY_TOLERANCE,
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)

    existing_metadata = None
    if args.overwrite:
        for path in [
            sft_path,
            dpo_path,
            ckpt_path,
            metadata_path,
            pair_journal_path,
            pair_journal_path.with_suffix(".tmp"),
        ]:
            if path.exists():
                path.unlink()
        for image_path in (output_dir / "images").glob("env_*.png"):
            image_path.unlink()

    if not args.overwrite and (sft_path.exists() or dpo_path.exists()):
        if not metadata_path.exists():
            raise RuntimeError(
                "refusing to resume a pre-v5 dataset in place; choose a new "
                "output directory or pass --overwrite"
            )
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert_resume_compatible(existing_metadata, expected_metadata)

    # Resume from the attempted environment id, not the SFT line count.  An
    # infeasible/failed environment may intentionally produce no SFT row, so
    # line-count resume can otherwise duplicate later sample ids.
    checkpoint_next_id = 0
    if ckpt_path.exists():
        try:
            checkpoint_next_id = int(
                ckpt_path.read_text(encoding="utf-8").strip()
            )
        except ValueError as exc:
            raise ValueError(f"invalid generation checkpoint: {ckpt_path}") from exc
    _recover_pending_pair(pair_journal_path, sft_path, dpo_path)
    record_state = paired_record_state(
        output_dir, sft_path.name, dpo_path.name
    )
    existing_sft = record_state["num_sft_records"]
    existing_dpo = record_state["num_dpo_records"]
    # A hard interruption can occur after both JSONL rows are appended but
    # before checkpoint.txt is advanced.  Never reuse an ID already present.
    start_id = max(checkpoint_next_id, record_state["next_environment_id"])
    if existing_sft >= num_samples and existing_dpo >= num_samples:
        if existing_metadata and existing_metadata.get(
            "generation_complete"
        ) is True:
            validated = validate_dataset_metadata(
                existing_metadata,
                data_dir=output_dir,
                expected_simulation=sim_cfg,
                expected_seed=args.seed,
            )
            print(f"All {num_samples} samples already exist at {output_dir}")
            print(
                "  content_fingerprint: "
                f"{validated['content_fingerprint']}"
            )
            return
        dataset_metadata = _finalize_dataset_metadata(
            existing_metadata or expected_metadata,
            output_dir=output_dir,
            sft_path=sft_path,
            dpo_path=dpo_path,
            num_sft_records=existing_sft,
            num_dpo_records=existing_dpo,
            next_environment_id=start_id,
            generation_complete=True,
        )
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(dataset_metadata, handle, indent=2)
        print(f"All {num_samples} samples already exist at {output_dir}")
        print(
            "  content_fingerprint: "
            f"{dataset_metadata['content_fingerprint']}"
        )
        return

    # 复用 text-grid baseline 的场景生成器与 SCA-FP solver，保证两条路线可对照。
    scenario_gen = build_oracle_scenario(sim_cfg, seed=args.seed)
    solver = build_oracle_solver(sim_cfg)
    generator = OracleDataGenerator(
        scenario_gen=scenario_gen,
        solver=solver,
        config=data_cfg,
        sim_config=sim_cfg,
    )

    dataset_metadata = expected_metadata
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_metadata, handle, indent=2)

    signal.signal(signal.SIGINT, _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    print("=" * 60)
    print("UAV-ISAC MLLM: BEV-image smoke data generator")
    print("=" * 60)
    print(
        f"  Samples:      {num_samples} target records "
        f"({existing_sft} existing, next env id {start_id})"
    )
    print(f"  Restarts/env: {generator.num_restarts}")
    print(f"  Image size:   {image_size}")
    print(f"  Output:       {output_dir}")
    print()

    t0 = time.time()
    n_sft = existing_sft
    n_dpo = existing_dpo
    sample_id = start_id
    attempts = 0
    max_attempts = max(num_samples * 10, num_samples + 100)

    while n_sft < num_samples:
        if _stop_requested:
            break
        if attempts >= max_attempts:
            raise RuntimeError(
                f"only generated {n_sft}/{num_samples} paired feasible records "
                f"after {attempts} attempts"
            )
        try:
            sft_sample, dpo_samples = _process_one(
                sample_id, generator, sim_cfg, output_dir, image_size
            )
        except Exception as exc:
            raise RuntimeError(
                f"unexpected Oracle generation failure at env {sample_id}"
            ) from exc
        if sft_sample is not None:
            if len(dpo_samples) != 1:
                raise RuntimeError(
                    "v5 generation requires exactly one DPO row per SFT row"
                )
            _append_paired_jsonl(
                pair_journal_path,
                sft_path,
                dpo_path,
                sft_sample,
                dpo_samples[0],
            )
            n_sft += 1
            n_dpo += 1

        with ckpt_path.open("w", encoding="utf-8") as f:
            f.write(f"{sample_id + 1}\n")

        elapsed = time.time() - t0
        attempts += 1
        rate = elapsed / max(attempts, 1)
        remaining = max(num_samples - n_sft, 0) * rate
        print(
            f"  [env {sample_id}] {n_sft}/{num_samples} paired SFT/DPO | "
            f"{elapsed:.0f}s elapsed, ~{remaining / 60:.1f}min remaining",
            flush=True,
        )
        sample_id += 1

    elapsed = time.time() - t0
    dataset_metadata = _finalize_dataset_metadata(
        dataset_metadata,
        output_dir=output_dir,
        sft_path=sft_path,
        dpo_path=dpo_path,
        num_sft_records=n_sft,
        num_dpo_records=n_dpo,
        next_environment_id=sample_id,
        generation_complete=(
            n_sft == num_samples and n_dpo == num_samples
        ),
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(dataset_metadata, handle, indent=2)
    print("\nDone")
    print(f"  SFT:   {n_sft} -> {sft_path}")
    print(f"  DPO:   {n_dpo} -> {dpo_path}")
    print(f"  Images: {output_dir / 'images'}")
    print(f"  Time:  {elapsed:.1f}s")


if __name__ == "__main__":
    main()
