# 2026-07-20 P0 multimodal mainline closure

## Scope

This change removes the mismatch where the project generated multimodal SFT/DPO
records but only had a control-only SFT loop and text-only DPO/evaluation paths.

## Implemented

- `train_sft_mm.py` supports optional response-token CE through
  `--lambda_lm_ce`; response mode requires trainable LoRA.
- Response SFT uses the Gemma multimodal chat template. Legacy control-only
  diagnostics keep their raw-prompt behavior for checkpoint compatibility.
- Only supervised response positions request vocabulary logits, avoiding full
  4096-token vocabulary-logit materialization.
- Added `MultimodalDPODataset` with image tensors, chosen/rejected sequences,
  control masks, geometry cues and Q-only preference masking.
- Added `src/training/train_dpo_mm.py` with a trainable multimodal policy,
  frozen multimodal reference, selected-position log probabilities, DPO loss,
  SFT anchor and control anchor.
- Added `scripts/evaluate_mm_solver.py` for image-conditioned model inference,
  warm/cold downstream solving, actual CRB, constraint feasibility, utility and
  iteration/time speedup.
- Legacy text DPO/evaluation entry points now fail fast when given a multimodal
  config instead of silently running the wrong model path.
- Projection checkpoint loading is strict by default. Architecture migration
  requires the explicit `--allow_partial_projection_load` flag, and diagnostic
  CLI/checkpoint mode conflicts are rejected.
- Mainline SFT/DPO/evaluation validates `dataset_metadata.json` and rejects
  stale/incomplete Oracle data. Legacy data requires an explicit diagnostic
  override.

## Verification status

- `python -m compileall -q src scripts tests`: PASS.
- Ruff undefined-name/syntax checks on changed files: PASS.
- Full Torch tests could not run on the Windows host because the temporary
  Torch wheel download timed out twice. This is an environment limitation, not
  a recorded pass.

Run on the existing `uavmllm` server environment before any long job:

```bash
python -m unittest \
  tests.test_multimodal_sequence_budget \
  tests.test_multimodal_dpo \
  tests.test_association_solver \
  tests.test_channel_model \
  tests.test_q_geometry_branch \
  tests.test_power_branch \
  tests.test_training_branch_freeze \
  tests.test_delta_diagnostics \
  -v
```

Then generate only a v5 train20/val20 preflight. Do not reuse the old v4
train500/val100 for mainline training.

