# 2026-07-21 P0 control input contract and state diagnostics

## Problem

Schema-v5 Q/A/P control-only SFT omitted response tokens when
`lambda_lm_ce == 0`.  `train_sft_mm.py` incorrectly reused that condition to
disable the Gemma multimodal chat template.  As a result, control-only SFT used
raw image-token plus prompt text while response SFT and DPO used the
instruction-tuned chat layout.  Diagnostics and forward smoke also defaulted
to raw prompts, so a checkpoint could be trained and evaluated under
inconsistent input contracts.

The existing delta diagnostic only reported per-dimension control-state
standard deviation.  That statistic cannot detect states that all point in
nearly the same high-dimensional direction, which previously appeared as
nearest-neighbor cosine values above 0.999.

## Repair

- Added one centralized `resolve_multimodal_chat_template` rule.
  - Fresh schema-v5 runs default to the multimodal chat template.
  - Existing checkpoints preserve `metadata.json::use_chat_template`.
  - Explicit overrides remain available for controlled A/B diagnostics.
- Decoupled response inclusion from prompt formatting in multimodal SFT.
- Stored the resolved input format in every intermediate and final checkpoint.
- Applied the same rule to delta analysis, forward smoke, solver evaluation,
  and sequence-length analysis.
- Added raw and mean-centered control-state nearest-neighbor cosine,
  near-duplicate pair count, and centered effective rank to delta diagnostics.
- Added warnings for nearly indistinguishable and low-rank control states.
- Added unit coverage for v5 defaults, legacy checkpoint compatibility,
  explicit diagnostic overrides, and collapsed/distinct state summaries.

## Validation performed locally

- `python -m py_compile` passed for every modified Python entry point.
- `git diff --check` passed.
- Dependency-free runtime tests in `tests.test_training_runtime` passed.
- Torch/NumPy-dependent tests could not run in the local Windows Python because
  that interpreter does not have `torch` or `numpy`; they must be run in the
  server `uavmllm` environment before any training.

## Runtime gate before training

1. Run the focused unit suite on the server.
2. Compare the same frozen checkpoint and dataset under raw versus chat
   formatting.  This isolates input formatting from optimization and LoRA.
3. Continue to a fresh 30-step Q-only run only if the chat-formatted control
   states are measurably distinguishable.  Otherwise stop and repair the
   control-token representation path first.

No LoRA experiment is authorized by this repair alone.
