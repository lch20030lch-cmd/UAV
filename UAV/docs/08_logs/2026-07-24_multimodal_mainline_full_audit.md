# 2026-07-24 multimodal mainline full audit

## Decision

Do not start another training run from the existing v3/v4/v5-revision-2
datasets or checkpoints.  The next executable step is one fresh revision-3
two-sample runtime gate.  A long Q, A, P, joint-SFT, or DPO run is not allowed
until that gate passes.

The production architecture remains:

```text
environment
  -> constraint-aware multi-restart Oracle
  -> sealed SFT/DPO records + BEV image
  -> Gemma multimodal chat encoding
  -> eight control-token states
  -> split Q / A / P projection head
  -> feasible warm start
  -> downstream constraint-aware solver
  -> final allocation
```

Q, A, and P isolation runs are unit/learnability gates for the three branches
of the same projection head.  They are not three independently trained final
models to concatenate.  LoRA and control-token offsets are shared
representation parameters, so independently trained copies cannot be merged
without a defined merge rule.  After all three branches pass train and
held-out gates from the same fresh anchor, one joint SFT trains the shared
representation once.

## Blocking defects found and repaired

### Oracle and physical simulation

- The alternating solver could return its last state even when an earlier
  visited state was the best feasible solution.  It now retains the best
  finite feasible state across the full trajectory.
- The effective communication threshold now includes both configured SINR and
  the SINR implied by the configured minimum rate.
- Undetected sensing targets are now excluded consistently from the Oracle,
  utility, constraint checks, prompt geometry, and BEV rendering.
- Sensing SINR used a hard-coded numerical floor that materially disagreed
  with the configured thermal-noise value.  It now divides by the exact
  configured noise.
- The Oracle contract is advanced to solver revision 3 and channel model
  `elevation_los_3gpp_pathloss_v3`.

### Dataset generation and provenance

- A serialized rounded Q/A/P tuple could carry feasibility and utility values
  computed before serialization.  Generation now re-evaluates the exact
  rounded tuple and rejects it when it is no longer feasible.
- Prior validation could silently repair a supplied A/P tuple before
  evaluation.  It now validates and evaluates the exact supplied tuple.
- DPO `utility_chosen`, `utility_rejected`, and `utility_gap` are now computed
  from the exact serialized chosen/rejected records.
- Generation is paired and crash-safe: SFT/DPO rows are journaled together,
  images are written only for accepted samples, and resume rejects inconsistent
  state.
- Completed datasets are sealed with a SHA-256 content fingerprint covering
  both JSONL files and all referenced BEV image bytes.
- Dataset validation now reproduces each environment and prompt from the
  configured seed and environment id; it also rejects missing, duplicated,
  escaped, or orphaned image paths.

### Model inputs, branch isolation, and optimization

- All current-schema SFT, DPO, diagnostics, and smoke paths use the same Gemma
  multimodal chat template and the same eight control-token layout.
- The processor smoke now exercises `MultimodalSFTDataset`, not a separate
  raw-prompt approximation.
- The projection head remains physically split into Q, A, and P branches.
- Branch-freeze flags now also isolate every effective loss.  This prevents an
  unrelated A or P loss from updating shared LoRA/control offsets during a
  nominal Q-only run, and vice versa.
- Trainable control-token offsets are saved and loaded as an explicit
  checkpoint component.
- Projection, control-offset, and LoRA gradients are clipped independently and
  logged before and after clipping.
- When the vision tower is frozen, its parameters and modules remain frozen and
  in evaluation mode even after the language model is switched to training
  mode.  Vision LoRA is forbidden in this configuration.

### Checkpoints, DPO, diagnostics, and evaluation

- Current-schema training checkpoints require exact projection-head,
  control-embedding, control-offset, dataset, and mode provenance.
- Held-out diagnostics require the same physical contract but may use a
  different seed/content fingerprint.  Training resume requires exact dataset
  identity.
- DPO now stores its own runtime progress instead of inheriting ambiguous
  Stage-I progress fields; optimizer-step resolution, scheduler metadata, and
  checkpoint rotation are explicit.
- Q/A/P diagnostics include collapse, baseline, feasibility, control-state
  diversity, and held-out metrics.
- Restart stability reports the full restart set and the near-optimal subset,
  including stored-versus-reproduced Q/A/P checks.
- `scripts/audit_mm_mainline.py` is the fail-fast executable contract for the
  configuration, exact dataset contents, serialized Oracle solutions, image
  references, DPO utility ordering, and optional checkpoint.

## Required branch-gate order

Every isolated gate starts from the same fresh revision-3 anchor.

1. Q head only: no LoRA and no control offsets.  This verifies the Q head,
   target, projection, and loss wiring.
2. A head only: no LoRA and no control offsets.  This verifies association
   indexing, targets, logits, and loss wiring.
3. P head only: no LoRA and no control offsets.  This verifies active/inactive
   masking, sensing power, simplex projection, and raw-KL wiring.
4. Only if a head can fit training targets but fails held-out targets, open the
   compact control offsets for that diagnostic.
5. Only after the compact adapter is insufficient, open language LoRA while
   keeping vision parameters and vision LoRA frozen.
6. After all three branches pass, run joint SFT once.  Run DPO only after joint
   SFT and solver evaluation pass.

This order distinguishes a broken head/loss from a representation bottleneck
and avoids using LoRA to hide a wiring defect.

## Metrics that gate progress

- Data/Oracle: exact content fingerprint, reproducible prompt/environment,
  `oracle_feasible=true`, maximum constraint violation at or below `1e-5`,
  positive recomputed DPO utility gap, no hidden-target leakage.
- Q: zero mobility violations, correct norm/bounds, train loss reduction,
  held-out 3D/XY cosine improvement over its untrained anchor.  A geometry
  baseline may be reported but is not silently substituted for learned Q.
- A: argmax accuracy and top-2 accuracy, gain over fixed-user majority,
  balanced prediction histogram, no fixed-user collapse, train versus held-out
  gap.
- P: total power budget, inactive leakage, active/sensing MSE, raw KL, entropy,
  cross-sample variance, train versus held-out gap.
- Representation: centered effective rank, centered nearest-neighbor cosine,
  per-dimension standard deviation, and duplicate-pair count.
- Runtime: no NaN/Inf/OOM, branch and loss metadata agree, only intended
  tensors are trainable, independent post-clip gradient norms are bounded.

## Artifact policy

All v3, v4, and v5 solver-revision-2 datasets/checkpoints are historical
diagnostics and must not initialize revision-3 training.  Do not delete them
until:

1. the revision-3 two-sample data/runtime gate passes;
2. a fresh revision-3 checkpoint can be saved, loaded, and diagnosed; and
3. its exact paths are recorded.

After those conditions, inventory candidate directories with `du -sh` and
delete only explicit obsolete paths.  Never delete the current selected
revision-3 anchor or the only copy of a passing diagnostic.

## Local verification

- `python -m compileall -q src scripts tests`: PASS.
- `git diff --check`: PASS.
- Dependency-free runtime helper tests: PASS.
- The Windows audit host does not have the server's NumPy/Torch/SciPy/Pillow
  environment, so the complete test suite and model forward must run once in
  the `uavmllm` server environment before training.

## Runtime-gate follow-up

The first server gate passed 98 tests and then exposed a configuration-boundary
defect before writing any records: PyYAML parsed `rate_min_bps: 1e6` as a
string, while the solver expected a float.  The repair:

- writes scientific notation as `1.0e6` in every maintained YAML config;
- canonicalizes and validates every physical simulation field before hashing
  or constructing the scenario/solver;
- makes the solver defensive when called directly with a string rate;
- casts numeric prompt fields explicitly; and
- adds regression coverage for YAML-style numeric strings through both the
  contract and the real runtime solver builder.
