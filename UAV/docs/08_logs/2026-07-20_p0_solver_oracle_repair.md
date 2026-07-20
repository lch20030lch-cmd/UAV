# 2026-07-20 P0 Solver / Oracle repair

## Scope

This change fixes the downstream contract that made learned A/P warm starts
ineffective and made Oracle labels inconsistent with the constraints stated in
the prompt.

## Implemented

- Replaced the misleading pseudo-SCA/FP body with a compatibility-preserving
  `constraint_aware_alternating_optimization` implementation.
- The first deployment update now consumes model-provided Q/A/P before any A/P
  update. Warm-start A/P are no longer immediately overwritten.
- Communication CSI is recomputed after Q moves by scaling the sampled CSI with
  a deterministic large-scale geometry ratio.
- Association initialization and updates use a capacity-constrained assignment.
- Power updates are association-aware, force inactive entries to zero, enforce
  the per-UAV budget, and allocate from communication/sensing SINR requirements
  instead of a fixed 70/30 split.
- Added hard deployment projection for area, altitude, mobility radius and
  minimum UAV separation.
- Added a single constraint report covering association, power, movement,
  boundary, separation, communication SINR and sensing SINR.
- Solver utility, Oracle ranking and end-to-end evaluation now share the same
  geometry-dependent channel model.
- Corrected the LoS probability parameters and removed the duplicated carrier
  frequency term in path loss.
- Oracle generation accepts only feasible solutions. Rejected DPO tuples are
  scored using the exact serialized Q/A/P tuple, rather than the utility of a
  different solver solution.
- New v5 generation uses a checkpointed next environment id, emits paired
  SFT/DPO rows, writes a dataset contract, and refuses unsafe in-place resume
  from pre-v5 data.

## Compatibility decision

The public names `SCAFPOptimizer`, `SCAFPConfig` and `SCAFPSolution` remain for
old imports. New solution metadata explicitly records the actual algorithm;
the implementation must not be described as a formal convex SCA/FP solver in
the paper unless a genuine SCA/FP implementation replaces it later.

## Verification

- `python -m compileall -q src scripts tests`: PASS.
- Ruff undefined-name/syntax checks on changed files: PASS.
- 10 local NumPy/SciPy tests: PASS.
  - warm-start A/P reach the first deployment update;
  - capacity-constrained association;
  - inactive-power structural zero;
  - Q-dependent channel gain;
  - minimum-separation projection;
  - LoS/path-loss invariants.
- One real generated environment with two restarts: Oracle SFT produced,
  `feasible=True`, maximum reported violation `0.0`.

## Required consequence

All v3/v4 Oracle targets were generated under the old solver/channel semantics.
They remain useful only as historical diagnostics. Mainline training must use a
new v5 train/validation dataset generated after this repair.

