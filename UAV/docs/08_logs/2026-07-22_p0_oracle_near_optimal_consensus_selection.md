# P0: Oracle near-optimal consensus target selection

Date: 2026-07-22

## Why this repair is required

The restart-stability gate reproduced every stored target exactly, and all
three restarts were feasible. The failure is therefore not random dataset
corruption or an infeasible solver output.

On five training environments, the mean top-versus-second utility gap was
0.00222 while the mean restart Q 3-D cosine was 0.68265 and cue agreement was
0.70. On five independent validation environments, the corresponding values
were 0.00237, 0.77787 and 0.83333, with one of five environments classified as
near-equal-utility but direction-divergent. A highest-utility-only rule was
turning sub-percent optimizer differences into different hard Q labels.

## Repair

- Solver revision is bumped from 1 to 2. Revision-1 datasets and checkpoints
  cannot silently enter the repaired mainline.
- Feasible candidates within 1% of the best utility are retained.
- The canonical label is the real candidate with the highest mean per-UAV 3-D
  direction cosine to the other near-optimal candidates (the Q medoid).
- Exact ties are resolved by utility and then deterministic candidate order.
- Q, A and P always come from the same selected solver solution. No averaged
  position or synthetic possibly-infeasible tuple is created.
- Both the generic Oracle generator and the multimodal BEV generator call the
  same selector.
- Dataset metadata records the selection mode and tolerance. Every SFT/DPO
  record stores candidate count, selected utility rank, relative utility gap,
  selected consensus, best-utility consensus and consensus gain.
- The restart audit reconstructs the same revision-2 selection instead of
  assuming that the highest-utility restart is the stored label.

## Tests and gates

Added tests cover:

- choosing a lower-ranked but more representative real candidate;
- excluding candidates outside the utility tolerance;
- recording selection semantics in the immutable dataset contract;
- refusing an in-place resume when the selection tolerance changes.

Local `py_compile` passed. Local unit-test collection is blocked because the
Windows Python environment has no NumPy; run the targeted tests in the server
`uavmllm` environment before regenerating data.

## Required next experiment

Generate new revision-2 train/validation smoke datasets in new directories.
Do not overwrite or resume revision-1 data. Audit record-level feasibility,
utility loss and consensus gain before starting any Q-head or LoRA training.
