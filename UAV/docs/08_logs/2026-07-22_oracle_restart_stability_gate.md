# 2026-07-22 Oracle restart stability gate

## Motivation

The compact-64 cached Q-cue probe memorized 20 training environments but
underperformed a fixed mixture on the independent validation seed.  Reducing
the bottleneck to 8 produced underfit.  Increasing model or LoRA capacity at
this point would not identify whether the supervision itself is stable.

For each environment, schema-v5 data selects the highest-utility feasible
solution from three deterministic random restarts of the constraint-aware
solver.  Different local optima can have nearly identical utility but very
different Q directions.  A single hard direction or derived cue label is then
ambiguous even though every candidate is physically valid.

## Added diagnostic

`scripts/analyze_oracle_restart_stability.py` reconstructs the exact stored
environments and restart seeds and reports:

- stored-target reproducibility;
- top-versus-second relative utility gap;
- pairwise 3D and XY Q-direction cosine;
- derived geometry-cue agreement across restarts;
- the ratio of environments with near-equal utility but divergent Q.

The script is a data-quality gate only.  It is not part of model training or
inference.  If restarts are unstable, the next repair must canonicalize or
soften Oracle Q supervision rather than add model capacity.
