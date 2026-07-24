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

After that repair, 100 tests and two-sample generation passed.  The exact-data
audit then found that rejected DPO utility was evaluated before the rejected
response was rounded to six JSON decimals.  Chosen and rejected priors now
follow one authoritative path:

```text
candidate arrays -> serialize JSON once -> parse stored JSON values
                 -> evaluate utility/constraints -> write record
```

Both the record tensors and their utility fields therefore describe the exact
same numeric tuple.  The invalid two-sample gate dataset must be regenerated;
changing its sealed JSONL fields in place is forbidden.

The regenerated revision-3 gate subsequently passed end to end:

- 101 tests: PASS;
- two paired SFT/DPO records and two BEV images: PASS;
- exact mainline audit: PASS;
- maximum constraint violation: `1.4987809876743086e-06`;
- minimum exact serialized DPO utility gap: `0.2167664679316772`;
- real multimodal control-only length: `2712..2724`, below `3072`;
- chat template and eight control tokens: PASS;
- Gemma forward control states: `(1, 8, 2560)`;
- split outputs Q/A/P: `(1,4,3)`, `(1,4,20)`, `(1,4,21)`.

This closes the non-training runtime gate.  The next permitted experiment is
the two-record Q-head-only overfit gate with LoRA and control offsets disabled.

## Revision-3 Q-head learnability gate

The isolated Q branch was trained on the sealed two-record revision-3 runtime
dataset with:

- split projection head, direct direction mode, and no geometry shortcut;
- only `readout_q` / `q_mlp` trainable;
- LoRA and control-token offsets disabled; and
- only `lambda_q_dir=1`, with every A/P and auxiliary Q loss set to zero.

The first 30-step run reduced the unit-direction MSE from approximately
`0.48` to `0.289619` but exhausted its short cosine schedule before fitting the
two records.  Continuing from that projection checkpoint with a fresh
100-step schedule resolved the apparent underfit:

- step 1: `loss_q_dir=0.289619`;
- step 25: `loss_q_dir=0.064822`;
- step 50: `loss_q_dir=0.000957`;
- step 75: `loss_q_dir=0.000108`;
- step 100: `loss_q_dir=0.000035`;
- all unrelated loss terms remained exactly zero; and
- no NaN, Inf, OOM, or runtime error occurred.

Reloading the final checkpoint reproduced the result:

- predicted/target displacement norm: `14.999998 / 14.999998`;
- mobility violation ratio: `0.0`;
- Q target 3D cosine: `0.999948`;
- Q target XY cosine: `0.999976`;
- raw Q direction cosine: `0.999948`; and
- predicted/target direction standard deviation: `0.334509 / 0.336234`.

Verdict: the direct Q projection branch, direction loss, backward path,
optimizer path, serialization, reload, and inference path pass the two-record
learnability gate.  This establishes implementation capacity and checkpoint
integrity only; it does not establish held-out generalization.  The passing
checkpoint is a diagnostic artifact and must not initialize an independent A
or P branch gate.

## Revision-3 A-head learnability gate

The independent A branch was started from the same clean model/configuration,
not from the diagnostic Q checkpoint.  Only `readout_a` / `a_mlp` were
trainable; LoRA and control-token offsets were disabled.  The initial gate
used only raw association cross-entropy so that classification capacity could
be established before involving the constrained association projection.

Training reduced `loss_a_raw_ce` from `1.386258` (the four-class random
baseline) to `0.013615` in 100 optimizer steps.  Every unrelated loss stayed
at zero and no numerical/runtime error occurred.

Reloading the final checkpoint established both raw and projected behavior:

- raw argmax/top-2 accuracy: `1.0 / 1.0`;
- projected argmax/top-2 accuracy: `1.0 / 1.0`;
- fixed-user-majority baseline: `0.625`;
- projected oracle probability mean: `0.999776`;
- raw oracle probability mean: `0.986536`; and
- predicted and target UAV histograms both equal
  `{'0': 16, '1': 9, '2': 11, '3': 4}`.

Verdict: the A readout, raw classification loss, constrained projection,
checkpoint serialization/reload, and inference path pass the two-record
learnability gate.  The remaining diagnostic warning belongs to the frozen P
branch and is not an A-gate failure.

## Revision-3 P-head learnability gate

The independent P branch was also started from the clean model/configuration.
Only `readout_p` / `p_mlp` were trainable; LoRA and control-token offsets were
disabled.  The gate used the raw power-distribution KL loss, whose temperature
and normalization match the final simplex power projection.

Training reduced `loss_p_raw_kl` from `2.431211` to `0.009261` in 100
optimizer steps.  Inactive communication-power leakage fell from `0.048377`
to `0.000570`; all unrelated losses stayed zero and no numerical/runtime error
occurred.

Reloading the final checkpoint produced:

- predicted/target per-dimension standard deviation:
  `0.040320 / 0.040855`;
- total power MSE: `3.2244e-06`;
- active communication MSE: `4.3110e-06`;
- inactive communication MSE: `7.5868e-07`;
- sensing MSE: `3.4778e-05`;
- inactive power leakage mean: `5.6993e-04`; and
- predicted/target total power mean: `1.0 / 0.999777`, with total-power MAE
  `2.2326e-04`.

Verdict: the P readout, raw KL loss, simplex projection, checkpoint
serialization/reload, and inference path pass the two-record learnability
gate.  The diagnostic warnings belong to the frozen random A branch.

Together, the Q, A, and P branches now pass their independent two-record
implementation gates.  These gates prove wiring, optimization capacity, and
artifact integrity, but not held-out generalization or joint-training
stability.

## Train20 replay precision repair

Fresh revision-3 train20 generation completed, but the strict replay audit
stopped at `env_7` because its stored utility differed from the utility
recomputed from the seeded environment by more than `1e-6`.

The serialized Q/A/P tuple was not the source of this mismatch.  Generation
converted `EnvironmentSample.user_weights` to float32 in
`OracleDataGenerator._env_sample_to_dict`, while the audit and downstream
solver evaluator independently reconstructed a dictionary with the original
float64 weights.  Since communication utility is explicitly weighted by
`user_weights`, the two paths could assign slightly different utilities to
the same otherwise identical record.

The repair adds one authoritative
`environment_sample_to_solver_dict()` conversion in `oracle_runtime.py` and
uses it for:

- Oracle data generation;
- exact dataset replay auditing; and
- downstream warm-start/cold-start solver evaluation.

The helper preserves the generator's established float32 source precision,
so the already generated train20 labels and stored utilities are unchanged.
The audit tolerance remains the strict absolute `1e-6`; no mismatch is hidden
by relaxing the gate.  Audit failures now include stored, recomputed, and
difference values.  A regression test covers the canonical user-weight
conversion.

Local `compileall` and `git diff --check` pass.  The Windows audit host lacks
NumPy, so the targeted runtime tests and the existing train20 re-audit must be
run in the server's `uavmllm` environment before val20 generation resumes.

## Revision-3 train20/val20 data gate

The server-side targeted tests and the repaired train20 exact replay audit
passed.  The existing train20 dataset was retained without regenerating or
editing its sealed records.  A fresh seed-2026 val20 dataset was then generated
with three Oracle restarts per environment and passed the same exact audit.

- both datasets contain 20 paired SFT/DPO records and 20 referenced images;
- both use schema 5, solver revision 3, and simulation fingerprint
  `bc6f4a5bd357aba7c0c57bee383bffc6facb83838cdc33b2dd4101d749308940`;
- train and validation content fingerprints differ;
- validation maximum constraint violation is `1.9282e-06`;
- validation minimum DPO utility gap is `0.091405`; and
- train/validation generation-complete flags are true.

Target distributions are aligned at the branch level:

- Q per-dimension standard deviation: `8.2256 / 8.3947`;
- A per-dimension standard deviation: `0.42383 / 0.42260`;
- P per-dimension standard deviation: `0.08467 / 0.09202`;
- inactive communication power mean and nonzero ratio: `0 / 0` on both;
- A fixed-user count: `0 / 0`; and
- A dominant ratio mean: `0.3375 / 0.3550`.

Real Gemma multimodal chat-template lengths, including image expansion and
eight control tokens, are `2697..2735` for train and `2694..2732` for
validation.  Every record fits within the selected `max_length=3072`.

The near-optimal medoid selector also respects its utility contract:

- chosen relative utility gap maximum: `0.005245 / 0.007240`, both below 1%;
- chosen Q consensus mean: `0.8142 / 0.8752`; and
- chosen Q consensus minimum: `0.2988 / 0.5529`.

The low train minimum is not a contract failure, but it identifies at least one
multi-modal near-optimal environment whose hard Q direction is less stable.
That record must be identified and reported before interpreting Q-only cosine
metrics; it must not be silently deleted after seeing model performance.

The low-consensus records are:

- train `env_2`: two candidates, consensus `0.2988`, selected utility rank 1;
- train `env_20`: two candidates, consensus `0.5567`, selected rank 1;
- train `env_9`: three candidates, selected rank 3 at a `0.524%` utility gap,
  improving consensus from `0.2769` to `0.5869`; and
- validation `env_22`: three candidates, selected rank 2 at a `0.240%`
  utility gap, improving consensus from `0.5149` to `0.5529`.

Thus low-consensus multi-candidate records account for 15% of train and 5% of
validation.  They remain below the declared 20% stop threshold and are kept
unchanged.  Full-set Q cosine must be interpreted together with these known
multi-solution cases.

Before the Q train20 fit gate, a loss-isolation audit found that Q isolation
intentionally permits the UAV separation penalty, but the SFT CLI had no way
to override the configured `lambda_sep=0.1`.  The two-record gate happened to
have zero separation penalty; train20 need not.  A `--lambda_sep` override was
therefore added consistently with the other loss overrides.  The formal
Q-only learnability run sets it explicitly to zero, while default joint
training behavior remains unchanged.
