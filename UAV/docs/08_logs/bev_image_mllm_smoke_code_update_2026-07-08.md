# BEV-Image MLLM Smoke Code Update Log

> Date: 2026-07-08  
> Scope: Continue the project according to `docs/09_code_modification_plans`, focusing on the RTX 5090 32GB BEV-image MLLM minimal smoke path.  
> Principle: Add the multimodal branch without breaking the existing text-grid SFT/DPO baseline.

---

## 1. Context

The paper requires a true MLLM path:

```text
communication summary + sensing summary + BEV image
  -> multimodal backbone / processor
  -> control-token hidden states
  -> projection head
  -> delta_q / delta_a / delta_p
  -> SCA-FP warm-start
```

The current implemented baseline is still text-grid based:

```text
Gemma3 text-only input
BEV grid serialized into prompt text
control tokens
projection head
SFT / DPO
```

The 09 planning documents recommend preserving that baseline and adding a separate BEV-image MLLM branch. This update implements the first code slice of that plan: BEV image data generation and processor smoke checks.

---

## 2. Files Added

### 2.1 BEV renderer

Added:

```text
src/env/bev_renderer.py
```

Purpose:

```text
Render UAV / user / target geometry into a simple BEV PNG.
Use fixed axes over the service area.
Use stable visual markers:
  UAV: blue triangle
  users: green dots
  targets: red X markers
  optional current association lines
  optional UAV coverage circles
```

Design notes:

```text
No text-heavy legend.
No decorative gradients.
No complex background.
The image is intended for spatial geometry, not presentation.
```

### 2.2 Multimodal smoke data generator

Added:

```text
scripts/generate_mm_smoke.py
```

Purpose:

```text
Generate a small BEV-image multimodal smoke dataset.
Reuse existing scenario generator, SCA-FP solver, oracle prior extraction, and JSON response format.
Write prompt_type="multimodal_bev_image" and relative bev_image_path into JSONL.
```

Output layout:

```text
/root/autodl-tmp/data/mm_smoke/
  sft_dataset.jsonl
  dpo_dataset.jsonl
  checkpoint.txt
  images/
    env_000000.png
    env_000001.png
```

### 2.3 Multimodal processor smoke

Added:

```text
scripts/smoke_mm_processor.py
```

Purpose:

```text
Read one multimodal JSONL sample.
Open the referenced BEV image.
Load AutoProcessor for the configured model.
Encode text + image.
Append control tokens.
Locate control tokens by token id.
Print input_ids, attention_mask, and image tensor shapes.
```

Important check:

```text
control_token_count must equal model.control_token.num_tokens.
```

### 2.4 RTX 5090 multimodal smoke config

Added:

```text
configs/rtx5090_multimodal_smoke.yaml
```

Purpose:

```text
Minimal 32GB smoke configuration.
Not intended for final multimodal training quality.
```

Key settings:

```text
use_4bit: true
freeze_vision_tower: true
max_seq_length: 1024
num_environments: 20
num_restarts: 3
image_size: 224
use_bev_text_grid: false
use_bev_image: true
```

---

## 3. Files Modified

### 3.1 Environment sample schema

Modified:

```text
src/env/isac_scenario.py
```

Change:

```python
bev_image_path: Optional[str] = None
```

Reason:

```text
Keep old text-grid samples compatible while allowing multimodal samples to carry a rendered BEV image path.
```

### 3.2 Env package exports

Modified:

```text
src/env/__init__.py
```

Change:

```python
from .bev_renderer import render_bev_image, render_bev_sample
```

Reason:

```text
Expose the BEV renderer through the existing env package.
```

### 3.3 Prompt builder

Modified:

```text
src/data/prompt_builder.py
```

Added:

```python
build_multimodal_prompt(env_sample, config)
```

Behavior:

```text
Preserve build_full_prompt() for the text-grid baseline.
For multimodal samples, keep communication and sensing summaries as text.
Replace full BEV text grid with a short description of the attached BEV image.
Do not hard-code model-specific image placeholders.
```

Reason:

```text
The image placeholder format depends on the actual processor / chat template.
The prompt builder should not guess it before processor smoke verification.
```

---

## 4. Validation Performed

### 4.1 Syntax checks

Passed:

```bash
python -m py_compile \
  scripts/smoke_mm_processor.py \
  scripts/generate_mm_smoke.py \
  src/env/bev_renderer.py \
  src/env/isac_scenario.py \
  src/data/prompt_builder.py
```

### 4.2 ASCII check for newly added files

Passed:

```text
scripts/smoke_mm_processor.py
scripts/generate_mm_smoke.py
src/env/bev_renderer.py
configs/rtx5090_multimodal_smoke.yaml
```

All newly added files are ASCII-only.

### 4.3 Local runtime limitation

Local import/runtime smoke could not be completed because the current local Python environment does not have `numpy` installed:

```text
ModuleNotFoundError: No module named 'numpy'
```

This is an environment limitation on the local machine, not a syntax error. The scripts should be run in the project training environment where `requirements.txt` dependencies are installed.

---

## 5. Recommended Server Commands

### 5.1 Generate BEV-image smoke data

```bash
cd /root/Projects/UAV/UAV
conda activate uavmllm

python scripts/generate_mm_smoke.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --output_dir /root/autodl-tmp/data/mm_smoke \
  --num_samples 20 \
  --num_restarts 3 \
  --overwrite
```

Expected checks:

```bash
ls -lh /root/autodl-tmp/data/mm_smoke
ls -lh /root/autodl-tmp/data/mm_smoke/images | head
head -1 /root/autodl-tmp/data/mm_smoke/sft_dataset.jsonl
```

### 5.2 Processor smoke

```bash
python scripts/smoke_mm_processor.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke
```

Expected output:

```text
input_ids shape
attention_mask shape
pixel_values or processor-specific image tensor shape
control_token_count: 8
```

---

## 6. Code Review Notes

The following issues were observed during review but not changed in this update, because they affect existing baseline behavior and should be handled as separate, tested changes.

### 6.1 Association discretization does not enforce capacity

File:

```text
src/model/projection_head.py
```

Observation:

```text
AssociationProjection.discretize() currently uses per-user argmax.
The comment mentions capacity post-processing, but the implementation does not enforce K_max.
```

Risk:

```text
Inference-time discrete association can violate per-UAV load capacity.
This is not fully aligned with the paper's CapAssign step.
```

Suggested follow-up:

```text
Implement a capacity-aware assignment step, preferably min-cost flow or a deterministic greedy fallback with tests.
```

### 6.2 Non-square area sampling caveat

File:

```text
src/env/uav_network.py
```

Observation:

```text
User cluster centers are sampled with scalar low/high values.
This is fine for the default 1000 x 1000 area, but can bias sampling if area_w != area_h.
```

Suggested follow-up:

```text
Use separate x/y ranges for non-square service areas.
```

### 6.3 Power floor projection edge case

File:

```text
src/model/projection_head.py
```

Observation:

```text
PowerProjection applies a communication power floor before returning the original sensing component.
In extreme cases, communication floor plus sensing power may exceed P_max.
```

Suggested follow-up:

```text
Either mask non-associated communication beams before projection, or renormalize the full communication+sensing vector after floor handling.
```

---

## 7. Current Status

Completed in this update:

```text
BEV renderer added.
Multimodal prompt builder added.
Multimodal smoke data generator added.
Multimodal processor smoke script added.
RTX 5090 multimodal smoke config added.
Syntax checks passed.
```

Not completed yet:

```text
MultimodalSFTDataset
Gemma3MultimodalISAC model wrapper
smoke_mm_forward.py
train_sft_mm.py
evaluate_mm.py
capacity-aware CapAssign implementation
```

Next recommended engineering step:

```text
Implement MultimodalSFTDataset and a single-batch smoke_mm_forward.py.
After forward smoke passes, add train_sft_mm.py for 10-30 step multimodal SFT smoke.
```

One-line summary:

```text
The project now has the first BEV-image MLLM smoke slice in place: image rendering, multimodal prompts, smoke data generation, and processor-level validation hooks, while preserving the existing text-grid baseline.
```

---

## 8. Server Smoke Results

> Date: 2026-07-12  
> Server path: `/root/Projects/UAV/UAV`  
> Data path: `/root/autodl-tmp/data/mm_smoke`  
> Model path: `/root/autodl-tmp/huggingface/models/gemma-3-4b-it`

### 8.1 BEV-image data generation

Command:

```bash
python scripts/generate_mm_smoke.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --output_dir /root/autodl-tmp/data/mm_smoke \
  --num_samples 20 \
  --num_restarts 3 \
  --overwrite
```

Result:

```text
sft_dataset.jsonl generated
dpo_dataset.jsonl generated
images/env_000000.png ... generated
checkpoint.txt generated
```

Observed files:

```text
/root/autodl-tmp/data/mm_smoke/sft_dataset.jsonl
/root/autodl-tmp/data/mm_smoke/dpo_dataset.jsonl
/root/autodl-tmp/data/mm_smoke/images/env_000000.png
```

The JSONL samples contain:

```text
prompt
response
bev_image_path
prompt_type="multimodal_bev_image"
q_current
delta_q / delta_a / delta_p
```

### 8.2 Processor smoke

Initial issue:

```text
ValueError: Prompt contained 0 image tokens but received 1 images.
```

Cause:

```text
Gemma3 processor requires a model-specific image token in the text prompt.
The script was patched to inject the Gemma BOI image token before the
[Bird's-Eye-View Image] marker.
```

Second issue:

```text
Mismatch in image token count between text and input_ids.
Likely due to truncation='max_length'.
```

Cause:

```text
The original max_seq_length=1024 was too short after Gemma3 expanded the image
tokens. The smoke was rerun with max_length=4096, then the config was updated
to max_seq_length=3072 for the next stage.
```

Successful command:

```bash
python scripts/smoke_mm_processor.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_length 4096
```

Successful output:

```text
OK: multimodal processor smoke
  data: /root/autodl-tmp/data/mm_smoke/sft_dataset.jsonl
  image: /root/autodl-tmp/data/mm_smoke/images/env_000000.png size=(224, 224)
  input_ids: (1, 2025)
  attention_mask: (1, 2025)
  token_type_ids: (1, 2017)
  pixel_values: (1, 3, 896, 896)
  control_token_count: 8
```

Conclusion:

```text
Prompt + BEV image can be encoded by the local Gemma3 processor.
Control tokens are correctly appended and located.
1024 tokens is insufficient; 3072 is the current smoke default.
Gemma3 internally converts the 224 x 224 BEV image to pixel_values shape
(1, 3, 896, 896), which increases multimodal memory pressure.
```

### 8.3 Multimodal model forward smoke

Added files used in this smoke:

```text
src/data/multimodal_dataset.py
src/model/gemma_multimodal_isac.py
scripts/smoke_mm_forward.py
```

Command:

```bash
python scripts/smoke_mm_forward.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_length 3072
```

Successful output:

```text
OK: multimodal model forward smoke
  data: /root/autodl-tmp/data/mm_smoke/sft_dataset.jsonl
  max_length: 3072
  input_ids: (1, 3072)
  attention_mask: (1, 3072)
  pixel_values: (1, 3, 896, 896)
  control_token_count: 8
  control_states: (1, 8, 2560)
  delta_q: (1, 4, 3)
  delta_a: (1, 4, 20)
  delta_p: (1, 4, 21)
```

Conclusion:

```text
The BEV-image MLLM minimal forward loop is now validated:

BEV image + text prompt
  -> Gemma3 multimodal processor
  -> Gemma3 multimodal model
  -> control-token hidden states
  -> projection head
  -> delta_q / delta_a / delta_p
```

This confirms that the solver-facing interface remains unchanged:

```text
delta_q: (B, M, 3)
delta_a: (B, M, K)
delta_p: (B, M, K+1)
```

### 8.4 Current milestone status

Completed:

```text
Step 1: generate_mm_smoke.py       PASS
Step 2: smoke_mm_processor.py      PASS
Step 3: smoke_mm_forward.py        PASS
```

Next step:

```text
Step 4: multimodal SFT smoke
```

Step 4 acceptance targets:

```text
10-30 training steps complete
no OOM
no NaN
loss_ctl has numeric values
grad_norm has numeric values
checkpoint can be saved
```

### 8.5 Multimodal SFT smoke

Added file used in this smoke:

```text
src/training/train_sft_mm.py
```

Training mode:

```text
projection-head-only CTL smoke
Gemma3 multimodal backbone frozen
vision tower frozen
no token-level CE loss
no LoRA update yet
```

Rationale:

```text
This is the lowest-risk RTX 5090 32GB training smoke. It validates the training
shell around the already-passing multimodal forward path before adding LoRA
or token-level SFT losses.
```

Command:

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_steps 10 \
  --max_length 3072
```

Observed training output:

```text
step=1  loss_ctl=72.270164   grad_norm_proj=348.923516
step=2  loss_ctl=92.060333   grad_norm_proj=354.783184
step=3  loss_ctl=68.496063   grad_norm_proj=295.408330
step=4  loss_ctl=83.245941   grad_norm_proj=305.034217
step=5  loss_ctl=56.014587   grad_norm_proj=239.805846
step=6  loss_ctl=76.060646   grad_norm_proj=263.420681
step=7  loss_ctl=71.136230   grad_norm_proj=248.398176
step=8  loss_ctl=107.080132  grad_norm_proj=281.475067
step=9  loss_ctl=75.074249   grad_norm_proj=246.545792
step=10 loss_ctl=76.983398   grad_norm_proj=244.714702
```

Final output:

```text
OK: multimodal SFT smoke complete
final_checkpoint: /root/autodl-tmp/outputs/mm_smoke/mm_sft_smoke_final
```

Checkpoint paths:

```text
/root/autodl-tmp/outputs/mm_smoke/mm_sft_smoke_final
/root/autodl-tmp/checkpoints/mm_smoke/mm_sft_smoke_step_10
```

Result:

```text
10 training steps completed.
No OOM observed.
No NaN observed.
loss_ctl produced numeric values.
grad_norm_proj produced numeric values.
Final checkpoint was saved.
```

Updated milestone status:

```text
Step 1: generate_mm_smoke.py              PASS
Step 2: smoke_mm_processor.py             PASS
Step 3: smoke_mm_forward.py               PASS
Step 4: train_sft_mm.py, 10-step smoke    PASS
```

Current conclusion:

```text
The BEV-image MLLM minimal training loop is now validated at projection-head
level. The project has moved beyond data/processor/forward smoke and can now
test longer CTL smoke or a LoRA-enabled multimodal SFT smoke.
```

Recommended next options:

```text
1. Run projection-head-only smoke for 30 steps to check stability.
2. Add a delta diagnostic path for mm_sft_smoke_final.
3. Add LoRA-enabled multimodal SFT smoke after confirming memory headroom.
```

### 8.6 Multimodal SFT smoke, 30-step stability check

Command:

```bash
python src/training/train_sft_mm.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --max_steps 30 \
  --max_length 3072
```

Training mode:

```text
projection-head-only CTL smoke
Gemma3 multimodal backbone frozen
vision tower frozen
no token-level CE loss
no LoRA update yet
```

Observed output summary:

```text
Loading weights completed.
30 / 30 training steps completed.
Runtime: about 26 seconds.
Throughput: about 1.12 it/s.
No OOM observed.
No NaN observed.
```

Selected metrics:

```text
step=1  loss_ctl=72.270164   grad_norm_proj=348.923516
step=10 loss_ctl=76.983398   grad_norm_proj=244.714702
step=20 loss_ctl=65.143181   grad_norm_proj=201.112136
step=30 loss_ctl=69.149162   grad_norm_proj=199.785992
```

Metric trend:

```text
loss_ctl remains noisy across the 20-sample smoke dataset, which is expected
for projection-head-only training with shuffled small data.

grad_norm_proj remains finite and generally decreases from the initial
~350 range toward the ~200 range, indicating that backward and optimizer
updates are functioning.
```

Final status:

```text
OK: multimodal SFT smoke complete
```

Updated milestone status:

```text
Step 1: generate_mm_smoke.py                    PASS
Step 2: smoke_mm_processor.py                   PASS
Step 3: smoke_mm_forward.py                     PASS
Step 4a: train_sft_mm.py, 10-step smoke         PASS
Step 4b: train_sft_mm.py, 30-step stability     PASS
```

Current conclusion:

```text
The projection-head-only BEV-image MLLM training smoke is stable for 30 steps
on the RTX 5090 setup. The next meaningful engineering step is either to add
delta-output diagnostics for the multimodal smoke checkpoint or to enable a
small LoRA training smoke.
```

### 8.7 Multimodal delta diagnostic

Command:

```bash
python scripts/analyze_mm_delta_outputs.py \
  --config configs/rtx5090_multimodal_smoke.yaml \
  --data_dir /root/autodl-tmp/data/mm_smoke \
  --model /root/autodl-tmp/huggingface/models/gemma-3-4b-it \
  --checkpoint /root/autodl-tmp/outputs/mm_smoke/mm_sft_smoke_final \
  --name mm_sft_smoke_30step \
  --num_samples 20 \
  --max_length 3072 \
  --output /root/autodl-tmp/outputs/mm_smoke/delta_diag_mm_sft_smoke_20.json \
  --save_raw
```

Observed summary:

```text
delta_q_per_dim_std_mean: 0.2534600794315338
delta_a_per_dim_std_mean: 0.028002941980957985
delta_p_per_dim_std_mean: 0.008766541257500648
delta_a_argmax_unique_per_user_mean: 1.15
delta_a_entropy_mean: 0.8596800911881917
delta_p_entropy_mean: 1.9475480959227327
warnings: ['delta_a_argmax_nearly_constant']
```

Interpretation:

```text
delta_q has clear cross-sample variation.
delta_p has nonzero cross-sample variation and a smooth power split.
delta_a has soft-value variation, but the argmax UAV choice is nearly fixed.
```

Conclusion:

```text
The projection-head-only multimodal smoke checkpoint does not show global
delta collapse, but association argmax behavior is still too conservative.
This is expected because the Gemma3 backbone is frozen, LoRA is not enabled,
and only 20 smoke samples / 30 projection-head steps were used.
```

Next action:

```text
Enable a small LoRA multimodal SFT smoke with CTL-only loss first.
The goal is not final performance; it is to verify memory, gradients, and
whether backbone adaptation can start improving environment-conditioned
association behavior.
```
