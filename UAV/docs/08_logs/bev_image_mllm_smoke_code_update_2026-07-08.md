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
