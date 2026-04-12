# AGENTS.md

This file provides guidance to Codex and other agentic coding tools when working with code in this repository.

## Project Overview

**SlideReasoner** is a pathology Whole-Slide Image (WSI) reasoning project built around an agent-style zoom-and-inspect workflow.

The intended product direction is:

- open one WSI at a time
- create an initial thumbnail observation
- let an LLM reason over the current observations
- let the model request new ROIs through `zoom_in_image`
- save each ROI as a new observation and continue until enough evidence is collected

This repository is **mid-refactor**. The architecture is visible and coherent, but the current codebase mixes newer WSI-specific logic with older notebook-derived agent code. Treat the repository as a research codebase with one emerging mainline, not as a fully stabilized package.

## Source Of Truth

When the repo appears inconsistent, use this priority order:

1. This `AGENTS.md`
2. `CLAUDE.md`
3. The current modular files under `slidereasoner/`
4. Notebooks and ad-hoc scripts as historical reference only

`README.md` is currently minimal and should not be treated as authoritative.

## Repository Map

- `slidereasoner/agent/`
  - `wsi_agent.py`: intended main WSI agent implementation
  - `crop_agent.py`: older crop-only agent prototype with legacy globals and inconsistencies
  - `_slide_agent_base.py`: incomplete shared base prototype
- `slidereasoner/utils/`
  - `image_utils.py`: WSI/image geometry, MPP inference, bbox validation, ROI extraction, resizing helpers
  - `print_utils.py`: multimodal console/Jupyter trace printing
  - `prompt_utils.py`: tiny prompt file loader
  - `agent_utils.py`: tool-result conversion helpers plus older prompt-loading leftovers
  - `logging_utils.py`: basic logger setup
- `slidereasoner/built_in_prompt/`
  - `prompt_wsi_agent.md`: current prompt contract for the WSI reasoning loop
- `slidereasoner/Formatter/`
  - `_slidereason_formatter.py`: custom OpenAI-style formatter with image promotion logic
- `slidereasoner/workspace/`
  - runtime output area for saved observation images; currently empty
- `test/`
  - ad-hoc scripts and notebooks, not a formal automated test suite
- `reasource/`
  - reference/demo image assets
- `result/`
  - generated outputs; currently empty in this checkout
- `datasets/`
  - large user data area; do not traverse or modify by default

## Mainline Architecture

The intended stable flow is centered on `slidereasoner/agent/wsi_agent.py` plus `slidereasoner/utils/image_utils.py`.

### WSI Agent Flow

1. `start_wsi_session(...)`
   - opens one slide with OpenSlide
   - infers level-0/native MPP
   - creates and saves a thumbnail observation
   - injects the thumbnail plus task text into agent memory

2. `reply(...)`
   - runs a ReAct-style reasoning/acting loop
   - optionally registers `generate_response` for structured output
   - stops when the model answers directly or emits the finish tool

3. `zoom_in_image(...)`
   - takes `observation_index`, `bbox_2d`, and target scale
   - maps the relative box back to slide-native coordinates
   - reads a new ROI from the WSI
   - saves the ROI as a new observation image
   - promotes the image back into memory as a user-visible observation

4. `print_multimodal_trace(...)`
   - prints text/tool/image traces in terminal or Jupyter-friendly form

### Geometry And Scale Rules

These are core invariants and should stay consistent across edits:

- bbox coordinates are on a **0-1000 relative scale**
- coordinates are defined in the **parent observation image space**
- internal WSI geometry is resolved against **OpenSlide level 0 / native resolution**
- the project uses canonical pathology magnification labels:
  - `0.5x -> 20.0`
  - `2x -> 5.0`
  - `5x -> 2.0`
  - `10x -> 1.0`
  - `20x -> 0.5`
  - `40x -> 0.25`
- Qwen-style image sizing assumes dimensions aligned to multiples of `32`

### Supporting Modules

- `image_utils.py` is the geometry backbone. Any change to ROI extraction, bbox validation, MPP handling, or pixel-budget logic should be coordinated there first.
- `prompt_wsi_agent.md` defines the model-facing contract. If tool signatures or observation semantics change, update the prompt at the same time.
- `_slidereason_formatter.py` handles OpenAI-compatible message conversion and promotion of tool-generated images into user messages.

## Current Stability Assessment

### Intended Stable Direction

The repo is clearly moving toward:

- `slidereasoner/agent/wsi_agent.py`
- `slidereasoner/utils/image_utils.py`
- `slidereasoner/built_in_prompt/prompt_wsi_agent.md`
- `slidereasoner/utils/prompt_utils.py`
- `slidereasoner/utils/print_utils.py`

If you need to extend or repair the project, prefer pushing this path forward instead of reviving notebook-local logic.

### Known Breakpoints

These issues are real and should be assumed until explicitly fixed:

- `wsi_agent.py` mixes two naming schemes:
  - dataclass fields are declared as `native_x/native_y/native_w/native_h`
  - session and zoom code instantiate and read `level0_x/level0_y/level0_w/level0_h`
- `wsi_agent.py` currently references helper names that are not defined in that module:
  - `_validate_relative_bbox`
  - `get_roi_at_mpp_optimized`
  - `_validate_patch_pixels_for_wsi`
- `wsi_agent.py` initializes `self.observation_meta` as a dict in `__init__`, then later uses it as a list
- `crop_agent.py` still depends on notebook-era globals and misspelled identifiers such as:
  - `obersvaton_list`
  - `work_dir`
  - `action_idx`
  - `validate_MinMax_pixels_test`
- `_slide_agent_base.py` is not complete and references missing state such as `analysis_prompt`
- there is no dependency manifest (`requirements.txt`, `pyproject.toml`, `environment.yml`, etc.)
- there is no formal CI or automated unit-test structure in this checkout

Because of these mismatches, do **not** assume the modular code currently runs end-to-end without reconciliation.

## Notebooks And Scripts

The notebooks are important historical context, but they are not equal in status:

- `slidereasoner/test_v2.ipynb`
  - older notebook snapshot
  - contains duplicated utility and tool code inline
  - useful for recovering earlier working behavior
- `slidereasoner/test_v3.ipynb`
  - newer exploratory notebook
  - closer to the modular direction, but still experiment-oriented
- `test/Qwen_VL_A800_MPP10.py` and `test/test_speed.py`
  - standalone experiment scripts
  - talk to OpenAI-compatible endpoints through custom base URLs
  - good for understanding prompt/API usage, not for defining architecture

If code and notebook behavior disagree, prefer the modular direction but inspect the notebook to recover intent before editing.

## Data Boundaries

The user explicitly treats most repository data as personal data. Work conservatively.

- Do not scan or summarize the full `datasets/` tree unless explicitly asked.
- If sampling is necessary, inspect only the approved subpath and only a few files.
- Do not edit `datasets/`, `result/`, or runtime workspaces unless the user explicitly asks.

The approved sample path examined so far is:

- `datasets/Clv1_Vev2_Sev2_ReRev3_32B_MPP10/`

Observed sample pattern from a few files:

- one case per JSON file
- fields include:
  - case/report/sample identifiers
  - raw `report_text`
  - `manifest`
  - `verify_clinical_with_clinical`
  - `image_select`
  - `wsi_relative_specimens`
  - `wsi_nonrelative_specimens`
  - `remove_mult`
  - `remove_reduant`
  - token/time accounting fields

This dataset looks like a downstream structured pathology-report processing product, not the direct runtime input format of `WSIReActAgent`.

## Safe Working Rules For Codex

- Prefer small, surgical edits over broad refactors.
- Before adding features, reconcile naming mismatches in the WSI mainline first.
- Keep the 0-1000 bbox convention unchanged unless the entire stack is updated together.
- Keep prompt/tool signatures synchronized.
- Do not silently migrate notebook experiments into package code without checking whether they represent the latest intent.
- Preserve user data, generated artifacts, and existing local changes.
- When unsure whether a file is mainline or legacy, assume `wsi_agent.py` and `image_utils.py` are the target direction, and `crop_agent.py` plus notebook-local copies are legacy reference.

## Recommended Repair Order

If future work is intended to stabilize the project, the safest order is:

1. unify `native_*` vs `level0_*` field names in `wsi_agent.py`
2. reconcile missing helper names between `wsi_agent.py` and `image_utils.py`
3. decide whether `observation_meta` is list-based or dict-based and keep it consistent
4. confirm one stable runtime entrypoint
5. only then extend features or prompts

## Verification Expectations

Before claiming a WSI-agent change is stable, try to verify at least:

- the edited Python modules import cleanly
- helper names referenced by `wsi_agent.py` actually exist
- prompt text matches current tool arguments
- no new edit depends on scanning or mutating `datasets/`

If runtime verification is blocked by environment or data availability, say so explicitly.
