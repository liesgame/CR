# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SlideReasoner** is a pathology Whole-Slide Image (WSI) reasoning system. It uses a ReAct (Reasoning + Acting) agent loop built on [AgentScope](https://github.com/modelscope/agentscope) to enable LLM-guided multi-scale exploration of WSI files. The agent can zoom from 0.5x (thumbnail) to 40x (cellular detail) by iteratively requesting regions of interest.

## Key Dependencies

No `requirements.txt` exists. The core dependencies are:
- `agentscope` — agent framework (ReAct loop, memory, tool registration)
- `openslide-python` — WSI file reading
- `Pillow` — image manipulation
- `pydantic` — tool argument validation
- `openai` — LLM API client
- `loguru` — logging

## Architecture

### Agent Layer (`slidereasoner/agent/`)

- **`wsi_agent.py` — `WSIReActAgent`**: The primary agent. Wraps a single WSI file and exposes `zoom_in_image()`, `backtrack()`, and `mark_roi()` as tools. Each call tracks an `ObservationMeta` with parent index, bounding box, and MPP. Implements `reply()` → `reasoning()` → `acting()` loop.
- **`crop_agent.py` — `BaseCropAgent`**: Simpler variant for generic image cropping. Uses global state (`action_idx`, `observation_list`, `work_dir`). Supports parallel tool calls and memory hints (`HINT`/`COMPRESSED` marks).
- **`_slide_agent_base.py`**: Shared base class for both agents.

### Coordinate System

All bounding boxes use a **0–1000 relative scale** on the *parent observation's* image space (not absolute pixel coordinates). Conversion to native WSI pixel coordinates is handled inside `get_roi_at_native_resolution()` in `image_utils.py`.

### Fixed Magnification Levels

The agent works with canonical magnification labels mapped to target MPP values:
```
0.5x → 20.0 µm/px   (thumbnail)
2x   →  5.0 µm/px
5x   →  2.0 µm/px
10x  →  1.0 µm/px
20x  →  0.5 µm/px
40x  →  0.25 µm/px  (highest detail)
```

### Image/Token Budget (`slidereasoner/utils/image_utils.py`)

- `IMAGE_MAX_TOKEN_NUM = 16384` — hard cap on vision tokens per image
- `MAX_RATIO = 200` — maximum allowed aspect ratio
- `patch_multiple = 32` — all output dimensions snapped to multiples of 32
- `smart_resize()` — resizes while preserving aspect ratio within token budget
- `fit_pixels_to_budget()` — constrains total pixel count for model limits
- Always reads from **level 0** (native resolution) of the WSI, then downsamples to target MPP

### Prompt System

The WSI agent system prompt lives in `slidereasoner/built_in_prompt/prompt_wsi_agent.md`. It instructs the model on:
- Observation indexing (reuse indices, don't re-zoom already-seen regions)
- Bbox coordinates are on the 0–1000 scale relative to the parent observation
- When to stop and emit a final answer

Load prompts via `slidereasoner/utils/prompt_utils.py:get_prompt_from_file()`.

## Development Workflow

The primary way to run and test the agent is through Jupyter notebooks:
- `slidereasoner/test_v2.ipynb` — current integration test notebook
- `slidereasoner/test_v3.ipynb` — latest test notebook

No formal test runner or CI configuration exists yet.

## Data

WSI datasets and JSON case files live under `datasets/`. The `result/` directory holds agent outputs. Neither should be committed (see `.gitignore`).
