# -*- coding: utf-8 -*-
"""WSI zoom-in agent built on top of AgentScope.

This module turns a whole-slide image (WSI) into an interactive pathology
reasoning session:

1. Create an initial thumbnail observation from the WSI.
2. Let the model reason over the current observation tree.
3. Allow the model to zoom into a relative bbox on an existing observation.
4. Save each zoom result as a new observation and feed it back to the model.

Design choice used here:
- The LLM only sees pathology-style magnification labels: 1x / 5x / 10x / 20x / 40x
- Internally, crops are generated with canonical target mpp:
    0.5x -> 20.0
    2x -> 5.0
    5x -> 2.0
    10x -> 1.0
    20x -> 0.5
    40x -> 0.25
- If the slide native resolution is lower than the requested canonical mpp,
  controlled upsampling is allowed, but capped by `max_upsample_ratio`.
"""


# test


#@ sfas fasdff
from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple, Type
from datetime import datetime
from zoneinfo import ZoneInfo



import openslide
import shortuuid
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from agentscope.agent import ReActAgentBase
from agentscope.formatter import FormatterBase
from agentscope.memory import InMemoryMemory, MemoryBase
from agentscope.message import ImageBlock, Msg, TextBlock, ToolResultBlock, ToolUseBlock, URLSource
from agentscope.model import ChatModelBase
from agentscope.tool import ToolResponse, Toolkit
from agentscope.tracing import trace_reply

from slidereasoner.utils.agent_utils import convert_tool_result_to_string
from slidereasoner.utils.image_utils import (
    IMAGE_MAX_TOKEN_NUM,
    IMAGE_MIN_TOKEN_NUM,
    MAX_RATIO,
    maybe_resize_bbox,
    smart_resize,
    to_rgb,
)
from slidereasoner.utils.logging_utils import logger
from slidereasoner.utils.print_utils import print_multimodal_trace


# -----------------------------
# Fixed pathology magnifications
# -----------------------------
# 这里直接固定到你想要的 6 档。
# LLM 层只操作这些倍率，底层统一用 canonical mpp 实现。
MAG_LITERAL = Literal["0.5x", "2x", "5x", "10x", "20x", "40x"]

MAG_TO_MPP: Dict[str, float] = {
    "0.5x": 20.0,
    "2x": 5.0,
    "5x": 2.0,
    "10x": 1.0,
    "20x": 0.5,
    "40x": 0.25,
}

MAG_ORDER: Tuple[str, ...] = ("0.5x", "2x", "5x", "10x", "20x", "40x")

# Qwen3-VL 的尺寸建议按 32 的倍数对齐
QWEN_SIZE_MULTIPLE = 32

# 最多允许的数字上采样倍数。
# 例如 20x-native -> 40x-display 是 2x，允许；
# 如果还想更激进，可以再调大，但我建议先固定成 2.2。
MAX_UPSAMPLE_FACTOR = 2.2

DEFAULT_WSI_AGENT_PROMPT = """You are a pathology whole-slide image reasoning agent.

You work on one WSI session at a time.
Each image is identified by an observation_index.

Available magnifications are strictly:
0.5x, 2x, 5x, 10x, 20x, 40x

Observation ids are hierarchical path ids such as:
- 0p5x_root
- 0p5x_root-2x_1
- 0p5x_root-2x_1-5x_1
- 0p5x_root-2x_1-5x_1-10x_3

Rules:
- Start from the currently available observations.
- The whole-slide overview starts at 0.5x and has observation_id = root.
- If you need more detail, call `zoom_in_image`.
- `bbox_2d` uses relative coordinates on a 0-1000 scale:
  [x1, y1, x2, y2]
- Reuse returned observation_index values exactly as given in later tool calls.
- Do not invent observation ids.
- If needed, you may backtrack to an earlier observation and zoom a different region.
- Avoid repeatedly requesting the same crop unless there is a clear reason.
- Use 0.5x/2x for overview and region selection.
- Use 5x/10x for structural confirmation.
- Use 20x/40x for the highest-detail inspection available in this system.
- When enough evidence is available, answer directly and clearly.
"""


@dataclass(slots=True)
class ObservationMeta:
    """Metadata for each rendered observation."""

    observation_id: str
    image_path: str
    label: str

    # Native WSI coordinates of the rendered observation.
    # native == OpenSlide level 0
    native_x: int
    native_y: int
    native_w: int
    native_h: int

    # image dim 
    image_w: int
    image_h: int

    # Effective display mpp of the returned observation image.
    # 如果发生了有限上采样，这里仍然记录“显示尺度”的 mpp，
    # 这样后续在该 observation 上再次框 bbox 时，几何映射仍然是自洽的。
    effective_mpp: float

    # Pathology-facing magnification label shown to the model.
    display_mag: str

    parent_observation_id: Optional[str]

    # Tree bookkeeping
    children_ids: List[str] = field(default_factory=list)
    local_child_index: int = 0

    # Extra bookkeeping for trace / explanation.
    is_upsampled: bool = False
    reason: str = ""
    is_marked_roi: bool = False

class ZoomInImageArgs(BaseModel):
    """Arguments for zooming into a child region."""

    observation_id: str = Field(..., description="Parent observation id.")
    bbox_2d: List[int] = Field(
        ...,
        description="Relative bbox [x1, y1, x2, y2] on a 0-1000 scale.",
        min_length=4,
        max_length=4,
    )
    target_mag: MAG_LITERAL = Field(..., description="Target pathology magnification.")
    label: str = Field(default="", description="Short label for this crop.")
    reason: str = Field(default="", description="Why this crop is requested.")

class BacktrackArgs(BaseModel):
    """Arguments for backtracking to an existing observation."""

    observation_id: str = Field(..., description="Observation to return to.")

class MarkROIArgs(BaseModel):
    """Arguments for marking an observation as a key ROI."""

    observation_id: str = Field(..., description="Observation to mark.")
    reason: str = Field(..., description="Why this observation is important.")

def infer_native_mpp(
    slide: openslide.OpenSlide,
    override_native_mpp: Optional[float] = None,
) -> float:
    """Infer native mpp from OpenSlide properties."""
    if override_native_mpp is not None:
        if override_native_mpp <= 0:
            raise ValueError(f"override_native_mpp must be positive, got {override_native_mpp}")
        return float(override_native_mpp)

    properties = slide.properties
    candidates = (
        "openslide.mpp-x",
        "aperio.MPP",
        "hamamatsu.XResolution",
    )

    for key in candidates:
        value = properties.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            if key == "hamamatsu.XResolution":
                # Hamamatsu often stores pixels per mm.
                return 1000.0 / parsed
            return parsed

    raise ValueError(
        "Failed to infer native mpp from slide properties. "
        "Please provide `override_native_mpp` explicitly when starting the session.",
    )


def validate_relative_bbox_1000(bbox_2d: Sequence[int]) -> Tuple[int, int, int, int]:
    """Validate a [x1, y1, x2, y2] bbox on a 0-1000 scale."""
    if len(bbox_2d) != 4:
        raise ValueError(f"bbox_2d must contain 4 values, got {bbox_2d}")

    x1, y1, x2, y2 = [int(v) for v in bbox_2d]
    values = (x1, y1, x2, y2)

    if any(v < 0 or v > 1000 for v in values):
        raise ValueError(f"bbox_2d values must all be within [0, 1000], got {bbox_2d}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox_2d must satisfy x2>x1 and y2>y1, got {bbox_2d}")

    return x1, y1, x2, y2



def snap_to_multiple(value: int, multiple: int, mode: Literal["expand", "nearest"] = "expand") -> int:
    """
    将整数向下对齐到指定倍数。

    这个函数主要用于“空间不足时”的兜底逻辑：
    当期望输出尺寸对应的 native ROI 已经放不进 slide 内，
    且又不想 padding，这时只能把输出尺寸缩小到“当前可行的最大倍数”。

    Args:
        value:
            原始整数值。
        multiple:
            目标倍数，例如 32。

    Returns:
        不大于 value 的最大 multiple 倍数。
    """
    value = int(max(1, value))
    if multiple <= 1:
        return value

    if mode == "expand":
        return max(multiple, math.ceil(value / multiple) * multiple)

    lower = int(max(multiple, math.floor(value / multiple) * multiple))
    upper = int(max(multiple, math.ceil(value / multiple) * multiple))
    return lower if abs(value - lower) <= abs(upper - value) else upper


def snap_down_to_multiple(value: int, multiple: int) -> int:
    """Snap a positive integer down to the nearest multiple."""
    value = max(1, int(value))
    if multiple <= 1:
        return value
    return int(max(multiple, math.floor(value / multiple) * multiple))


def fit_pixels_to_budget(
    width: int,
    height: int,
    multiple: int = QWEN_SIZE_MULTIPLE,
    min_token_num: int = IMAGE_MIN_TOKEN_NUM,
    max_token_num: int = IMAGE_MAX_TOKEN_NUM,
) -> Tuple[int, int, float]:
    """
    Resize an image shape into the model pixel budget while preserving aspect ratio.

    This is mainly used for the initial 0.5x overview, which may otherwise be too large.
    """
    width = max(1, int(width))
    height = max(1, int(height))

    min_pixels = min_token_num * multiple * multiple
    max_pixels = max_token_num * multiple * multiple

    pixels = width * height
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / pixels)
    else:
        scale = 1.0

    out_w = snap_to_multiple(max(1, round(width * scale)), multiple, mode="nearest")
    out_h = snap_to_multiple(max(1, round(height * scale)), multiple, mode="nearest")
    return out_w, out_h, scale


def place_interval_without_resizing(center: float, size: int, limit: int) -> int:
    """
    Place a fixed-size interval inside [0, limit) by shifting only.

    This helper never changes `size`.
    If `size` does not fit, the caller must reduce the requested output size first.
    """
    if size > limit:
        raise ValueError(
            f"Cannot place interval of size={size} inside limit={limit} without resizing."
        )

    start = int(round(center - size / 2.0))
    start = max(0, min(start, limit - size))
    return start

def validate_patch_pixels_for_wsi(
    width: int,
    height: int,
    bbox_2d: Sequence[int],
    observation_id: str,
    target_mag: MAG_LITERAL,
    factor: int = QWEN_SIZE_MULTIPLE,
) -> None:
    """Validate model-side patch size constraints."""
    min_pixels = IMAGE_MIN_TOKEN_NUM * factor * factor
    max_pixels = IMAGE_MAX_TOKEN_NUM * factor * factor
    pixels = width * height

    if min(width, height) <= 0:
        raise ValueError(f"Invalid patch size {width}x{height} for bbox_2d={bbox_2d}")

    if max(width, height) / min(width, height) > MAX_RATIO:
        raise ValueError(
            "WSI_PATCH_INVALID_ASPECT_RATIO: "
            f"bbox_2d={list(bbox_2d)}; size={width}x{height}; max_ratio={MAX_RATIO}. "
            "Action: adjust bbox to reduce the aspect ratio."
        )

    if pixels > max_pixels:
        raise ValueError(
            "WSI_PATCH_TOO_LARGE: "
            f"observation_id={observation_id}; "
            f"target_mag={target_mag}; "
            f"bbox_2d={list(bbox_2d)}; "
            f"patch_wh={width}x{height}; "
            f"patch_pixels={pixels}; "
            f"max_pixels={max_pixels}. "
            "Action: shrink bbox and retry."
        )

    if pixels < min_pixels:
        raise ValueError(
            "WSI_PATCH_TOO_SMALL: "
            f"observation_id={observation_id}; "
            f"target_mag={target_mag}; "
            f"bbox_2d={list(bbox_2d)}; "
            f"patch_wh={width}x{height}; "
            f"patch_pixels={pixels}; "
            f"min_pixels={min_pixels}. "
            "Action: expand bbox and retry."
        )
    
def next_child_mag(parent_mag_idx: int) -> Optional[str]:
    """
    Return the next allowed child magnification.

    The zoom tree is fixed:
    0.5x -> 2x -> 5x -> 10x -> 20x -> 40x
    """
    if parent_mag_idx < 0 or parent_mag_idx >= len(MAG_ORDER):
        raise ValueError(f"Unknown parent magnification index: {parent_mag_idx}, valid range is [0, {len(MAG_ORDER)-1}]")
    
    idx = MAG_ORDER[parent_mag_idx]
    if idx == len(MAG_ORDER) - 1:
        return None
    return MAG_ORDER[idx + 1]

def make_child_observation_id(
    parent_observation_id: str,
    target_mag: str,
    child_index: int,
) -> str:
    """
    Create a hierarchical path-style observation id.

    Examples:
        parent=root, target_mag=2x, child_index=1
            -> 2x_1

        parent=2x_1__5x_1, target_mag=10x, child_index=3
            -> 2x_1__5x_1__10x_3
    """
    node = f"{target_mag}_{child_index}"
    if parent_observation_id == ROOT_OBSERVATION_ID:
        return node
    return f"{parent_observation_id}-{node}"


def get_roi_at_fixed_mag(
    slide: openslide.OpenSlide,
    source_bbox_1000: Tuple[int, int, int, int],
    source_native_x: int,
    source_native_y: int,
    source_native_w: int,
    source_native_h: int,
    target_mag: MAG_LITERAL,
    native_mpp: float,
    min_pixels: int = QWEN_SIZE_MULTIPLE,
    patch_multiple: int = QWEN_SIZE_MULTIPLE,
    max_upsample_factor: float = MAX_UPSAMPLE_FACTOR,
) -> Tuple[Image.Image, Tuple[int, int, int, int], float, bool, str]:
    """
    Extract a child ROI from the parent observation using fixed magnification labels.

    Design:
    - LLM requests one of the fixed pathology magnifications.
    - Internally we convert that magnification to canonical mpp.
    - We always read from native resolution (OpenSlide level 0).
    - Limited upsampling is allowed for display consistency.
    - Output size is snapped to a multiple of `patch_multiple`.

    Returns:
        crop:
            Output RGB patch.
        native_roi:
            Actual crop region in native coordinates: (x, y, w, h).
        effective_mpp:
            Display mpp of the returned patch.
        is_upsampled:
            Whether the returned patch is digitally upsampled beyond native resolution.
    """
    if native_mpp <= 0:
        raise ValueError(f"native_mpp must be positive, got {native_mpp}")

    x1_1000, y1_1000, x2_1000, y2_1000 = validate_relative_bbox_1000(source_bbox_1000)

    if source_native_x < 0 or source_native_y < 0 or source_native_w <= 0 or source_native_h <= 0:
        raise ValueError(
            f"source_native_x {source_native_x}/source_native_y {source_native_y} must be non-negative and "
            f"source_native_w {source_native_w}/source_native_h {source_native_h} must be positive."
        )

    slide_w, slide_h = slide.dimensions
    parent_x2 = source_native_x + source_native_w
    parent_y2 = source_native_y + source_native_h

    if parent_x2 > slide_w or parent_y2 > slide_h:
        raise ValueError(
            f"Parent native extent exceeds slide bounds: "
            f"parent=(x1={source_native_x}, y1={source_native_y}, x2={parent_x2}, y2={parent_y2}), "
            f"slide_size={slide.dimensions}"
        )

    # 1) Map the relative bbox back to native WSI coordinates.
    native_x1 = source_native_x + math.floor(source_native_w * x1_1000 / 1000.0)
    native_y1 = source_native_y + math.floor(source_native_h * y1_1000 / 1000.0)
    native_x2 = source_native_x + math.ceil(source_native_w * x2_1000 / 1000.0)
    native_y2 = source_native_y + math.ceil(source_native_h * y2_1000 / 1000.0)

    native_x1 = max(source_native_x, native_x1)
    native_y1 = max(source_native_y, native_y1)
    native_x2 = min(parent_x2, native_x2)
    native_y2 = min(parent_y2, native_y2)

    base_native_w = max(1, native_x2 - native_x1)
    base_native_h = max(1, native_y2 - native_y1)

    # 2) Convert fixed magnification to canonical target mpp.
    canonical_target_mpp = MAG_TO_MPP[target_mag]

    # 3) Allow limited digital upsampling.
    # Example:
    #   native_mpp=0.50, target=0.25 -> 2x upsample, allowed
    #   native_mpp=1.00, target=0.25 -> 4x upsample, clamp to 0.50
    effective_mpp = max(canonical_target_mpp, native_mpp / max_upsample_factor)
    is_upsampled = effective_mpp < native_mpp

    # 4) Compute the desired output size at the requested display scale.
    raw_out_w = max(min_pixels, round(base_native_w * native_mpp / effective_mpp))
    raw_out_h = max(min_pixels, round(base_native_h * native_mpp / effective_mpp))

    out_w = snap_to_multiple(raw_out_w, patch_multiple, mode="expand")
    out_h = snap_to_multiple(raw_out_h, patch_multiple, mode="expand")

    # 5) Slightly adjust the native ROI so the final output naturally matches
    #    the snapped size. This reduces extra distortion from later resizing.
    desired_native_w = max(1, round(out_w * effective_mpp / native_mpp))
    desired_native_h = max(1, round(out_h * effective_mpp / native_mpp))

    center_x = native_x1 + base_native_w / 2.0
    center_y = native_y1 + base_native_h / 2.0

    # If the requested native ROI is too large for the slide, shrink the output
    # size first, then re-compute the native crop size. No padding is used.
    if desired_native_w > slide_w:
        max_out_w = snap_down_to_multiple(
            max(min_pixels, math.floor(slide_w * native_mpp / effective_mpp)),
            patch_multiple,
        )
        out_w = max(min_pixels, max_out_w)
        desired_native_w = max(1, round(out_w * effective_mpp / native_mpp))

    if desired_native_h > slide_h:
        max_out_h = snap_down_to_multiple(
            max(min_pixels, math.floor(slide_h * native_mpp / effective_mpp)),
            patch_multiple,
        )
        out_h = max(min_pixels, max_out_h)
        desired_native_h = max(1, round(out_h * effective_mpp / native_mpp))

    aligned_native_x = place_interval_without_resizing(center_x, desired_native_w, slide_w)
    aligned_native_y = place_interval_without_resizing(center_y, desired_native_h, slide_h)

    native_roi = (
        aligned_native_x,
        aligned_native_y,
        desired_native_w,
        desired_native_h,
    )

    # 6) Read from native resolution only.
    crop = slide.read_region(
        (aligned_native_x, aligned_native_y),
        0,
        (desired_native_w, desired_native_h),
    ).convert("RGB")

    # 7) 根据放大还是缩小，动态选择最优的重采样算法
    if crop.size != (out_w, out_h):
        if is_upsampled:
            # 上采样：使用 BICUBIC，避免在细胞核等高反差边缘产生 Lanczos 振铃效应干扰模型
            crop = crop.resize((out_w, out_h), Image.Resampling.BICUBIC)
        else:
            # 下采样：使用 LANCZOS，提供最强的抗锯齿（Anti-aliasing）效果
            crop = crop.resize((out_w, out_h), Image.Resampling.LANCZOS)

    # 判断是否发生了“微观妥协”（模型要 40x，但实际给不到）
    delivery_status = "success"
    if effective_mpp > canonical_target_mpp * 1.1: # 给 10% 的容差
        # 计算实际交付的相当于传统病理的多少倍 (10.0 / effective_mpp 近似等于倍率)
        realized_mag_val = 10.0 / effective_mpp 
        delivery_status = f"capped_at_{realized_mag_val:.1f}x"

    return crop, native_roi, effective_mpp, is_upsampled, delivery_status

 


class WSIReActAgent(ReActAgentBase):
    """WSI zooming agent with fixed pathology magnifications and path-style observation ids."""

    finish_function_name: str = "generate_response"

    def __init__(
        self,
        name: str,
        model: ChatModelBase,
        formatter: FormatterBase,
        sys_prompt: str = DEFAULT_WSI_AGENT_PROMPT,
        toolkit: Optional[Toolkit] = None,
        memory: Optional[MemoryBase] = None,
        work_dir: Optional[str] = None,
        max_iters: int = 50,
        parallel_tool_calls: bool = False,
        min_pixels: int = 32,
    ) -> None:
        super().__init__()

        self.name = name
        self.model = model
        self.formatter = formatter
        self._sys_prompt = sys_prompt
        self.max_iters = max_iters
        self.parallel_tool_calls = parallel_tool_calls
        self.min_pixels = min_pixels

        self._stream_prefix: dict = {}
        self.memory = memory or InMemoryMemory()
        self.toolkit = toolkit or Toolkit()
        self._required_structured_model: Type[BaseModel] | None = None

        self.workspace_root = work_dir

        self.slide: Optional[openslide.OpenSlide] = None
        self.current_slide_path: Optional[str] = None
        self.current_slide_label: Optional[str] = None
        self.native_mpp: Optional[float] = None
        self.session_dir: Optional[Path] = None


        self.action_idx: int = 0

        # Keep image paths in creation order if you still need this outside.
        self.observation_list: List[str] = []

        # Stable creation order for trace export / debugging
        self.observation_order: List[str] = []

        # observation_id -> image path
        self.observation_dict: Dict[str, str] = {}

         # Main observation store:
        # observation_id -> metadata
        self.observation_meta: Dict[str, ObservationMeta] = {}


        self.current_observation_id: Optional[str] = None

        self.toolkit.register_tool_function(self.zoom_in_image)
        self.toolkit.register_tool_function(self.backtrack_to_observation)
        self.toolkit.register_tool_function(self.mark_roi)


        self.register_state("name")
        self.register_state("_sys_prompt")

    @property
    def sys_prompt(self) -> str:
        return self._sys_prompt

    def reset_session(self) -> None:
        """Reset current WSI session state while keeping the agent instance."""

        self.memory = InMemoryMemory()
        self._stream_prefix.clear()
        self.action_idx = 0

        if self.slide is not None:
            self.slide.close()
        self.slide = None
        self.current_slide_path: Optional[str] = None
        self.current_slide_label: Optional[str] = None
    
        self.observation_list: List[str] = []
        self.observation_order: List[str] = []
        self.observation_dict: Dict[str, str] = {}
        self.observation_meta: Dict[str, ObservationMeta] = {}
        self.current_observation_id: Optional[str] = None

        self.workspace_root = None
        self.native_mpp = None
        self.session_dir = None

    def require_observation(self, observation_id: str) -> ObservationMeta:
        """Return an observation metadata object or raise a clear error."""
        if observation_id not in self.observation_meta:
            raise ValueError(
                f"Invalid observation_id={observation_id}. "
                f"Available ids: {self.observation_order}"
            )
        return self.observation_meta[observation_id]
    

    def render_overview_thumbnail(
        self,
        slide: openslide.OpenSlide,
        native_mpp: float,
        target_mag: MAG_LITERAL = "0.5x",
    ) -> Tuple[Image.Image, float, str]:
        """
        Render the initial whole-slide overview at nominal 0.5x.

        We use thumbnail rendering here for efficiency.
        Patch zooming still uses native-resolution cropping.
        """
        slide_w, slide_h = slide.dimensions
        target_mpp = MAG_TO_MPP[target_mag]

        raw_w = max(QWEN_SIZE_MULTIPLE, round(slide_w * native_mpp / target_mpp))
        raw_h = max(QWEN_SIZE_MULTIPLE, round(slide_h * native_mpp / target_mpp))

        out_w, out_h, scale = fit_pixels_to_budget(
            raw_w,
            raw_h,
            multiple=QWEN_SIZE_MULTIPLE,
            min_token_num=IMAGE_MIN_TOKEN_NUM,
            max_token_num=IMAGE_MAX_TOKEN_NUM,
        )


        # 核心修复点：计算真实的 effective_mpp
        # 因为 out_w 可能被 snap_to_multiple 进行了微调，所以最精确的做法是直接用物理宽度除以像素宽度
        # slide_w * native_mpp = 切片的绝对物理宽度 (微米)
        effective_mpp_w = (slide_w * native_mpp) / out_w
        effective_mpp_h = (slide_h * native_mpp) / out_h
        
        # 取平均值或宽度的 MPP 作为最终的有效 MPP（由于长宽比几乎保持不变，两者差距极小）
        effective_mpp = (effective_mpp_w + effective_mpp_h) / 2.0


        # 1. 先获取一个稍大于或等于目标的缩略图 (利用 OpenSlide 的金字塔层级加速)
        thumb = slide.get_thumbnail((out_w, out_h))
        thumb = to_rgb(thumb)

        # 2. 强制 Resize 到严格对齐 32 倍数的尺寸
        if thumb.size != (out_w, out_h):
            # 概览图下采样使用 LANCZOS 保持清晰度
            thumb = thumb.resize((out_w, out_h), Image.Resampling.LANCZOS)      

        # 判断是否因为 Token 限制发生了严重的“宏观妥协”
        # 注意逻辑反转：MPP 越大，说明压缩越狠，倍率越低
        delivery_status = "success"
        if effective_mpp > target_mpp * 1.1:
            logger.warning(
                f"Effective MPP {effective_mpp:.4f} is significantly higher than target {target_mpp:.4f}. "
                f"The overview image may be too blurry for detailed inspection. "
                f"Consider increasing the model pixel budget or allowing larger overview sizes."
            )

            # 计算实际被压缩到了相当于多少倍率
            realized_mag_val = 10.0 / effective_mpp 
            delivery_status = f"compressed_to_{realized_mag_val:.1f}x"


        return thumb, effective_mpp, delivery_status
    

    def start_wsi_session(
        self,
        wsi_path: str,
        question: str,
        native_mpp: Optional[float] = None,
        thumbnail_size: Tuple[int, int] = (1024, 1024),
        slide_label: Optional[str] = None,
    ) -> Msg:
        """Open a WSI and create the initial thumbnail observation message."""

        self.reset_session()

        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")

        self.current_slide_path = os.path.abspath(wsi_path)
        self.current_slide_label = slide_label or Path(wsi_path).stem
        self.slide = openslide.OpenSlide(self.current_slide_path)
        self.native_mpp = infer_native_mpp(self.slide, native_mpp)

        overview_image, overview_mpp, delivery_status = self.render_overview_thumbnail(self.slide, self.native_mpp, MAG_ORDER[0])

        time_str = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d%H%M%S")
        self.session_dir = os.path.join(self.workspace_root, self.current_slide_label, time_str, shortuuid.uuid())
        os.makedirs(self.session_dir, exist_ok=True)

        if delivery_status.startswith("capped_at_"):
            actual_mag = delivery_status.split("_")[-1].replace(".", "p")
        else:
            actual_mag = MAG_ORDER[0].replace(".", "p")

        self.ROOT_OBSERVATION_ID = f"obs_{actual_mag}_root"

        thumbnail_path = os.path.join(self.session_dir, f"{self.ROOT_OBSERVATION_ID}.png")

        overview_image.save(thumbnail_path)


        level0_w, level0_h = self.slide.dimensions
        thumb_w, thumb_h = overview_image.size

        meta = ObservationMeta(
            observation_id=self.ROOT_OBSERVATION_ID,
            image_path=thumbnail_path,
            label="whole-slide thumbnail",
            native_x=0,
            native_y=0,
            native_w=level0_w,
            native_h=level0_h,
            image_w=thumb_w,
            image_h=thumb_h,
            effective_mpp=overview_mpp,
            parent_observation_id=None,
            display_mag=actual_mag,
            local_child_index=0,
            is_upsampled=False,
            reason="initial whole-slide overview",
        )
        self.observation_list.append(thumbnail_path)
        self.observation_meta.append(meta)
        return Msg(
            name="user",
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"The following image is observation_id {self.ROOT_OBSERVATION_ID} for WSI "
                        f"'{self.slide_label}'.\n"
                        f"- slide_path: {self.slide_path}\n"
                        f"- level0_dimensions: {level0_w}x{level0_h}\n"
                        f"- level0_mpp: {self.level0_mpp:.6f}\n"
                        f"- observation_mpp: {effective_mpp:.6f}\n"
                        f"- Use zoom_in_image with observation_index values to inspect ROIs."
                    ),
                ),
                ImageBlock(
                    type="image",
                    source=URLSource(type="url", url=thumbnail_path),
                ),
                TextBlock(type="text", text=question),
            ],
            role="user",
        )

    async def run_on_wsi(
        self,
        wsi_path: str,
        question: str,
        level0_mpp: Optional[float] = None,
        thumbnail_size: Tuple[int, int] = (1024, 1024),
        slide_label: Optional[str] = None,
        structured_model: Type[BaseModel] | None = None,
    ) -> Msg:
        """Convenience entrypoint: open the WSI and run the full reply loop."""

        initial_msg = self.start_wsi_session(
            wsi_path=wsi_path,
            question=question,
            level0_mpp=level0_mpp,
            thumbnail_size=thumbnail_size,
            slide_label=slide_label,
        )
        return await self.reply(initial_msg, structured_model=structured_model)

    @trace_reply
    async def reply(
        self,
        msg: Msg | List[Msg] | None = None,
        structured_model: Type[BaseModel] | None = None,
    ) -> Msg:
        """Run the reasoning/acting loop."""

        await self.memory.add(msg)

        tool_choice: Literal["auto", "none", "required"] | None = None
        self._required_structured_model = structured_model

        if structured_model:
            if self.finish_function_name not in self.toolkit.tools:
                self.toolkit.register_tool_function(
                    getattr(self, self.finish_function_name),
                )
            self.toolkit.set_extended_model(
                self.finish_function_name,
                structured_model,
            )
            tool_choice = "required"
        else:
            if self.finish_function_name in self.toolkit.tools:
                self.toolkit.remove_tool_function(self.finish_function_name)

        structured_output = None
        reply_msg: Optional[Msg] = None

        for _ in range(self.max_iters):
            msg_reasoning = await self.reasoning(tool_choice=tool_choice)
            tool_calls = msg_reasoning.get_content_blocks("tool_use")

            if not tool_calls:
                msg_reasoning.metadata = structured_output
                reply_msg = msg_reasoning
                break

            if self.parallel_tool_calls and len(tool_calls) > 1:
                structured_outputs = await asyncio.gather(
                    *(self.acting(tool_call) for tool_call in tool_calls),
                )
            else:
                structured_outputs = []
                for tool_call in tool_calls:
                    structured_outputs.append(await self.acting(tool_call))

            for item in structured_outputs:
                if item is not None:
                    structured_output = item

            if structured_output is not None:
                msg_reasoning.metadata = structured_output
                reply_msg = msg_reasoning
                break

        if reply_msg is None:
            raise RuntimeError(
                f"WSIReActAgent reached max_iters={self.max_iters} without producing a final reply.",
            )

        return reply_msg

    async def reasoning(
        self,
        tool_choice: Literal["auto", "none", "required"] | None = None,
    ) -> Msg:
        """Perform a single reasoning step."""

        prompt = await self.formatter.format(
            msgs=[
                Msg("system", self.sys_prompt, "system"),
                *await self.memory.get_memory(),
            ],
        )

        res = await self.model(
            prompt,
            tools=self.toolkit.get_json_schemas(),
            tool_choice=tool_choice,
        )

        msg = Msg(name=self.name, content=[], role="assistant")
        if self.model.stream:
            async for content_chunk in res:
                msg.content = content_chunk.content
                await print_multimodal_trace(self._stream_prefix, msg, False)
        else:
            msg.content = list(res.content)

        await print_multimodal_trace(self._stream_prefix, msg, True)
        await asyncio.sleep(0.001)
        await self.memory.add(msg)
        return msg

    async def acting(self, tool_call: ToolUseBlock) -> Dict[str, Any] | None:
        """Execute a tool call and feed the result back into memory."""

        tool_name = tool_call["name"]
        tool_res_msg = Msg(
            "system",
            [
                ToolResultBlock(
                    type="tool_result",
                    id=tool_call["id"],
                    name=tool_name,
                    output=[],
                ),
            ],
            "system",
        )

        if tool_name not in self.toolkit.tools:
            tool_res_msg.content[0]["output"] = [
                TextBlock(
                    type="text",
                    text=(
                        f"FunctionNotFoundError: {tool_name} is not available. "
                        f"Available tools: {list(self.toolkit.tools.keys())}."
                    ),
                ),
            ]
            await print_multimodal_trace(self._stream_prefix, tool_res_msg, True)
            await self.memory.add(tool_res_msg)
            return None

        print(f"[TOOL_CALL] {tool_name}")
        tool_res = await self.toolkit.call_tool_function(tool_call)

        last_chunk = None
        async for chunk in tool_res:
            tool_res_msg.content[0]["output"] = chunk.content
            await print_multimodal_trace(self._stream_prefix, tool_res_msg, chunk.is_last)
            last_chunk = chunk
            if chunk.is_interrupted:
                raise asyncio.CancelledError()

        if last_chunk is None:
            raise RuntimeError(f"Tool {tool_name} returned no chunks.")

        if tool_name == self.finish_function_name:
            await self.memory.add(tool_res_msg)
            metadata = last_chunk.metadata or {}
            return metadata.get("structured_output")

        if tool_name == "zoom_in_image" and last_chunk.metadata.get("success"):
            print(f"[TOOL_RESPONSE] {tool_name}")
            textual_output, multimodal_data = convert_tool_result_to_string(
                tool_res_msg.content[0]["output"],
            )

            if len(textual_output) != 1:
                raise ValueError(
                    "zoom_in_image must return exactly one text block, "
                    f"got {len(textual_output)}",
                )
            if len(multimodal_data) != 1:
                raise ValueError(
                    "zoom_in_image must return exactly one image block, "
                    f"got {len(multimodal_data)}",
                )

            text_tuple = textual_output[0]
            image_url, image_block = multimodal_data[0]

            text_only_tool_msg = Msg(
                "system",
                [
                    ToolResultBlock(
                        type="tool_result",
                        id=tool_res_msg.content[0]["id"],
                        name=tool_res_msg.content[0]["name"],
                        output=[text_tuple[1]],
                    ),
                ],
                "system",
            )
            await print_multimodal_trace(self._stream_prefix, text_only_tool_msg, True)
            await self.memory.add(text_only_tool_msg)

            promoted_msg = Msg(
                name="user",
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            "<system-info>The following image was generated from "
                            f"tool result '{tool_name}'."
                        ),
                    ),
                    TextBlock(
                        type="text",
                        text=(
                            f"- observation_index: {last_chunk.metadata['observation_index']}\n"
                            f"- parent_observation_index: {last_chunk.metadata['source_observation_index']}\n"
                            f"- target_mpp: {last_chunk.metadata['effective_mpp']:.6f}\n"
                            f"- label: {last_chunk.metadata['label']}"
                        ),
                    ),
                    ImageBlock(
                        type="image",
                        source=URLSource(type="url", url=image_url),
                    ),
                    TextBlock(type="text", text="</system-info>"),
                ],
                role="user",
            )
            await print_multimodal_trace(self._stream_prefix, promoted_msg, True)
            await self.memory.add(promoted_msg)
            return None

        await self.memory.add(tool_res_msg)
        return None

    def generate_response(self, **kwargs: Any) -> ToolResponse:
        """Validate and return final structured output when requested."""

        structured_output = None
        if self._required_structured_model:
            try:
                structured_output = self._required_structured_model.model_validate(
                    kwargs,
                ).model_dump()
            except ValidationError as exc:
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=f"Arguments Validation Error: {exc}",
                        ),
                    ],
                    metadata={
                        "success": False,
                        "structured_output": {},
                    },
                )
        else:
            logger.warning(
                "generate_response was called without a required structured model.",
            )

        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text="Successfully generated response.",
                ),
            ],
            metadata={
                "success": True,
                "structured_output": structured_output,
            },
            is_last=True,
        )

    def zoom_in_image(
        self,
        bbox_2d: Annotated[List[float], Field(min_length=4, max_length=4)],
        target_mpp: Annotated[float, Field(gt=0)],
        label: str,
        observation_index: Annotated[int, Field(ge=0)],
    ) -> ToolResponse:
        """Zoom in on a WSI observation using true level-aware WSI reads."""

        try:
            if self.slide is None or self.level0_mpp is None or self.session_dir is None:
                raise RuntimeError("No active WSI session. Call start_wsi_session first.")

            if observation_index >= len(self.observation_meta):
                raise IndexError(
                    f"observation_index={observation_index} is out of range. "
                    f"Available indices: 0..{len(self.observation_meta) - 1}",
                )

            rel_x1, rel_y1, rel_x2, rel_y2 = _validate_relative_bbox(bbox_2d)
            parent_meta = self.observation_meta[observation_index]

            if not os.path.exists(parent_meta.image_path):
                raise FileNotFoundError(
                    f"Observation image does not exist: {parent_meta.image_path}",
                )

            parent_image = to_rgb(Image.open(parent_meta.image_path))
            img_width, img_height = parent_image.size

            abs_x1 = math.floor(rel_x1 / 1000.0 * img_width)
            abs_y1 = math.floor(rel_y1 / 1000.0 * img_height)
            abs_x2 = math.ceil(rel_x2 / 1000.0 * img_width)
            abs_y2 = math.ceil(rel_y2 / 1000.0 * img_height)

            left, top, right, bottom = maybe_resize_bbox(
                abs_x1,
                abs_y1,
                abs_x2,
                abs_y2,
                img_width,
                img_height,
            )

            roi_image, level0_roi, effective_mpp = get_roi_at_mpp_optimized(
                slide=self.slide,
                source_roi=(left, top, right - left, bottom - top),
                source_mpp=parent_meta.effective_mpp,
                source_level0_x=parent_meta.level0_x,
                source_level0_y=parent_meta.level0_y,
                target_mpp=target_mpp,
                level0_mpp=self.level0_mpp,
                min_pixels=self.min_pixels,
            )

            resized_h, resized_w = smart_resize(
                roi_image.height,
                roi_image.width,
                factor=32,
            )
            _validate_patch_pixels_for_wsi(
                bbox_2d=bbox_2d,
                width=resized_w,
                height=resized_h,
                observation_index=observation_index,
                source_mpp=parent_meta.effective_mpp,
                target_mpp=target_mpp,
            )

            if roi_image.size != (resized_w, resized_h):
                roi_image = roi_image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)

            output_path = str(
                self.session_dir / f"observation_{self.action_idx}_{shortuuid.uuid()}.png",
            )
            roi_image.save(output_path)

            new_observation_index = len(self.observation_meta)
            level0_x, level0_y, level0_w, level0_h = level0_roi
            child_meta = ObservationMeta(
                observation_index=new_observation_index,
                image_path=output_path,
                label=label,
                level0_x=level0_x,
                level0_y=level0_y,
                level0_w=level0_w,
                level0_h=level0_h,
                effective_mpp=effective_mpp,
                parent_observation_index=observation_index,
            )

            self.observation_list.append(output_path)
            self.observation_meta.append(child_meta)
            self.action_idx += 1

            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=(
                            "zoom_in_image succeeded.\n"
                            "Generated a zoomed-in WSI ROI view.\n"
                            f"- returned observation_index: {new_observation_index}\n"
                            f"- source observation_index: {observation_index}\n"
                            f"- label: {label}\n"
                            f"- effective_mpp: {effective_mpp:.6f}\n"
                            f"- level0_roi_xywh: {level0_roi}\n"
                            f"Use observation_index={new_observation_index} for later follow-up tool calls."
                        ),
                    ),
                    ImageBlock(
                        type="image",
                        source=URLSource(type="url", url=output_path),
                    ),
                ],
                metadata={
                    "success": True,
                    "observation_index": new_observation_index,
                    "source_observation_index": observation_index,
                    "effective_mpp": effective_mpp,
                    "level0_roi": level0_roi,
                    "label": label,
                },
            )

        except Exception as exc:  # pylint: disable=broad-except
            return ToolResponse(
                content=[
                    TextBlock(
                        type="text",
                        text=f"Failure to execute zoom_in_image, error: {exc}",
                    ),
                ],
                metadata={"success": False},
            )
