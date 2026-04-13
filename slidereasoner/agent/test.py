from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import openslide
import shortuuid
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from agentscope.agent import ReActAgentBase
from agentscope.formatter import FormatterBase
from agentscope.memory import InMemoryMemory, MemoryBase
from agentscope.model import ChatModelBase
from agentscope.tool import ToolResponse, Toolkit

from slidereasoner.utils.image_utils import (
    IMAGE_MAX_TOKEN_NUM,
    IMAGE_MIN_TOKEN_NUM,
    MAX_RATIO,
    to_rgb,
)


# -----------------------------
# Fixed pathology magnifications
# -----------------------------
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

QWEN_SIZE_MULTIPLE = 32
MAX_UPSAMPLE_FACTOR = 2.0

ROOT_OBSERVATION_ID = "root"


DEFAULT_WSI_AGENT_PROMPT = """You are a pathology whole-slide image reasoning agent.

You work on one WSI session at a time.
Each image is identified by an observation_id.

Observation ids are hierarchical path ids such as:
- root
- 2x_1
- 2x_1__5x_1
- 2x_1__5x_1__10x_3

Rules:
- Start from the currently available observations.
- The whole-slide overview starts at 0.5x and has observation_id = root.
- If you need more detail, call `zoom_in_image`.
- `bbox_2d` uses relative coordinates on a 0-1000 scale:
  [x1, y1, x2, y2]
- Reuse returned observation_id values exactly as given.
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

    # native == OpenSlide level 0
    native_x: int
    native_y: int
    native_w: int
    native_h: int

    # Display-scale mpp of the returned observation image.
    # If limited upsampling happens, this still records the display-scale mpp,
    # so later bbox mapping remains self-consistent inside this observation.
    effective_mpp: float

    # Pathology-facing magnification shown to the model.
    display_mag: str

    parent_observation_id: Optional[str]

    # Tree bookkeeping
    children_ids: List[str] = field(default_factory=list)
    local_child_index: int = 0

    # Extra trace / explanation state
    is_upsampled: bool = False
    reason: str = ""
    is_marked_roi: bool = False


class ZoomInImageArgs(BaseModel):
    """Arguments for zooming into a child region."""

    observation_id: str = Field(..., description="Parent observation id.")
    bbox_2d: List[float] = Field(
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

    observation_id: str = Field(..., description="Observation id to return to.")


class MarkROIArgs(BaseModel):
    """Arguments for marking an observation as a key ROI."""

    observation_id: str = Field(..., description="Observation id to mark.")
    reason: str = Field(..., description="Why this observation is important.")


def _infer_native_mpp(
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
        "Please provide `level0_mpp` explicitly when starting the session.",
    )


def _validate_relative_bbox_1000(bbox_2d: Sequence[float]) -> Tuple[float, float, float, float]:
    """Validate a [x1, y1, x2, y2] bbox on a 0-1000 scale."""
    if len(bbox_2d) != 4:
        raise ValueError(f"bbox_2d must contain 4 values, got {bbox_2d}")

    x1, y1, x2, y2 = [float(v) for v in bbox_2d]
    values = (x1, y1, x2, y2)

    if any(v < 0 or v > 1000 for v in values):
        raise ValueError(f"bbox_2d values must all be within [0, 1000], got {bbox_2d}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox_2d must satisfy x2>x1 and y2>y1, got {bbox_2d}")

    return x1, y1, x2, y2


def _snap_to_multiple(value: int, multiple: int, mode: Literal["expand", "nearest"] = "expand") -> int:
    """Snap a positive integer to a given multiple."""
    value = max(1, int(value))
    if multiple <= 1:
        return value

    if mode == "expand":
        return max(multiple, math.ceil(value / multiple) * multiple)

    lower = max(multiple, math.floor(value / multiple) * multiple)
    upper = max(multiple, math.ceil(value / multiple) * multiple)
    return lower if abs(value - lower) <= abs(upper - value) else upper


def _snap_down_to_multiple(value: int, multiple: int) -> int:
    """Snap a positive integer down to the nearest multiple."""
    value = max(1, int(value))
    if multiple <= 1:
        return value
    return max(multiple, math.floor(value / multiple) * multiple)


def _fit_pixels_to_budget(
    width: int,
    height: int,
    multiple: int = QWEN_SIZE_MULTIPLE,
    min_token_num: int = IMAGE_MIN_TOKEN_NUM,
    max_token_num: int = IMAGE_MAX_TOKEN_NUM,
) -> Tuple[int, int]:
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

    out_w = _snap_to_multiple(max(1, round(width * scale)), multiple, mode="nearest")
    out_h = _snap_to_multiple(max(1, round(height * scale)), multiple, mode="nearest")
    return out_w, out_h


def _place_interval_without_resizing(center: float, size: int, limit: int) -> int:
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


def _validate_patch_pixels_for_wsi(
    width: int,
    height: int,
    bbox_2d: Sequence[float],
    observation_id: str,
    target_mag: str,
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


def _sanitize_observation_id_for_filename(observation_id: str) -> str:
    """Make an observation id safe and tidy for filenames."""
    return observation_id.replace(".", "p")


def _next_child_mag(parent_mag: str) -> Optional[str]:
    """
    Return the next allowed child magnification.

    The zoom tree is fixed:
    0.5x -> 2x -> 5x -> 10x -> 20x -> 40x
    """
    if parent_mag not in MAG_ORDER:
        raise ValueError(f"Unknown parent magnification: {parent_mag}")

    idx = MAG_ORDER.index(parent_mag)
    if idx == len(MAG_ORDER) - 1:
        return None
    return MAG_ORDER[idx + 1]


def _make_child_observation_id(
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
    return f"{parent_observation_id}__{node}"


def get_roi_at_fixed_mag(
    slide: openslide.OpenSlide,
    source_bbox_1000: Tuple[float, float, float, float],
    source_native_x: int,
    source_native_y: int,
    source_native_w: int,
    source_native_h: int,
    target_mag: MAG_LITERAL,
    native_mpp: float,
    min_pixels: int = QWEN_SIZE_MULTIPLE,
    patch_multiple: int = QWEN_SIZE_MULTIPLE,
    max_upsample_factor: float = MAX_UPSAMPLE_FACTOR,
) -> Tuple[Image.Image, Tuple[int, int, int, int], float, bool]:
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

    x1_1000, y1_1000, x2_1000, y2_1000 = _validate_relative_bbox_1000(source_bbox_1000)

    if source_native_x < 0 or source_native_y < 0 or source_native_w <= 0 or source_native_h <= 0:
        raise ValueError(
            "source_native_x/source_native_y must be non-negative and "
            "source_native_w/source_native_h must be positive."
        )

    slide_w, slide_h = slide.dimensions
    parent_x2 = source_native_x + source_native_w
    parent_y2 = source_native_y + source_native_h

    if parent_x2 > slide_w or parent_y2 > slide_h:
        raise ValueError(
            f"Parent native extent exceeds slide bounds: "
            f"parent=({source_native_x}, {source_native_y}, {source_native_w}, {source_native_h}), "
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
    effective_mpp = max(canonical_target_mpp, native_mpp / max_upsample_factor)
    is_upsampled = effective_mpp < native_mpp

    # 4) Compute the desired output size at the requested display scale.
    raw_out_w = max(min_pixels, round(base_native_w * native_mpp / effective_mpp))
    raw_out_h = max(min_pixels, round(base_native_h * native_mpp / effective_mpp))

    out_w = _snap_to_multiple(raw_out_w, patch_multiple, mode="expand")
    out_h = _snap_to_multiple(raw_out_h, patch_multiple, mode="expand")

    # 5) Slightly adjust the native ROI so the final output naturally matches
    #    the snapped size. This reduces extra distortion from later resizing.
    desired_native_w = max(1, round(out_w * effective_mpp / native_mpp))
    desired_native_h = max(1, round(out_h * effective_mpp / native_mpp))

    center_x = native_x1 + base_native_w / 2.0
    center_y = native_y1 + base_native_h / 2.0

    # If the requested native ROI is too large for the slide, shrink the output
    # size first, then re-compute the native crop size. No padding is used.
    if desired_native_w > slide_w:
        max_out_w = _snap_down_to_multiple(
            max(min_pixels, math.floor(slide_w * native_mpp / effective_mpp)),
            patch_multiple,
        )
        out_w = max(min_pixels, max_out_w)
        desired_native_w = max(1, round(out_w * effective_mpp / native_mpp))

    if desired_native_h > slide_h:
        max_out_h = _snap_down_to_multiple(
            max(min_pixels, math.floor(slide_h * native_mpp / effective_mpp)),
            patch_multiple,
        )
        out_h = max(min_pixels, max_out_h)
        desired_native_h = max(1, round(out_h * effective_mpp / native_mpp))

    aligned_native_x = _place_interval_without_resizing(center_x, desired_native_w, slide_w)
    aligned_native_y = _place_interval_without_resizing(center_y, desired_native_h, slide_h)

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

    # 7) Resize exactly once to the target output size.
    if crop.size != (out_w, out_h):
        crop = crop.resize((out_w, out_h), Image.Resampling.LANCZOS)

    return crop, native_roi, effective_mpp, is_upsampled


class WSIReActAgent(ReActAgentBase):
    """WSI zooming agent with fixed pathology magnifications and path-style observation ids."""

    def __init__(
        self,
        name: str,
        model: ChatModelBase,
        sys_prompt: str = DEFAULT_WSI_AGENT_PROMPT,
        formatter: Optional[FormatterBase] = None,
        memory: Optional[MemoryBase] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            sys_prompt=sys_prompt,
            formatter=formatter,
            memory=memory or InMemoryMemory(),
            **kwargs,
        )

        self.slide: Optional[openslide.OpenSlide] = None
        self.native_mpp: Optional[float] = None
        self.current_slide_path: Optional[str] = None
        self.current_slide_label: Optional[str] = None
        self.work_dir: Optional[Path] = None

        # Keep image paths in creation order if you still need this outside.
        self.observation_list: List[str] = []

        # Main observation store:
        # observation_id -> metadata
        self.observations: Dict[str, ObservationMeta] = {}

        # Stable creation order for trace export / debugging
        self.observation_order: List[str] = []

        # Current focus node
        self.current_observation_id: Optional[str] = None

        self.toolkit = Toolkit()
        self.toolkit.register(self.zoom_in_image)
        self.toolkit.register(self.backtrack_to_observation)
        self.toolkit.register(self.mark_roi)

    def _reset_wsi_session(self) -> None:
        """Clear the current WSI session state."""
        if self.slide is not None:
            try:
                self.slide.close()
            except Exception:
                pass

        self.slide = None
        self.native_mpp = None
        self.current_slide_path = None
        self.current_slide_label = None
        self.work_dir = None

        self.observation_list = []
        self.observations = {}
        self.observation_order = []
        self.current_observation_id = None

    def _require_observation(self, observation_id: str) -> ObservationMeta:
        """Return an observation metadata object or raise a clear error."""
        if observation_id not in self.observations:
            raise ValueError(
                f"Invalid observation_id={observation_id}. "
                f"Available ids: {self.observation_order}"
            )
        return self.observations[observation_id]

    def _render_overview_at_0p5x(
        self,
        slide: openslide.OpenSlide,
        native_mpp: float,
    ) -> Tuple[Image.Image, float]:
        """
        Render the initial whole-slide overview at nominal 0.5x.

        We use thumbnail rendering here for efficiency.
        Patch zooming still uses native-resolution cropping.
        """
        slide_w, slide_h = slide.dimensions
        target_mpp = MAG_TO_MPP["0.5x"]

        raw_w = max(QWEN_SIZE_MULTIPLE, round(slide_w * native_mpp / target_mpp))
        raw_h = max(QWEN_SIZE_MULTIPLE, round(slide_h * native_mpp / target_mpp))

        out_w, out_h = _fit_pixels_to_budget(
            raw_w,
            raw_h,
            multiple=QWEN_SIZE_MULTIPLE,
            min_token_num=IMAGE_MIN_TOKEN_NUM,
            max_token_num=IMAGE_MAX_TOKEN_NUM,
        )

        thumb = slide.get_thumbnail((out_w, out_h))
        thumb = to_rgb(thumb)

        return thumb, target_mpp

    def start_wsi_session(
        self,
        wsi_path: str,
        question: str,
        level0_mpp: Optional[float] = None,
        slide_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Start a new WSI session and create the initial 0.5x overview.

        Returns a small session summary that can be fed into the conversation.
        """
        self._reset_wsi_session()

        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"WSI not found: {wsi_path}")

        slide = openslide.OpenSlide(wsi_path)
        native_mpp = _infer_native_mpp(slide, level0_mpp)

        overview_image, overview_mpp = self._render_overview_at_0p5x(slide, native_mpp)

        work_dir = Path("runs") / "wsi_agent" / shortuuid.uuid()
        work_dir.mkdir(parents=True, exist_ok=True)

        image_path = str(work_dir / "observation_root_0p5x.png")
        overview_image.save(image_path)

        self.slide = slide
        self.native_mpp = native_mpp
        self.current_slide_path = wsi_path
        self.current_slide_label = slide_label or Path(wsi_path).stem
        self.work_dir = work_dir

        self.observation_list.append(image_path)

        root_meta = ObservationMeta(
            observation_id=ROOT_OBSERVATION_ID,
            image_path=image_path,
            label="whole-slide overview",
            native_x=0,
            native_y=0,
            native_w=slide.dimensions[0],
            native_h=slide.dimensions[1],
            effective_mpp=overview_mpp,
            display_mag="0.5x",
            parent_observation_id=None,
            local_child_index=0,
            is_upsampled=False,
            reason="initial whole-slide overview",
        )

        self.observations[ROOT_OBSERVATION_ID] = root_meta
        self.observation_order.append(ROOT_OBSERVATION_ID)
        self.current_observation_id = ROOT_OBSERVATION_ID

        return {
            "observation_id": ROOT_OBSERVATION_ID,
            "image_path": image_path,
            "display_mag": "0.5x",
            "effective_mpp": overview_mpp,
            "question": question,
            "slide_label": self.current_slide_label,
        }

    def zoom_in_image(
        self,
        observation_id: str,
        bbox_2d: List[float],
        target_mag: MAG_LITERAL,
        label: str = "",
        reason: str = "",
    ) -> ToolResponse:
        """
        Create a child observation by zooming into a bbox on a parent observation.

        The bbox is always [x1, y1, x2, y2] on a 0-1000 scale.

        Observation ids are hierarchical path ids such as:
        - root
        - 2x_1
        - 2x_1__5x_1
        - 2x_1__5x_1__10x_3
        """
        try:
            if self.slide is None or self.native_mpp is None or self.work_dir is None:
                raise ValueError("No active WSI session. Call start_wsi_session(...) first.")

            x1, y1, x2, y2 = _validate_relative_bbox_1000(bbox_2d)
            parent = self._require_observation(observation_id)

            expected_child_mag = _next_child_mag(parent.display_mag)
            if expected_child_mag is None:
                raise ValueError(
                    f"Observation {observation_id} is already at the deepest level "
                    f"({parent.display_mag}); no further zoom is allowed."
                )

            if target_mag != expected_child_mag:
                raise ValueError(
                    f"Invalid target_mag={target_mag} for parent {observation_id} "
                    f"at {parent.display_mag}. Expected next level: {expected_child_mag}"
                )

            crop, native_roi, effective_mpp, is_upsampled = get_roi_at_fixed_mag(
                slide=self.slide,
                source_bbox_1000=(x1, y1, x2, y2),
                source_native_x=parent.native_x,
                source_native_y=parent.native_y,
                source_native_w=parent.native_w,
                source_native_h=parent.native_h,
                target_mag=target_mag,
                native_mpp=self.native_mpp,
                min_pixels=QWEN_SIZE_MULTIPLE,
                patch_multiple=QWEN_SIZE_MULTIPLE,
                max_upsample_factor=MAX_UPSAMPLE_FACTOR,
            )

            _validate_patch_pixels_for_wsi(
                width=crop.size[0],
                height=crop.size[1],
                bbox_2d=bbox_2d,
                observation_id=observation_id,
                target_mag=target_mag,
                factor=QWEN_SIZE_MULTIPLE,
            )

            # Child index is local to this parent node only.
            child_index = len(parent.children_ids) + 1
            child_observation_id = _make_child_observation_id(
                parent_observation_id=observation_id,
                target_mag=target_mag,
                child_index=child_index,
            )

            safe_id = _sanitize_observation_id_for_filename(child_observation_id)
            file_name = f"observation_{safe_id}.png"
            image_path = str(self.work_dir / file_name)
            crop.save(image_path)

            child_meta = ObservationMeta(
                observation_id=child_observation_id,
                image_path=image_path,
                label=label or f"zoom from {observation_id}",
                native_x=native_roi[0],
                native_y=native_roi[1],
                native_w=native_roi[2],
                native_h=native_roi[3],
                effective_mpp=effective_mpp,
                display_mag=target_mag,
                parent_observation_id=observation_id,
                local_child_index=child_index,
                is_upsampled=is_upsampled,
                reason=reason,
            )

            self.observation_list.append(image_path)
            self.observations[child_observation_id] = child_meta
            self.observation_order.append(child_observation_id)
            self.observations[observation_id].children_ids.append(child_observation_id)
            self.current_observation_id = child_observation_id

            return ToolResponse(
                content=(
                    f"Created observation {child_observation_id} from {observation_id}. "
                    f"display_mag={target_mag}, "
                    f"effective_mpp={effective_mpp:.4f}, "
                    f"is_upsampled={is_upsampled}, "
                    f"label={child_meta.label}. "
                    f"Current observation is now {child_observation_id}."
                )
            )

        except ValidationError as e:
            return ToolResponse(content=f"Validation error: {str(e)}", is_error=True)
        except Exception as e:
            return ToolResponse(content=f"zoom_in_image failed: {str(e)}", is_error=True)

    def backtrack_to_observation(self, observation_id: str) -> ToolResponse:
        """
        Switch the current focus back to an existing observation.

        This does not re-read the WSI.
        It simply tells the agent to branch from an earlier node.
        """
        try:
            meta = self._require_observation(observation_id)
            self.current_observation_id = observation_id

            return ToolResponse(
                content=(
                    f"Backtracked to observation {observation_id}. "
                    f"display_mag={meta.display_mag}, "
                    f"label={meta.label}. "
                    f"Current observation is now {observation_id}."
                )
            )
        except Exception as e:
            return ToolResponse(content=f"backtrack_to_observation failed: {str(e)}", is_error=True)

    def mark_roi(self, observation_id: str, reason: str) -> ToolResponse:
        """
        Mark an observation as a key ROI / evidence node.
        """
        try:
            meta = self._require_observation(observation_id)
            meta.is_marked_roi = True
            meta.reason = reason

            return ToolResponse(
                content=f"Marked observation {observation_id} as ROI. reason={reason}"
            )
        except Exception as e:
            return ToolResponse(content=f"mark_roi failed: {str(e)}", is_error=True)

    def export_trace_json(self) -> List[Dict[str, Any]]:
        """Export the current observation tree as serializable trace data."""
        trace: List[Dict[str, Any]] = []
        for observation_id in self.observation_order:
            meta = self.observations[observation_id]
            trace.append(
                {
                    "observation_id": meta.observation_id,
                    "image_path": meta.image_path,
                    "label": meta.label,
                    "display_mag": meta.display_mag,
                    "effective_mpp": meta.effective_mpp,
                    "native_x": meta.native_x,
                    "native_y": meta.native_y,
                    "native_w": meta.native_w,
                    "native_h": meta.native_h,
                    "parent_observation_id": meta.parent_observation_id,
                    "children_ids": list(meta.children_ids),
                    "local_child_index": meta.local_child_index,
                    "is_upsampled": meta.is_upsampled,
                    "is_marked_roi": meta.is_marked_roi,
                    "reason": meta.reason,
                }
            )
        return trace

    def build_case_bootstrap_text(
        self,
        question: str,
        session_info: Dict[str, Any],
    ) -> str:
        """Build the first user-facing text for a new WSI case."""
        return (
            f"WSI session started.\n"
            f"- slide_label: {session_info['slide_label']}\n"
            f"- question: {question}\n"
            f"- initial observation_id: {session_info['observation_id']}\n"
            f"- initial magnification: {session_info['display_mag']}\n"
            f"- note: the whole-slide overview starts at 0.5x\n"
            f"- available zoom levels: 0.5x, 2x, 5x, 10x, 20x, 40x\n"
            f"- observation ids are path ids such as 2x_1__5x_1__10x_3\n"
            f"Use zoom_in_image, backtrack_to_observation, and mark_roi when needed."
        )

    def prepare_wsi_case(
        self,
        wsi_path: str,
        question: str,
        level0_mpp: Optional[float] = None,
        slide_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepare a new WSI case.

        This helper starts the session and returns everything the outer ReAct loop
        needs to bootstrap the conversation.
        """
        session_info = self.start_wsi_session(
            wsi_path=wsi_path,
            question=question,
            level0_mpp=level0_mpp,
            slide_label=slide_label,
        )
        bootstrap_text = self.build_case_bootstrap_text(question, session_info)

        return {
            "bootstrap_text": bootstrap_text,
            "observation_id": session_info["observation_id"],
            "image_path": session_info["image_path"],
            "display_mag": session_info["display_mag"],
            "trace": self.export_trace_json(),
        }