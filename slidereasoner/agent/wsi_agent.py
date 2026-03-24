# -*- coding: utf-8 -*-
"""WSI zoom-in agent built on top of AgentScope.

This module turns a whole-slide image (WSI) into an interactive visual
reasoning session:

1. Create an initial thumbnail observation from the WSI.
2. Let the model reason over the current image observations.
3. Allow the model to call ``zoom_in_image`` with a bbox and target mpp.
4. Convert the crop result into a new observation and feed it back into
   the conversation as a new user-visible image.

The implementation borrows the overall reasoning/acting loop from the
project notebook and packages it into a reusable Python module.
"""

from __future__ import annotations

import asyncio
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple, Type

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


DEFAULT_WSI_AGENT_PROMPT = """You are a pathology whole-slide image reasoning agent.

You work on one WSI session at a time. The conversation may contain one or
more observations, and each image is identified by an observation_index.

When solving the task:
- Start from the currently available observations and inspect them carefully.
- If you need more detail, call `zoom_in_image`.
- `bbox_2d` uses relative coordinates on a 0-1000 scale in the current
  observation image: [x1, y1, x2, y2].
- `target_mpp` is the target microns-per-pixel for the new crop. Smaller
  mpp means higher magnification / more detail.
- Reuse the returned observation_index in later tool calls.
- Avoid repeatedly requesting the same region unless there is a clear reason.
- If the crop tool reports the patch is too small or too large, adjust the
  bbox and retry.

When you have enough evidence, answer directly and clearly."""


@dataclass(slots=True)
class ObservationMeta:
    """Metadata for each rendered observation image."""

    observation_index: int
    image_path: str
    label: str
    level0_x: int
    level0_y: int
    level0_w: int
    level0_h: int
    effective_mpp: float
    parent_observation_index: Optional[int]


def _infer_level0_mpp(
    slide: openslide.OpenSlide,
    override_level0_mpp: Optional[float] = None,
) -> float:
    """Infer level-0 mpp from OpenSlide properties."""

    if override_level0_mpp is not None:
        if override_level0_mpp <= 0:
            raise ValueError(
                f"override_level0_mpp must be positive, got {override_level0_mpp}",
            )
        return float(override_level0_mpp)

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
                # Hamamatsu often stores pixels per mm; convert to um/px.
                return 1000.0 / parsed
            return parsed

    raise ValueError(
        "Failed to infer level0 mpp from slide properties. "
        "Please provide `level0_mpp` explicitly when starting the session.",
    )


def _validate_relative_bbox(bbox_2d: Sequence[float]) -> Tuple[float, float, float, float]:
    """Validate a 0-1000 relative bbox."""

    if len(bbox_2d) != 4:
        raise ValueError(f"bbox_2d must contain 4 values, got {bbox_2d}")

    rel_x1, rel_y1, rel_x2, rel_y2 = [float(v) for v in bbox_2d]
    values = (rel_x1, rel_y1, rel_x2, rel_y2)
    if any(v < 0 or v > 1000 for v in values):
        raise ValueError(
            f"bbox_2d values must all be within [0, 1000], got {bbox_2d}",
        )
    if rel_x2 <= rel_x1 or rel_y2 <= rel_y1:
        raise ValueError(
            f"bbox_2d must satisfy x2>x1 and y2>y1, got {bbox_2d}",
        )
    return rel_x1, rel_y1, rel_x2, rel_y2


def _validate_patch_pixels_for_wsi(
    bbox_2d: Sequence[float],
    width: int,
    height: int,
    observation_index: int,
    source_mpp: float,
    target_mpp: float,
    factor: int = 32,
) -> None:
    """Validate image size constraints for model-side vision ingestion."""

    min_pixels = IMAGE_MIN_TOKEN_NUM * factor * factor
    max_pixels = IMAGE_MAX_TOKEN_NUM * factor * factor
    pixels = width * height
    rel_x1, rel_y1, rel_x2, rel_y2 = bbox_2d

    if min(width, height) <= 0:
        raise ValueError(
            f"Invalid patch size {width}x{height} for bbox_2d={bbox_2d}",
        )

    if max(width, height) / min(width, height) > MAX_RATIO:
        raise ValueError(
            "WSI_PATCH_INVALID_ASPECT_RATIO: "
            f"bbox_2d={list(bbox_2d)}; size={width}x{height}; "
            f"max_ratio={MAX_RATIO}. Action: adjust bbox to reduce the aspect ratio.",
        )

    if pixels > max_pixels:
        raise ValueError(
            "WSI_PATCH_TOO_LARGE: "
            f"observation_index={observation_index}; "
            f"source_mpp={source_mpp:.6f}; target_mpp={target_mpp:.6f}; "
            f"bbox_2d=[{rel_x1},{rel_y1},{rel_x2},{rel_y2}]; "
            f"patch_wh={width}x{height}; patch_pixels={pixels}; "
            f"max_pixels={max_pixels}. "
            "Action: shrink bbox (reduce area) and retry.",
        )

    if pixels < min_pixels:
        raise ValueError(
            "WSI_PATCH_TOO_SMALL: "
            f"observation_index={observation_index}; "
            f"source_mpp={source_mpp:.6f}; target_mpp={target_mpp:.6f}; "
            f"bbox_2d=[{rel_x1},{rel_y1},{rel_x2},{rel_y2}]; "
            f"patch_wh={width}x{height}; patch_pixels={pixels}; "
            f"min_pixels={min_pixels}. "
            "Action: expand bbox (increase area) and retry.",
        )


def get_roi_at_mpp_optimized(
    slide: openslide.OpenSlide,
    source_roi: Tuple[int, int, int, int],
    source_mpp: float,
    source_level0_x: int,
    source_level0_y: int,
    target_mpp: float,
    level0_mpp: float,
    min_pixels: int = 32,
) -> Tuple[Image.Image, Tuple[int, int, int, int], float]:
    """Read a ROI from WSI at an efficient level without upsampling.

    Args:
        slide: OpenSlide instance.
        source_roi: ROI in the parent observation pixel coordinates, as
            ``(x, y, width, height)``.
        source_mpp: Effective mpp of the parent observation.
        source_level0_x: Parent observation origin at level-0 x.
        source_level0_y: Parent observation origin at level-0 y.
        target_mpp: Requested target mpp for the child crop.
        level0_mpp: Native level-0 mpp.
        min_pixels: Minimum output side length.

    Returns:
        The rendered RGB image, its level-0 ROI, and the effective mpp.
    """

    if source_mpp <= 0 or target_mpp <= 0 or level0_mpp <= 0:
        raise ValueError(
            "source_mpp, target_mpp, and level0_mpp must all be positive.",
        )

    native_mpp = level0_mpp
    width, height = slide.dimensions

    src_x, src_y, src_w, src_h = source_roi
    if src_x < 0 or src_y < 0 or src_w <= 0 or src_h <= 0:
        raise ValueError(
            f"Invalid source_roi={source_roi}; all dimensions must be positive.",
        )
    if source_level0_x < 0 or source_level0_y < 0:
        raise ValueError(
            "source_level0_x and source_level0_y must be non-negative.",
        )

    scale_factor_to_level0 = source_mpp / native_mpp
    level0_x = math.floor(src_x * scale_factor_to_level0) + source_level0_x
    level0_y = math.floor(src_y * scale_factor_to_level0) + source_level0_y
    level0_w = max(1, math.floor(src_w * scale_factor_to_level0))
    level0_h = max(1, math.floor(src_h * scale_factor_to_level0))
    output_level0_roi = (level0_x, level0_y, level0_w, level0_h)

    if level0_x + level0_w > width or level0_y + level0_h > height:
        raise ValueError(
            "Requested level-0 ROI exceeds slide bounds: "
            f"roi={output_level0_roi}, slide_size={slide.dimensions}",
        )

    if target_mpp < native_mpp:
        effective_mpp = native_mpp
        optimal_level = 0
        optimal_level_w = max(min_pixels, level0_w)
        optimal_level_h = max(min_pixels, level0_h)
        target_w = max(min_pixels, level0_w)
        target_h = max(min_pixels, level0_h)
    else:
        effective_mpp = target_mpp
        level_mpps = [native_mpp * ds for ds in slide.level_downsamples]
        valid_levels = [idx for idx, mpp in enumerate(level_mpps) if mpp <= target_mpp]
        if not valid_levels:
            optimal_level = 0
        else:
            optimal_level = valid_levels[-1]

        optimal_level_downsample = slide.level_downsamples[optimal_level]
        optimal_level_w = max(min_pixels, math.floor(level0_w / optimal_level_downsample))
        optimal_level_h = max(min_pixels, math.floor(level0_h / optimal_level_downsample))

        scale_factor_to_target = native_mpp / target_mpp
        target_w = max(min_pixels, math.floor(level0_w * scale_factor_to_target))
        target_h = max(min_pixels, math.floor(level0_h * scale_factor_to_target))

    intermediate_image = slide.read_region(
        (level0_x, level0_y),
        optimal_level,
        (optimal_level_w, optimal_level_h),
    ).convert("RGB")

    if intermediate_image.size != (target_w, target_h):
        target_image = intermediate_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    else:
        target_image = intermediate_image

    return target_image, output_level0_roi, effective_mpp


class WSIReActAgent(ReActAgentBase):
    """A reusable WSI agent with a real zoom-in tool."""

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
        max_iters: int = 20,
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
        self.min_pixels = max(1, min_pixels)

        self._stream_prefix: Dict[str, Dict[str, str]] = {}
        self.memory = memory or InMemoryMemory()
        self.toolkit = toolkit or Toolkit()
        self._required_structured_model: Type[BaseModel] | None = None

        self.workspace_root = Path(
            work_dir
            or "/data/home/zhangchen/project/RL/SlideReasoner/slidereasoner/workspace/wsi_agent",
        )
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        self.slide: Optional[openslide.OpenSlide] = None
        self.slide_path: Optional[str] = None
        self.slide_label: Optional[str] = None
        self.level0_mpp: Optional[float] = None
        self.session_dir: Optional[Path] = None
        self.action_idx: int = 1
        self.observation_list: List[str] = []
        self.observation_meta: List[ObservationMeta] = []

        self.toolkit.register_tool_function(self.zoom_in_image)

        self.register_state("name")
        self.register_state("_sys_prompt")

    @property
    def sys_prompt(self) -> str:
        return self._sys_prompt

    def reset_session(self) -> None:
        """Reset current WSI session state while keeping the agent instance."""

        self.memory = InMemoryMemory()
        self._stream_prefix.clear()
        self.action_idx = 1
        self.observation_list = []
        self.observation_meta = []
        if self.slide is not None:
            self.slide.close()
        self.slide = None
        self.slide_path = None
        self.slide_label = None
        self.level0_mpp = None
        self.session_dir = None

    def start_wsi_session(
        self,
        wsi_path: str,
        question: str,
        level0_mpp: Optional[float] = None,
        thumbnail_size: Tuple[int, int] = (1024, 1024),
        slide_label: Optional[str] = None,
    ) -> Msg:
        """Open a WSI and create the initial thumbnail observation message."""

        self.reset_session()

        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")

        self.slide_path = os.path.abspath(wsi_path)
        self.slide_label = slide_label or Path(wsi_path).stem
        self.slide = openslide.OpenSlide(self.slide_path)
        self.level0_mpp = _infer_level0_mpp(self.slide, level0_mpp)

        session_name = f"{self.slide_label}_{shortuuid.uuid()}"
        self.session_dir = self.workspace_root / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        thumbnail = to_rgb(self.slide.get_thumbnail(thumbnail_size))
        thumbnail_path = str(self.session_dir / f"observation_0_{shortuuid.uuid()}.png")
        thumbnail.save(thumbnail_path)

        level0_w, level0_h = self.slide.dimensions
        thumb_w, thumb_h = thumbnail.size
        scale_x = level0_w / max(1, thumb_w)
        scale_y = level0_h / max(1, thumb_h)
        effective_mpp = self.level0_mpp * max(scale_x, scale_y)

        meta = ObservationMeta(
            observation_index=0,
            image_path=thumbnail_path,
            label="whole slide thumbnail",
            level0_x=0,
            level0_y=0,
            level0_w=level0_w,
            level0_h=level0_h,
            effective_mpp=effective_mpp,
            parent_observation_index=None,
        )
        self.observation_list.append(thumbnail_path)
        self.observation_meta.append(meta)

        return Msg(
            name="user",
            content=[
                TextBlock(
                    type="text",
                    text=(
                        f"The following image is observation_index 0 for WSI "
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
