import asyncio
from enum import Enum
import os
import base64
import json
from typing import Literal, Tuple, Annotated, Optional, Union, List, Any, Dict, Sequence, Type

from pydantic import BaseModel, BaseModel, ValidationError, Field

import math
import base64
import copy
from io import BytesIO
from PIL import ImageColor, Image, ImageDraw, ImageFont

from transformers import Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration, AutoProcessor

from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import OpenAIChatFormatter, OpenAIMultiAgentFormatter, FormatterBase
from agentscope.memory import InMemoryMemory, MemoryBase
from agentscope.model import OpenAIChatModel, ChatModelBase
from agentscope.agent import ReActAgent, ReActAgentBase
from agentscope.tool import ToolResponse, Toolkit, execute_python_code, write_text_file
from agentscope.message import (
    Msg,
    Base64Source,
    TextBlock,
    URLSource,
    ThinkingBlock,
    ImageBlock,
    ToolUseBlock,
    ToolResultBlock,
    VideoBlock,
    AudioBlock
)
from agentscope.tracing import trace_reply
import requests
import openai
import openslide
import uuid
import shortuuid
import numpy as np
import cv2
import matplotlib.pyplot as plt
from qwen_vl_utils import process_vision_info, extract_vision_info

from slidereasoner.utils.agent_utils import (
    convert_tool_result_to_string
)

# image tools utils
from slidereasoner.utils.image_utils import (
    round_by_factor,
    ceil_by_factor, 
    floor_by_factor, 
    to_rgb, 
    smart_resize, 
    maybe_resize_bbox,
    display_image_from_source
)

from slidereasoner.utils.logging_utils import logger

from slidereasoner.utils.print_utils import print_multimodal_trace


class _MemoryMark(str, Enum):
    """The memory marks used in the ReAct agent."""

    HINT = "hint"
    """Used to mark the hint messages that will be cleared after use."""

    COMPRESSED = "compressed"
    """Used to mark the compressed messages in the memory."""



class BaseCropAgent(ReActAgentBase):
    """A BaseCropAgent agent implementation in AgentScope, which supports

    - Realtime steering
    - API-based (parallel) tool calling
    - Hooks around reasoning, acting, reply, observe and print functions
    - Structured output generation
    """

    finish_function_name: str = "generate_response"
    """The name of the function used to generate structured output. Only
    registered when structured output model is provided in the reply call."""



    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: ChatModelBase,
        formatter: FormatterBase,
        toolkit: Toolkit | None = None,
        memory: MemoryBase | None = None,
        parallel_tool_calls: bool = False,
        print_hint_msg: bool = False,
        max_iters: int = 40,
    ) -> None:
        """Initialize the ReAct agent

        Args:
            name (`str`):
                The name of the agent.
            sys_prompt (`str`):
                The system prompt of the agent.
            model (`ChatModelBase`):
                The chat model used by the agent.
            formatter (`FormatterBase`):
                The formatter used to format the messages into the required
                format of the model API provider.
            toolkit (`Toolkit | None`, optional):
                A `Toolkit` object that contains the tool functions. If not
                provided, a default empty `Toolkit` will be created.
            memory (`MemoryBase | None`, optional):
                The memory used to store the dialogue history. If not provided,
                a default `InMemoryMemory` will be created, which stores
                messages in a list in memory.

            parallel_tool_calls (`bool`, defaults to `False`):
                When LLM generates multiple tool calls, whether to execute
                them in parallel.
            max_iters (`int`, defaults to `10`):
                The maximum number of iterations of the reasoning-acting loops.

        """
        super().__init__()

        # Static variables in the agent
        self.name = name
        self._sys_prompt = sys_prompt

        # The maximum number of iterations of the reasoning-acting loops
        self.max_iters = max_iters
        self.model = model
        self.formatter = formatter


        # -------------- Multi-Modulity print--------------
        # override in AgentBase
        # The prefix used in streaming printing, which will save the
        # accumulated text and audio streaming data for each message id.
        # e.g. {"text": "xxx", "audio": (stream_obj, "{base64_data}")}
        self._stream_prefix = {}


        # -------------- Memory management --------------
        # Record the dialogue history in the memory
        self.memory = memory or InMemoryMemory()

        # -------------- Tool management --------------
        # If None, a default Toolkit will be created
        self.toolkit = toolkit or Toolkit()


        self.parallel_tool_calls = parallel_tool_calls


        # Variables to record the intermediate state

        # If required structured output model is provided
        self._required_structured_model: Type[BaseModel] | None = None

        # -------------- State registration and hooks --------------
        # Register the status variables
        self.register_state("name")
        self.register_state("_sys_prompt")


        # tools
        self.obersvaton_list = []


    @property
    def sys_prompt(self) -> str:
        """The dynamic system prompt of the agent."""
        # agent_skill_prompt = self.toolkit.get_agent_skill_prompt()
        # if agent_skill_prompt:
        #     return self._sys_prompt + "\n\n" + agent_skill_prompt
        # else:
        #     return self._sys_prompt
        return self._sys_prompt
    
    @trace_reply
    async def reply(  # pylint: disable=too-many-branches
        self,
        msg: Msg | list[Msg] | None = None,
        structured_model: Type[BaseModel] | None = None,
    ) -> Msg:
        """Generate a reply based on the current state and input arguments.

        Args:
            msg (`Msg | list[Msg] | None`, optional):
                The input message(s) to the agent.
            structured_model (`Type[BaseModel] | None`, optional):
                The required structured output model. If provided, the agent
                is expected to generate structured output in the `metadata`
                field of the output message.

        Returns:
            `Msg`:
                The output message generated by the agent.
        """
        # Record the input message(s) in the memory
        await self.memory.add(msg)



        # Control if LLM generates tool calls in each reasoning step
        tool_choice: Literal["auto", "none", "required"] | None = None


        # -------------- Structured output management --------------
        self._required_structured_model = structured_model
        # Record structured output model if provided
        if structured_model:
            # Register generate_response tool only when structured output
            # is required
            if self.finish_function_name not in self.toolkit.tools:
                self.toolkit.register_tool_function(
                    getattr(self, self.finish_function_name),
                )

            # Set the structured output model
            self.toolkit.set_extended_model(
                self.finish_function_name,
                structured_model,
            )
            tool_choice = "required"
        else:
            # Remove generate_response tool if no structured output is required
            self.toolkit.remove_tool_function(self.finish_function_name)


    # pylint: disable=too-many-branches

    async def reasoning(
            self, 
            tool_choice: Literal["auto", "none", "required"] | None = None,
    ) -> Msg:
        """Perform the reasoning process."""

        prompt = await self.formatter.format(
            msgs=[
                Msg("system", self.sys_prompt, "system"),
                *await self.memory.get_memory(),

            ],
        )


        # Clear the hint messages after use (maybe need. I prefer mulit memory version)
        # await self.memory.delete_by_mark(mark=_MemoryMark.HINT)

        res = await self.model(
            prompt,
            tools=self.toolkit.get_json_schemas(),
            tool_choice=tool_choice,
        )

        # handle output from the model
        interrupted_by_user = False
        msg = None


        msg = Msg(name=self.name, content=[], role="assistant")


        if self.model.stream:
            async for content_chunk in res:
                msg.content = content_chunk.content
                await self.print_mulit(msg, False)

        else:
            msg.content = list(res.content)

        await print_multimodal_trace(self._stream_prefix, msg, True)

        # Add a tiny sleep to yield the last message object in the message queue
        await asyncio.sleep(0.001)

        # None will be ignored by the memory
        await self.memory.add(msg)

        return msg



    async def acting(self, tool_call: ToolUseBlock) -> dict | None:

        """Perform the acting process, and return the structured output if
        it's generated and verified in the finish function call.

        Args:
            tool_call (`ToolUseBlock`):
                The tool use block to be executed.

        Returns:
            `Union[dict, None]`:
                Return the structured output if it's verified in the finish
                function call, otherwise return None.
        """

        tool_res_msg = Msg(
            "system",
            [
                ToolResultBlock(
                    type="tool_result",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    output=[],
                ),
            ],
            "system",
        )

        try:
            # Execute the tool call
            print(f"[TOOL_CALL] {tool_call["name"]}")
            if tool_call["name"] not in self.toolkit.tools:
                tool_res_msg.content[0][  # type: ignore[index]
                    "output"
                ] = [
                    TextBlock(
                        type="text",
                        text=f"FunctionNotFoundError: {tool_call["name"]} is not available. You must not call {tool_call["name"]} again. Available tools: {f'{ [ _ for _ in toolkit.tools.keys() ]}.'}"
                    )
                    ]
                await print_multimodal_trace(self._stream_prefix, tool_res_msg, True)
                return None
                
            tool_res = await self.toolkit.call_tool_function(tool_call)

            # Async generator handling
            async for chunk in tool_res:
                # Turn into a tool result block
                tool_res_msg.content[0][  # type: ignore[index]
                    "output"
                ] = chunk.content

                await print_multimodal_trace(self._stream_prefix, tool_res_msg, chunk.is_last)

                # Raise the CancelledError to handle the interruption in the
                # handle_interrupt function
                if chunk.is_interrupted:
                    raise asyncio.CancelledError()
                
            return None

        finally:
            # Record the tool result message in the memory

            if tool_call['name'] == "zoom_in_image":

                if chunk.metadata['success']:
                    print(f"[TOOL_RESPONSE] {tool_call["name"]}")
                    (
                        textual_output,
                        multimodal_data,
                    ) = convert_tool_result_to_string(tool_res_msg.content[0]["output"])

                    if len(textual_output) != 1:
                        raise ValueError(f"the respone of zoom_in_image must contain only 1 textual_output, but got len(textual_output) = {len(textual_output)}")

                    if len(multimodal_data) != 1:
                        raise ValueError(f"the respone of zoom_in_image must contain only 1 multimodal_data, but got len(multimodal_data) = {len(multimodal_data)}")
                    textual_output = textual_output[0]
                    multimodal_data = multimodal_data[0]
                    
                    tool_res_msg_text = Msg(
                        "system",
                        [
                            ToolResultBlock(
                                type="tool_result",
                                id=tool_res_msg.content[0]['id'],
                                name=tool_res_msg.content[0]['name'],
                                output=[
                                    textual_output[1]
                                ],
                            ),
                        ],
                        "system",
                    )

                    await print_multimodal_trace(self._stream_prefix, tool_res_msg_text, True)

                    await self.memory.add(tool_res_msg_text)

                    promoted_blocks_image: list = []

                    url, multimodal_block = multimodal_data
                    if (
                        multimodal_block["type"] == "image"
                    ):
                        promoted_blocks_image.extend(
                            [
                                TextBlock(
                                    type="text",
                                    text=f"\n- This image is labeled with observation_index {chunk.metadata['observation_index']}, generated by applying the zoom_in_image function to the image with observation_index {chunk.metadata['source_observation_index']}: ",
                                ),
                                ImageBlock(
                                    type="image",
                                    source=URLSource(
                                        type="url",
                                        url=url,
                                    ),
                                ),
                            ],
                        )
                    if promoted_blocks_image:
                        # Insert promoted blocks as new user message(s)
                        promoted_blocks_image = [
                            TextBlock(
                                type="text",
                                text="<system-info>The following are "
                                "the image contents from the tool "
                                f"result of '{tool_res_msg.content[0]['name']}':",
                            ),
                            *promoted_blocks_image,
                            TextBlock(
                                type="text",
                                text="</system-info>",
                            ),
                        ]
                        promoted_msg_image = Msg(
                            name="user",
                            content=promoted_blocks_image,
                            role="user",
                        )
                        await print_multimodal_trace(self._stream_prefix, promoted_msg_image, True)

                        await self.memory.add(promoted_msg_image)
            else:
                await self.memory.add(tool_res_msg)



    def generate_response(
        self,
        **kwargs: Any,
    ) -> ToolResponse:
        """
        Generate required structured output by this function and return it
        """

        structured_output = None
        # Prepare structured output
        if self._required_structured_model:
            try:
                # Use the metadata field of the message to store the
                # structured output
                structured_output = (
                    self._required_structured_model.model_validate(
                        kwargs,
                    ).model_dump()
                )

            except ValidationError as e:
                return ToolResponse(
                    content=[
                        TextBlock(
                            type="text",
                            text=f"Arguments Validation Error: {e}",
                        ),
                    ],
                    metadata={
                        "success": False,
                        "structured_output": {},
                    },
                )
        else:
            logger.warning(
                "The generate_response function is called when no structured "
                "output model is required.",
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
        bbox_2d: Annotated[list[float], Field(min_length=4, max_length=4)],
        label: str,
        observation_index: Annotated[int, Field(ge=0)],
    ) -> ToolResponse:

        """Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.

        Args:
            bbox_2d (`list[float]`):
                The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner (x1 is left and y1 is top) and (x2, y2) is the bottom-right corner (x2 is right and y2 is bottom). The bounding box uses the relative coordinated with range 0-1000.
            label (`str`):
                The name or label of the object in the specified bounding box.
            observation_index (`int`):
                The index of the image to zoom-in(starting from 0). The index of the image to crop.

        Returns:
            `ToolResponse`:
                The tool response containing the result of the writing operation.
        """
        global action_idx
        try: 
            img_file_path = self.obersvaton_list[observation_index]

            if not os.path.exists(img_file_path):
                raise ValueError(f'img_file_path: {img_file_path} is not exist')

            original_image = to_rgb(Image.open(img_file_path))

            img_width, img_height  = original_image.size

            rel_x1, rel_y1, rel_x2, rel_y2 = bbox_2d

            abs_x1 = math.floor(rel_x1 / 1000. * img_width)
            abs_y1 = math.floor(rel_y1 / 1000. * img_height)
            abs_x2 = math.ceil(rel_x2 / 1000. * img_width)
            abs_y2 = math.ceil(rel_y2 / 1000. * img_height)

            # maybe we can directly crop * 32 insteaded of resize ?

            validated_bbox = maybe_resize_bbox(abs_x1, abs_y1, abs_x2, abs_y2, img_width, img_height)


            left, top, right, bottom = validated_bbox


            # Crop the image

            cropped_image = original_image.crop((left, top, right, bottom))

            new_w, new_h = smart_resize((right - left), (bottom - top), factor=32)

            validate_MinMax_pixels_test(bbox_2d, new_w, new_h, max_pixels= 16384 * 32 * 32, min_pixels= 4 * 32 * 32)    
            
            cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)

            # save crop image

            output_path = os.path.abspath(os.path.join(work_dir, f'observation_{action_idx}_{shortuuid.uuid()}.png'))
            cropped_image.save(output_path)

            new_img_idx = len(obersvaton_list)
            obersvaton_list.append(output_path)
            action_idx += 1
            return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=(
                        "zoom_in_image succeeded.\n"
                        "Generated a zoomed-in ROI view (image) from the selected bounding box.\n"
                        f"- returned observation_index: {new_img_idx}\n"
                        f"- source observation_index: {observation_index}\n"
                        f"- label: {label}\n"
                        f"To reference this ROI view in later tool calls, use observation_index={new_img_idx}."
                    ),
                ),
                ImageBlock(
                    type="image",
                    source=URLSource(
                        type="url",
                        url=output_path
                    )
                )
            ],
            metadata={
                "success": True,
                "observation_index": new_img_idx,
                "source_observation_index" : observation_index
                }
            )
        
        except Exception as e:
            obs = f'Tool Execution Error {str(e)}'
            return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=f"Failure to execute zoom_in_image, error: {obs}.",
                )
            ],
            metadata={"success": False}
            )
