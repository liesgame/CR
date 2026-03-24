# -*- coding: utf-8 -*-
import asyncio
import json
import time
import traceback
from typing import Any, Optional, Literal, Type, Dict

from pydantic import BaseModel, Field, ValidationError

from loguru import logger

from agentscope.agent import ReActAgent
from agentscope.model import ChatModelBase
from agentscope.formatter import FormatterBase
from agentscope.memory import MemoryBase, LongTermMemoryBase, InMemoryMemory
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
from agentscope.tool import Toolkit, ToolResponse
from agentscope.agent import ReActAgentBase, ReActAgent
from agentscope.tracing import trace_reply


from slidereasoner.utils.image_utils import _in_jupyter, _resize_keep_ratio_max_side
from slidereasoner.utils.agent_utils import convert_tool_result_to_string

class SimpleReActAgent(ReActAgentBase):
    """A ReAct agent implementation in AgentScope, which supports

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
        parallel_tool_calls: bool = False,
        memory: MemoryBase | None = None,
        max_iters: int = 40, 
        print_hint_msg: bool = False       
            
    ) -> None:

        super().__init__()

        self.name = name
        self._sys_prompt = sys_prompt
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

        self.intermediate_memory = []

        # -------------- Tool management --------------
        # If None, a default Toolkit will be created
        self.toolkit = toolkit or Toolkit()

        self.parallel_tool_calls = parallel_tool_calls

        # If print the reasoning hint messages
        self.print_hint_msg = print_hint_msg
        
        # The maximum number of iterations of the reasoning-acting loops
        self.max_iters = max_iters

        # The hint messages that will be attached to the prompt to guide the
        # agent's behavior before each reasoning step, and cleared after
        # each reasoning step, meaning the hint messages is one-time use only.
        # We use an InMemoryMemory instance to store the hint messages
        self._reasoning_hint_msgs = InMemoryMemory()


        # Variables to record the intermediate state
        # If required structured output model is provided
        self._required_structured_model: Type[BaseModel] | None = None


        # -------------- State registration and hooks --------------
        # Register the status variables
        self.register_state("name")
        self.register_state("_sys_prompt")
        


    @property
    def sys_prompt(self) -> str:
        return self._sys_prompt
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

        # -------------- The reasoning-acting loop --------------
        # Cache the structured output generated in the finish function call
        structured_output = None
        reply_msg = None

        for _ in range(self.max_iters):
            pass
            # -------------- Memory compression --------------


            # -------------- The reasoning process --------------

             
    async def _reasoning(
        self,
        tool_choice: Literal["auto", "none", "required"] | None = None,
    ) -> Msg:
        """Perform the reasoning process."""
        prompt = await self.formatter.format(
            msg = [
                Msg("system", self.analysis_prompt, "system"),
                *await self.memory.get_memory(),
            ]
        )

        # # Clear the hint messages after use
        # await self.memory.delete_by_mark(mark=_MemoryMark.HINT)

        res = await self.model(
            prompt,
            tools=self.toolkit.get_json_schemas(),
            tool_choice=tool_choice,                
        )

        msg = None
        msg = Msg(name=self.name, content=[], role="assistant")

        if self.model.stream:
            async for content_chunk in res:
                msg.content = content_chunk.content

                await self.multimodal_typewriter_print(msg, False)

        else:
            msg.content = list(res.content)

        await self.multimodal_typewriter_print(msg, True)

        # Add a tiny sleep to yield the last message object in the
        # message queue
        await asyncio.sleep(0.001)

        await self.memory.add(msg)

        return msg

    async def _acting(self, tool_call: ToolUseBlock) -> dict | None:
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
                await self.multimodal_typewriter_print(tool_res_msg, True)
                return None
                
            tool_res = await self.toolkit.call_tool_function(tool_call)

            # Async generator handling
            async for chunk in tool_res:
                # Turn into a tool result block
                tool_res_msg.content[0][  # type: ignore[index]
                    "output"
                ] = chunk.content

                await self.multimodal_typewriter_print(tool_res_msg, chunk.is_last)

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

                    await self.multimodal_typewriter_print(tool_res_msg_text, True)

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
                        await self.multimodal_typewriter_print(promoted_msg_image, True)

                        await self.memory.add(promoted_msg_image)
            else:
                await self.memory.add(tool_res_msg)        

    def _print_text_block(
        self,
        msg_id: str,
        name_prefix: str,
        text_content: str,
        thinking_and_text_to_print: list[str],
    ) -> None:
        """Print the text block and thinking block content.

        Args:
            msg_id (`str`):
                The unique identifier of the message
            name_prefix (`str`):
                The prefix for the message, e.g. "{name}: " for text block and
                "{name}(thinking): " for thinking block.
            text_content (`str`):
                The textual content to be printed.
            thinking_and_text_to_print (`list[str]`):
                A list of textual content to be printed together. Here we
                gather the text and thinking blocks to print them together.
        """
        thinking_and_text_to_print.append(
            f"{name_prefix}: {text_content}",
        )
        # The accumulated text and thinking blocks to print
        to_print = "\n".join(thinking_and_text_to_print)

        # The text prefix that has been printed
        if msg_id not in self._stream_prefix:
            self._stream_prefix[msg_id] = {}

        text_prefix = self._stream_prefix[msg_id].get("text", "")

        # Only print when there is new text content
        if len(to_print) > len(text_prefix):
            print(to_print[len(text_prefix) :], end="")

            # Save the printed text prefix
            self._stream_prefix[msg_id]["text"] = to_print


    def _display_image_from_source(
        source: Dict[str, Any],
        print_image: bool = True,
        save_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Returns:
            - Jupyter: None (already displayed)
            - Terminal: returns file path if saved, or None if only printed URL
        """
        if not print_image:
            return None

        src_type = source.get("type")
        if src_type not in ("url", "base64"):
            raise ValueError(f"image source type must be 'url' or 'base64', got: {src_type}")

        is_jup = _in_jupyter()

        # Lazy imports (optional deps)
        PIL_OK = True
        try:
            from PIL import Image  # type: ignore
        except Exception:
            PIL_OK = False

        if src_type == "url":
            url = source.get("url")
            if not url:
                return None

            if is_jup and PIL_OK:
                try:
                    import requests  # type: ignore
                    from IPython.display import display  # type: ignore

                    # img = Image.open(requests.get(url, stream=True, timeout=20).raw)
                    img = Image.open(url)
                    img = _resize_keep_ratio_max_side(img)
                    display(img)
                    return None
                except Exception as e:
                    print(f"[IMAGE] failed to display from url: {url} ({e})")
                    print(f"[IMAGE] {url}")
                    return None

            # terminal / no PIL
            print(f"[IMAGE] {url}")
            return None
        else:
            raise NotImplementedError(f"src_type : {src_type} is not implemented")
        
    def _print_last_block(
        self,
        block: ToolUseBlock
        | ToolResultBlock
        | ImageBlock
        | VideoBlock
        | AudioBlock,
        msg: Msg,
        print_image: bool = False
    ) -> None:
        """Process and print the last content block, and the block type
        is not text, or thinking.

        Args:
            block (`ToolUseBlock | ToolResultBlock | ImageBlock | VideoBlock \
            | AudioBlock`):
                The content block to be printed
            msg (`Msg`):
                The message object
        """
        # TODO: We should consider how to handle the multimodal blocks in the
        #  terminal, since the base64 data may be too long to display.

        if block.get("type") == "image":
            if print_image:
                block_source =  block.get("source")

                self._display_image_from_source(block_source, print_image=print_image)
            else:
                return
        
        if block.get("type") in ["video", "audio"]:
                return

        text_prefix = self._stream_prefix.get(msg.id, {}).get("text", "")

        if text_prefix:
            # Add a newline to separate from previous text content
            print_newline = "" if text_prefix.endswith("\n") else "\n"
            print(
                f"{print_newline}"
                f"{json.dumps(block, indent=4, ensure_ascii=False)}",
            )
        else:
            print(
                f"{msg.name}:"
                f" {json.dumps(block, indent=4, ensure_ascii=False)}",
            )


    async def multimodal_typewriter_print(
            self,
            msg: Msg,
            last: bool = True,
            print_image=True
    ) -> None:
        """The function to display the message.

        Args:
            msg (`Msg`):
                The message object to be printed.
            last (`bool`, defaults to `True`):
                Whether this is the last one in streaming messages. For
                non-streaming message, this should always be `True`.
        """

        # The accumulated textual content to print, including the text blocks and the thinking blocks
        thinking_and_text_to_print = []
        # Todo: We need to handle the multimodal blocks in terminal (images like qwen-agent)

        for block in msg.get_content_blocks():
            if block["type"] == "text":
                self._print_text_block(
                    msg.id,
                    name_prefix=msg.name,
                    text_content=block["text"],
                    thinking_and_text_to_print=thinking_and_text_to_print,
                )
        

            elif block["type"] == "thinking":
                self._print_text_block(
                    msg.id,
                    name_prefix=f"{msg.name}(thinking)",
                    text_content=block["thinking"],
                    thinking_and_text_to_print=thinking_and_text_to_print,
                )
            
            elif last:
                self._print_last_block(block, msg, print_image=print_image)


        # Clean up resources if this is the last message in streaming
        if last and msg.id in self._stream_prefix:
            stream_prefix = self._stream_prefix.pop(msg.id)
            if "text" in stream_prefix and not stream_prefix["text"].endswith(
                "\n",
            ):
                print()