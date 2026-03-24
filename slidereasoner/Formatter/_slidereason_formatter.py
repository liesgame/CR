# -*- coding: utf-8 -*-
# pylint: disable=too-many-branches, too-many-nested-blocks
"""The OpenAI formatter for agentscope."""
import base64
import json
import os
from typing import Any
from urllib.parse import urlparse

import requests

from agentscope.formatter import TruncatedFormatterBase
from agentscope._logging import logger
from agentscope.memory import (
    Msg,
    URLSource,
    TextBlock,
    ImageBlock,
    AudioBlock,
    Base64Source,
    ToolUseBlock,
    ToolResultBlock,
)
from agentscope.token import TokenCounterBase


def _format_openai_image_block(
    image_block: ImageBlock,
) -> dict[str, Any]:
    """Format an image block for OpenAI API.

    Args:
        image_block (`ImageBlock`):
            The image block to format.

    Returns:
        `dict[str, Any]`:
            A dictionary with "type" and "image_url" keys in OpenAI format.

    Raises:
        `ValueError`:
            If the source type is not supported.
    """
    source = image_block["source"]
    if source["type"] == "url":
        url = _to_openai_image_url(source["url"])
    elif source["type"] == "base64":
        data = source["data"]
        media_type = source["media_type"]
        url = f"data:{media_type};base64,{data}"
    else:
        raise ValueError(
            f"Unsupported image source type: {source['type']}",
        )

    return {
        "type": "image_url",
        "image_url": {
            "url": url,
        },
    }


def _to_openai_image_url(url: str) -> str:
    """Convert an image url to openai format. If the given url is a local
    file, it will be converted to base64 format. Otherwise, it will be
    returned directly.

    Args:
        url (`str`):
            The local or public url of the image.
    """
    # See https://platform.openai.com/docs/guides/vision for details of
    # support image extensions.
    support_image_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
    )

    parsed_url = urlparse(url)

    lower_url = url.lower()

    # Web url
    if not os.path.exists(url) and parsed_url.scheme != "":
        if any(lower_url.endswith(_) for _ in support_image_extensions):
            return url

    # Check if it is a local file
    elif os.path.exists(url) and os.path.isfile(url):
        if any(lower_url.endswith(_) for _ in support_image_extensions):
            with open(url, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode(
                    "utf-8",
                )
            extension = parsed_url.path.lower().split(".")[-1]
            mime_type = f"image/{extension}"
            return f"data:{mime_type};base64,{base64_image}"

    raise TypeError(f'"{url}" should end with {support_image_extensions}.')


def _to_openai_audio_data(source: URLSource | Base64Source) -> dict:
    """Covert an audio source to OpenAI format."""
    if source["type"] == "url":
        extension = source["url"].split(".")[-1].lower()
        if extension not in ["wav", "mp3"]:
            raise TypeError(
                f"Unsupported audio file extension: {extension}, "
                "wav and mp3 are supported.",
            )

        parsed_url = urlparse(source["url"])

        if os.path.exists(source["url"]):
            with open(source["url"], "rb") as audio_file:
                data = base64.b64encode(audio_file.read()).decode("utf-8")

        # web url
        elif parsed_url.scheme != "":
            response = requests.get(source["url"])
            response.raise_for_status()
            data = base64.b64encode(response.content).decode("utf-8")

        else:
            raise ValueError(
                f"Unsupported audio source: {source['url']}, "
                "it should be a local file or a web URL.",
            )

        return {
            "data": data,
            "format": extension,
        }

    if source["type"] == "base64":
        data = source["data"]
        media_type = source["media_type"]

        if media_type not in ["audio/wav", "audio/mp3"]:
            raise TypeError(
                f"Unsupported audio media type: {media_type}, "
                "only audio/wav and audio/mp3 are supported.",
            )

        return {
            "data": data,
            "format": media_type.split("/")[-1],
        }

    raise TypeError(f"Unsupported audio source: {source['type']}.")


class SlideReasonChatFormatter(TruncatedFormatterBase):
    """The OpenAI formatter class for chatbot scenario, where only a user
    and an agent are involved. We use the `name` field in OpenAI API to
    identify different entities in the conversation.
    """

    support_tools_api: bool = True
    """Whether support tools API"""

    support_multiagent: bool = True
    """Whether support multi-agent conversation"""

    support_vision: bool = True
    """Whether support vision models"""

    supported_blocks: list[type] = [
        TextBlock,
        ImageBlock,
        AudioBlock,
        ToolUseBlock,
        ToolResultBlock,
    ]
    """Supported message blocks for OpenAI API"""

    def __init__(
        self,
        promote_tool_result_images: bool = True,
        token_counter: TokenCounterBase | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the OpenAI chat formatter.

        Args:
            promote_tool_result_images (`bool`, defaults to `True`):
                Whether to promote images from tool results to user messages.
                Most LLM APIs don't support images in tool result blocks, but
                do support them in user message blocks. When `True`, images are
                extracted and appended as a separate user message with
                explanatory text indicating their source.
            token_counter (`TokenCounterBase | None`, optional):
                A token counter instance used to count tokens in the messages.
                If not provided, the formatter will format the messages
                without considering token limits.
            max_tokens (`int | None`, optional):
                The maximum number of tokens allowed in the formatted
                messages. If not provided, the formatter will not truncate
                the messages.
        """
        super().__init__(token_counter=token_counter, max_tokens=max_tokens)
        self.promote_tool_result_images = promote_tool_result_images

    async def _format(
        self,
        msgs: list[Msg],
    ) -> list[dict[str, Any]]:
        """Format message objects into OpenAI API required format.

        Args:
            msgs (`list[Msg]`):
                The list of Msg objects to format.

        Returns:
            `list[dict[str, Any]]`:
                A list of dictionaries, where each dictionary has "name",
                "role", and "content" keys.
        """
        self.assert_list_of_msgs(msgs)

        messages: list[dict] = []
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            content_blocks = []
            tool_calls = []

            for block in msg.get_content_blocks():
                typ = block.get("type")
                if typ == "text":
                    content_blocks.append({**block})

                elif typ == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id"),
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(
                                    block.get("input", {}),
                                    ensure_ascii=False,
                                ),
                            },
                        },
                    )

                elif typ == "tool_result":
                    (
                        textual_output,
                        multimodal_data,
                    ) = self.convert_tool_result_to_string(block["output"])

                    metadata = block["metadata"]
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("id"),
                            "content": (  # type: ignore[arg-type]
                                textual_output
                            ),
                            "name": block.get("name"),
                        },
                    )

                    # Then, handle the multimodal data if any
                    promoted_blocks: list = []
                    for idx, url, multimodal_block in enumerate(multimodal_data):
                        if (
                            multimodal_block["type"] == "image"
                            and self.promote_tool_result_images
                        ):
                            promoted_blocks.extend(
                                [
                                    TextBlock(
                                        type="text",
                                        text=f"\n- The image from '{url} (The observation_index of this image is {metadata['observation_index'][idx]} by execting zoom_in_image from the image with observation_index={metadata['source_observation_index'][idx]}) ': ",
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

                    if promoted_blocks:
                        # Insert promoted blocks as new user message(s)
                        promoted_blocks = [
                            TextBlock(
                                type="text",
                                text="<system-info>The following are "
                                "the image contents from the tool "
                                f"result of '{block['name']}':",
                            ),
                            *promoted_blocks,
                            TextBlock(
                                type="text",
                                text="</system-info>",
                            ),
                        ]

                        msgs.insert(
                            i + 1,
                            Msg(
                                name="user",
                                content=promoted_blocks,
                                role="user",
                            ),
                        )

                elif typ == "image":
                    content_blocks.append(
                        _format_openai_image_block(
                            block,  # type: ignore[arg-type]
                        ),
                    )

                elif typ == "audio":
                    input_audio = _to_openai_audio_data(block["source"])
                    content_blocks.append(
                        {
                            "type": "input_audio",
                            "input_audio": input_audio,
                        },
                    )

                else:
                    logger.warning(
                        "Unsupported block type %s in the message, skipped.",
                        typ,
                    )

            msg_openai = {
                "role": msg.role,
                "name": msg.name,
                "content": content_blocks or None,
            }

            if tool_calls:
                msg_openai["tool_calls"] = tool_calls

            # When both content and tool_calls are None, skipped
            if msg_openai["content"] or msg_openai.get("tool_calls"):
                messages.append(msg_openai)

            # Move to next message
            i += 1

        return messages