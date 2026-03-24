
import json
from typing import Literal, Tuple, Annotated, Optional, Union, List, Any, Dict, Sequence
from agentscope.message import (
    Msg,
    TextBlock,
    ImageBlock,
    AudioBlock,
    VideoBlock,
    ToolResultBlock,
    ToolUseBlock,
    ThinkingBlock
)

def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        ip = get_ipython()
        return ip is not None and "IPKernelApp" in ip.config
    except Exception:
        return False

def _resize_keep_ratio_max_side(img, max_side: int = 400):
    """Keep aspect ratio; ensure longest side <= max_side."""
    if not max_side or max_side <= 0:
        return img

    try:
        resample = __import__("PIL.Image").Image.Resampling.LANCZOS  # Pillow >= 9
    except Exception:
        from PIL import Image  # type: ignore
        resample = getattr(Image, "LANCZOS", 1)

    w, h = img.size
    if max(w, h) <= max_side:
        return img

    img = img.copy()
    img.thumbnail((max_side, max_side), resample=resample)
    return img

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
    


def _print_text_block(
    _stream_prefix: dict,
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
    if msg_id not in _stream_prefix:
        _stream_prefix[msg_id] = {}

    text_prefix = _stream_prefix[msg_id].get("text", "")

    # Only print when there is new text content
    if len(to_print) > len(text_prefix):
        print(to_print[len(text_prefix) :], end="")

        # Save the printed text prefix
        _stream_prefix[msg_id]["text"] = to_print


def _print_last_block(
    _stream_prefix: dict,
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

            _display_image_from_source(block_source, print_image=print_image)
        else:
            return
    
    if block.get("type") in ["video", "audio"]:
            return

    text_prefix = _stream_prefix.get(msg.id, {}).get("text", "")

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


async def print_multimodal_trace(
        _stream_prefix: Dict,    
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
            _print_text_block(
                _stream_prefix,
                msg.id,
                name_prefix=msg.name,
                text_content=block["text"],
                thinking_and_text_to_print=thinking_and_text_to_print,
            )
    

        elif block["type"] == "thinking":
            _print_text_block(
                _stream_prefix,
                msg.id,
                name_prefix=f"{msg.name}(thinking)",
                text_content=block["thinking"],
                thinking_and_text_to_print=thinking_and_text_to_print,
            )
        
        elif last:
            _print_last_block(_stream_prefix, block, msg, print_image=print_image)


    # Clean up resources if this is the last message in streaming
    if last and msg.id in _stream_prefix:
        stream_prefix = _stream_prefix.pop(msg.id)
        if "text" in stream_prefix and not stream_prefix["text"].endswith(
            "\n",
        ):
            print()