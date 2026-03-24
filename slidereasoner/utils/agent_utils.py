import json
import os
import re
from typing import Any, Sequence, Type, Union, List, Tuple

from agentscope.tool import Toolkit, ToolResponse
from agentscope.message import (
    TextBlock,
    ImageBlock,
    AudioBlock,
    VideoBlock
)
from pydantic import BaseModel



def convert_tool_result_to_string(
    output: str | List[TextBlock | ImageBlock | AudioBlock | VideoBlock],
) -> tuple[
    str,
    Sequence[
        Tuple[
            str,
            ImageBlock | AudioBlock | TextBlock | VideoBlock,
        ]
    ],
]:
    """Turn the tool result list into a textual output to be compatible
    with the LLM API that doesn't support multimodal data in the tool
    result.

    For URL-based images, the URL is included in the list. For
    base64-encoded images, the local file path where the image is saved
    is included in the returned list.

    Args:
        output (`str | List[TextBlock | ImageBlock | AudioBlock | \
        VideoBlock]`):
            The output of the tool response, including text and multimodal
            data like images and audio.

    Returns:
        `tuple[str, list[Tuple[str, ImageBlock | AudioBlock | VideoBlock \
        TextBlock]]]`:
            A tuple containing the textual representation of the tool
            result and a list of tuples. The first element of each tuple
            is the local file path or URL of the multimodal data, and the
            second element is the corresponding block.
    """

    if isinstance(output, str):
        return output, []

    textual_output = []
    multimodal_data = []
    for block in output:
        assert isinstance(block, dict) and "type" in block, (
            f"Invalid block: {block}, a TextBlock, ImageBlock, "
            f"AudioBlock, or VideoBlock is expected."
        )
        if block["type"] == "text":
            # textual_output.append(block["text"])
            textual_output.append((block["text"], block))

        elif block["type"] in ["image", "audio", "video"]:
            assert "source" in block, (
                f"Invalid {block['type']} block: {block}, 'source' key "
                "is required."
            )
            source = block["source"]
            # Save the image locally and return the file path
            if source["type"] == "url":

                path_multimodal_file = source["url"]

            # elif source["type"] == "base64":
            #     path_multimodal_file = _save_base64_data(
            #         source["media_type"],
            #         source["data"],
            #     )
            #     textual_output.append(
            #         f"The returned {block['type']} can be found "
            #         f"at: {path_multimodal_file}",
            #     )

            else:
                raise ValueError(
                    f"Invalid image source: {block['source']}, "
                    "expected 'url' or 'base64'.",
                )

            multimodal_data.append(
                (path_multimodal_file, block),
            )

        else:
            raise ValueError(
                f"Unsupported block type: {block['type']}, "
                "expected 'text', 'image', 'audio', or 'video'.",
            )

    # if len(textual_output) == 1:
    #     return textual_output[0], multimodal_data

    # else:
    #     return "\n".join("- " + _ for _ in textual_output), multimodal_data

    return textual_output, multimodal_data


def get_prompt_from_file(
    file_path: str,
    return_json: bool,
) -> Union[str, dict]:
    """Get prompt from file"""
    with open(os.path.join(file_path), "r", encoding="utf-8") as f:
        if return_json:
            prompt = json.load(f)
        else:
            prompt = f.read()
    return prompt


def load_prompt_dict() -> dict:
    """Load prompt into dict"""
    prompt_dict = {}
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    prompt_dict["add_note"] = get_prompt_from_file(
        file_path=os.path.join(
            cur_dir,
            "built_in_prompt/prompt_worker_additional_sys_prompt.md",
        ),
        return_json=False,
    )

    prompt_dict["tool_use_rule"] = get_prompt_from_file(
        file_path=os.path.join(
            cur_dir,
            "built_in_prompt/prompt_tool_usage_rules.md",
        ),
        return_json=False,
    )

    prompt_dict["decompose_sys_prompt"] = get_prompt_from_file(
        file_path=os.path.join(
            cur_dir,
            "built_in_prompt/prompt_decompose_subtask.md",
        ),
        return_json=False,
    )

    prompt_dict["expansion_sys_prompt"] = get_prompt_from_file(
        file_path=os.path.join(
            cur_dir,
            "built_in_prompt/prompt_deeper_expansion.md",
        ),
        return_json=False,
    )

    prompt_dict["summarize_sys_prompt"] = get_prompt_from_file(
        file_path=os.path.join(
            cur_dir,
            "built_in_prompt/prompt_inprocess_report.md",
        ),
        return_json=False,
    )

    prompt_dict["reporting_sys_prompt"] = get_prompt_from_file(
        file_path=os.path.join(
            cur_dir,
            "built_in_prompt/prompt_deepresearch_summary_report.md",
        ),
        return_json=False,
    )

    prompt_dict["reflect_sys_prompt"] = get_prompt_from_file(
        file_path=os.path.join(
            cur_dir,
            "built_in_prompt/prompt_reflect_failure.md",
        ),
        return_json=False,
    )

    prompt_dict["reasoning_prompt"] = (
        "## Current Subtask:\n{objective}\n"
        "## Working Plan:\n{meta_planner_agent}\n"
        "{knowledge_gap}\n"
        "## Research Depth:\n{depth}"
    )

    prompt_dict["previous_plan_inst"] = (
        "## Previous Plan:\n{previous_plan}\n"
        "## Current Subtask:\n{objective}\n"
    )

    prompt_dict["max_depth_hint"] = (
        "The search depth has reached the maximum limit. So the "
        "current subtask can not be further decomposed and "
        "expanded anymore. I need to find another way to get it "
        "done no matter what."
    )

    prompt_dict["expansion_inst"] = (
        "Review the web search results and identify whether "
        "there is any information that can potentially help address "
        "checklist items or fulfill knowledge gaps of the task, "
        "but whose content is limited or only briefly mentioned.\n"
        "**Task Description:**\n{objective}\n"
        "**Checklist:**\n{checklist}\n"
        "**Knowledge Gaps:**\n{knowledge_gaps}\n"
        "**Search Results:**\n{search_results}\n"
        "**Output:**\n"
    )

    prompt_dict["follow_up_judge_sys_prompt"] = (
        "To provide sufficient external information for the user's "
        "query, you have conducted a web search to obtain additional "
        "data. However, you found that some of the information, while "
        "important, was insufficient. Consequently, you extracted the "
        "entire content from one of the URLs to gather more "
        "comprehensive information. Now, you must rigorously and "
        "carefully assess whether, after both the web search and "
        "extraction process, the information content is adequate to "
        "address the given task. Be aware that any arbitrary decisions "
        "may result in unnecessary and unacceptable time costs.\n"
    )

    prompt_dict[
        "retry_hint"
    ] = "Something went wrong when {state}. I need to retry."

    prompt_dict["need_deeper_hint"] = (
        "The information is insufficient and I need to make deeper "
        "research to fill the knowledge gap."
    )

    prompt_dict[
        "sufficient_hint"
    ] = "The information after web search and extraction is sufficient enough!"

    prompt_dict["no_result_hint"] = (
        "I mistakenly called the `summarize_intermediate_results` tool as "
        "there exists no milestone result to summarize now."
    )

    prompt_dict["summarize_hint"] = (
        "Based on your work history above, examine which step in the "
        "following working meta_planner_agent has been completed. Mark the completed "
        "step with [DONE] at the end of its line (e.g., k. step k [DONE]) "
        "and leave the uncompleted steps unchanged. You MUST return only "
        "the updated meta_planner_agent, preserving exactly the same format as the "
        "original meta_planner_agent. Do not include any explanations, reasoning, "
        "or section headers such as '## Working Plan:', just output the"
        "updated meta_planner_agent itself."
        "\n\n## Working Plan:\n{meta_planner_agent}"
    )

    prompt_dict["summarize_inst"] = (
        "**Task Description:**\n{objective}\n"
        "**Checklist:**\n{knowledge_gaps}\n"
        "**Knowledge Gaps:**\n{working_plan}\n"
        "**Search Results:**\n{tool_result}"
    )

    prompt_dict["update_report_hint"] = (
        "Due to the overwhelming quantity of information, I have replaced the "
        "original bulk search results from the research phase with the "
        "following report that consolidates and summarizes the essential "
        "findings:\n {intermediate_report}\n\n"
        "Such report has been saved to the {report_path}. "
        "I will now **proceed to the next item** in the working meta_planner_agent."
    )

    prompt_dict["save_report_hint"] = (
        "The milestone results of the current item in working meta_planner_agent "
        "are summarized into the following report:\n{intermediate_report}"
    )

    prompt_dict["reflect_instruction"] = (
        "## Work History:\n{conversation_history}\n"
        "## Working Plan:\n{meta_planner_agent}\n"
    )

    prompt_dict["subtask_complete_hint"] = (
        "Subtask ‘{cur_obj}’ is completed. Now the current subtask "
        "fallbacks to '{next_obj}'"
    )

    return prompt_dict


