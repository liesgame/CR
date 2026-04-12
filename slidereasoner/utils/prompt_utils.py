import os
import json
import re
from typing import Union, Sequence, Any, Type
from pydantic import BaseModel


from agentscope.tool import Toolkit, ToolResponse


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