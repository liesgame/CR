import openslide
from PIL import Image
import os
import json
from transformers import AutoProcessor
import re
import time # 建議在重試之間加入短暫的延遲
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import gc
import torch

from openai import OpenAI
import openai
import base64
import mimetypes
def image_path_to_base64_uri(filepath: str) -> str:
    """
    将本地图片文件路径转换为 Base64 编码的 Data URI。
    """
    # 1. 获取文件的 MIME 类型 (例如 'image/png')
    mime_type, _ = mimetypes.guess_type(filepath)
    if mime_type is None:
        mime_type = "application/octet-stream"  # 如果无法确定，使用通用二进制流类型

    # 2. 读取文件内容 (二进制模式)
    with open(filepath, "rb") as image_file:
        # 3. 进行 Base64 编码
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    # 4. 组装成 Data URI 格式
    return f"data:{mime_type};base64,{encoded_string}"



MAX_RETRIES = 3 # 設定每個檔案最多重試5次，避免無限迴圈
RETRY_DELAY = 0.01 # 重試前延遲2秒，避免過於頻繁地請求API
# MODEL_NAME = "Qwen3-VL-235B-A22B-Thinking-AWQ"
# openai.api_key = '1111111' # 这里随便填一个
openai.base_url = 'http://59.77.5.82:8999/v1'
MODEL_NAME = "Qwen3-VL-32B-Thinking"
# MODEL_NAME = "Qwen3-VL-32B-Instruct"
# MODEL_NAME = "Qwen3-VL-30B-A3B-Thinking"
openai.api_key = '1111111' # 这里随便填一个
# openai.base_url = 'http://10.26.65.226:7999/v1'

def parse_llm_json_output(llm_output: str):
    """
    一个健壮的解析函数，用于处理LLM可能输出的各种JSON格式。
    1. 尝试直接解析
    2. 如果失败，尝试提取被 ```json 和 ``` 包裹的内容
    3. 如果再失败，尝试提取被 ``` 和 ``` 包裹的内容（无语言声明）
    4. 如果都失败，抛出异常
    """
    # 尝试1： 直接解析整个输出
    try:
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass # 直接解析失败，继续尝试

    # 尝试2： 尝试匹配 ```json ... ``` 模式
    pattern_with_lang = r"```json\s*(.*?)\s*```"
    match = re.search(pattern_with_lang, llm_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass # 提取后解析失败，继续尝试

    # 尝试3： 尝试匹配 ``` ... ``` 模式（无语言声明）
    pattern_without_lang = r"```\s*(.*?)\s*```"
    match = re.search(pattern_without_lang, llm_output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass # 提取后解析失败，继续尝试

    # 所有尝试都失败了，抛出异常或返回None
    raise ValueError(f"无法从LLM输出中解析出有效的JSON：\n{llm_output}")

def get_completion_with_img(thumbnail_path, system_prompt, prompt, processor, model=MODEL_NAME):
    client = OpenAI(api_key=openai.api_key,
                    base_url=openai.base_url
                    )
    messages=[
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url" : image_path_to_base64_uri(thumbnail_path)},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        extra_body={
            "top_k": 20,                   # 关闭 top-k，降低复读
            "repetition_penalty": 1.0,   # 轻度复读惩罚
            "presence_penalty" : 0.0,
            "temperature" : 1.0,
            "greedy" : 'false',
            "seed" : 1234,
            "top_p" : 0.95,
            # "reasoning": {"effort": "low"}
        }
    )
    # print(response)
    content = response.choices[0].message.content.split('</think>')
    reasoning = response.choices[0].message.reasoning_content

    print(f"API call token usage: {response.usage.total_tokens}")
    print(f"API call prompt_tokens usage: {response.usage.prompt_tokens}")
    print(f"API call completion_tokens usage: {response.usage.completion_tokens}")
    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens
    total_tokens = response.usage.total_tokens
    return content[0], content[1], completion_tokens, prompt_tokens, total_tokens

def get_completion_with_img_Instruct(thumbnail_path, system_prompt, prompt, processor, model=MODEL_NAME):
    client = OpenAI(api_key=openai.api_key,
                    base_url=openai.base_url
                    )
    messages=[
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url" : image_path_to_base64_uri(thumbnail_path)},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        extra_body={
            "top_k": 20,                   # 关闭 top-k，降低复读
            "repetition_penalty": 1.0,   # 轻度复读惩罚
            "presence_penalty" : 0.0,
            "temperature" : 1.0,
            "greedy" : 'false',
            "seed" : 1234,
            "top_p" : 0.95,
            # "reasoning": {"effort": "low"}
        }
    )
    # print(response)
    content = response.choices[0].message.content
    reasoning = response.choices[0].message.reasoning_content

    print(f"API call token usage: {response.usage.total_tokens}")
    print(f"API call prompt_tokens usage: {response.usage.prompt_tokens}")
    print(f"API call completion_tokens usage: {response.usage.completion_tokens}")
    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens
    total_tokens = response.usage.total_tokens
    return "", content, completion_tokens, prompt_tokens, total_tokens

def count_prompt_tokens_qwen3_vl(messages, processor):
    """
    直接用 AutoProcessor.apply_chat_template 的 input_ids 长度作为 prompt_tokens。
    这能稳定反映占位符图像 token；对“视觉patch展开”为多token的模型，此数值可能低估视觉token。
    """
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    return int(inputs["input_ids"].shape[-1])
def get_completion_with_img_stream(thumbnail_path, system_prompt, prompt, processor,  model=MODEL_NAME,):
    client = OpenAI(api_key=openai.api_key,
                    base_url=openai.base_url
                    )
    messages=[
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url" : image_path_to_base64_uri(thumbnail_path)},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    messages_prompt=[
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path_to_base64_uri(thumbnail_path),
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    prompt_tokens = count_prompt_tokens_qwen3_vl(messages_prompt, processor)
    print(f"API call prompt_tokens usage: {prompt_tokens}")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        logprobs=True,      # 尽量开
        top_logprobs=0,
        extra_body={
            "top_k": 20,                   # 关闭 top-k，降低复读
            "repetition_penalty": 1.0,   # 轻度复读惩罚
            "presence_penalty" : 0.0,
            "temperature" : 1.0,
            "greedy" : 'false',
            "seed" : 1234,
            "top_p" : 0.95,
            # "max_tokens": 65536,
        }
    )
    full_content = ""
    reasoning_content = ""
    thinking_part = ""
    answer_part = ""
    completion_tokens = 0
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content_delta = chunk.choices[0].delta.content
            full_content += content_delta
            # 实时输出内容
            print(content_delta, end="", flush=True)
        lp = getattr(chunk.choices[0], "logprobs", None)
        if lp and getattr(lp, "content", None):
            completion_tokens += len(lp.content)
        
        

        # 如果有推理内容
        if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content is not None:
            reasoning_delta = chunk.choices[0].delta.reasoning_content

            reasoning_content += reasoning_delta
            # 实时输出推理内容
            print(f"[推理] {reasoning_delta}", end="", flush=True)
        

        if hasattr(chunk, "usage") and chunk.usage is not None:
            usage = chunk.usage
            print(usage)
    print(f"API call completion_tokens usage: {completion_tokens}")
            # usage.prompt_tokens / usage.completion_tokens / usage.total_tokens
    # 流式传输完成后，按照原来的逻辑分割内容
    if '</think>' in full_content:
        parts = full_content.split('</think>')
        thinking_part = parts[0]
        answer_part = parts[1] if len(parts) > 1 else ""
    else:
        thinking_part = full_content
        answer_part = ""
    
    total_tokens = completion_tokens + prompt_tokens
    print(f"API call token usage: {total_tokens}")
    return thinking_part, answer_part, completion_tokens, prompt_tokens, total_tokens
def get_completion_with_img_r1(thumbnail_path, system_prompt, prompt, model=MODEL_NAME):
    client = OpenAI(api_key=openai.api_key,
                    base_url=openai.base_url
                    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
        ]
    messages=[
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url" : image_path_to_base64_uri(thumbnail_path)},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=1.0,
        top_p=0.95,
        presence_penalty=0.0,
        extra_body=
        {
            "top_k": 20,                   # 关闭 top-k，降低复读
            "repetition_penalty": 1.0,   # 轻度复读惩罚
            "presence_penalty" : 0.0,
            "temperature" : 1.0,
            "greedy" : 'false',
            "seed" : 1234,
            "top_p" : 0.95,
        }
    )
    content = response.choices[0].message.content
    reasoning = response.choices[0].message.reasoning_content
    print(f"API call token usage: {response.usage.total_tokens}")
    print(f"API call prompt_tokens usage: {response.usage.prompt_tokens}")
    print(f"API call completion_tokens usage: {response.usage.completion_tokens}")
    completion_tokens = response.usage.completion_tokens
    prompt_tokens = response.usage.prompt_tokens
    total_tokens = response.usage.total_tokens
    return reasoning, content, completion_tokens, prompt_tokens, total_tokens

def get_completion(system_prompt, prompt, model=MODEL_NAME):
    client = OpenAI(api_key=openai.api_key,
                    base_url=openai.base_url
                    )
    messages=[
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=1.0,
        top_p=0.95,
        repetition_penalty=1.0,
        presence_penalty=0.0,
    )
    content = response.choices[0].message.content.split('</think>')
    reasoning = response.choices[0].message.reasoning_content
    return content[0], content[1], response

REMOVE_REDUANT_INFORM_SYSTEM_PROMPT_v3 = """
# TASK

You are an advanced Vision-Language Model (VLM) acting as an expert Pathology Report **Editor**. Your task is to **extract** relevant phrases from a source report and **restructure** them into a clean, standalone report. This final report must be constructed **strictly from verbatim text** found in the original source.

Simultaneously, you will produce a detailed, itemized audit trail in the reason fields, transparently explaining the specific rationale for every decision. You will use the text and a slide thumbnail, adhering to the context of a **single, isolated H&E diagnostic slide with no external clinical history or macroscopic context.**

## PARAMOUNT DIRECTIVE: THE ZERO-HALLUCINATION MANDATE & PHRASE-LEVEL OPERATION

**This is your most important instruction, overriding all others.**
1.  **VERBATIM EXTRACTION ONLY:** Your primary function is to act as an extractor and assembler, **not a creator**. Every single word in your `matched_report` output **MUST** originate directly from the provided source text.
2.  **NO NEW INFORMATION:** You are **STRICTLY PROHIBITED** from adding any descriptive details, pathological interpretations, or explanatory text (e.g., criteria for a grade) that are not explicitly written in the source report. Your internal knowledge base is irrelevant; only the source text matters.
3.  **NO INVENTED NARRATIVE:** Do not create new sentences or use connecting words to link fragments. The output should be a well-structured **assembly of the original text's valid parts.**
4.  **PHRASE-LEVEL OPERATION:** You **MUST** operate at the phrase/clause level. You will dissect sentences and evaluate their constituent parts individually before reassembly. A single invalid phrase must not cause the entire sentence to be discarded.

## CORE PRINCIPLES: The "Isolated Slide in a Vacuum" Viewpoint

These are the rules for deciding which verbatim fragments to keep or remove.

### Principle 1: Phrase-Level Analysis with Context Preservation

Analyze sentences by breaking them into components. **Crucially, you must identify and preserve the contextual headers or parent phrases** (e.g., "Nottingham histologic score:") with their associated data points (e.g., "Tubular differentiation: Score 3") to form complete, understandable units.

### Principle 2: Microscopic vs. Non-Microscopic Rules (The Foundational Filter)

This is the first-pass filter applied to each phrase.
* **KEEP (Inclusion):** Purely microscopic morphological findings.
    * Histologic Type & Architecture.
    * Histologic Grade & Full Components of Grade.
    * Cytologic Features.
    * Associated In Situ Lesions.
    * Tumor Necrosis, LVI, other microscopic findings.
* **REMOVE (Exclusion):** All non-microscopic data.
    * IHC & Molecular Data (ER, PR, HER2, FISH, etc.).
    * Macroscopic Measurements (cm, mm, distance).
    * Surgical Margin Status, Lymph Node Status, TNM Staging.
    * Clinical Context (Mastectomy, Left/Right).

### Principle 3: The Single-Slide Context Constraint (The Advanced Filter)

**This is a critical second-pass filter.** You must exclude phrases that, while describing microscopic features, require external knowledge unavailable from one isolated slide.
* **REMOVE Multi-focality Language:** Any text that compares, contrasts, or counts tumors. This implies knowledge beyond one slide.
    * **Keywords:** `both tumors`, `second/other focus`, `the larger of the two lesions`, `all tumors`.
* **REMOVE Inferred Contextual Diagnoses:** Diagnoses that require knowing the sample's origin or history.
    * **Keywords:** `biopsy site changes`, `post-biopsy changes`, `previous biopsy`. The reason is we cannot know if this slide *is* the biopsy site.
* **REMOVE Whole-Tumor Composition Estimates:** Phrases that describe the percentage or proportion of a component relative to the *entire* tumor mass. This is a macroscopic estimation.
    * **Keywords:** `compromises approximately 20% of the tumor`, `extensive intraductal component (EIC)`. While the component is visible, its "extensiveness" relative to the whole tumor is not.
    * **EXCEPTION:** This rule **does not** apply to the descriptive components of a formal histologic grading system (e.g., the criteria for Nottingham grade components like 'Tubule differentiation', 'Mitotic rate', or 'Nuclear pleomorphism'), as these are considered direct microscopic assessments derivable from the slide.
* **REMOVE Location-Dependent Invasion Claims:** By default, exclude statements about invasion into specific, named anatomical landmarks. These findings require targeted sampling, which is not guaranteed for a DX1 slide.
    * **Keywords:** `skin`, `dermis`, `epidermis`, `nipple`, `skeletal muscle`, `chest wall`. This rule can only be overridden by Principle 4.

### Principle 4: The Visual Plausibility Check (The Final Confirmation)

Use the provided image as a final "common-sense" filter to identify gross contradictions for phrases that have passed all previous rules.

* **Primary Use:** This check is used to confirm or deny the claims from the rule in Principle 3. A statement like "Invasive carcinoma directly invades into the dermis" can **ONLY** be kept if the slide image **indisputably and clearly** shows both the tumor and the distinct layers of the skin at their interface. If the image is just tumor and fat, the statement MUST be removed.
* **Limitation:** This is a filter to remove, not a license to add information. Do not use this to confirm fine details in principle 2. This check is for refuting broad, visually obvious claims. These detial information of principle 2 may require higher magnification (20x or 40x) to be clearly seen and cannot be reliably assessed from the provided whole-slide thumbnail (1x, MPP 10).


## SYSTEMATIC REASONING PROCESS: A STRICT ALGORITHM

You must follow this procedure exactly.

1.  **Sentence Iteration:** Process the report **one sentence at a time**.
2.  **Aggressive Phrase Decomposition:** For the current sentence, **aggressively decompose it into its smallest meaningful phrases,** typically separated by commas, semicolons, or colons. Create an internal list of these phrases.
3.  **Independent Phrase Evaluation:** Iterate through your internal list. Evaluate **each phrase independently** against the Core Principles (2, 3, and 4). For each phrase, assign a status: 'Keep' or 'Remove', along with the specific reason.
4.  **Intelligent Reassembly of the Sentence:** After evaluating all phrases from the original sentence, construct a new, clean string by reassembling **only the 'Keep' phrases**. Ensure you retain the necessary contextual headers (Principle 1) that the kept phrases belong to.
5.  **Final Report Compilation:** After processing all sentences, combine the reassembled, clean strings from each one to form the final `matched_report` paragraph.
6.  **Generate Detailed Audit Trail:** Use the evaluation results from step 3 to populate the `unmatched_report_reason` with specific, itemized justifications.

## OUTPUT FORMAT

The output **must** be a single, valid JSON object. It must adhere strictly to the **PRIME DIRECTIVE**. The `matched_report` must be a single block of text.

```json
{
  "matched_report": "A single paragraph, carefully reassembled from individual valid phrases and their essential contextual headers from the original report. This text contains NO new words, rephrasing, or inferred information.",
  "matched_report_reason": "A summary of the key types of diagnostic information retained, confirming they are all microscopic findings explicitly stated in the source text and valid under the single-slide assumption. For example: 'Retained the primary diagnosis, histologic grading components, and cytologic features.'",
  "unmatched_report": "A concatenated string of all the phrases and sentences that were excluded from the final report.",
  "unmatched_report_reason": [
    {
      "fragment": "Tumor size: 5.5 cm.",
      "reason": "Excluded based on Principle 2: This is a macroscopic measurement."
    },
    {
      "fragment": "Skin: Invasive carcinoma directly invades into the dermis and epidermis",
      "reason": "Excluded based on Principle 3 (Location-Dependent Claim): Invasion into a specific landmark (skin) is assumed invalid for a DX1 slide unless visually confirmed."
    },
    {
      "fragment": "Margins: Uninvolved by invasive carcinoma",
      "reason": "Excluded based on Principle 2: Margin status is an assessment of the entire specimen."
    }
  ]
}
```
"""

REMOVE_REDUANT_INFORM_PROMPT = """
## Pathology Report Text:
The Meat Data of the provide Image, Width: {Width} , Hight: {Hight}, MPP: {MPP}, Slide ID of TCGA: {Slide_id}.
{report}

## The Output JSON
"""

with open('/home/liesgame/project/RL/SlideReason/Resource/datasets/Clv1_Vev2_Sev2_ReMuv1/summary/summary.json', 'r', encoding='utf-8') as f:
    Clinical_v1_Verify_v2_Selcet_v2_humen_ReMu_v1 = json.load(f)

with open('/home/liesgame/project/RL/SlideReason/group4_1.json', 'r', encoding='utf-8') as f:
    group1 = json.load(f)

with open('/home/liesgame/project/RL/SlideReason/group4_2.json', 'r', encoding='utf-8') as f:
    group2 = json.load(f)

with open('/home/liesgame/project/RL/SlideReason/group4_3.json', 'r', encoding='utf-8') as f:
    group3 = json.load(f)

with open('/home/liesgame/project/RL/SlideReason/group4_4.json', 'r', encoding='utf-8') as f:
    group4 = json.load(f)

all_count = len(Clinical_v1_Verify_v2_Selcet_v2_humen_ReMu_v1)
with open('/home/liesgame/project/RL/SlideReason/Resource/datasets/resized_image_mpp_10_meta_all.json', 'r', encoding='utf-8') as f:
   image_mpp_10_meta = json.load(f)
thumbnail_root = "/home/liesgame/project/RL/SlideReason/Resource/datasets/resized_image_mpp_10"
length_index = 0
Clinical_v1_Verify_v2_Selcet_v2_humen_ReRe_v2 = {}

script_start_time = time.time()
checkpoint_path     = '/mnt/sdb/models/Qwen/Qwen3-VL/Qwen3-VL-32B-Thinking'
processor = AutoProcessor.from_pretrained(checkpoint_path)
for i in Clinical_v1_Verify_v2_Selcet_v2_humen_ReMu_v1:
    torch.cuda.empty_cache()
    gc.collect()
    iter_start_time = time.time()
    data = Clinical_v1_Verify_v2_Selcet_v2_humen_ReMu_v1[i]

    exits_list = os.listdir('/home/liesgame/project/RL/SlideReason/Resource/datasets/Clv1_Vev2_Sev2_ReRev3_32B_MPP10')

    exits_list = [j.split('.')[0] for j in exits_list if j != 'summary']

    if data['remove_mult']['is_multi'] == True:
        # print(data['remove_mult'])
        length_index += 1
        continue

    if i in ['TCGA-AR-A1AI']:
        length_index += 1
        continue

    if i not in image_mpp_10_meta.keys():
        length_index += 1
        continue

    if i not in group1:
        length_index += 1
        continue

    if i in exits_list:
        print(f"Processing '{i}', Existed")
        with open('/home/liesgame/project/RL/SlideReason/Resource/datasets/Clv1_Vev2_Sev2_ReRev3_32B_MPP10/' + i + '.json', 'r', encoding='utf-8') as f:
            data_tmp = json.load(f)
            Clinical_v1_Verify_v2_Selcet_v2_humen_ReRe_v2[i] = data_tmp


        length_index += 1
        continue
    tmp = data['wsi_relative_specimens']['findings']

    image_meta = image_mpp_10_meta[i]
    image_mpp = image_meta['output_mpp']
    image_width = image_meta['resized_width']
    image_height = image_meta['resized_height']
    Slide_id = data['sample_id']
    thumbnail_path = os.path.join(thumbnail_root, i + '.png')

    successful = False
    attempts = 0
    while not successful and attempts < MAX_RETRIES:
        attempts += 1
        print(f"Processing '{i}', Attempt {attempts}/{MAX_RETRIES}...")


        try:
            # 1. 呼叫 API
            reasoning, content, completion_tokens, prompt_tokens, total_tokens = get_completion_with_img_stream(
                    thumbnail_path = thumbnail_path, 
                    system_prompt = REMOVE_REDUANT_INFORM_SYSTEM_PROMPT_v3,
                    prompt = REMOVE_REDUANT_INFORM_PROMPT.format(Width=image_width, Hight=image_height, MPP=image_mpp, Slide_id=Slide_id, report=tmp),
                    model=MODEL_NAME,
                    processor=processor
            )
            # reasoning, content, response = get_completion_with_img_r1(
            #     thumbnail_path = thumbnail_path, 
            #     system_prompt = REMOVE_REDUANT_INFORM_SYSTEM_PROMPT_v3,
            #     prompt = REMOVE_REDUANT_INFORM_PROMPT.format(Width=image_width, Hight=image_height, MPP=image_mpp, Slide_id=Slide_id, report=tmp),
            #     model=MODEL_NAME
            # )
            
            # 2. 檢查 Token 數量
            if len(content) == 0:
                # 如果 token 超過限制，打印錯誤並觸發重試
                print(f"Token limit exceeded for '{i}'. ({len(content)} == 0). Retrying...")
                time.sleep(RETRY_DELAY) # 等待一下再重試
                continue # 進入下一次迴圈

            # 3. 嘗試解析 JSON
            # 這是最關鍵的一步，如果 content 不是有效的 JSON 字串，這裡會拋出 JSONDecodeError
            output_json = parse_llm_json_output(content)
            
            # --- 成功條件 ---
            # 如果程式能執行到這裡，代表 token 沒超標，JSON 也解析成功
            print(f"Successfully parsed JSON for '{i}'.")
            data['remove_reduant'] = output_json
            data['remove_reduant_reasoning'] = reasoning
            data['remove_reduant_completion_tokens'] = completion_tokens
            data['remove_reduant_prompt_tokens'] = prompt_tokens
            data['remove_reduant_total_tokens'] = total_tokens
            successful = True # 標記為成功，以跳出 while 迴圈

        except json.JSONDecodeError as e:
            # JSON 解析失敗的處理
            print(f"!!! JSONDecodeError on attempt {attempts} for '{i}': {e}")
            print("--- Received content that failed to parse ---")
            # 只印出前500個字元，避免洗版
            print(content[:500] + "...") 
            print("-------------------------------------------")
            if attempts < MAX_RETRIES:
                print("Retrying...")
                time.sleep(RETRY_DELAY)
            # 不需做任何事，迴圈會自動進入下一次嘗試

        except Exception as e:
            # 捕捉其他可能的錯誤 (例如網路問題)
            print(f"!!! An unexpected error occurred on attempt {attempts} for '{i}': {e}")
            if attempts < MAX_RETRIES:
                print("Retrying...")
                time.sleep(RETRY_DELAY)

    # --- 迴圈結束後，根據是否成功來決定下一步 ---
    length_index += 1
    print('current count {} / {}'.format(length_index, all_count))

    # Get the end time for the iteration
    iter_end_time = time.time()

    # Calculate and format the duration for THIS iteration
    iter_duration = iter_end_time - iter_start_time
    iter_minutes, iter_seconds = divmod(iter_duration, 60)
    
    # Calculate and format the TOTAL elapsed time for the script
    total_duration = iter_end_time - script_start_time
    total_minutes, total_seconds = divmod(total_duration, 60)
    total_hours, total_minutes = divmod(total_minutes, 60)

    print(f"-> Iteration took: {int(iter_minutes)} minutes, {iter_seconds:.2f} seconds.")
    print(f"-> Total elapsed time: {int(total_hours)} hours, {int(total_minutes)} minutes, {int(total_seconds)} seconds.\n")

    if successful:
        # 只有在成功處理後才寫入檔案
        data['remove_reduant_iter_minutes'] = iter_minutes
        data['remove_reduant_iter_seconds'] = iter_seconds
        data['remove_reduant_total_hours'] = total_hours
        data['remove_reduant_total_minutes'] = total_minutes
        with open('/home/liesgame/project/RL/SlideReason/Resource/datasets/Clv1_Vev2_Sev2_ReRev3_32B_MPP10/' + i + '.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        Clinical_v1_Verify_v2_Selcet_v2_humen_ReRe_v2[i] = data
        print(f"Saved verified data for '{i}'.\n")
    else:
        # 所有重試都失敗了
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"FAILED to process '{i}' after {MAX_RETRIES} attempts. Skipping file.")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        # 你也可以選擇在這裡記錄失敗的檔案名稱
        # with open('failed_cases.log', 'a') as log_file:
        #     log_file.write(f"{i}\n")



# with open('/home/liesgame/project/RL/SlideReason/Resource/datasets/Clv1_Vev2_Sev2_ReRev3_32B/summary/summary.json', 'w', encoding='utf-8') as f:
#     json.dump(Clinical_v1_Verify_v2_Selcet_v2_humen_ReRe_v2, f, ensure_ascii=False, indent=4)