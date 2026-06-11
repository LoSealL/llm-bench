# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""BFCL v4 constants, enums, and prompt templates.

Ported from bfcl-lite for standalone evaluation.
"""

from enum import Enum
from pathlib import Path


class Language(Enum):
    """Supported ground-truth languages."""

    PYTHON = "python"


class ReturnFormat(Enum):
    """Model output format variants."""

    PYTHON = "python"
    JSON = "json"
    VERBOSE_XML = "verbose_xml"
    CONCISE_XML = "concise_xml"


VERSION_PREFIX = "BFCL_v4"

NON_LIVE_CATEGORY = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
]

LIVE_CATEGORY = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
]

SINGLE_TURN_CATEGORY = NON_LIVE_CATEGORY + LIVE_CATEGORY
ALL_CATEGORIES = SINGLE_TURN_CATEGORY

TEST_COLLECTION_MAPPING = {
    "all": ALL_CATEGORIES,
    "single_turn": SINGLE_TURN_CATEGORY,
    "live": LIVE_CATEGORY,
    "non_live": NON_LIVE_CATEGORY,
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "scripts" / "BFCL"
PROMPT_PATH = DATA_ROOT / "prompts"
POSSIBLE_ANSWER_PATH = DATA_ROOT / "ground_truth"

OUTPUT_FORMAT_MAPPING = {
    "python": (
        "[func_name1(params_name1=params_value1, params_name2=params_value2...), "
        "func_name2(params)]"
    ),
    "json": (
        '```json\n[{"function":"func_name1","parameters":{"param1":"value1",'
        '"param2":"value2"...}},{"function":"func_name2","parameters":'
        '{"param":"value"}}]\n```'
    ),
    "verbose_xml": (
        '<functions><function name="func_name1"><params><param name="param1" '
        'value="value1" type="type1"/><param name="param2" value="value2" '
        'type="type2"/>...</params></function><function name="func_name2">'
        '<params><param name="param3" value="value3" type="type3"/></params>'
        '</function></functions>'
    ),
    "concise_xml": (
        '<functions><function name="func_name1"><param name="param1" '
        'type="type1">value1</param><param name="param2" type="type2">value2'
        '</param>...</function><function name="func_name2"><param name="param3" '
        'type="type3">value</param></function></functions>'
    ),
}

PARAM_TYPE_MAPPING = {
    "python": "",
    "json": "",
    "verbose_xml": (
        "The type fields of the parameters in your function calls must be one of: "
        "string, integer, float, boolean, array, dict, or tuple."
    ),
    "concise_xml": (
        "The type fields of the parameters in your function calls must be one of: "
        "string, integer, float, boolean, array, dict, or tuple."
    ),
}

PROMPT_STYLE_TEMPLATES = {
    "classic": {
        "persona": "You are an expert in composing functions.",
        "task": (
            "You are given a question and a set of possible functions. "
            "Based on the question, you will need to make one or more "
            "function/tool calls to achieve the purpose. If none of the "
            "functions can be used, point it out. If the given question "
            "lacks the parameters required by the function, also point it out."
        ),
        "tool_call_no_tag": (
            "You should only return the function calls in your response.\n\n"
            "If you decide to invoke any of the function(s), you MUST put it in "
            "the format of {output_format}. {param_types} You SHOULD NOT include "
            "any other text in the response."
        ),
        "tool_call_with_tag": (
            "You should only return the function calls in the <TOOLCALL> section. "
            "If you decide to invoke any of the function(s), you MUST put it in "
            "the format of <TOOLCALL>{output_format}</TOOLCALL>. {param_types} "
            "You SHOULD NOT include any other text in the response."
        ),
        "multiturn_behavior": (
            "At each turn, you should try your best to complete the "
            "tasks requested by the user within the current turn."
        ),
        "available_tools": (
            "Here is a list of functions in {format} format "
            "that you can invoke.\n{functions}\n"
        ),
    },
}

_PLAINTEXT_SYSTEM_PROMPT_TEMPLATE = (
    "{persona}{task}\n\n{tool_call_format}\n\n{multiturn_behavior}\n\n{available_tools}"
)

PROMPT_TEMPLATE_MAPPING = {
    "plaintext": _PLAINTEXT_SYSTEM_PROMPT_TEMPLATE,
}

DEFAULT_SYSTEM_PROMPT_FORMAT = (
    "ret_fmt=python"
    "&tool_call_tag=False"
    "&func_doc_fmt=json"
    "&prompt_fmt=plaintext"
    "&style=classic"
)
