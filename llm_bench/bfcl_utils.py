# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""BFCL v4 utilities.

Ported from bfcl-lite for standalone evaluation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from llm_bench.bfcl_constants import (
    ALL_CATEGORIES,
    DEFAULT_SYSTEM_PROMPT_FORMAT,
    OUTPUT_FORMAT_MAPPING,
    PARAM_TYPE_MAPPING,
    POSSIBLE_ANSWER_PATH,
    PROMPT_PATH,
    PROMPT_STYLE_TEMPLATES,
    PROMPT_TEMPLATE_MAPPING,
    TEST_COLLECTION_MAPPING,
    VERSION_PREFIX,
)


def extract_test_category(
    input_string: str | Path, raise_error: bool = True
) -> str | None:
    input_string = str(input_string)
    pattern = rf".*{VERSION_PREFIX}_(\w+?)(?:_score|_result)?\.json"
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    if raise_error:
        raise ValueError(f"Could not extract test category from: {input_string}")
    return None


def extract_test_category_from_id(test_entry_id: str) -> str:
    if ":" in test_entry_id:
        test_entry_id = test_entry_id.split(":")[0]
    return test_entry_id.rsplit("_", 1)[0]


def get_file_name_by_category(test_category: str, is_result_file: bool = False) -> str:
    suffix = "_result.json" if is_result_file else ".json"
    return f"{VERSION_PREFIX}_{test_category}{suffix}"


def parse_test_category_argument(test_category_args: list[str]) -> list[str]:
    test_name_total: set[str] = set()
    for tc in test_category_args:
        if tc in TEST_COLLECTION_MAPPING:
            test_name_total.update(TEST_COLLECTION_MAPPING[tc])
        elif tc in ALL_CATEGORIES:
            test_name_total.add(tc)
        else:
            raise ValueError(f"Invalid test category: {tc}")
    return sorted(list(test_name_total))


def is_relevance_or_irrelevance(test_category: str) -> bool:
    return "relevance" in test_category or "irrelevance" in test_category


def is_live(test_category: str) -> bool:
    return "live" in test_category


def is_non_live(test_category: str) -> bool:
    return not is_live(test_category)


def load_file(file_path: str | Path, sort_by_id: bool = False) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    if sort_by_id:
        result.sort(key=sort_key)
    return result


def write_list_of_dicts_to_file(
    filename: str, data: list[dict[str, Any]], subdir: str | Path | None = None
) -> None:
    if subdir:
        subdir_path = Path(subdir)
        subdir_path.mkdir(parents=True, exist_ok=True)
        filename = str(subdir_path / Path(filename).name)
    abs_filename = Path(filename).resolve()
    with open(abs_filename, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def sort_key(entry: dict[str, Any]) -> tuple[str, int]:
    entry_id = entry["id"].split(":")[0]
    parts = entry_id.rsplit("_", 1)
    test_category, index = parts[0], parts[1]
    if "-" in index:
        index = index.split("-")[0]
    return (test_category, int(index))


def is_function_calling_format_output(decoded_output: Any) -> bool:
    if not isinstance(decoded_output, list):
        return False
    for item in decoded_output:
        if not isinstance(item, dict):
            return False
        if len(item) != 1:
            return False
        if not isinstance(list(item.values())[0], dict):
            return False
    return True


def is_empty_output(decoded_output: Any) -> bool:
    if not is_function_calling_format_output(decoded_output):
        return True
    if len(decoded_output) == 0:
        return True
    if len(decoded_output) == 1 and len(decoded_output[0]) == 0:
        return True
    return False


def load_dataset_entry(test_category: str) -> list[dict[str, Any]]:
    file_name = get_file_name_by_category(test_category)
    file_path = PROMPT_PATH / file_name
    logger.debug("Loading BFCL dataset entry for {} from {}", test_category, file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    entries = load_file(file_path)
    for entry in entries:
        if "function" in entry:
            entry["function"] = _add_language_hint(entry["function"], test_category)
    return entries


def load_ground_truth_entry(test_category: str) -> list[dict[str, Any]]:
    file_name = get_file_name_by_category(test_category)
    file_path = POSSIBLE_ANSWER_PATH / file_name
    logger.debug(
        "Loading BFCL ground truth entry for {} from {}",
        test_category,
        file_path,
    )
    if not file_path.exists():
        if is_relevance_or_irrelevance(test_category):
            return []
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")
    return load_file(file_path)


def _add_language_hint(
    functions: list[dict[str, Any]], test_category: str
) -> list[dict[str, Any]]:
    if len(functions) == 0:
        return functions
    hint = " Note that the provided function is in Python 3 syntax."
    for item in functions:
        item["description"] = item.get("description", "") + hint
    return functions


def extract_prompt_format_from_id(test_entry_id: str) -> str:
    if ":" not in test_entry_id:
        return DEFAULT_SYSTEM_PROMPT_FORMAT
    parts = test_entry_id.split(":")
    assert len(parts) == 3, f"Invalid format sensitivity id: {test_entry_id}"
    return parts[1]


def system_prompt_pre_processing_chat_model(
    prompts: list[dict[str, str]],
    function_docs: list[dict[str, Any]],
    test_entry_id: str,
) -> list[dict[str, str]]:
    prompt_format = extract_prompt_format_from_id(test_entry_id)
    system_prompt = _formulate_system_prompt(prompt_format, function_docs)
    if prompts[0]["role"] == "system":
        prompts[0]["content"] = system_prompt + "\n\n" + prompts[0]["content"]
    else:
        prompts.insert(0, {"role": "system", "content": system_prompt})
    return prompts


def _formulate_system_prompt(
    format_sensitivity_config: str, functions: list[dict[str, Any]]
) -> str:
    (
        return_format,
        has_tool_call_tag,
        function_doc_format,
        prompt_format,
        prompt_style,
    ) = parse_prompt_variation_params(format_sensitivity_config)
    formatted_function_doc = _format_function_doc(functions, function_doc_format)
    prompt_template = PROMPT_TEMPLATE_MAPPING.get(
        prompt_format, PROMPT_TEMPLATE_MAPPING["plaintext"]
    )
    style_template = PROMPT_STYLE_TEMPLATES.get(
        prompt_style, PROMPT_STYLE_TEMPLATES["classic"]
    )
    persona = style_template["persona"]
    task = style_template["task"]
    if has_tool_call_tag:
        tool_call_format = style_template["tool_call_with_tag"].format(
            output_format=OUTPUT_FORMAT_MAPPING[return_format],
            param_types=PARAM_TYPE_MAPPING[return_format],
        )
    else:
        tool_call_format = style_template["tool_call_no_tag"].format(
            output_format=OUTPUT_FORMAT_MAPPING[return_format],
            param_types=PARAM_TYPE_MAPPING[return_format],
        )
    multiturn_behavior = style_template["multiturn_behavior"]
    available_tools = style_template["available_tools"].format(
        format=function_doc_format,
        functions=formatted_function_doc,
    )
    return prompt_template.format(
        persona=persona,
        task=task,
        tool_call_format=tool_call_format,
        multiturn_behavior=multiturn_behavior,
        available_tools=available_tools,
    )


def parse_prompt_variation_params(input_str: str) -> tuple[str, bool, str, str, str]:
    _PATTERN = re.compile(
        r"^"
        r"ret_fmt=(?P<return_format>python|json|verbose_xml|concise_xml)"
        r"&tool_call_tag=(?P<has_tool_call_tag>True|False)"
        r"&func_doc_fmt=(?P<function_doc_format>python|xml|json)"
        r"&prompt_fmt=(?P<prompt_format>plaintext|markdown)"
        r"&style=(?P<prompt_style>classic|experimental)"
        r"$"
    )
    match = _PATTERN.match(input_str)
    if not match:
        raise ValueError(f"Invalid query format: {input_str!r}")
    return (
        match.group("return_format"),
        match.group("has_tool_call_tag") == "True",
        match.group("function_doc_format"),
        match.group("prompt_format"),
        match.group("prompt_style"),
    )


def _format_function_doc(
    functions: list[dict[str, Any]], function_doc_format: str
) -> str:
    if function_doc_format == "json":
        return json.dumps(functions, indent=4)
    if function_doc_format == "python":
        return _generate_function_doc_python(functions)
    if function_doc_format == "xml":
        return _generate_function_doc_xml(functions)
    raise ValueError(f"Invalid function doc format: {function_doc_format}")


def _generate_function_doc_python(functions: list[dict[str, Any]]) -> str:
    def _to_py_type(meta: dict[str, Any]) -> str:
        t = meta.get("type", "string")
        primitive_map = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
        }
        if t in primitive_map:
            return primitive_map[t]
        if t in {"array", "list", "tuple"} and "items" in meta:
            return f"list[{_to_py_type(meta['items'])}]"
        if t in {"object", "dict"}:
            return "dict"
        return t

    INDENT = " " * 8
    docs: list[str] = []
    for fn in functions:
        lines: list[str] = []
        lines.append(f"# Function: {fn['name']}\n")
        lines.append('    """\n')
        lines.append(f"    {fn.get('description', '')}\n\n")
        params = fn.get("parameters", {}).get("properties", {})
        if params:
            lines.append("    Args:\n")
            for pname, pmeta in params.items():
                py_type = _to_py_type(pmeta)
                desc = pmeta.get("description", "")
                if "enum" in pmeta:
                    desc += f" Enum values: {pmeta['enum']}."
                default_note = ""
                if "default" in pmeta:
                    default_note = f", default={pmeta['default']!r}"
                lines.append(f"{INDENT}{pname} ({py_type}{default_note}): {desc}\n")
        lines.append('    """\n')
        docs.append("".join(lines))
    return "\n\n".join(docs)


def _generate_function_doc_xml(functions: list[dict[str, Any]]) -> str:
    def _param_xml(
        name: str,
        meta: dict[str, Any],
        required_set: set[str] | None,
        indent_lvl: int = 2,
    ) -> str:
        indent = " " * indent_lvl * 2
        p_type = meta.get("type", "string")
        p_desc = meta.get("description", "")
        is_required = (
            "true" if required_set is None or name in required_set else "false"
        )
        if "enum" in meta:
            p_desc += f" Enum values: {meta['enum']}."
        if "items" in meta and "type" in meta["items"]:
            p_type = f"{p_type}[{meta['items']['type']}]"
        attrs = [f'name="{name}" type="{p_type}" required="{is_required}"']
        if "default" in meta:
            attrs.append(f'default="{meta["default"]!r}"')
        open_tag = f"{indent}<param " + " ".join(attrs).replace(",", "") + ">\n"
        parts = [open_tag, f"{indent}  <desc>{p_desc}</desc>\n"]
        if "properties" in meta:
            child_required = meta.get("required", None)
            child_set = set(child_required) if child_required else None
            parts.append(f"{indent}  <params>\n")
            for cname, cmeta in meta["properties"].items():
                parts.append(_param_xml(cname, cmeta, child_set, indent_lvl + 2))
            parts.append(f"{indent}  </params>\n")
        parts.append(f"{indent}</param>\n")
        return "".join(parts)

    blocks: list[str] = []
    for fn in functions:
        name = fn["name"]
        desc = fn.get("description", "")
        params_schema = fn["parameters"]
        top_props = params_schema.get("properties", {})
        top_required = params_schema.get("required", None)
        top_set = set(top_required) if top_required else None
        xml = f'<function name="{name}">\n'
        xml += f"  <desc>{desc}</desc>\n"
        xml += "  <params>\n"
        for pname, pmeta in top_props.items():
            xml += _param_xml(pname, pmeta, top_set, 2)
        xml += "  </params>\n"
        xml += "</function>\n"
        blocks.append(xml)
    return "\n".join(blocks)
