# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""BFCL v4 utilities.

Ported from bfcl-lite for standalone evaluation.
"""

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
    """Extract the BFCL test category from a file path or string.

    Args:
        input_string: Path or string containing a BFCL filename.
        raise_error: Whether to raise if extraction fails.

    Returns:
        Test category name, or ``None`` if extraction fails and
        ``raise_error`` is ``False``.
    """
    input_string = str(input_string)
    pattern = rf".*{VERSION_PREFIX}_(\w+?)(?:_score|_result)?\.json"
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)
    if raise_error:
        raise ValueError(f"Could not extract test category from: {input_string}")
    return None


def extract_test_category_from_id(test_entry_id: str) -> str:
    """Extract the test category from an entry identifier.

    Args:
        test_entry_id: Entry identifier, optionally with a suffix
            separated by ``:``.

    Returns:
        Test category name.
    """
    if ":" in test_entry_id:
        test_entry_id = test_entry_id.split(":")[0]
    return test_entry_id.rsplit("_", 1)[0]


def get_file_name_by_category(test_category: str, is_result_file: bool = False) -> str:
    """Build the BFCL filename for a category.

    Args:
        test_category: Category name.
        is_result_file: Whether to use the result-file suffix.

    Returns:
        Filename string.
    """
    suffix = "_result.json" if is_result_file else ".json"
    return f"{VERSION_PREFIX}_{test_category}{suffix}"


def parse_test_category_argument(test_category_args: list[str]) -> list[str]:
    """Expand category arguments into concrete category names.

    Args:
        test_category_args: Category names or collection aliases.

    Returns:
        Sorted list of concrete category names.
    """
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
    """Return whether the category is relevance or irrelevance."""
    return "relevance" in test_category or "irrelevance" in test_category


def is_live(test_category: str) -> bool:
    """Return whether the category is a live evaluation."""
    return "live" in test_category


def is_non_live(test_category: str) -> bool:
    """Return whether the category is not a live evaluation."""
    return not is_live(test_category)


def load_file(file_path: str | Path, sort_by_id: bool = False) -> list[dict[str, Any]]:
    """Load a JSON Lines file.

    Args:
        file_path: Path to the JSONL file.
        sort_by_id: Whether to sort entries by identifier.

    Returns:
        List of parsed dictionaries.
    """
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
    """Write a list of dictionaries to a JSON Lines file.

    Args:
        filename: Output filename.
        data: Dictionaries to serialise.
        subdir: Optional subdirectory for the output file.
    """
    if subdir:
        subdir_path = Path(subdir)
        subdir_path.mkdir(parents=True, exist_ok=True)
        filename = str(subdir_path / Path(filename).name)
    abs_filename = Path(filename).resolve()
    with open(abs_filename, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def make_json_serializable(value: Any) -> Any:
    """Recursively convert a value to a JSON-serialisable form.

    Args:
        value: Value of any type.

    Returns:
        JSON-serialisable value.
    """
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
    """Build a sort key for an entry based on its identifier.

    Args:
        entry: Dictionary with an ``id`` key.

    Returns:
        Tuple of ``(test_category, index)``.
    """
    entry_id = entry["id"].split(":")[0]
    parts = entry_id.rsplit("_", 1)
    test_category, index = parts[0], parts[1]
    if "-" in index:
        index = index.split("-")[0]
    return (test_category, int(index))


def is_function_calling_format_output(decoded_output: Any) -> bool:
    """Return whether decoded output follows the function-calling format."""
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
    """Return whether the decoded output is effectively empty."""
    if not is_function_calling_format_output(decoded_output):
        return True
    if len(decoded_output) == 0:
        return True
    if len(decoded_output) == 1 and len(decoded_output[0]) == 0:
        return True
    return False


def load_dataset_entry(test_category: str) -> list[dict[str, Any]]:
    """Load prompt entries for a BFCL category.

    Args:
        test_category: Category name.

    Returns:
        List of prompt dictionaries.
    """
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
    """Load ground-truth entries for a BFCL category.

    Args:
        test_category: Category name.

    Returns:
        List of ground-truth dictionaries.
    """
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
    functions: list[dict[str, Any]], _test_category: str
) -> list[dict[str, Any]]:
    """Append a Python syntax hint to function descriptions.

    Args:
        functions: Function schema list.
        test_category: Category name.

    Returns:
        Modified function schema list.
    """
    if len(functions) == 0:
        return functions
    hint = " Note that the provided function is in Python 3 syntax."
    for item in functions:
        item["description"] = item.get("description", "") + hint
    return functions


def extract_prompt_format_from_id(test_entry_id: str) -> str:
    """Extract the prompt-format suffix from an entry identifier.

    Args:
        test_entry_id: Entry identifier.

    Returns:
        Prompt-format string, or the default format if absent.
    """
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
    """Prepend a system prompt with formatted function documentation.

    Args:
        prompts: Existing message list.
        function_docs: Function schemas.
        test_entry_id: Entry identifier.

    Returns:
        Message list with the updated or inserted system prompt.
    """
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
    """Build the system prompt for a given format configuration.

    Args:
        format_sensitivity_config: Format configuration string.
        functions: Function schemas.

    Returns:
        Rendered system prompt.
    """
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
    """Parse a format-sensitivity configuration string.

    Args:
        input_str: Configuration string.

    Returns:
        Tuple of ``(return_format, has_tool_call_tag,
        function_doc_format, prompt_format, prompt_style)``.
    """
    _pattern = re.compile(
        r"^"
        r"ret_fmt=(?P<return_format>python|json|verbose_xml|concise_xml)"
        r"&tool_call_tag=(?P<has_tool_call_tag>True|False)"
        r"&func_doc_fmt=(?P<function_doc_format>python|xml|json)"
        r"&prompt_fmt=(?P<prompt_format>plaintext|markdown)"
        r"&style=(?P<prompt_style>classic|experimental)"
        r"$"
    )
    match = _pattern.match(input_str)
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
    """Render function schemas in the requested documentation format.

    Args:
        functions: Function schemas.
        function_doc_format: Target format (``python``, ``xml``, or ``json``).

    Returns:
        Rendered documentation string.
    """
    if function_doc_format == "json":
        return json.dumps(functions, indent=4)
    if function_doc_format == "python":
        return _generate_function_doc_python(functions)
    if function_doc_format == "xml":
        return _generate_function_doc_xml(functions)
    raise ValueError(f"Invalid function doc format: {function_doc_format}")


def _generate_function_doc_python(functions: list[dict[str, Any]]) -> str:
    """Render function schemas as Python-style docstrings.

    Args:
        functions: Function schemas.

    Returns:
        Rendered Python documentation string.
    """

    def _to_py_type(meta: dict[str, Any]) -> str:
        """Convert a JSON schema type to a Python type annotation."""
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

    indent = " " * 8
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
                lines.append(f"{indent}{pname} ({py_type}{default_note}): {desc}\n")
        lines.append('    """\n')
        docs.append("".join(lines))
    return "\n\n".join(docs)


def _generate_function_doc_xml(functions: list[dict[str, Any]]) -> str:
    """Render function schemas as XML documentation.

    Args:
        functions: Function schemas.

    Returns:
        Rendered XML documentation string.
    """

    def _param_xml(
        name: str,
        meta: dict[str, Any],
        required_set: set[str] | None,
        indent_lvl: int = 2,
    ) -> str:
        """Render a single parameter as XML."""
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
