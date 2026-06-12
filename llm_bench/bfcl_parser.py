# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""BFCL v4 output parsers.

Ported from bfcl-lite for standalone evaluation.
"""

import ast
import json
import re
import xml.etree.ElementTree as ET
from typing import Any

from loguru import logger

from llm_bench.bfcl_constants import ReturnFormat


def ast_parse(
    input_str: str,
    language: ReturnFormat = ReturnFormat.PYTHON,
    has_tool_call_tag: bool = False,
) -> list[dict[str, Any]]:
    """Parse a raw model response into structured function calls.

    Args:
        input_str: Raw model output.
        language: Expected output format.
        has_tool_call_tag: Whether to look for ``<TOOLCALL>`` tags.

    Returns:
        List of function-call dictionaries.
    """
    logger.debug("Parsing BFCL output with language {}", language)
    if has_tool_call_tag:
        match = re.search(r"<TOOLCALL>(.*?)</TOOLCALL>", input_str, re.DOTALL)
        if match:
            input_str = match.group(1).strip()
        else:
            raise ValueError(f"No tool call tag found: {input_str}")

    if language == ReturnFormat.PYTHON:
        cleaned_input = input_str.strip().strip("'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted: list[dict[str, Any]] = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(_resolve_ast_call(parsed.body))
        elif isinstance(parsed.body, (ast.List, ast.Tuple)):
            for elem in parsed.body.elts:
                if not isinstance(elem, ast.Call):
                    raise ValueError(f"Expected ast.Call, got {type(elem)}")
                extracted.append(_resolve_ast_call(elem))
        else:
            raise ValueError(f"Unsupported AST body type: {type(parsed.body)}")
        logger.debug("Extracted {} function call(s) from Python output", len(extracted))
        return extracted

    if language == ReturnFormat.JSON:
        json_match = re.search(r"\[.*\]", input_str, re.DOTALL)
        if json_match:
            input_str = json_match.group(0)
        return _parse_json_function_call(input_str)

    if language == ReturnFormat.VERBOSE_XML:
        match = re.search(r"<functions>(.*?)</functions>", input_str, re.DOTALL)
        if not match:
            raise ValueError(f"No XML found: {input_str}")
        return _parse_verbose_xml_function_call(match.group(0))

    if language == ReturnFormat.CONCISE_XML:
        match = re.search(r"<functions>(.*?)</functions>", input_str, re.DOTALL)
        if not match:
            raise ValueError(f"No XML found: {input_str}")
        return _parse_concise_xml_function_call(match.group(0))

    raise NotImplementedError(f"Unsupported language: {language}")


def _resolve_ast_call(elem: ast.Call) -> dict[str, Any]:
    """Convert an ``ast.Call`` node into a function-call dictionary.

    Args:
        elem: AST call expression.

    Returns:
        Dictionary mapping function name to keyword arguments.
    """
    func_parts: list[str] = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict: dict[str, Any] = {}
    for arg in elem.keywords:
        if arg.arg is not None:
            args_dict[arg.arg] = _resolve_ast_by_type(arg.value)
    return {func_name: args_dict}


def _safe_eval_binop(node: ast.BinOp) -> Any:
    """Evaluate a simple binary operation AST node.

    Args:
        node: Binary operation AST node.

    Returns:
        Result of the binary operation.
    """
    left = _resolve_ast_by_type(node.left)
    right = _resolve_ast_by_type(node.right)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        return left / right
    if isinstance(node.op, ast.FloorDiv):
        return left // right
    if isinstance(node.op, ast.Mod):
        return left % right
    if isinstance(node.op, ast.Pow):
        return left**right
    raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")


def _resolve_ast_by_type(value: ast.AST) -> Any:
    """Recursively resolve an AST node to its Python value.

    Args:
        value: AST node to resolve.

    Returns:
        Python value represented by the node.
    """
    if isinstance(value, ast.Constant):
        if value.value is ...:
            return "..."
        return value.value
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
        operand = value.operand
        if isinstance(operand, ast.Constant) and isinstance(
            operand.value, (int, float)
        ):
            return -operand.value
        if isinstance(operand, ast.Num) and isinstance(operand.n, (int, float)):
            return -operand.n
        raise ValueError("Unsupported unary operand")
    if isinstance(value, ast.List):
        return [_resolve_ast_by_type(v) for v in value.elts]
    if isinstance(value, ast.Dict):
        keys = value.keys or []
        values = value.values or []
        return {
            _resolve_ast_by_type(k): _resolve_ast_by_type(v)
            for k, v in zip(keys, values)
            if k is not None
        }
    if isinstance(value, ast.NameConstant):
        return value.value
    if isinstance(value, ast.BinOp):
        return _safe_eval_binop(value)
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            return ast.unparse(value)
        return _resolve_ast_call(value)
    if isinstance(value, ast.Tuple):
        return tuple(_resolve_ast_by_type(v) for v in value.elts)
    if isinstance(value, ast.Ellipsis):
        return "..."
    if isinstance(value, ast.Subscript):
        return ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    raise ValueError(f"Unsupported AST type: {type(value)}")


def _parse_json_function_call(input_str: str) -> list[dict[str, Any]]:
    """Parse a JSON-encoded function call string.

    Args:
        input_str: JSON array of function calls.

    Returns:
        List of function-call dictionaries.
    """
    data = json.loads(input_str)
    if not isinstance(data, list):
        data = [data]
    result: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and "function" in item and "parameters" in item:
            result.append({item["function"]: item["parameters"]})
        elif isinstance(item, dict) and len(item) == 1:
            result.append(item)
        else:
            raise ValueError(f"Invalid JSON function call format: {item}")
    return result


def _parse_verbose_xml_function_call(xml_str: str) -> list[dict[str, Any]]:
    """Parse a verbose XML function call representation.

    Args:
        xml_str: XML string containing ``<functions>``.

    Returns:
        List of function-call dictionaries.
    """
    root = ET.fromstring(xml_str)
    result: list[dict[str, Any]] = []
    for func in root.findall("function"):
        func_name = func.get("name")
        params: dict[str, Any] = {}
        params_elem = func.find("params")
        if params_elem is not None:
            for param in params_elem.findall("param"):
                name = param.get("name")
                value = param.get("value")
                ptype = param.get("type", "string")
                if name is not None:
                    params[name] = _convert_xml_value(value, ptype)
        if func_name is not None:
            result.append({func_name: params})
    return result


def _parse_concise_xml_function_call(xml_str: str) -> list[dict[str, Any]]:
    """Parse a concise XML function call representation.

    Args:
        xml_str: XML string containing ``<functions>``.

    Returns:
        List of function-call dictionaries.
    """
    root = ET.fromstring(xml_str)
    result: list[dict[str, Any]] = []
    for func in root.findall("function"):
        func_name = func.get("name")
        params: dict[str, Any] = {}
        for param in func.findall("param"):
            name = param.get("name")
            ptype = param.get("type", "string")
            value = param.text
            if name is not None:
                params[name] = _convert_xml_value(value, ptype)
        if func_name is not None:
            result.append({func_name: params})
    return result


def _convert_xml_value(value: str | None, ptype: str) -> Any:
    """Convert an XML parameter value to its native Python type.

    Args:
        value: Raw XML attribute or text value.
        ptype: Declared parameter type.

    Returns:
        Native Python representation of the value.
    """
    if value is None:
        return None
    if ptype == "integer":
        return int(value)
    if ptype == "float":
        return float(value)
    if ptype == "boolean":
        return value.lower() == "true"
    if ptype == "array":
        try:
            return json.loads(value)
        except Exception:
            return value.split(",") if "," in value else [value]
    if ptype == "dict":
        try:
            return json.loads(value)
        except Exception:
            logger.warning(
                "Failed to parse dict value {!r}, returning empty dict", value
            )
            return {}
    return value
