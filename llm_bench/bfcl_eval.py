# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""BFCL v4 evaluation checker.

Evaluates native tool-call results (structured function-call lists)
against BFCL ground truth. Type checking is lenient: values are
compared via normalized string representations so that ``"5"`` matches
``5`` and ``"True"`` matches ``True``.
"""

import re
from typing import Any

from loguru import logger

from llm_bench.bfcl_utils import (
    is_empty_output,
    is_function_calling_format_output,
    is_relevance_or_irrelevance,
    load_dataset_entry,
    load_ground_truth_entry,
)


PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}


def _normalize_for_compare(value: Any) -> Any:
    """Recursively normalize a value for type-agnostic comparison.

    Native tool-call arguments may encode values as strings even when
    the schema declares a number or boolean (e.g. ``"5"`` for an
    ``integer``, ``"True"`` for a ``boolean``). Rather than coercing
    types — which is ambiguous for values like ``'"5"'`` or nested
    structures — we normalize every leaf to a comparable string and
    recurse into containers. This lets ``"5"`` match ``5``,
    ``"True"`` match ``True``, and ``["1", "2"]`` match ``[1, 2]``
    without any type parsing.

    Args:
        value: Any Python value (scalar, list, or dict).

    Returns:
        A normalized form where scalars are lowercased strings and
        containers are normalized recursively.
    """
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, list):
        return [_normalize_for_compare(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_for_compare(v) for k, v in value.items()}
    return str(value)


def _value_in_possible(value: Any, possible_answers: list[Any]) -> bool:
    """Check whether *value* matches any entry in *possible_answers*.

    Comparison is type-agnostic: both sides are normalized via
    :func:`_normalize_for_compare` so that ``"5"`` matches ``5`` and
    ``"True"`` matches ``True``. An empty-string entry (``""``) is
    treated as a wildcard (optional parameter).

    Args:
        value: The model-provided parameter value.
        possible_answers: List of acceptable ground-truth values.

    Returns:
        ``True`` when *value* matches at least one acceptable answer.
    """
    norm_value = _normalize_for_compare(value)
    for ans in possible_answers:
        if ans == "":
            return True
        if norm_value == _normalize_for_compare(ans):
            return True
    return False


def evaluate_task(
    test_category: str,
    model_result: list[dict[str, Any]],
    model_name: str,
) -> dict[str, Any]:
    """Evaluate a single BFCL category.

    Args:
        test_category: Category name.
        model_result: List of prediction dicts with ``id`` and ``result``
            keys. Each ``result`` is a structured list of function-call
            dictionaries (``[{func_name: {param: value}}, ...]``) produced
            by native tool calling.
        model_name: Model identifier.

    Returns:
        Dictionary with ``accuracy``, ``correct_count``, ``total_count``,
        and ``errors`` keys.
    """
    prompt = load_dataset_entry(test_category)
    logger.debug(
        "Evaluating BFCL category {} with {} model results",
        test_category,
        len(model_result),
    )

    if is_relevance_or_irrelevance(test_category):
        prompt, _ = _subset_entries(model_result, prompt, None)
        accuracy, total_count, errors = _relevance_file_runner(
            model_result, prompt, model_name, test_category
        )
    else:
        possible_answer = load_ground_truth_entry(test_category)
        if possible_answer:
            assert len(prompt) == len(possible_answer), (
                "Prompt and ground truth length mismatch"
            )
        prompt, possible_answer = _subset_entries(model_result, prompt, possible_answer)
        accuracy, total_count, errors = _function_call_runner(
            model_result,
            prompt,
            possible_answer,
            test_category,
            model_name,
        )

    correct_count = int(accuracy * total_count) if total_count else 0
    logger.debug(
        "BFCL category {} accuracy: {:.2%} ({}/{})",
        test_category,
        accuracy,
        correct_count,
        total_count,
    )
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "errors": errors,
    }


def _subset_entries(
    model_result_entries: list[dict[str, Any]],
    prompt_entries: list[dict[str, Any]],
    ground_truth_entries: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter prompt and ground-truth entries to IDs present in predictions.

    Args:
        model_result_entries: Prediction entries with ``id`` keys.
        prompt_entries: Full prompt entries.
        ground_truth_entries: Optional full ground-truth entries.

    Returns:
        Filtered prompt entries and ground-truth entries.
    """
    if not model_result_entries:
        return [], []
    present_ids = {entry["id"]: entry for entry in model_result_entries}
    filtered_prompt: list[dict[str, Any]] = []
    filtered_gt: list[dict[str, Any]] = []
    for idx, p in enumerate(prompt_entries):
        if p["id"] in present_ids:
            filtered_prompt.append(p)
            if ground_truth_entries is not None:
                filtered_gt.append(ground_truth_entries[idx])
    return filtered_prompt, filtered_gt


def _relevance_file_runner(
    model_result: list[dict[str, Any]],
    prompt: list[dict[str, Any]],
    model_name: str,
    test_category: str,
) -> tuple[float, int, list[dict[str, Any]]]:
    """Evaluate relevance or irrelevance categories.

    Args:
        model_result: Prediction entries with structured ``result`` lists.
        prompt: Corresponding prompt entries.
        model_name: Model identifier.
        test_category: Category name.

    Returns:
        Tuple of ``(accuracy, total_count, errors)``.
    """
    correct_count = 0
    errors: list[dict[str, Any]] = []
    logger.debug("Running relevance check for {} entries", len(model_result))
    for i, entry in enumerate(model_result):
        index = entry["id"]
        model_result_item = entry["result"]
        prompt_entry = prompt[i]

        decoded_result = (
            model_result_item if isinstance(model_result_item, list) else []
        )
        contain_func_call = not is_empty_output(decoded_result)

        if "irrelevance" in test_category:
            success = not contain_func_call
        else:
            success = contain_func_call

        if success:
            correct_count += 1
        else:
            temp: dict[str, Any] = {
                "id": index,
                "model_name": model_name,
                "test_category": test_category,
                "valid": success,
                "prompt": prompt_entry,
                "model_result": model_result_item,
                "decoded_result": decoded_result,
            }
            if "irrelevance" in test_category:
                temp["error"] = ["Tool call returned when none was expected."]
                temp["error_type"] = "irrelevance_error:tool_call_present"
            else:
                temp["error"] = ["No tool call returned when one was expected."]
                temp["error_type"] = "relevance_error:no_tool_call"
            errors.append(temp)

    accuracy = correct_count / len(model_result) if model_result else 0.0
    return accuracy, len(model_result), errors


def _function_call_runner(
    model_result: list[dict[str, Any]],
    prompt: list[dict[str, Any]],
    possible_answer: list[dict[str, Any]],
    test_category: str,
    model_name: str,
) -> tuple[float, int, list[dict[str, Any]]]:
    """Evaluate standard function-calling categories.

    Args:
        model_result: Prediction entries with structured ``result`` lists.
        prompt: Corresponding prompt entries.
        possible_answer: Ground-truth function calls.
        test_category: Category name.
        model_name: Model identifier.

    Returns:
        Tuple of ``(accuracy, total_count, errors)``.
    """
    assert len(model_result) == len(prompt) == len(possible_answer), "Length mismatch"

    correct_count = 0
    errors: list[dict[str, Any]] = []
    logger.debug("Running function-call evaluation for {} entries", len(model_result))
    for i, entry in enumerate(model_result):
        index = entry["id"]
        model_result_item = entry["result"]
        prompt_entry = prompt[i]
        possible_answer_item = possible_answer[i]["ground_truth"]

        entry_result = _evaluate_single_entry(
            index,
            model_result_item,
            possible_answer_item,
            prompt_entry,
            model_name,
            test_category,
        )

        if entry_result["valid"]:
            correct_count += 1
        else:
            errors.append(entry_result)

    accuracy = correct_count / len(model_result) if model_result else 0.0
    return accuracy, len(model_result), errors


def _evaluate_single_entry(
    index: str,
    model_result_item: Any,
    possible_answer_item: list[dict[str, Any]],
    prompt_entry: dict[str, Any],
    _model_name: str,
    test_category: str,
) -> dict[str, Any]:
    """Evaluate a single prediction entry.

    Args:
        index: Entry identifier.
        model_result_item: Structured function-call list from native
            tool calling (``[{func_name: {param: value}}, ...]``).
        possible_answer_item: Ground-truth function calls.
        prompt_entry: Prompt entry with function definitions.
        _model_name: Model name.
        test_category: Category name.

    Returns:
        Validation result dictionary.
    """
    prompt_function = prompt_entry["function"]
    decoded_output = model_result_item if isinstance(model_result_item, list) else []

    if not is_function_calling_format_output(decoded_output):
        return {
            "id": index,
            "model_name": _model_name,
            "test_category": test_category,
            "valid": False,
            "error": ["Did not output in the specified function-calling format."],
            "error_type": "eval:wrong_output_format",
            "prompt": prompt_entry,
            "model_result_raw": model_result_item,
            "possible_answer": possible_answer_item,
        }

    checker_result = function_call_checker(
        prompt_function,
        decoded_output,
        possible_answer_item,
        test_category,
        _model_name,
    )

    if not checker_result["valid"]:
        return {
            "id": index,
            "model_name": _model_name,
            "test_category": test_category,
            "valid": checker_result["valid"],
            "error": checker_result["error"],
            "error_type": checker_result["error_type"],
            "prompt": prompt_entry,
            "model_result_raw": model_result_item,
            "model_result_decoded": decoded_output,
            "possible_answer": possible_answer_item,
        }
    return {"valid": True}


def function_call_checker(
    func_description: list[dict[str, Any]] | dict[str, Any],
    model_output: list[dict[str, Any]],
    possible_answer: list[dict[str, Any]],
    test_category: str,
    _model_name: str,
) -> dict[str, Any]:
    """Dispatch to the appropriate function-call checker.

    Args:
        func_description: Function schema or list of schemas.
        model_output: Decoded model function calls.
        possible_answer: Ground-truth function calls.
        test_category: Category name.
        _model_name: Model identifier.

    Returns:
        Validation result dictionary.
    """
    if "parallel" in test_category:
        return _parallel_function_checker_no_order(
            func_description, model_output, possible_answer
        )
    if "multiple" in test_category:
        return _multiple_function_checker(
            func_description, model_output, possible_answer
        )
    if len(model_output) != 1:
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "simple_function_checker:wrong_count",
        }
    return _simple_function_checker(
        func_description[0] if isinstance(func_description, list) else func_description,
        model_output[0],
        possible_answer[0],
    )


def _find_description(
    func_descriptions: list[dict[str, Any]] | dict[str, Any], name: str
) -> dict[str, Any] | None:
    """Find a function description by name.

    Args:
        func_descriptions: Function schema or list of schemas.
        name: Function name to find.

    Returns:
        Matching function description, or ``None``.
    """
    if isinstance(func_descriptions, list):
        for fd in func_descriptions:
            if fd["name"] == name:
                return fd
        return None
    return func_descriptions


def _standardize_string(input_string: str) -> str:
    """Normalise a string for fuzzy comparison.

    Args:
        input_string: Raw string value.

    Returns:
        Lowercase string with punctuation and whitespace removed.
    """
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def _string_checker(
    param: str, model_output: str, possible_answer: list[Any]
) -> dict[str, Any]:
    """Check a string parameter against a list of acceptable values.

    Args:
        param: Parameter name.
        model_output: Model-provided string value.
        possible_answer: List of acceptable values.

    Returns:
        Validation result dictionary.
    """
    standardize_possible_answer: list[str] = []
    standardize_model_output = _standardize_string(model_output)
    for ans in possible_answer:
        if isinstance(ans, str):
            standardize_possible_answer.append(_standardize_string(ans))
    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {param!r}: {model_output!r}. "
                f"Expected one of {possible_answer}."
            ],
            "error_type": "value_error:string",
        }
    return {"valid": True, "error": []}


def _list_checker(
    param: str, model_output: list[Any], possible_answer: list[Any]
) -> dict[str, Any]:
    """Check a list parameter against acceptable list values.

    Args:
        param: Parameter name.
        model_output: Model-provided list value.
        possible_answer: List of acceptable list values.

    Returns:
        Validation result dictionary.
    """
    standardize_model_output = [_normalize_for_compare(item) for item in model_output]
    standardize_possible_answer: list[list[Any]] = [
        [_normalize_for_compare(val) for val in ans] for ans in possible_answer
    ]
    if standardize_model_output not in standardize_possible_answer:
        return {
            "valid": False,
            "error": [
                f"Invalid value for parameter {param!r}: {model_output!r}. "
                f"Expected one of {possible_answer}."
            ],
            "error_type": "value_error:list/tuple",
        }
    return {"valid": True, "error": []}


def _dict_checker(
    _param: str, model_output: dict[str, Any], possible_answers: list[Any]
) -> dict[str, Any]:
    """Check a dict parameter against acceptable dict values.

    Args:
        _param: Parameter name.
        model_output: Model-provided dict value.
        possible_answers: List of acceptable dict values.

    Returns:
        Validation result dictionary.
    """
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for possible_answer in possible_answers:
        if possible_answer == "":
            continue
        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        if not isinstance(possible_answer, dict):
            continue
        for key, value in model_output.items():
            if key not in possible_answer:
                result["valid"] = False
                result["error"].append(f"Unexpected dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break
            norm_value = _normalize_for_compare(value)
            norm_possible = [_normalize_for_compare(v) for v in possible_answer[key]]
            if norm_value not in norm_possible:
                result["valid"] = False
                result["error"].append(
                    f"Invalid value for parameter {key!r}: {value!r}. "
                    f"Expected one of {possible_answer[key]}."
                )
                result["error_type"] = "value_error:dict_value"
                flag = False
                break
        for key, value in possible_answer.items():
            if key not in model_output and "" not in value:
                result["valid"] = False
                result["error"].append(f"Missing dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def _list_dict_checker(
    param: str, model_output: list[Any], possible_answers: list[Any]
) -> dict[str, Any]:
    """Check a list-of-dicts parameter against acceptable values.

    Args:
        param: Parameter name.
        model_output: Model-provided list of dicts.
        possible_answers: List of acceptable list-of-dicts values.

    Returns:
        Validation result dictionary.
    """
    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}
    for _, possible_answer in enumerate(possible_answers):
        flag = True
        if len(model_output) != len(possible_answer):
            result["valid"] = False
            result["error"] = ["Wrong number of dictionaries in the list."]
            result["error_type"] = "value_error:list_dict_count"
            flag = False
            continue
        for dict_index, model_output_item in enumerate(model_output):
            result = _dict_checker(
                param,
                model_output_item,
                [possible_answer[dict_index]],
            )
            if not result["valid"]:
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}
    return result


def _simple_function_checker(
    func_description: dict[str, Any],
    model_output: dict[str, Any],
    possible_answer: dict[str, Any],
) -> dict[str, Any]:
    """Validate a single function call against its schema and ground truth.

    Dispatches to type-specific checkers (dict, list, string) when the
    model value's type matches the schema-declared type. For all other
    cases — including type mismatches common in native tool calls —
    falls back to :func:`_value_in_possible` for type-agnostic
    comparison.

    Args:
        func_description: Function schema dictionary.
        model_output: Decoded function call dictionary.
        possible_answer: Ground-truth parameter values.

    Returns:
        Validation result dictionary.
    """
    possible_answer = list(possible_answer.values())[0]
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"]["required"]
    result = {
        "valid": True,
        "error": [],
        "error_type": "simple_function_checker:unclear",
    }

    if func_name not in model_output:
        result["valid"] = False
        result["error"].append(
            f"Function name {func_name!r} not found in model output."
        )
        result["error_type"] = "simple_function_checker:wrong_func_name"
        return result

    model_params = model_output[func_name]
    for param in required_params:
        if param not in model_params:
            result["valid"] = False
            result["error"].append(f"Missing required parameter: {param!r}.")
            result["error_type"] = "simple_function_checker:missing_required"
            return result

    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            result["valid"] = False
            result["error"].append(f"Unexpected parameter: {param!r}.")
            result["error_type"] = "simple_function_checker:unexpected_param"
            return result

        full_param_details = param_details[param]
        expected_type_description = full_param_details["type"]
        expected_type = PYTHON_TYPE_MAPPING[expected_type_description]

        if expected_type_description == "tuple" and isinstance(value, tuple):
            value = list(value)

        if expected_type_description == "float" and isinstance(value, int):
            value = float(value)

        nested_type: type | None = None
        if expected_type_description in ("array", "tuple"):
            items = full_param_details.get("items", {})
            nested_type = PYTHON_TYPE_MAPPING.get(items.get("type", "any"), str)

        check_result = _check_param_value(
            param,
            value,
            possible_answer[param],
            expected_type,
            nested_type,
        )
        if check_result is not None:
            if not check_result["valid"]:
                return check_result
            continue

        if not _value_in_possible(value, possible_answer[param]):
            result["valid"] = False
            result["error"].append(
                f"Invalid value for parameter {param!r}: {value!r}. "
                f"Expected one of {possible_answer[param]}."
            )
            result["error_type"] = "value_error:others"
            return result

    for param in possible_answer:
        if param not in model_params and "" not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Optional parameter {param!r} not provided and not marked as optional."
            )
            result["error_type"] = "simple_function_checker:missing_optional"
            return result

    return result


def _check_param_value(
    param: str,
    value: Any,
    possible: list[Any],
    expected_type: type,
    nested_type: type | None,
) -> dict[str, Any] | None:
    """Dispatch to a type-specific checker when types align.

    Returns ``None`` when no type-specific checker applies (type
    mismatch or scalar types like int/float/bool), signaling the caller
    to use the generic :func:`_value_in_possible` fallback.

    Args:
        param: Parameter name.
        value: Model-provided value.
        possible: Acceptable ground-truth values for this parameter.
        expected_type: Declared Python type from the schema.
        nested_type: Declared element type for list/tuple params.

    Returns:
        Checker result dict, or ``None`` for fallback.
    """
    if expected_type is dict and isinstance(value, dict):
        return _dict_checker(param, value, possible)
    if expected_type is list and isinstance(value, list):
        if nested_type is dict:
            return _list_dict_checker(param, value, possible)
        return _list_checker(param, value, possible)
    if expected_type is str and isinstance(value, str):
        return _string_checker(param, value, possible)
    return None


def _parallel_function_checker_no_order(
    func_descriptions: list[dict[str, Any]] | dict[str, Any],
    model_output: list[dict[str, Any]],
    possible_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate parallel function calls without enforcing order.

    Args:
        func_descriptions: Function schemas.
        model_output: Decoded function calls.
        possible_answers: Ground-truth function calls.

    Returns:
        Validation result dictionary.
    """
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "parallel_function_checker_no_order:wrong_count",
        }
    matched_indices: list[int] = []
    all_errors: list[Any] = []
    for i, possible_answer in enumerate(possible_answers):
        func_name_expected = list(possible_answer.keys())[0]
        func_description = _find_description(func_descriptions, func_name_expected)
        if func_description is None:
            return {
                "valid": False,
                "error": [f"Function description not found for {func_name_expected!r}"],
                "error_type": "parallel_function_checker_no_order:missing_desc",
            }
        result: dict[str, Any] = {"valid": False, "error": [], "error_type": ""}
        for index, model_output_item in enumerate(model_output):
            if index in matched_indices:
                continue
            result = _simple_function_checker(
                func_description, model_output_item, possible_answer
            )
            if result["valid"]:
                matched_indices.append(index)
                break
            all_errors.append(
                {
                    f"Model Result Index {index}": {
                        "sub_error": result["error"],
                        "sub_error_type": result["error_type"],
                        "model_output_item": model_output_item,
                        "possible_answer_item": possible_answer,
                    }
                }
            )
        if not result["valid"]:
            considered_indices = [
                idx for idx in range(len(model_output)) if idx not in matched_indices
            ]
            all_errors.insert(
                0,
                f"Could not find a matching function among index {considered_indices} "
                f"for index {i} of possible answers.",
            )
            return {
                "valid": False,
                "error": all_errors,
                "error_type": "parallel_function_checker_no_order:cannot_find_match",
            }
    return {"valid": True, "error": []}


def _multiple_function_checker(
    func_descriptions: list[dict[str, Any]] | dict[str, Any],
    model_output: list[dict[str, Any]],
    possible_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate multiple sequential function calls.

    Args:
        func_descriptions: Function schemas.
        model_output: Decoded function calls.
        possible_answers: Ground-truth function calls.

    Returns:
        Validation result dictionary.
    """
    if len(model_output) != len(possible_answers):
        return {
            "valid": False,
            "error": ["Wrong number of functions."],
            "error_type": "multiple_function_checker:wrong_count",
        }
    for i, possible_answer in enumerate(possible_answers):
        func_name_expected = list(possible_answer.keys())[0]
        func_description = _find_description(func_descriptions, func_name_expected)
        if func_description is None:
            return {
                "valid": False,
                "error": [f"Function description not found for {func_name_expected!r}"],
                "error_type": f"multiple_function_checker:index_{i}:missing_desc",
            }
        result = _simple_function_checker(
            func_description, model_output[i], possible_answer
        )
        if not result["valid"]:
            result["error"].insert(0, f"Function call at index {i} failed validation.")
            result["error_type"] = (
                f"multiple_function_checker:index_{i}:{result['error_type']}"
            )
            return result
    return {"valid": True, "error": []}
