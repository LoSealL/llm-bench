# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""BFCL v4 evaluation checker.

Ported from bfcl-lite for standalone evaluation.
"""

import re
from typing import Any, Protocol

from loguru import logger

from llm_bench.bfcl_constants import Language, ReturnFormat
from llm_bench.bfcl_utils import (
    is_empty_output,
    is_function_calling_format_output,
    is_relevance_or_irrelevance,
    load_dataset_entry,
    load_ground_truth_entry,
)


class DecodeHandler(Protocol):
    """Protocol for objects that can decode model output."""

    def decode_ast(
        self,
        result: str,
        language: ReturnFormat = ReturnFormat.PYTHON,
        has_tool_call_tag: bool = False,
    ) -> list[dict[str, Any]]: ...

    def decode_execute(
        self, result: str, has_tool_call_tag: bool = False
    ) -> list[str]: ...


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

PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]


def evaluate_task(
    test_category: str,
    model_result: list[dict[str, Any]],
    model_name: str,
    handler: DecodeHandler,
) -> dict[str, Any]:
    """Evaluate a single BFCL category.

    Args:
        test_category: Category name.
        model_result: List of prediction dicts with ``id`` and ``result`` keys.
        model_name: Model identifier.
        handler: Handler with ``decode_ast`` / ``decode_execute`` methods.

    Returns:
        Dictionary with ``accuracy``, ``correct_count``, ``total_count``,
        and ``errors`` keys.
    """
    prompt = load_dataset_entry(test_category)
    logger.debug(
        "Evaluating BFCL category %s with %d model results",
        test_category,
        len(model_result),
    )

    if is_relevance_or_irrelevance(test_category):
        model_result, prompt = _subset_entries(model_result, prompt, None)
        accuracy, total_count, errors = _relevance_file_runner(
            handler, model_result, prompt, model_name, test_category
        )
    else:
        possible_answer = load_ground_truth_entry(test_category)
        if possible_answer:
            assert len(prompt) == len(possible_answer), (
                "Prompt and ground truth length mismatch"
            )
        prompt, possible_answer = _subset_entries(model_result, prompt, possible_answer)
        accuracy, total_count, errors = _ast_file_runner(
            handler,
            model_result,
            prompt,
            possible_answer,
            test_category,
            model_name,
        )

    correct_count = int(accuracy * total_count) if total_count else 0
    logger.debug(
        "BFCL category %s accuracy: %.2f%% (%d/%d)",
        test_category,
        accuracy * 100,
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
    handler: DecodeHandler,
    model_result: list[dict[str, Any]],
    prompt: list[dict[str, Any]],
    model_name: str,
    test_category: str,
) -> tuple[float, int, list[dict[str, Any]]]:
    """Evaluate relevance or irrelevance categories.

    Args:
        handler: Decoder handler.
        model_result: Prediction entries.
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

        contain_func_call = False
        decoded_result = None
        decode_error = None
        try:
            decoded_result = handler.decode_ast(
                model_result_item,
                language=ReturnFormat.PYTHON,
                has_tool_call_tag=False,
            )
            contain_func_call = True
            if is_empty_output(decoded_result):
                contain_func_call = False
        except Exception as e:
            contain_func_call = False
            decode_error = str(e)
            logger.debug("Decode error for entry {}: {}", index, decode_error)

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
                temp["error"] = [
                    "Valid syntax. Successfully decode AST when it should not."
                ]
                temp["error_type"] = "irrelevance_error:decoder_success"
            else:
                temp["error"] = [
                    f"Invalid syntax. Failed to decode AST. {decode_error}"
                ]
                temp["error_type"] = "relevance_error:decoder_failed"
            errors.append(temp)

    accuracy = correct_count / len(model_result) if model_result else 0.0
    return accuracy, len(model_result), errors


def _ast_file_runner(
    handler: DecodeHandler,
    model_result: list[dict[str, Any]],
    prompt: list[dict[str, Any]],
    possible_answer: list[dict[str, Any]],
    test_category: str,
    model_name: str,
) -> tuple[float, int, list[dict[str, Any]]]:
    """Evaluate standard AST-based function-calling categories.

    Args:
        handler: Decoder handler.
        model_result: Prediction entries.
        prompt: Corresponding prompt entries.
        possible_answer: Ground-truth function calls.
        test_category: Category name.
        model_name: Model identifier.

    Returns:
        Tuple of ``(accuracy, total_count, errors)``.
    """
    assert len(model_result) == len(prompt) == len(possible_answer), "Length mismatch"

    language = Language.PYTHON
    return_format = ReturnFormat.PYTHON

    correct_count = 0
    errors: list[dict[str, Any]] = []
    logger.debug("Running AST evaluation for {} entries", len(model_result))
    for i, entry in enumerate(model_result):
        index = entry["id"]
        model_result_item = entry["result"]
        prompt_entry = prompt[i]
        possible_answer_item = possible_answer[i]["ground_truth"]

        entry_result = _evaluate_single_ast_entry(
            handler,
            index,
            model_result_item,
            possible_answer_item,
            prompt_entry,
            model_name,
            test_category,
            language,
            return_format,
        )

        if entry_result["valid"]:
            correct_count += 1
        else:
            errors.append(entry_result)

    accuracy = correct_count / len(model_result) if model_result else 0.0
    return accuracy, len(model_result), errors


def _evaluate_single_ast_entry(
    handler: DecodeHandler,
    index: str,
    model_result_item: str,
    possible_answer_item: list[dict[str, Any]],
    prompt_entry: dict[str, Any],
    _model_name: str,
    test_category: str,
    language: Language,
    return_format: ReturnFormat,
) -> dict[str, Any]:
    """Evaluate a single AST-based prediction entry.

    Args:
        handler: Decoder handler.
        index: Entry identifier.
        model_result_item: Raw model output.
        possible_answer_item: Ground-truth function calls.
        prompt_entry: Prompt entry with function definitions.
        model_name: Model identifier.
        test_category: Category name.
        language: Expected language.
        return_format: Expected return format.

    Returns:
        Validation result dictionary.
    """
    prompt_function = prompt_entry["function"]
    model_result_item_raw = model_result_item
    try:
        decoded_output = handler.decode_ast(model_result_item, return_format, False)
    except Exception as e:
        return {
            "id": index,
            "model_name": _model_name,
            "test_category": test_category,
            "valid": False,
            "error": [f"Invalid syntax. Failed to decode AST. {e!s}"],
            "error_type": "ast_decoder:decoder_failed",
            "prompt": prompt_entry,
            "model_result_raw": model_result_item_raw,
            "possible_answer": possible_answer_item,
        }

    if not is_function_calling_format_output(decoded_output):
        return {
            "id": index,
            "model_name": _model_name,
            "test_category": test_category,
            "valid": False,
            "error": ["Did not output in the specified format."],
            "error_type": "ast_decoder:decoder_wrong_output_format",
            "prompt": prompt_entry,
            "model_result_raw": str(model_result_item_raw),
            "model_result_decoded": str(decoded_output),
            "possible_answer": possible_answer_item,
        }

    checker_result = ast_checker(
        prompt_function,
        decoded_output,
        possible_answer_item,
        language,
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
            "model_result_raw": model_result_item_raw,
            "model_result_decoded": model_result_item,
            "possible_answer": possible_answer_item,
        }
    return {"valid": True}


def ast_checker(
    func_description: list[dict[str, Any]] | dict[str, Any],
    model_output: list[dict[str, Any]],
    possible_answer: list[dict[str, Any]],
    language: Language,
    test_category: str,
    _model_name: str,
) -> dict[str, Any]:
    """Dispatch to the appropriate function-call checker.

    Args:
        func_description: Function schema or list of schemas.
        model_output: Decoded model function calls.
        possible_answer: Ground-truth function calls.
        language: Ground-truth language.
        test_category: Category name.
        model_name: Model identifier.

    Returns:
        Validation result dictionary.
    """
    if "parallel" in test_category:
        return _parallel_function_checker_no_order(
            func_description, model_output, possible_answer, language
        )
    if "multiple" in test_category:
        return _multiple_function_checker(
            func_description, model_output, possible_answer, language
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
        language,
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


def _get_possible_answer_type(possible_answer: list[Any]) -> type | None:
    """Determine the Python type of the first non-empty possible answer.

    Args:
        possible_answer: List of possible answer values.

    Returns:
        The inferred type, or ``None`` if all values are empty.
    """
    for answer in possible_answer:
        if answer != "":
            return type(answer)
    return None


def _type_checker(
    param: str,
    value: Any,
    possible_answer: list[Any],
    expected_type_description: str,
    expected_type_converted: type,
    nested_type_converted: type | None,
) -> dict[str, Any]:
    """Check whether a parameter value has the expected type.

    Args:
        param: Parameter name.
        value: Model-provided parameter value.
        possible_answer: List of acceptable values.
        expected_type_description: Expected type as a string.
        expected_type_converted: Expected Python type.
        nested_type_converted: Expected nested element type for lists/tuples.

    Returns:
        Type-check result dictionary.
    """
    result = {
        "valid": True,
        "error": [],
        "is_variable": False,
        "error_type": "type_error:simple",
    }
    is_variable = False
    possible_answer_type = _get_possible_answer_type(possible_answer)
    if (
        possible_answer_type is not None
        and possible_answer_type != expected_type_converted
    ):
        is_variable = True
    if isinstance(value, expected_type_converted):
        if nested_type_converted is None:
            result["is_variable"] = is_variable
            return result
        for possible_answer_item in possible_answer:
            flag = True
            if isinstance(possible_answer_item, list):
                for value_item in value:
                    checker_result = _type_checker(
                        param,
                        value_item,
                        possible_answer_item,
                        str(nested_type_converted),
                        nested_type_converted,
                        None,
                    )
                    if not checker_result["valid"]:
                        flag = False
                        break
                if flag:
                    return {"valid": True, "error": [], "is_variable": is_variable}
        result["valid"] = False
        result["error"] = [f"Nested type checking failed for parameter {param!r}."]
        result["error_type"] = "type_error:nested"
    possible_answer_type = _get_possible_answer_type(possible_answer)
    if possible_answer_type is not None and isinstance(value, possible_answer_type):
        result["is_variable"] = True
        return result
    result["valid"] = False
    result["error"].append(
        f"Incorrect type for parameter {param!r}. "
        f"Expected type {expected_type_description}, got {type(value).__name__}. "
        f"Parameter value: {value!r}."
    )
    result["error_type"] = "type_error:simple"
    return result


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
    standardize_model_output = list(model_output)
    for i, item in enumerate(standardize_model_output):
        if isinstance(item, str):
            standardize_model_output[i] = _standardize_string(item)
    standardize_possible_answer: list[list[Any]] = []
    for i, ans in enumerate(possible_answer):
        standardize_possible_answer.append([])
        for _, val in enumerate(ans):
            if isinstance(val, str):
                standardize_possible_answer[i].append(_standardize_string(val))
            else:
                standardize_possible_answer[i].append(val)
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
        param: Parameter name.
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
            standardize_value = value
            if isinstance(value, str):
                standardize_value = _standardize_string(value)
            standardize_possible_answer: list[Any] = []
            for j in range(len(possible_answer[key])):
                if isinstance(possible_answer[key][j], str):
                    standardize_possible_answer.append(
                        _standardize_string(possible_answer[key][j])
                    )
                else:
                    standardize_possible_answer.append(possible_answer[key][j])
            if standardize_value not in standardize_possible_answer:
                result["valid"] = False
                result["error"].append(
                    f"Invalid value for parameter {key!r}: {value!r}. "
                    f"Expected one of {standardize_possible_answer}."
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
    _language: Language,
) -> dict[str, Any]:
    """Validate a single function call against its schema and ground truth.

    Args:
        func_description: Function schema dictionary.
        model_output: Decoded function call dictionary.
        possible_answer: Ground-truth parameter values.
        language: Ground-truth language.

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
        is_variable = False
        nested_type_converted = None

        expected_type_converted = PYTHON_TYPE_MAPPING[expected_type_description]
        if expected_type_description in PYTHON_NESTED_TYPE_CHECK_LIST:
            nested_type = param_details[param]["items"]["type"]
            nested_type_converted = PYTHON_TYPE_MAPPING[nested_type]

        if expected_type_description == "tuple" and isinstance(value, tuple):
            value = list(value)

        if expected_type_description == "float" and isinstance(value, int):
            value = float(value)

        type_check_result = _type_checker(
            param,
            value,
            possible_answer[param],
            expected_type_description,
            expected_type_converted,
            nested_type_converted,
        )
        is_variable = type_check_result["is_variable"]
        if not type_check_result["valid"]:
            return type_check_result

        if not is_variable:
            if expected_type_converted is dict:
                assert isinstance(value, dict)
                result = _dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue
            if expected_type_converted is list and nested_type_converted is dict:
                assert isinstance(value, list)
                result = _list_dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue
            if expected_type_converted is str:
                assert isinstance(value, str)
                result = _string_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue
            if expected_type_converted is list:
                assert isinstance(value, list)
                result = _list_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

        if value not in possible_answer[param]:
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


def _parallel_function_checker_no_order(
    func_descriptions: list[dict[str, Any]] | dict[str, Any],
    model_output: list[dict[str, Any]],
    possible_answers: list[dict[str, Any]],
    _language: Language,
) -> dict[str, Any]:
    """Validate parallel function calls without enforcing order.

    Args:
        func_descriptions: Function schemas.
        model_output: Decoded function calls.
        possible_answers: Ground-truth function calls.
        language: Ground-truth language.

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
                func_description, model_output_item, possible_answer, _language
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
    _language: Language,
) -> dict[str, Any]:
    """Validate multiple sequential function calls.

    Args:
        func_descriptions: Function schemas.
        model_output: Decoded function calls.
        possible_answers: Ground-truth function calls.
        language: Ground-truth language.

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
            func_description, model_output[i], possible_answer, _language
        )
        if not result["valid"]:
            result["error"].insert(0, f"Function call at index {i} failed validation.")
            result["error_type"] = (
                f"multiple_function_checker:index_{i}:{result['error_type']}"
            )
            return result
    return {"valid": True, "error": []}
