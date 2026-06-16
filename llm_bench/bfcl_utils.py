# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""BFCL v4 utilities.

Data loading and helpers for native tool-call evaluation.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from llm_bench.bfcl_constants import (
    ALL_CATEGORIES,
    POSSIBLE_ANSWER_PATH,
    PROMPT_PATH,
    TEST_COLLECTION_MAPPING,
    VERSION_PREFIX,
)


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
        _test_category: Category name.

    Returns:
        Modified function schema list.
    """
    if len(functions) == 0:
        return functions
    hint = " Note that the provided function is in Python 3 syntax."
    for item in functions:
        item["description"] = item.get("description", "") + hint
    return functions
