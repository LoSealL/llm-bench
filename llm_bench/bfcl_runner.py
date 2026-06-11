# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""BFCL v4 benchmark runner.

Evaluates function-calling ability via prompting mode using the
OpenAI-compatible client.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from llm_bench.bfcl_constants import ReturnFormat
from llm_bench.bfcl_eval import evaluate_task
from llm_bench.bfcl_parser import ast_parse
from llm_bench.bfcl_utils import (
    load_dataset_entry,
    parse_test_category_argument,
    system_prompt_pre_processing_chat_model,
)
from llm_bench.client import LLMClient
from llm_bench.reporter import ensure_dir


class MockHandler:
    """Minimal handler for BFCL prompting-mode evaluation.

    Provides ``decode_ast`` and ``decode_execute`` so the original
    ``evaluate_task`` can be reused without a full model-handler
    implementation.
    """

    @staticmethod
    def decode_ast(
        result: str,
        language: ReturnFormat = ReturnFormat.PYTHON,
        has_tool_call_tag: bool = False,
    ) -> list[dict[str, Any]]:
        """Decode raw model text into structured function calls."""
        result = result.strip("`\n ")
        if not (result.startswith("[") and result.endswith("]")):
            result = "[" + result + "]"
        return ast_parse(result, language, has_tool_call_tag)

    @staticmethod
    def decode_execute(result: str, has_tool_call_tag: bool = False) -> list[str]:
        """Decode raw model text into executable Python strings."""
        result = result.strip("`\n ")
        if not (result.startswith("[") and result.endswith("]")):
            result = "[" + result + "]"
        decoded_output = ast_parse(
            result, language=ReturnFormat.PYTHON, has_tool_call_tag=has_tool_call_tag
        )
        return _decoded_output_to_execution_list(decoded_output)


def _decoded_output_to_execution_list(
    decoded_output: list[dict[str, Any]],
) -> list[str]:
    execution_list: list[str] = []
    for function_call in decoded_output:
        for key, value in function_call.items():
            args_str = ", ".join(
                f"{k}={_parse_nested_value(v)}" for k, v in value.items()
            )
            execution_list.append(f"{key}({args_str})")
    return execution_list


def _parse_nested_value(value: Any) -> str:
    if isinstance(value, dict):
        if all(isinstance(v, dict) for v in value.values()):
            func_name = list(value.keys())[0]
            args = value[func_name]
            args_str = ", ".join(
                f"{k}={_parse_nested_value(v)}" for k, v in args.items()
            )
            return f"{func_name}({args_str})"
        return (
            "{"
            + ", ".join(f"'{k}': {_parse_nested_value(v)}" for k, v in value.items())
            + "}"
        )
    return repr(value)


class BFCLRunner:
    """Execute the BFCL v4 benchmark suite.

    Attributes:
        _client: :class:`LLMClient` instance for API calls.
        _output_dir: Directory where ``.jsonl`` predictions are saved.
        _categories: List of BFCL categories to evaluate.
        _limit: Optional sample limit per category.
    """

    def __init__(
        self,
        client: LLMClient,
        output_dir: str | Path,
        categories: list[str] | None = None,
        limit: int | None = None,
            max_tokens: int = 32000,
    ) -> None:
        """Prepare the runner.

        Args:
            client: Initialized LLM client.
            output_dir: Base output directory; predictions are written
                to ``output_dir/bfcl/``.
            categories: BFCL categories to evaluate. ``None`` defaults
                to ``["simple_python", "multiple"]``.
            limit: If set, evaluate only the first *N* samples per
                category.
            max_tokens: Maximum number of new tokens to generate.
        """
        self._client = client
        self._limit = limit
        self._max_tokens = max_tokens
        self._output_dir = Path(output_dir) / "bfcl"
        ensure_dir(self._output_dir)
        if categories is None:
            categories = ["simple_python", "multiple"]
        self._categories = parse_test_category_argument(categories)
        logger.info(
            "BFCL runner initialised: categories={}, limit={}, max_tokens={}",
            self._categories,
            limit,
            max_tokens,
        )

    def _build_messages(self, entry: dict[str, Any]) -> list[dict[str, str]]:
        """Build the message list for a single entry.

        Args:
            entry: A dataset entry with ``question`` and ``function``.

        Returns:
            List of messages with roles (system, user, etc.).
        """
        messages: list[dict[str, str]] = list(entry["question"][0])
        function_docs: list[dict[str, Any]] = entry.get("function", [])
        return system_prompt_pre_processing_chat_model(
            messages, function_docs, entry["id"]
        )

    def _predict_category(self, category: str) -> list[dict[str, Any]]:
        """Run inference on a single BFCL category.

        Args:
            category: BFCL category name.

        Returns:
            List of prediction dicts with ``id`` and ``result`` keys.
        """
        dataset = load_dataset_entry(category)
        logger.info("Loaded {} entries for category '{}'", len(dataset), category)
        if self._limit is not None:
            dataset = dataset[: self._limit]
            logger.info("Limited to {} samples", self._limit)

        results: list[dict[str, Any]] = []
        for entry in tqdm(dataset, desc=f"BFCL-{category}"):
            messages = self._build_messages(entry)
            raw_response = self._client.chat(
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=0.1,
            )
            if not raw_response:
                logger.warning("Empty response for entry {}", entry["id"])
            results.append(
                {
                    "id": entry["id"],
                    "result": raw_response,
                }
            )
        logger.info(
            "Completed predictions for '{}' ({} samples)", category, len(results)
        )
        return results

    def _score_category(
        self, category: str, predictions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Score predictions for a single category.

        Args:
            category: BFCL category name.
            predictions: Prediction dicts from :meth:`_predict_category`.

        Returns:
            Result dictionary with ``accuracy``, ``correct_count``,
            ``total_count``, and ``errors``.
        """
        handler = MockHandler()
        model_name = self._client._model
        return evaluate_task(category, predictions, model_name, handler)

    def run(self) -> dict[str, dict[str, Any]]:
        """Run the BFCL v4 benchmark.

        Returns:
            Mapping ``category_name -> {accuracy, correct_count,
            total_count, errors}``.
        """
        all_results: dict[str, dict[str, Any]] = {}
        for category in self._categories:
            predictions = self._predict_category(category)

            stats = self._score_category(category, predictions)
            all_results[category] = stats
            logger.info(
                "BFCL {}: {:.2%} ({}/{})",
                category,
                stats["accuracy"],
                stats["correct_count"],
                stats["total_count"],
            )

        return all_results
