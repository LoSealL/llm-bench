# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""BFCL v4 benchmark runner.

Evaluates function-calling ability via the OpenAI-compatible
``/v1/chat/completions`` API using native tool calling. Function
definitions are passed through the ``tools`` parameter and tool-call
results are read directly from the structured ``message.tool_calls``
field rather than parsed from free-form text.
"""

import copy
import json
from pathlib import Path
from typing import Any

from loguru import logger

from llm_bench.bfcl_eval import evaluate_task
from llm_bench.bfcl_utils import (
    load_dataset_entry,
    parse_test_category_argument,
)
from llm_bench.client import ChatResponse, LLMClient
from llm_bench.runners import BaseRunner, _JsonlWriter


def _tool_calls_to_decoded(tool_calls: list[Any]) -> list[dict[str, Any]]:
    """Convert native OpenAI tool-call objects to the BFCL decoded format.

    Reads the function name and arguments directly from the structured
    ``message.tool_calls`` field — no free-form text parsing.

    Args:
        tool_calls: Tool-call objects from the chat completion response.

    Returns:
        List of ``{function_name: {param: value}}`` dictionaries.
    """
    decoded: list[dict[str, Any]] = []
    for call in tool_calls:
        name = call.function.name
        raw_args = call.function.arguments
        try:
            args = json.loads(raw_args) if raw_args else {}
        except (json.JSONDecodeError, ValueError):
            logger.warning("Malformed tool arguments for {}: {!r}", name, raw_args)
            args = {}
        decoded.append({name: args})
    return decoded


def _response_to_result(response: ChatResponse) -> list[dict[str, Any]]:
    """Extract structured function calls from a native tool-call response.

    Args:
        response: Structured chat response.

    Returns:
        List of ``{function_name: {param: value}}`` dictionaries, or an
        empty list when the model returned no tool calls (e.g. declined
        to call a function).
    """
    if response.tool_calls:
        return _tool_calls_to_decoded(response.tool_calls)
    return []


def _bfcl_function_to_openai_tool(function: dict[str, Any]) -> dict[str, Any]:
    """Convert a BFCL function schema to an OpenAI tool definition.

    Args:
        function: BFCL function schema using ``"type": "dict"``.

    Returns:
        OpenAI tool definition ``{"type": "function", "function": ...}``.
    """
    converted = _convert_dict_type_to_object(copy.deepcopy(function))
    return {"type": "function", "function": converted}


def _convert_dict_type_to_object(node: Any) -> Any:
    """Recursively rewrite non-standard JSON-schema types.

    BFCL data uses ``"type": "dict"``, ``"type": "float"``,
    ``"type": "tuple"``, and ``"type": "any"`` — none of which are
    valid JSON Schema types. This function rewrites them to
    ``"object"``, ``"number"``, ``"array"``, and removes the key
    respectively so OpenAI-compatible providers accept the schema.

    Args:
        node: A decoded JSON-schema node.

    Returns:
        The node with non-standard types replaced.
    """
    if isinstance(node, dict):
        if node.get("type") == "dict":
            node["type"] = "object"
        elif node.get("type") == "float":
            node["type"] = "number"
        elif node.get("type") == "tuple":
            node["type"] = "array"
        elif node.get("type") == "any":
            del node["type"]
        return {k: _convert_dict_type_to_object(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_convert_dict_type_to_object(item) for item in node]
    return node


class BFCLRunner(BaseRunner):
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
        max_tokens: int = 1024,
        temperature: float = 0.0,
        *,
        force: bool = False,
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
            temperature: Sampling temperature.
            force: If ``True``, re-run even when cached JSONL exists.
        """
        super().__init__(client, output_dir, "bfcl", limit, force=force)
        self._max_tokens = max_tokens
        self._temperature = temperature
        if categories is None:
            categories = ["simple_python", "multiple"]
        self._categories = parse_test_category_argument(categories)
        logger.info(
            "BFCL runner initialised: categories={}, limit={}, max_tokens={}",
            self._categories,
            limit,
            self._max_tokens,
        )

    def _build_messages(self, entry: dict[str, Any]) -> list[dict[str, str]]:
        """Return the original message list for a single entry unchanged.

        The dataset's own messages (including any system prompt) are
        respected verbatim; BFCL no longer prepends its own system
        prompt. Functions are exposed to the model via the native
        ``tools`` parameter instead.

        Args:
            entry: A dataset entry with a ``question`` key.

        Returns:
            The original message list from the dataset.
        """
        return list(entry["question"][0])

    def dry_run(self, **kwargs: Any) -> None:
        """Load dataset and print metadata without API calls."""
        for category in self._categories:
            dataset = load_dataset_entry(category)
            dataset = self._apply_limit(dataset)
            logger.info(
                "BFCL '{}' — {} samples",
                category,
                len(dataset),
            )
            for item in dataset:
                sample_id = item.get("id", "unknown")
                logger.info("  Sample: {}", sample_id)
                messages = item.get("question", [])
                if messages:
                    logger.info(
                        "    Question: {}",
                        messages[0][-1].get("content", "")[:200]
                        if isinstance(messages[0], list) and messages[0]
                        else str(messages)[:200],
                    )
                functions = item.get("function", [])
                logger.info(
                    "    Functions: {}", [f.get("name", "?") for f in functions]
                )

    def _build_tools(self, entry: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Build OpenAI tool definitions for a single entry.

        Args:
            entry: A dataset entry with a ``function`` key.

        Returns:
            OpenAI-style tool list, or ``None`` when the entry exposes
            no functions (e.g. irrelevance probes).
        """
        functions: list[dict[str, Any]] = entry.get("function", [])
        if not functions:
            return None
        return [_bfcl_function_to_openai_tool(fn) for fn in functions]

    def _predict_category(
        self,
        category: str,
        skip: int = 0,
        writer: _JsonlWriter | None = None,
    ) -> list[dict[str, Any]]:
        """Run inference on a single BFCL category.

        Args:
            category: BFCL category name.
            skip: Number of samples to skip (already cached).
            writer: Optional streaming JSONL writer.

        Returns:
            List of prediction dicts with ``id`` and ``result`` keys.
        """
        dataset = load_dataset_entry(category)
        logger.info("Loaded {} entries for category '{}'", len(dataset), category)
        dataset = self._apply_limit(dataset)
        if self._limit is not None:
            logger.info("Limited to {} samples", self._limit)

        if skip:
            dataset = dataset[skip:]
            logger.info("Skipping {} cached samples for {}", skip, category)

        results: list[dict[str, Any]] = []
        for entry in self._progress(dataset, desc=f"BFCL-{category}"):
            messages = self._build_messages(entry)
            tools = self._build_tools(entry)
            response = self._chat(
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                tools=tools,
            )
            if not response:
                logger.warning(
                    "Invalid response for entry {} (finish_reason={})",
                    entry["id"],
                    response.finish_reason,
                )
            record = {
                "id": entry["id"],
                "result": _response_to_result(response),
                "valid": response.valid,
                "finish_reason": response.finish_reason,
            }
            results.append(record)
            if writer is not None:
                writer.write(record)
        logger.info(
            "Completed predictions for '{}' ({} samples)", category, len(results)
        )
        return results

    def _score_category(
        self, category: str, predictions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Score predictions for a single category.

        Only valid predictions are counted toward the accuracy.

        Args:
            category: BFCL category name.
            predictions: Prediction dicts from :meth:`_predict_category`.

        Returns:
            Result dictionary with ``accuracy``, ``correct_count``,
            ``total_count``, and ``errors``.
        """
        model_name = self._client._model
        valid_predictions = [p for p in predictions if p.get("valid", True)]
        return evaluate_task(category, valid_predictions, model_name)

    def run(self, **kwargs: Any) -> dict[str, dict[str, Any]]:
        """Run the BFCL v4 benchmark.

        Returns:
            Mapping ``category_name -> {accuracy, correct_count,
            total_count, errors}``.
        """
        all_results: dict[str, dict[str, Any]] = {}
        for category in self._categories:
            filename = f"{category}.jsonl"
            existing, writer = self._resume_jsonl(filename)
            try:
                new_predictions = self._predict_category(
                    category, skip=len(existing), writer=writer
                )
            finally:
                writer.close()
            predictions = existing + new_predictions

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
