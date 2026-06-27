# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Omni AI OCR benchmark runner.

Evaluates structured document extraction on the getomni-ai/ocr-benchmark
dataset (1,000 document images) using OpenAI-compatible multimodal chat
APIs. Scoring compares the model's parsed JSON output against each
sample's ``true_json_output`` under a normalized exact-match rule.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image  # type: ignore[import-untyped]

from llm_bench.client import LLMClient
from llm_bench.runners import BaseRunner, _JsonlWriter

_DATASET_NAME = "getomni-ai/ocr-benchmark"
_DATASET_SPLIT = "test"


class OmniOCRBenchRunner(BaseRunner):
    """Execute the Omni AI OCR benchmark suite.

    Evaluates structured document extraction across varied document
    formats (tables, charts, invoices, checks, etc.). The model is asked
    to output JSON matching a per-sample schema, and the output is
    scored via normalized exact-match against ``true_json_output``.

    Attributes:
        _client: :class:`LLMClient` instance for API calls.
        _output_dir: Directory where predictions are saved.
        _max_tokens: Maximum new tokens for answer generation.
        _temperature: Sampling temperature.
    """

    _SYSTEM_PROMPT = (
        "You are a document extraction assistant. "
        "Extract the structured information from the provided document image "
        "and output it as valid JSON matching the given JSON schema. "
        "Output ONLY the JSON, no explanations or markdown formatting."
    )

    def __init__(
        self,
        client: LLMClient | None,
        output_dir: str | Path,
        limit: int | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        image_width: int | None = None,
        image_height: int | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Prepare the runner.

        Args:
            client: Initialized LLM client (``None`` for dry-run mode).
            output_dir: Base output directory; results go to
                ``output_dir/ocrbench_omni/``.
            limit: If set, cap the number of evaluated samples.
            max_tokens: Max new tokens for the answer generation.
            temperature: Sampling temperature.
            image_width: If set, resize images to this width.
            image_height: If set, resize images to this height.
            force: If ``True``, re-run even when cached JSONL exists.
        """
        super().__init__(client, output_dir, "ocrbench_omni", limit, force=force)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._image_size = (
            (image_width, image_height)
            if image_width is not None and image_height is not None
            else None
        )

    def _build_messages(
        self,
        image: Image.Image,
        json_schema: str,
    ) -> list[dict[str, Any]]:
        """Build multimodal messages for an Omni AI OCR sample.

        Args:
            image: PIL Image instance.
            json_schema: JSON schema string for this sample.

        Returns:
            OpenAI-compatible messages list with image content and the
            extraction instruction embedding the schema.
        """
        data_uri = self._prepare_image_data_uri(image, self._image_size)
        question = (
            "Extract the structured data from this document image.\n\n"
            f"Output JSON matching this schema:\n{json_schema}"
        )
        return self._build_vision_messages(data_uri, question, self._SYSTEM_PROMPT)

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        """Remove code fences and extract the JSON payload.

        Handles `````json ... `````` fences and surrounding prose.

        Args:
            text: Raw model output (after thinking-strip).

        Returns:
            Cleaned text that should be parseable as JSON.
        """
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _normalize_json_value(value: Any) -> Any:
        """Recursively normalize a parsed JSON value for comparison.

        Lowercases all strings; sorts dict keys. Lists keep element order
        (order-insensitive comparison is handled at the dict level only;
        arrays represent ordered data in documents).

        Args:
            value: Parsed JSON value (dict, list, str, etc.).

        Returns:
            Normalized value.
        """
        if isinstance(value, str):
            return value.lower()
        if isinstance(value, dict):
            return {
                k: OmniOCRBenchRunner._normalize_json_value(v)
                for k, v in sorted(value.items())
            }
        if isinstance(value, list):
            return [OmniOCRBenchRunner._normalize_json_value(item) for item in value]
        return value

    @classmethod
    def _compare_json(
        cls,
        pred_text: str,
        answer_text: str,
    ) -> tuple[bool, Any, Any]:
        """Compare model output to ground-truth JSON.

        Parses both, normalizes them, and checks structural equality.
        Order-insensitive at the dict-key level, case-insensitive for
        string values, and whitespace/indentation-insensitive (parsed
        comparison).

        Args:
            pred_text: Raw model output (after thinking/fence strip).
            answer_text: Ground-truth JSON string.

        Returns:
            3-tuple ``(correct, pred_json, answer_json)``. On parse
            failure, ``correct`` is ``False`` and the json values may be
            ``None``.
        """
        try:
            pred_obj = json.loads(pred_text)
        except (json.JSONDecodeError, TypeError):
            return False, None, None

        try:
            answer_obj = json.loads(answer_text)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Ground-truth JSON is unparseable")
            return False, pred_obj, None

        return (
            cls._normalize_json_value(pred_obj)
            == cls._normalize_json_value(answer_obj),
            pred_obj,
            answer_obj,
        )

    def dry_run(self, **kwargs: Any) -> None:
        """Load dataset and print metadata without API calls."""
        dataset = self._load_hf_dataset(
            _DATASET_NAME,
            _DATASET_SPLIT,
            "Omni AI OCR",
        )
        self._inspect_dataset(
            dataset,
            label="Omni AI OCR",
            image_field="image",
            fields=[
                "id",
                "metadata",
                "true_json_output",
            ],
            image_size=self._image_size,
        )

    @staticmethod
    def _extract_format(metadata: str) -> str:
        """Extract the document format from the metadata JSON string.

        Args:
            metadata: JSON string from the dataset's ``metadata`` field.

        Returns:
            The ``format`` value, or ``"unknown"`` if not found.
        """
        try:
            meta = json.loads(metadata)
            return meta.get("format", "unknown")
        except (json.JSONDecodeError, TypeError):
            return "unknown"

    def _predict(
        self,
        skip: int = 0,
        writer: _JsonlWriter | None = None,
    ) -> list[dict[str, Any]]:
        """Run inference on Omni AI OCR.

        Args:
            skip: Number of samples to skip (already cached).
            writer: Optional streaming JSONL writer.

        Returns:
            List of prediction dicts.
        """
        import base64
        import io

        data_all = self._load_hf_dataset(
            _DATASET_NAME,
            _DATASET_SPLIT,
            "Omni AI OCR",
        )

        if skip:
            data_all = data_all[skip:]
            logger.info("Skipping {} cached samples", skip)

        results: list[dict[str, Any]] = []
        for item in self._progress(data_all, desc="Omni AI OCR"):
            row = dict(item)
            raw_image = row.get("image")
            if isinstance(raw_image, dict) and "bytes" in raw_image:
                image = Image.open(io.BytesIO(raw_image["bytes"]))
            elif isinstance(raw_image, str):
                image = Image.open(io.BytesIO(base64.b64decode(raw_image)))
            else:
                image = raw_image

            if not isinstance(image, Image.Image):
                image_valid = False
            else:
                image_valid = self._validate_image(image)
            doc_format = self._extract_format(row.get("metadata", ""))
            json_schema = row.get("json_schema", "")
            answer_text = row.get("true_json_output", "")

            if image_valid:
                assert isinstance(image, Image.Image)
                messages = self._build_messages(image, json_schema)
                response = self._chat(
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                if response.valid:
                    raw_pred = response.content
                    cleaned = self._strip_thinking(raw_pred)
                    cleaned = self._strip_json_fences(cleaned)
                    correct, pred_json, _ = self._compare_json(cleaned, answer_text)
                    pred_text = cleaned
                else:
                    correct = False
                    pred_json = None
                    pred_text = ""
                finish_reason = response.finish_reason
                response_text = response.content
                valid = response.valid
            else:
                logger.warning("Skipping invalid image for id {}", row.get("id"))
                correct = False
                pred_json = None
                pred_text = ""
                finish_reason = None
                response_text = ""
                valid = False

            record = {
                "id": row.get("id"),
                "data_id": f"ocrbench_omni_{row.get('id')}",
                "format": doc_format,
                "pred": pred_text,
                "pred_json": pred_json,
                "answer": answer_text,
                "correct": correct,
                "valid": valid,
                "image_valid": image_valid,
                "finish_reason": finish_reason,
                "response": response_text,
            }
            results.append(record)
            if writer is not None:
                writer.write(record)
        return results

    def _compute_stats(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate accuracy by document format and overall.

        Args:
            data: Predictions from :meth:`_predict`.

        Returns:
            Dictionary with ``overall`` and ``by_format`` keys.
        """
        return self._grouped_stats(
            data,
            group_fn=lambda item: item.get("format") or "unknown",
            group_label="format",
        )

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run the Omni AI OCR benchmark.

        Returns:
            Dictionary with keys ``overall`` and ``by_format``.
        """
        filename = "predictions.jsonl"
        existing, writer = self._resume_jsonl(filename)
        try:
            new_data = self._predict(skip=len(existing), writer=writer)
        finally:
            writer.close()
        data = existing + new_data

        stats = self._compute_stats(data)
        o = stats["overall"]
        logger.info(
            "Omni AI OCR: {:.2f}% ({}/{})",
            o["accuracy"],
            o["correct"],
            o["total"],
        )
        return stats
