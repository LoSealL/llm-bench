# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""OCRBench v2 benchmark runner.

Evaluates OCR capabilities on the ling99/OCRBench_v2 dataset (10,000
image-question pairs across 30 task types) using OpenAI-compatible
multimodal chat APIs. Scoring is rule-based, keyed on each sample's
``eval`` field.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image  # type: ignore[import-untyped]

from llm_bench.client import LLMClient
from llm_bench.runners import BaseRunner, _JsonlWriter

_DATASET_NAME = "ling99/OCRBench_v2"
_DATASET_SPLIT = "test"


class OCRBenchV2Runner(BaseRunner):
    """Execute the OCRBench v2 benchmark suite.

    Evaluates visual text recognition and reasoning across diverse
    image-question pairs. Per-sample scoring is determined by the
    ``eval`` field: ``exact match``, ``multiple choice``,
    ``case sensitive``, ``regression``, or ``None`` (containment).

    Attributes:
        _client: :class:`LLMClient` instance for API calls.
        _output_dir: Directory where predictions are saved.
        _max_tokens: Maximum new tokens for answer generation.
        _temperature: Sampling temperature.
    """

    _SYSTEM_PROMPT = (
        "Answer the question based on the image. "
        "Provide only the answer, do not explain."
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
                ``output_dir/ocrbench_v2/``.
            limit: If set, cap the number of evaluated samples.
            max_tokens: Max new tokens for the answer generation.
            temperature: Sampling temperature.
            image_width: If set, resize images to this width.
            image_height: If set, resize images to this height.
            force: If ``True``, re-run even when cached JSONL exists.
        """
        super().__init__(client, output_dir, "ocrbench_v2", limit, force=force)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._image_size = (
            (image_width, image_height)
            if image_width is not None and image_height is not None
            else None
        )

    def _build_messages(
        self, image: Image.Image, question: str
    ) -> list[dict[str, Any]]:
        """Build multimodal messages for an OCRBench v2 sample.

        Args:
            image: PIL Image instance.
            question: Question text.

        Returns:
            OpenAI-compatible messages list with image content.
        """
        data_uri = self._prepare_image_data_uri(image, self._image_size)
        return self._build_vision_messages(data_uri, question, self._SYSTEM_PROMPT)

    @staticmethod
    def _extract_answer(response: str) -> str:
        """Clean model output by removing reasoning tags and markdown.

        Args:
            response: Raw model response.

        Returns:
            Cleaned answer string.
        """
        if not response:
            return ""

        text = BaseRunner._strip_thinking(response)
        text = re.sub(r"[*_`#]", "", text)
        text = re.sub(
            r"^(Answer[：:]?\s*)+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return text.strip()

    @staticmethod
    def _score_sample(
        pred: str,
        answers: list[str],
        eval_method: str,
    ) -> bool:
        """Score a single sample against accepted answers.

        Dispatches on the ``eval`` field value:

        - ``exact match`` / ``multiple choice``: case-insensitive exact
          match against any accepted answer.
        - ``case sensitive``: case-sensitive exact match.
        - ``regression`` / ``None``: containment check (any accepted
          answer is a substring of the prediction, case-insensitive).

        Args:
            pred: Predicted answer string.
            answers: List of accepted answers.
            eval_method: The sample's ``eval`` field value.

        Returns:
            ``True`` if the prediction passes the scoring rule.
        """
        if not pred:
            return False

        method = eval_method.strip() if eval_method else "None"

        if method in ("exact match", "multiple choice"):
            pred_lower = pred.lower()
            return any(pred_lower == ans.strip().lower() for ans in answers)

        if method == "case sensitive":
            return any(pred == ans.strip() for ans in answers)

        if method in ("regression", "None"):
            pred_lower = pred.lower()
            return any(ans.strip().lower() in pred_lower for ans in answers)

        logger.warning("Unknown eval method '{}', using containment", method)
        pred_lower = pred.lower()
        return any(ans.strip().lower() in pred_lower for ans in answers)

    def dry_run(self, **kwargs: Any) -> None:
        """Load dataset and print metadata without API calls."""
        dataset = self._load_hf_dataset(
            _DATASET_NAME,
            _DATASET_SPLIT,
            "OCRBench v2",
        )
        self._inspect_dataset(
            dataset,
            label="OCRBench v2",
            image_field="image",
            fields=[
                "id",
                "question",
                "answers",
                "eval",
                "type",
                "dataset_name",
            ],
            image_size=self._image_size,
        )

    def _predict(
        self,
        skip: int = 0,
        writer: _JsonlWriter | None = None,
    ) -> list[dict[str, Any]]:
        """Run inference on OCRBench v2.

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
            "OCRBench v2",
        )

        if skip:
            data_all = data_all[skip:]
            logger.info("Skipping {} cached samples", skip)

        results: list[dict[str, Any]] = []
        for item in self._progress(data_all, desc="OCRBench v2"):
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
            answers = row.get("answers") or []
            eval_method = row.get("eval") or "None"

            if image_valid:
                assert isinstance(image, Image.Image)
                messages = self._build_messages(image, row["question"])
                response = self._chat(
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                pred = self._extract_answer(response.content) if response.valid else ""
                finish_reason = response.finish_reason
                response_text = response.content
                valid = response.valid
            else:
                logger.warning("Skipping invalid image for id {}", row.get("id"))
                pred = ""
                finish_reason = None
                response_text = ""
                valid = False

            correct = self._score_sample(pred, answers, eval_method) if valid else False
            record = {
                "id": row.get("id"),
                "data_id": f"ocrbench_v2_{row.get('id')}",
                "task_type": row.get("type", ""),
                "dataset_name": row.get("dataset_name", ""),
                "eval_method": eval_method,
                "question": row["question"],
                "pred": pred,
                "answers": answers,
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
        """Aggregate accuracy by task type and overall.

        Args:
            data: Predictions from :meth:`_predict`.

        Returns:
            Dictionary with ``overall`` and ``by_task_type`` keys.
        """
        return self._grouped_stats(
            data,
            group_fn=lambda item: item.get("task_type") or "unknown",
            group_label="task_type",
        )

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run the OCRBench v2 benchmark.

        Returns:
            Dictionary with keys ``overall`` and ``by_task_type``.
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
            "OCRBench v2: {:.2f}% ({}/{})",
            o["accuracy"],
            o["correct"],
            o["total"],
        )
        return stats
