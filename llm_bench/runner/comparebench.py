# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""CompareBench benchmark runner.

Evaluates visual comparison reasoning on qiuzhangTiTi/CompareBench
using OpenAI-compatible multimodal chat APIs.
"""

from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image  # type: ignore[import-untyped]

from llm_bench.client import LLMClient
from llm_bench.runners import BaseRunner, _JsonlWriter

_SPLITS = [
    "CompareTallyBench",
    "CompareGeometryBench",
    "CompareSpatialBench",
    "CompareTemporalBench",
]

# Splits that existed in earlier versions of the dataset.
# Kept for CLI choices so historical SQLite records remain queryable.
_LEGACY_SPLITS = [
    "CompareHistBench",
    "CompareCelebrityBench",
    "CompareLandmarkBench",
]

_ALL_KNOWN_SPLITS = _SPLITS + _LEGACY_SPLITS


class CompareBenchRunner(BaseRunner):
    """Execute the CompareBench benchmark suite.

    Attributes:
        _client: :class:`LLMClient` instance for API calls.
        _output_dir: Directory where predictions are saved.
        _max_tokens: Maximum new tokens for answer generation.
        _image_size: Optional (width, height) tuple for resizing.
    """

    _SYSTEM_PROMPT = (
        "You are a helpful visual reasoning assistant. "
        "Answer the question with a single letter: A, B, C, or D. "
        "Do not explain your reasoning."
    )

    def __init__(
        self,
        client: LLMClient,
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
            client: Initialized LLM client (must support vision inputs).
            output_dir: Base output directory; results go to
                ``output_dir/comparebench/``.
            limit: If set, cap the number of evaluated samples per split.
            max_tokens: Max new tokens for the answer generation.
            temperature: If set, override the default sampling temperature.
            image_width: If set, resize images to this width.
            image_height: If set, resize images to this height.
            force: If ``True``, re-run even when cached JSONL exists.
        """
        super().__init__(client, output_dir, "comparebench", limit, force=force)
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
        """Build multimodal messages for a VQA sample.

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
        """Extract a single-letter answer (A/B/C/D) from model output.

        Tries explicit markers first, then falls back to looking for a
        lone letter.

        Args:
            response: Raw model response.

        Returns:
            Upper-case single letter or the cleaned raw text.
        """
        if not response:
            return ""

        text = BaseRunner._strip_thinking(response)

        answer = BaseRunner._extract_letter_answer(
            text,
            patterns=[
                r"(?:answer|choice|option)[\s:：是为]*([A-D])",
                r"\(?([A-D])\)?",
                r"([A-D])[.、)]",
            ],
            fallback=False,
            flags=re.IGNORECASE,
        )
        if answer:
            return answer

        cleaned = re.sub(r"[^A-Da-d]", "", text.strip())
        if len(cleaned) == 1:
            return cleaned.upper()

        return text.strip()

    def _compare(self, pred: str, answer: str) -> bool:
        """Case-insensitive letter comparison.

        Args:
            pred: Predicted answer.
            answer: Ground-truth answer.

        Returns:
            ``True`` if letters match after uppercasing.
        """
        return pred.strip().upper() == answer.strip().upper()

    def dry_run(self, **kwargs: Any) -> None:
        """Load dataset and print metadata without API calls."""
        splits = kwargs.get("selected_splits") or _SPLITS
        for split_name in splits:
            dataset = self._load_hf_dataset(
                "qiuzhangTiTi/CompareBench",
                split_name,
                f"CompareBench/{split_name}",
            )
            self._inspect_dataset(
                dataset,
                label=f"CompareBench/{split_name}",
                image_field="image",
                fields=[
                    "image_name",
                    "vlm_question",
                    "gt_answer",
                ],
            )

    def _predict_split(
        self,
        split_name: str,
        skip: int = 0,
        writer: _JsonlWriter | None = None,
    ) -> list[dict[str, Any]]:
        """Run inference on a single CompareBench split.

        Args:
            split_name: Name of the dataset split to evaluate.
            skip: Number of samples to skip (already cached).
            writer: Optional streaming JSONL writer.

        Returns:
            List of prediction dicts.
        """
        dataset = self._load_hf_dataset(
            "qiuzhangTiTi/CompareBench",
            split_name,
            f"CompareBench/{split_name}",
        )

        if skip:
            dataset = dataset[skip:]
            logger.info("Skipping {} cached samples for {}", skip, split_name)

        results: list[dict[str, Any]] = []
        for idx, item in enumerate(
            self._progress(dataset, desc=f"CompareBench/{split_name}"),
            start=skip,
        ):
            row = dict(item)
            raw_image = row["image"]
            if isinstance(raw_image, dict) and "bytes" in raw_image:
                image = Image.open(io.BytesIO(raw_image["bytes"]))
            elif isinstance(raw_image, str):
                image = Image.open(io.BytesIO(base64.b64decode(raw_image)))
            else:
                image = raw_image
            assert isinstance(image, Image.Image)

            image_valid = self._validate_image(image)
            if image_valid:
                messages = self._build_messages(image, row["vlm_question"])
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
                logger.warning(
                    "Skipping invalid image for data_id {}",
                    f"{split_name}_{idx}",
                )
                pred = ""
                finish_reason = None
                response_text = ""
                valid = False

            answer = str(row.get("gt_answer", "")).strip().upper()
            record = {
                "data_id": f"{split_name}_{idx}",
                "split": split_name,
                "image_name": row.get("image_name", ""),
                "question": row["vlm_question"],
                "pred": pred,
                "answer": answer,
                "correct": self._compare(pred, answer) if valid else False,
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
        """Aggregate accuracy by split and overall.

        Args:
            data: Predictions from all splits.

        Returns:
            Dictionary with ``overall`` and ``by_split`` keys.
        """
        return self._grouped_stats(
            data,
            group_fn=lambda item: item["split"],
            group_label="split",
        )

    def run(
        self,
        selected_splits: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the CompareBench benchmark.

        Args:
            selected_splits: Subset of splits to evaluate.  ``None``
                evaluates all splits.
            **kwargs: Unused (for API compatibility).

        Returns:
            Dictionary with keys ``overall`` and ``by_split``.
        """
        splits = selected_splits if selected_splits is not None else _SPLITS
        filename = "predictions.jsonl"

        existing, writer = self._resume_jsonl(filename)
        try:
            # Count existing records per split for partial resume
            existing_by_split: dict[str, int] = {}
            for rec in existing:
                s = rec.get("split", "")
                existing_by_split[s] = existing_by_split.get(s, 0) + 1

            new_results: list[dict[str, Any]] = []
            for split_name in splits:
                skip = existing_by_split.get(split_name, 0)
                if skip:
                    logger.info(
                        "Skipping {} cached samples for {}",
                        skip,
                        split_name,
                    )
                try:
                    split_results = self._predict_split(
                        split_name, skip=skip, writer=writer
                    )
                    new_results.extend(split_results)
                except ValueError as exc:
                    if "Unknown split" in str(exc):
                        logger.warning(
                            "Split '{}' not found in dataset — skipping "
                            "(historical records still in report)",
                            split_name,
                        )
                        continue
                    raise
        finally:
            writer.close()
        all_results = existing + new_results

        stats = self._compute_stats(all_results)
        o = stats["overall"]
        logger.info(
            "CompareBench: {:.2f}% ({}/{})",
            o["accuracy"],
            o["correct"],
            o["total"],
        )
        for split, s in stats["by_split"].items():
            logger.info(
                "  {}: {:.2f}% ({}/{})",
                split,
                s["accuracy"],
                s["correct"],
                s["total"],
            )
        return stats
