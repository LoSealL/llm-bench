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
from llm_bench.runners import BaseRunner

_SPLITS = [
    "CompareTallyBench",
    "CompareGeometryBench",
    "CompareSpatialBench",
    "CompareHistBench",
    "CompareCelebrityBench",
    "CompareLandmarkBench",
]


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
        """
        super().__init__(client, output_dir, "comparebench", limit)
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

        text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

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

    def _predict_split(self, split_name: str) -> list[dict[str, Any]]:
        """Run inference on a single CompareBench split.

        Args:
            split_name: Name of the dataset split to evaluate.

        Returns:
            List of prediction dicts.
        """
        dataset = self._load_hf_dataset(
            "qiuzhangTiTi/CompareBench",
            split_name,
            f"CompareBench/{split_name}",
        )

        results: list[dict[str, Any]] = []
        for idx, item in enumerate(
            self._progress(dataset, desc=f"CompareBench/{split_name}")
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

            messages = self._build_messages(image, row["vlm_question"])
            response = self._chat(
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            pred = self._extract_answer(response)
            answer = str(row.get("gt_answer", "")).strip().upper()
            results.append(
                {
                    "data_id": f"{split_name}_{idx}",
                    "split": split_name,
                    "image_name": row.get("image_name", ""),
                    "question": row["vlm_question"],
                    "pred": pred,
                    "answer": answer,
                    "correct": pred == answer,
                    "response": response,
                },
            )
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
        all_results: list[dict[str, Any]] = []
        for split_name in splits:
            split_results = self._predict_split(split_name)
            all_results.extend(split_results)

        self._write_jsonl(all_results, "predictions.jsonl")

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
