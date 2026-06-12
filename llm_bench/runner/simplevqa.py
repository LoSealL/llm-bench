# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""SimpleVQA benchmark runner.

Evaluates vision-language understanding on m-a-p/SimpleVQA using
OpenAI-compatible multimodal chat APIs.
"""

from __future__ import annotations

import re
import string
import unicodedata
from pathlib import Path
from typing import Any

from loguru import logger

from llm_bench.client import LLMClient
from llm_bench.runners import BaseRunner


class SimpleVQARunner(BaseRunner):
    """Execute the SimpleVQA benchmark suite.

    Attributes:
        _client: :class:`LLMClient` instance for API calls.
        _output_dir: Directory where predictions are saved.
        _max_tokens: Maximum new tokens for answer generation.
    """

    _SYSTEM_PROMPT = "请根据图片内容回答问题，直接给出答案即可，不要解释。"

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
                ``output_dir/simplevqa/``.
            limit: If set, cap the number of evaluated samples.
            max_tokens: Max new tokens for the answer generation.
            temperature: If set, override the default sampling temperature.
            image_width: If set, resize images to this width.
            image_height: If set, resize images to this height.
        """
        super().__init__(client, output_dir, "simplevqa", limit)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._image_size = (
            (image_width, image_height)
            if image_width is not None and image_height is not None
            else None
        )

    def _build_messages(self, image_b64: str, question: str) -> list[dict[str, Any]]:
        """Build multimodal messages for a VQA sample.

        Args:
            image_b64: Raw base64 string (without data URI prefix).
            question: Question text.

        Returns:
            OpenAI-compatible messages list with image content.
        """
        data_uri = self._prepare_image_data_uri(image_b64, self._image_size)
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

        text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        text = re.sub(r"[*_`#]", "", text)
        text = re.sub(
            r"^(答案[：:]?\s*|Answer[：:]?\s*)+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return text.strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for fair comparison.

        Handles Chinese punctuation, full-width characters, spaces,
        and ASCII punctuation.

        Args:
            text: Raw text.

        Returns:
            Normalized text.
        """
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)
        text = text.lower().replace(" ", "")
        text = re.sub(
            r"[，。！？、；：\"\"''（）【】《》…—～・]"
            r"|[%s]" % re.escape(string.punctuation),
            "",
            text,
        )
        return text.strip()

    def _exact_match(self, pred: str, answer: str) -> bool:
        """Check normalized exact match.

        Args:
            pred: Predicted answer.
            answer: Ground-truth answer.

        Returns:
            ``True`` if normalized strings match.
        """
        return self._normalize_text(pred) == self._normalize_text(answer)

    def _predict(self) -> list[dict[str, Any]]:
        """Run inference on SimpleVQA.

        Returns:
            List of prediction dicts with ``data_id``, ``question``,
            ``pred``, ``answer``, ``correct``, ``response``,
            ``original_category``.
        """
        data_all = self._load_hf_dataset(
            "m-a-p/SimpleVQA",
            "test",
            "SimpleVQA",
        )

        results: list[dict[str, Any]] = []
        for item in self._progress(data_all, desc="SimpleVQA"):
            row = dict(item)
            image_valid = self._validate_image(row["image"])
            if image_valid:
                messages = self._build_messages(row["image"], row["question"])
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
                logger.warning("Skipping invalid image for data_id {}", row["data_id"])
                pred = ""
                finish_reason = None
                response_text = ""
                valid = False

            answer = str(row.get("answer", "")).strip()
            results.append(
                {
                    "data_id": row["data_id"],
                    "question": row["question"],
                    "pred": pred,
                    "answer": answer,
                    "correct": self._exact_match(pred, answer) if valid else False,
                    "valid": valid,
                    "image_valid": image_valid,
                    "finish_reason": finish_reason,
                    "response": response_text,
                    "original_category": row.get("original_category", ""),
                    "language": row.get("language", ""),
                },
            )
        return results

    def _compute_stats(self, data: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate accuracy by category and overall.

        Args:
            data: Predictions from :meth:`_predict`.

        Returns:
            Dictionary with ``overall`` and ``by_category`` keys.
        """
        return self._grouped_stats(
            data,
            group_fn=lambda item: item.get("original_category") or "unknown",
            group_label="category",
        )

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run the SimpleVQA benchmark.

        Returns:
            Dictionary with keys ``overall`` and ``by_category``.
        """
        data = self._predict()
        self._write_jsonl(data, "predictions.jsonl")

        stats = self._compute_stats(data)
        o = stats["overall"]
        logger.info(
            "SimpleVQA: {:.2f}% ({}/{})",
            o["accuracy"],
            o["correct"],
            o["total"],
        )
        return stats
