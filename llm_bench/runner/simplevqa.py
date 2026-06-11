# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""SimpleVQA benchmark runner.

Evaluates vision-language understanding on m-a-p/SimpleVQA using
OpenAI-compatible multimodal chat APIs.
"""

from __future__ import annotations

import base64
import io
import re
import string
import unicodedata
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from loguru import logger
from PIL import Image  # type: ignore[import-untyped]

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

    def _prepare_image(self, b64_str: str) -> str:
        """Decode base64, optionally resize, and return a JPEG data URI.

        Some vision APIs do not support WebP; this helper always
        transcodes to JPEG.  If ``self._image_size`` is set the image
        is resized before encoding.

        Args:
            b64_str: Raw base64 string (without data URI prefix).

        Returns:
            A ``data:image/jpeg;base64,...`` URI.
        """
        if b64_str.startswith("data:"):
            return b64_str

        raw = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(raw))

        if self._image_size is not None:
            resample = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]
            img = img.resize(self._image_size, resample)
            logger.debug("Resized image to {}x{}", *self._image_size)

        buf = io.BytesIO()
        rgb_img = img.convert("RGB")
        rgb_img.save(buf, format="JPEG")
        raw = buf.getvalue()

        b64_out = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{b64_out}"

    def _build_messages(self, image_b64: str, question: str) -> list[dict[str, Any]]:
        """Build multimodal messages for a VQA sample.

        Args:
            image_b64: Raw base64 string (without data URI prefix).
            question: Question text.

        Returns:
            OpenAI-compatible messages list with image content.
        """
        data_uri = self._prepare_image(image_b64)
        return [
            {
                "role": "system",
                "content": self._SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {"type": "text", "text": question},
                ],
            },
        ]

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
        dataset = load_dataset("m-a-p/SimpleVQA", split="test")
        data_all = self._apply_limit(list(dataset))
        logger.info("Loaded SimpleVQA with {} rows", len(data_all))

        results: list[dict[str, Any]] = []
        for item in self._progress(data_all, desc="SimpleVQA"):
            row = dict(item)
            messages = self._build_messages(row["image"], row["question"])
            response = self._client.chat(
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            pred = self._extract_answer(response)
            answer = str(row.get("answer", "")).strip()
            results.append(
                {
                    "data_id": row["data_id"],
                    "question": row["question"],
                    "pred": pred,
                    "answer": answer,
                    "correct": self._exact_match(pred, answer),
                    "response": response,
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
        overall_correct = sum(1 for item in data if item["correct"])
        overall_total = len(data)

        by_category: dict[str, dict[str, Any]] = {}
        for item in data:
            cat = item["original_category"] or "unknown"
            if cat not in by_category:
                by_category[cat] = {"correct": 0, "total": 0}
            by_category[cat]["total"] += 1
            if item["correct"]:
                by_category[cat]["correct"] += 1

        return {
            "overall": {
                "accuracy": self._accuracy(overall_correct, overall_total),
                "correct": overall_correct,
                "total": overall_total,
            },
            "by_category": {
                cat: {
                    "accuracy": self._accuracy(stats["correct"], stats["total"]),
                    "correct": stats["correct"],
                    "total": stats["total"],
                }
                for cat, stats in by_category.items()
            },
        }

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
