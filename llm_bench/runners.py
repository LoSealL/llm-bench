# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Package-level shared types and helpers."""

import base64
import io
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

from datasets import load_dataset  # type: ignore[import-untyped]
from loguru import logger
from PIL import Image  # type: ignore[import-untyped]
from tqdm import tqdm

from llm_bench.client import ChatResponse, LLMClient

T = TypeVar("T")


class _JsonlWriter:
    """Streaming JSONL writer with crash-safe flushing.

    Opens the file once, writes one record per call, and flushes
    after each write so partial results survive a crash.
    """

    def __init__(self, path: Path, *, truncate: bool = True) -> None:
        """Open the file for writing.

        Args:
            path: Output file path.
            truncate: If ``True``, overwrite existing content. If
                ``False``, append to the file.
        """
        self._fh = path.open("w" if truncate else "a", encoding="utf-8")
        self.count = 0

    def write(self, record: dict[str, Any]) -> None:
        """Write a single JSON record and flush.

        Args:
            record: Dictionary to serialise.
        """
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        """Close the underlying file handle."""
        self._fh.close()

    def __enter__(self) -> "_JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class BenchmarkResults:
    """Aggregate results from all benchmark runners.

    Attributes:
        model: The evaluated model identifier.
        lveval: Mapping ``dataset_name -> {length_level: score}``.
        longbench: Mapping of category names to accuracy percentages.
        matharena: Mapping with keys ``accuracy``, ``correct``, ``total``.
        bfcl: Mapping ``category -> {accuracy, correct_count, total_count}``.
    """

    model: str = ""
    lveval: dict[str, dict[str, float]] = field(default_factory=dict)
    longbench: dict[str, float] = field(default_factory=dict)
    matharena: dict[str, Any] = field(default_factory=dict)
    bfcl: dict[str, Any] = field(default_factory=dict)
    simplevqa: dict[str, Any] = field(default_factory=dict)
    comparebench: dict[str, Any] = field(default_factory=dict)


class BaseRunner(ABC):
    """Abstract base class for benchmark runners.

    Encapsulates common initialization, directory creation,
    sample limiting, result persistence, and accuracy helpers.
    """

    def __init__(
        self,
        client: LLMClient,
        output_dir: str | Path,
        benchmark_name: str,
        limit: int | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Prepare the runner.

        Args:
            client: Initialized LLM client.
            output_dir: Base output directory; a subdirectory named
                *benchmark_name* is created automatically.
            benchmark_name: Directory name for this benchmark's outputs.
            limit: If set, cap the number of evaluated samples.
            force: If ``True``, re-run even when cached JSONL exists.
        """
        self._client = client
        self._limit = limit
        self._force = force
        model_name = client._model.replace("/", "_")
        self._output_dir = Path(output_dir) / model_name / benchmark_name
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _apply_limit(self, data: list[T]) -> list[T]:
        """Return at most ``self._limit`` items from *data*.

        Args:
            data: Full list of samples.

        Returns:
            Possibly truncated list.
        """
        if self._limit is not None:
            return data[: self._limit]
        return data

    def _write_jsonl(
        self,
        records: list[dict[str, Any]],
        filename: str,
    ) -> Path:
        """Persist predictions as newline-delimited JSON.

        Args:
            records: List of prediction dictionaries.
            filename: Output file name (e.g. ``"predictions.jsonl"``).

        Returns:
            Absolute path to the written file.
        """
        path = self._output_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Saved {} records to {}", len(records), path)
        return path

    def _open_jsonl(
        self,
        filename: str,
        *,
        truncate: bool = True,
    ) -> _JsonlWriter:
        """Open a JSONL file for streaming writes.

        Args:
            filename: Output file name (e.g. ``"predictions.jsonl"``).
            truncate: If ``True``, overwrite existing content. If
                ``False``, append to the file.

        Returns:
            A :class:`_JsonlWriter` context manager.
        """
        return _JsonlWriter(self._output_dir / filename, truncate=truncate)

    def _load_existing_jsonl(
        self,
        filename: str,
    ) -> list[dict[str, Any]] | None:
        """Load an existing JSONL file if it is valid.

        Returns ``None`` when the file does not exist, is empty, or
        contains malformed JSON (e.g. from a crashed run).

        Args:
            filename: Output file name.

        Returns:
            List of records, or ``None``.
        """
        path = self._output_dir / filename
        if not path.exists():
            return None
        records: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load {} — {} — will re-run", path, exc)
            return None
        if not records:
            return None
        logger.info("Loaded {} existing records from {}", len(records), path)
        return records

    def _resume_jsonl(
        self,
        filename: str,
    ) -> tuple[list[dict[str, Any]], _JsonlWriter]:
        """Load existing records and open file for appending.

        When ``--force`` is set, existing data is discarded and the
        file is truncated. Otherwise, existing records are loaded and
        the file is opened in append mode so new records continue
        where the previous run left off.

        Args:
            filename: Output file name.

        Returns:
            A 2-tuple of ``(existing_records, writer)``. The writer
            is already entered as a context manager — the caller
            **must** close it (or use a ``with`` block on the
            returned writer).
        """
        if self._force:
            return [], self._open_jsonl(filename, truncate=True)

        existing = self._load_existing_jsonl(filename)
        if existing is not None:
            logger.info(
                "Resuming from {} existing records in {}",
                len(existing),
                filename,
            )
            return existing, self._open_jsonl(filename, truncate=False)

        return [], self._open_jsonl(filename, truncate=True)

    @staticmethod
    def _accuracy(
        correct: float,
        total: float,
        decimals: int = 2,
    ) -> float:
        """Compute a safe percentage.

        Args:
            correct: Number of correct samples.
            total: Total number of samples.
            decimals: Rounding precision.

        Returns:
            Percentage, or ``0.0`` when *total* is zero.
        """
        if total == 0:
            return 0.0
        return round(100 * correct / total, decimals)

    @staticmethod
    def _progress(
        iterable,
        desc: str | None = None,
        **kwargs: Any,
    ):
        """Wrap an iterable with ``tqdm``.

        Args:
            iterable: Collection to iterate over.
            desc: Progress bar description.
            **kwargs: Forwarded to ``tqdm``.

        Returns:
            ``tqdm`` iterator.
        """
        return tqdm(iterable, desc=desc, **kwargs)

    def _chat(
        self,
        prompt: str | None = None,
        *,
        messages: Any | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        tools: Any | None = None,
    ) -> ChatResponse:
        """Send a chat request through the wrapped client.

        Logs a warning when the model returns an invalid or empty
        response so each runner does not need to repeat the check.

        Args:
            prompt: Raw user message content. Used when ``messages`` is
                not provided.
            messages: Full message list to send directly. Takes precedence
                over ``prompt``.
            max_tokens: Maximum number of *new* tokens to generate.
            temperature: Sampling temperature.
            tools: Optional OpenAI-style tool definitions enabling native
                function calling via ``/v1/chat/completions``.

        Returns:
            A :class:`ChatResponse` with content, finish reason, and
            validity flag.
        """
        response = self._client.chat_with_meta(
            prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
        )
        if not response:
            logger.warning(
                "Invalid response from model (finish_reason={})",
                response.finish_reason,
            )
        return response

    def _validate_image(self, image: str | Image.Image) -> bool:
        """Check whether an image can be decoded successfully.

        Args:
            image: Base64-encoded string (with or without a data URI
                prefix) or a ``PIL.Image.Image`` instance.

        Returns:
            ``True`` if the image is decodable, otherwise ``False``.
        """
        try:
            if isinstance(image, str):
                if image.startswith("data:"):
                    b64_part = image.split(",", 1)[1]
                    raw = base64.b64decode(b64_part)
                else:
                    raw = base64.b64decode(image)
                Image.open(io.BytesIO(raw)).verify()
                return True
            if isinstance(image, Image.Image):
                image.verify()
                return True
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("Invalid image: {}", exc)
            return False

    def _load_hf_dataset(
        self,
        name: str,
        split: str,
        desc: str,
    ) -> list[dict[str, Any]]:
        """Load a HuggingFace dataset, apply the limit, and log.

        Args:
            name: HuggingFace dataset name.
            split: Dataset split to load.
            desc: Human-readable name used in log messages.

        Returns:
            List of dataset rows, capped at ``self._limit`` when set.
        """
        dataset = load_dataset(name, split=split)
        if self._limit is not None and hasattr(dataset, "select"):
            dataset = dataset.select(range(min(self._limit, len(dataset))))
        data = cast(list[dict[str, Any]], list(dataset))
        if self._limit is not None and len(data) > self._limit:
            data = data[: self._limit]
        logger.info("Loaded {} ({}) with {} rows", desc, name, len(data))
        return data

    @staticmethod
    def _extract_letter_answer(
        response: str,
        patterns: list[str] | None = None,
        fallback: bool = True,
        flags: int = 0,
    ) -> str | None:
        """Extract a single-letter answer (A-D) from model output.

        Args:
            response: Raw model response.
            patterns: Ordered list of regex patterns. The first capturing
                group should contain the letter. Defaults to common
                multiple-choice patterns.
            fallback: If ``True``, scan for a standalone ``A``-``D`` letter
                when none of the explicit patterns match.
            flags: Regex flags forwarded to ``re.search`` and ``re.findall``.

        Returns:
            Uppercase letter ``A``-``D``, or ``None``.
        """
        if not response:
            return None

        cleaned = response.replace("*", "")
        if patterns is None:
            patterns = [
                r"The correct answer is \(([A-D])\)",
                r"The correct answer is ([A-D])",
                r"(?:答案是|Answer:|answer:)\s*\(?([A-D])\)?",
            ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags)
            if match:
                return match.group(1).upper()

        if fallback:
            letters = re.findall(r"\b([A-D])\b", cleaned, flags)
            if letters:
                return letters[-1].upper()
        return None

    _THINKING_TAGS: tuple[str, ...] = (
        "think",
        "reasoning",
        "analysis",
        "thought",
        "inner_monologue",
    )

    @staticmethod
    def _strip_thinking(response: str) -> str:
        """Remove chain-of-thought and code-fence blocks from output.

        Handles XML-style thinking tags commonly used by open-weight
        models (e.g. ``<think>``, ``<reasoning>``, ``<analysis>``).
        Also strips fenced code blocks.

        Three cases are handled for each tag:

        1. Paixed: ``<tag>...</tag>`` — both removed.
        2. Missing close: ``<tag>...`` — tag and everything after removed.
        3. Missing open: ``...</tag>`` — everything before the closing
           tag is also removed (the closing tag itself is removed by
           the paired pattern in a subsequent pass or by this rule).

        Args:
            response: Raw model response.

        Returns:
            Response with thinking blocks removed.
        """
        text = response
        for tag in BaseRunner._THINKING_TAGS:
            text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL)
            text = re.sub(rf"<{tag}>.*$", "", text, flags=re.DOTALL)
            text = re.sub(rf"^.*?</{tag}>", "", text, flags=re.DOTALL)
        text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
        return text.strip()

    def _compare(self, pred: str, answer: str) -> bool:
        """Default exact-match comparison.

        Subclasses may override this to add custom normalization
        (e.g. case-folding, punctuation removal).

        Args:
            pred: Predicted answer string.
            answer: Ground-truth answer string.

        Returns:
            ``True`` if *pred* matches *answer* after default
            normalization.
        """
        return pred.strip() == answer.strip()

    def _prepare_image_data_uri(
        self,
        image: str | Image.Image,
        image_size: tuple[int, int] | None = None,
    ) -> str:
        """Convert an image to a JPEG data URI.

        Accepts either a base64-encoded string (with or without a data URI
        prefix) or a ``PIL.Image.Image``. The image is optionally resized
        and always transcoded to JPEG for API compatibility.

        Args:
            image: Base64 string or PIL Image.
            image_size: Optional ``(width, height)`` resize target.

        Returns:
            A ``data:image/jpeg;base64,...`` URI.
        """
        if isinstance(image, str):
            if image.startswith("data:"):
                return image
            raw = base64.b64decode(image)
            img = Image.open(io.BytesIO(raw))
        else:
            img = image

        if image_size is not None:
            resample = getattr(Image, "Resampling", Image).LANCZOS  # type: ignore[attr-defined]
            img = img.resize(image_size, resample)
            logger.debug("Resized image to {}x{}", *image_size)

        buf = io.BytesIO()
        rgb_img = img.convert("RGB")
        rgb_img.save(buf, format="JPEG")
        raw = buf.getvalue()
        b64_out = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{b64_out}"

    def _build_vision_messages(
        self,
        data_uri: str,
        question: str,
        system_prompt: str,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-compatible vision messages.

        Args:
            data_uri: Image data URI.
            question: Question text.
            system_prompt: System prompt content.

        Returns:
            Messages list with a system message and a multimodal user
            message.
        """
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": question},
                ],
            },
        ]

    def _overall_stats(
        self,
        data: list[dict[str, Any]],
        correct_key: str = "correct",
        valid_key: str = "valid",
        decimals: int = 2,
    ) -> dict[str, Any]:
        """Compute overall accuracy statistics over valid records.

        Args:
            data: Prediction records.
            correct_key: Key holding the boolean correctness value.
            valid_key: Key holding the boolean validity value. Only valid
                records are counted in the denominator.
            decimals: Rounding precision for the accuracy percentage.

        Returns:
            Dictionary with ``accuracy``, ``correct``, and ``total``.
        """
        valid_items = [item for item in data if item.get(valid_key, True)]
        correct = sum(1 for item in valid_items if item.get(correct_key))
        total = len(valid_items)
        return {
            "accuracy": self._accuracy(correct, total, decimals=decimals),
            "correct": correct,
            "total": total,
        }

    def _grouped_stats(
        self,
        data: list[dict[str, Any]],
        group_fn: Callable[[dict[str, Any]], str],
        correct_key: str = "correct",
        valid_key: str = "valid",
        group_label: str = "group",
        decimals: int = 2,
    ) -> dict[str, Any]:
        """Compute overall and per-group accuracy over valid records.

        Args:
            data: Prediction records.
            group_fn: Callable that returns a group name for each record.
            correct_key: Key holding the boolean correctness value.
            valid_key: Key holding the boolean validity value. Only valid
                records are counted.
            group_label: Label used for the grouped result key.

        Returns:
            Dictionary with ``overall`` and ``by_<group_label>`` keys.
        """
        valid_items = [item for item in data if item.get(valid_key, True)]
        overall = self._overall_stats(
            valid_items, correct_key, valid_key, decimals=decimals
        )
        by_group: dict[str, dict[str, int]] = {}
        for item in valid_items:
            group = group_fn(item)
            if group not in by_group:
                by_group[group] = {"correct": 0, "total": 0}
            by_group[group]["total"] += 1
            if item.get(correct_key):
                by_group[group]["correct"] += 1

        return {
            "overall": overall,
            f"by_{group_label}": {
                group: {
                    "accuracy": self._accuracy(
                        stats["correct"], stats["total"], decimals=decimals
                    ),
                    "correct": stats["correct"],
                    "total": stats["total"],
                }
                for group, stats in by_group.items()
            },
        }

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the benchmark and return aggregated results.

        Returns:
            Benchmark-specific result dictionary.
        """
        ...
