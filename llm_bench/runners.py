# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Package-level shared types and helpers."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger
from tqdm import tqdm

from llm_bench.client import LLMClient

T = TypeVar("T")


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
    ) -> None:
        """Prepare the runner.

        Args:
            client: Initialized LLM client.
            output_dir: Base output directory; a subdirectory named
                *benchmark_name* is created automatically.
            benchmark_name: Directory name for this benchmark's outputs.
            limit: If set, cap the number of evaluated samples.
        """
        self._client = client
        self._limit = limit
        self._output_dir = Path(output_dir) / benchmark_name
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

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the benchmark and return aggregated results.

        Returns:
            Benchmark-specific result dictionary.
        """
        ...
