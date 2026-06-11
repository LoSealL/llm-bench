# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""LongBench-v2 benchmark runner.

Reads prompt templates from ``scripts/LongBench/prompts`` without
modifying third-party code, then evaluates the model on the
LongBench-v2 multiple-choice dataset.
"""

import re
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from loguru import logger
from tqdm import tqdm

from llm_bench.client import LLMClient
from llm_bench.runners import BaseRunner


class LongBenchRunner(BaseRunner):
    """Execute the LongBench-v2 benchmark suite.

    Attributes:
        _client: :class:`LLMClient` instance for API calls.
        _output_dir: Directory where ``.jsonl`` predictions are saved.
        _template: Zero-shot prompt template read from the third-party
            prompts directory.
    """

    def __init__(
        self,
        client: LLMClient,
        output_dir: str | Path,
        limit: int | None = None,
    ) -> None:
        """Prepare the runner.

        Args:
            client: Initialized LLM client.
            output_dir: Base output directory; predictions are written
                to ``output_dir/longbench/``.
            limit: If set, evaluate only the first *N* samples.
        """
        super().__init__(client, output_dir, "longbench", limit)

        repo_root = Path(__file__).resolve().parents[2]
        prompt_path = repo_root / "scripts" / "LongBench" / "prompts" / "0shot.txt"
        self._template = prompt_path.read_text(encoding="utf-8")

    def _build_prompt(self, item: dict[str, Any]) -> str:
        """Substitute placeholders in the zero-shot template.

        Args:
            item: A single dataset row.

        Returns:
            Fully rendered prompt string.
        """
        return (
            self._template.replace("$DOC$", item["context"].strip())
            .replace("$Q$", item["question"].strip())
            .replace("$C_A$", item["choice_A"].strip())
            .replace("$C_B$", item["choice_B"].strip())
            .replace("$C_C$", item["choice_C"].strip())
            .replace("$C_D$", item["choice_D"].strip())
        )

    @staticmethod
    def _extract_answer(response: str) -> str | None:
        """Extract the multiple-choice letter from model output.

        Tries explicit format patterns first, then falls back to the
        last occurrence of a standalone ``A``-``D`` letter.

        Args:
            response: Raw model response.

        Returns:
            Uppercase letter ``A``-``D``, or ``None``.
        """
        cleaned = response.replace("*", "")
        for pattern in (
            r"The correct answer is \(([A-D])\)",
            r"The correct answer is ([A-D])",
            r"(?:答案是|Answer:|answer:)\s*\(?([A-D])\)?",
        ):
            match = re.search(pattern, cleaned)
            if match:
                return match.group(1)
        letters = re.findall(r"\b([A-D])\b", cleaned)
        return letters[-1] if letters else None

    def _predict(self) -> list[dict[str, Any]]:
        """Run inference on the full LongBench-v2 dataset.

        Returns:
            List of result dictionaries with ``pred``, ``answer``,
            ``judge``, and metadata fields.
        """
        dataset = load_dataset("THUDM/LongBench-v2", split="train")
        data_all = []
        for item in dataset:
            row = dict(item)
            data_all.append(
                {
                    "_id": row["_id"],
                    "domain": row["domain"],
                    "sub_domain": row["sub_domain"],
                    "difficulty": row["difficulty"],
                    "length": row["length"],
                    "question": row["question"],
                    "choice_A": row["choice_A"],
                    "choice_B": row["choice_B"],
                    "choice_C": row["choice_C"],
                    "choice_D": row["choice_D"],
                    "answer": row["answer"],
                    "context": row["context"],
                }
            )
        data_all = self._apply_limit(data_all)
        logger.info("Loaded LongBench-v2 dataset with {} rows", len(data_all))

        results: list[dict[str, Any]] = []
        for item in tqdm(data_all, desc="LongBench-v2"):
            prompt = self._build_prompt(item)
            prompt = self._client.truncate_prompt(prompt, 32000)
            response = self._client.chat(
                prompt,
                max_tokens=128,
                temperature=0.1,
            )
            pred = self._extract_answer(response)
            results.append(
                {
                    **item,
                    "response": response,
                    "pred": pred,
                    "judge": pred == item["answer"],
                },
            )
        return results

    def _compute_stats(
        self,
        data: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Aggregate accuracy by difficulty and length.

        Args:
            data: Predictions from :meth:`_predict`.

        Returns:
            Dictionary with keys ``overall``, ``easy``, ``hard``,
            ``short``, ``medium``, ``long``.
        """
        logger.debug("Computing LongBench-v2 statistics for {} predictions", len(data))
        counters: dict[str, dict[str, float]] = {
            "easy": {"correct": 0.0, "total": 0.0},
            "hard": {"correct": 0.0, "total": 0.0},
            "short": {"correct": 0.0, "total": 0.0},
            "medium": {"correct": 0.0, "total": 0.0},
            "long": {"correct": 0.0, "total": 0.0},
        }
        total_correct = 0.0

        for item in data:
            acc = 1.0 if item["judge"] else 0.0
            if item["pred"] is None:
                acc = 0.0
            total_correct += acc

            diff = item["difficulty"]
            if diff in counters:
                counters[diff]["correct"] += acc
                counters[diff]["total"] += 1.0

            length = item["length"]
            if length in counters:
                counters[length]["correct"] += acc
                counters[length]["total"] += 1.0

        total = len(data)
        return {
            "overall": self._accuracy(total_correct, total, decimals=1),
            "easy": self._accuracy(counters["easy"]["correct"], counters["easy"]["total"], decimals=1),
            "hard": self._accuracy(counters["hard"]["correct"], counters["hard"]["total"], decimals=1),
            "short": self._accuracy(counters["short"]["correct"], counters["short"]["total"], decimals=1),
            "medium": self._accuracy(counters["medium"]["correct"], counters["medium"]["total"], decimals=1),
            "long": self._accuracy(counters["long"]["correct"], counters["long"]["total"], decimals=1),
        }

    def run(self, **kwargs: Any) -> dict[str, float]:
        """Run the LongBench-v2 benchmark.

        Returns:
            Aggregated accuracy statistics.
        """
        data = self._predict()
        self._write_jsonl(data, "predictions.jsonl")

        stats = self._compute_stats(data)
        logger.info("LongBench-v2 Overall: {:.1f}%", stats["overall"])
        return stats
