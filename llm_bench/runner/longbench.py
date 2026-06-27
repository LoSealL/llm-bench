# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""LongBench-v2 benchmark runner.

Reads prompt templates from ``scripts/LongBench/prompts`` without
modifying third-party code, then evaluates the model on the
LongBench-v2 multiple-choice dataset.
"""

from pathlib import Path
from typing import Any

from loguru import logger

from llm_bench.client import LLMClient
from llm_bench.runners import (
    ArgSpec,
    BaseRunner,
    PersistenceSpec,
    RunnerMetadata,
    _JsonlWriter,
)


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
        max_tokens: int = 1024,
        temperature: float = 0.0,
        *,
        force: bool = False,
    ) -> None:
        """Prepare the runner.

        Args:
            client: Initialized LLM client.
            output_dir: Base output directory; predictions are written
                to ``output_dir/longbench/``.
            limit: If set, evaluate only the first *N* samples.
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            force: If ``True``, re-run even when cached JSONL exists.
        """
        super().__init__(client, output_dir, "longbench", limit, force=force)
        self._max_tokens = max_tokens
        self._temperature = temperature

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
        text = BaseRunner._strip_thinking(response)
        return BaseRunner._extract_letter_answer(
            text,
            patterns=[
                r"The correct answer is \(([A-D])\)",
                r"The correct answer is ([A-D])",
                r"(?:答案是|Answer:|answer:)\s*\(?([A-D])\)?",
            ],
        )

    def _compare(self, pred: str | None, answer: str) -> bool:
        """Case-insensitive letter comparison.

        Args:
            pred: Predicted letter or ``None``.
            answer: Ground-truth letter.

        Returns:
            ``True`` if letters match after uppercasing.
        """
        if pred is None:
            return False
        return pred.strip().upper() == answer.strip().upper()

    def dry_run(self, **kwargs: Any) -> None:
        """Load dataset and print metadata without API calls."""
        dataset = self._load_hf_dataset(
            "THUDM/LongBench-v2",
            "train",
            "LongBench-v2",
        )
        self._inspect_dataset(
            dataset,
            label="LongBench-v2",
            fields=[
                "_id",
                "domain",
                "question",
                "choice_A",
                "choice_B",
                "choice_C",
                "choice_D",
                "answer",
            ],
        )

    def _predict(
        self,
        skip: int = 0,
        writer: _JsonlWriter | None = None,
    ) -> list[dict[str, Any]]:
        """Run inference on the full LongBench-v2 dataset.

        Args:
            skip: Number of samples to skip (already cached).
            writer: Optional streaming JSONL writer.

        Returns:
            List of result dictionaries with ``pred``, ``answer``,
            ``judge``, and metadata fields.
        """
        dataset = self._load_hf_dataset(
            "THUDM/LongBench-v2",
            "train",
            "LongBench-v2",
        )
        data_all: list[dict[str, Any]] = []
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

        if skip:
            data_all = data_all[skip:]
            logger.info("Skipping {} cached samples", skip)

        results: list[dict[str, Any]] = []
        for item in self._progress(data_all, desc="LongBench-v2"):
            prompt = self._build_prompt(item)
            prompt = self._client.truncate_prompt(prompt, 32000)
            response = self._chat(
                prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            pred = self._extract_answer(response.content) if response.valid else None
            record = {
                **item,
                "response": response.content,
                "pred": pred,
                "judge": self._compare(pred, item["answer"])
                if response.valid
                else False,
                "valid": response.valid,
                "finish_reason": response.finish_reason,
            }
            results.append(record)
            if writer is not None:
                writer.write(record)
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
        valid_data = [item for item in data if item["valid"]]
        overall = self._overall_stats(valid_data, correct_key="judge", decimals=1)

        counters: dict[str, dict[str, float]] = {
            "easy": {"correct": 0.0, "total": 0.0},
            "hard": {"correct": 0.0, "total": 0.0},
            "short": {"correct": 0.0, "total": 0.0},
            "medium": {"correct": 0.0, "total": 0.0},
            "long": {"correct": 0.0, "total": 0.0},
        }

        for item in valid_data:
            acc = 1.0 if item["judge"] and item["pred"] is not None else 0.0

            diff = item["difficulty"]
            if diff in counters:
                counters[diff]["correct"] += acc
                counters[diff]["total"] += 1.0

            length = item["length"]
            if length in counters:
                counters[length]["correct"] += acc
                counters[length]["total"] += 1.0

        return {
            "overall": overall["accuracy"],
            "easy": self._accuracy(
                counters["easy"]["correct"], counters["easy"]["total"], decimals=1
            ),
            "hard": self._accuracy(
                counters["hard"]["correct"], counters["hard"]["total"], decimals=1
            ),
            "short": self._accuracy(
                counters["short"]["correct"], counters["short"]["total"], decimals=1
            ),
            "medium": self._accuracy(
                counters["medium"]["correct"], counters["medium"]["total"], decimals=1
            ),
            "long": self._accuracy(
                counters["long"]["correct"], counters["long"]["total"], decimals=1
            ),
        }

    def run(self, **kwargs: Any) -> dict[str, float]:
        """Run the LongBench-v2 benchmark.

        Returns:
            Aggregated accuracy statistics.
        """
        filename = "predictions.jsonl"
        existing, writer = self._resume_jsonl(filename)
        try:
            new_data = self._predict(skip=len(existing), writer=writer)
        finally:
            writer.close()
        data = existing + new_data

        stats = self._compute_stats(data)
        logger.info("LongBench-v2 Overall: {:.1f}%", stats["overall"])
        return stats


# ---- Registry configuration -------------------------------------------------


class Metadata(RunnerMetadata):
    """Self-registration metadata for the LongBench-v2 runner."""

    name = "longbench"
    dataset = "longbench_v2"
    runner_cls = LongBenchRunner
    cli_args = [
        ArgSpec(
            name="longbench",
            flag="--longbench",
            help="Run the LongBench-v2 benchmark.",
            is_flag=True,
        ),
    ]
    persistence = PersistenceSpec(
        layout="single",
        categories=[],
        filename="predictions.jsonl",
        id_key="_id",
    )

    @classmethod
    def build_runner(cls, client, output_dir, args):
        """Construct a LongBench-v2 runner from parsed CLI args."""
        return LongBenchRunner(
            client,  # type: ignore[arg-type]
            output_dir,
            limit=args.limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            force=args.force,
        )

    @classmethod
    def to_scores(cls, result):
        """Wrap flat ``category -> accuracy_float`` dict."""
        scores: dict[str, dict[str, Any]] = {}
        for cat, acc in result.items():
            scores[cat] = {
                "accuracy": acc,
                "correct": None,
                "total": None,
            }
        return scores
