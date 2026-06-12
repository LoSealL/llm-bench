# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""MathArena AIME 2026 benchmark runner.

Evaluates mathematical reasoning by exact-match on the final integer
answer extracted from model output.
"""

import re
from pathlib import Path
from typing import Any

from loguru import logger

from llm_bench.client import LLMClient
from llm_bench.runners import BaseRunner


class MathArenaRunner(BaseRunner):
    """Execute the MathArena AIME 2026 benchmark.

    Attributes:
        _client: :class:`LLMClient` instance for API calls.
        _output_dir: Directory where results are saved.
    """

    _SYSTEM_PROMPT = (
        "Solve the following math problem. "
        "Provide only the final numerical answer as an integer.\n\n"
    )

    def __init__(
        self,
        client: LLMClient,
        output_dir: str | Path,
        limit: int | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        """Prepare the runner.

        Args:
            client: Initialized LLM client.
            output_dir: Base output directory; results are written to
                ``output_dir/matharena/``.
            limit: If set, evaluate only the first *N* samples.
            max_tokens: If set, override the default generation limit.
            temperature: If set, override the default sampling temperature.
        """
        super().__init__(client, output_dir, "matharena", limit)
        self._max_tokens = max_tokens
        self._temperature = temperature

    def _build_prompt(self, problem: str) -> str:
        """Wrap a problem statement with the system instruction.

        Args:
            problem: Raw LaTeX problem text.

        Returns:
            Complete prompt for the model.
        """
        return f"{self._SYSTEM_PROMPT}{problem}\n\nFinal Answer:"

    @staticmethod
    def _extract_number(response: str) -> str | None:
        """Extract the last integer from model output.

        Args:
            response: Raw model response.

        Returns:
            The last matched integer string, or ``None``.
        """
        text = BaseRunner._strip_thinking(response)
        numbers = re.findall(r"\b\d+\b", text)
        return numbers[-1] if numbers else None

    def _compare(self, pred: str | None, answer: str) -> bool:
        """Numeric string comparison.

        Args:
            pred: Predicted number string or ``None``.
            answer: Ground-truth number string.

        Returns:
            ``True`` if both match after stripping.
        """
        if pred is None:
            return False
        return pred.strip() == answer.strip()

    def _predict(self) -> list[dict[str, Any]]:
        """Run inference on the AIME 2026 dataset.

        Returns:
            List of dictionaries with ``problem_idx``, ``pred``,
            ``answer``, and ``correct`` fields.
        """
        dataset = self._load_hf_dataset(
            "MathArena/aime_2026",
            "train",
            "MathArena",
        )
        results: list[dict[str, Any]] = []

        for item in self._progress(dataset, desc="MathArena"):
            row = dict(item)
            prompt = self._build_prompt(row["problem"])
            response = self._chat(
                prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            pred = self._extract_number(response.content) if response.valid else None
            answer = str(row["answer"])
            results.append(
                {
                    "problem_idx": row["problem_idx"],
                    "pred": pred,
                    "answer": answer,
                    "correct": self._compare(pred, answer) if response.valid else False,
                    "valid": response.valid,
                    "finish_reason": response.finish_reason,
                    "response": response.content,
                },
            )
        return results

    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Run the MathArena benchmark.

        Returns:
            Dictionary with keys ``accuracy``, ``correct``, ``total``.
        """
        data = self._predict()
        self._write_jsonl(data, "predictions.jsonl")

        logger.debug("Computing MathArena accuracy for {} predictions", len(data))
        stats = self._overall_stats(data)

        logger.info(
            "MathArena: {:.2f}% ({}/{})",
            stats["accuracy"],
            stats["correct"],
            stats["total"],
        )
        return stats
