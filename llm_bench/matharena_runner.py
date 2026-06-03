# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""MathArena AIME 2026 benchmark runner.

Evaluates mathematical reasoning by exact-match on the final integer
answer extracted from model output.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]
from tqdm import tqdm

from llm_bench.client import LLMClient
from llm_bench.reporter import ensure_dir


class MathArenaRunner:
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
    ) -> None:
        """Prepare the runner.

        Args:
            client: Initialized LLM client.
            output_dir: Base output directory; results are written to
                ``output_dir/matharena/``.
            limit: If set, evaluate only the first *N* samples.
        """
        self._client = client
        self._limit = limit
        self._output_dir = Path(output_dir) / "matharena"
        ensure_dir(self._output_dir)

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
        numbers = re.findall(r"\b\d+\b", response)
        return numbers[-1] if numbers else None

    def _predict(self) -> list[dict[str, Any]]:
        """Run inference on the AIME 2026 dataset.

        Returns:
            List of dictionaries with ``problem_idx``, ``pred``,
            ``answer``, and ``correct`` fields.
        """
        dataset = load_dataset("MathArena/aime_2026", split="train")
        if self._limit is not None:
            dataset = dataset.select(range(min(self._limit, len(dataset))))
        results: list[dict[str, Any]] = []

        for item in tqdm(dataset, desc="MathArena"):
            prompt = self._build_prompt(item["problem"])
            response = self._client.chat(
                prompt,
                max_tokens=1024,
                temperature=0.1,
            )
            pred = self._extract_number(response)
            answer = str(item["answer"])
            results.append(
                {
                    "problem_idx": item["problem_idx"],
                    "pred": pred,
                    "answer": answer,
                    "correct": pred == answer,
                    "response": response,
                },
            )
        return results

    def run(self) -> dict[str, Any]:
        """Run the MathArena benchmark.

        Returns:
            Dictionary with keys ``accuracy``, ``correct``, ``total``.
        """
        out_file = self._output_dir / "results.jsonl"

        if out_file.exists():
            print("Loading cached MathArena predictions")
            with out_file.open("r", encoding="utf-8") as fh:
                data = [json.loads(line) for line in fh]
        else:
            data = self._predict()
            with out_file.open("w", encoding="utf-8") as fh:
                for item in data:
                    fh.write(
                        json.dumps(item, ensure_ascii=False) + "\n",
                    )

        correct = sum(1 for item in data if item["correct"])
        total = len(data)
        accuracy = round(100 * correct / total, 2) if total else 0.0

        print(f"MathArena: {accuracy:.2f}% ({correct}/{total})")
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
