# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""LVEval benchmark runner.

Dynamically imports evaluation configuration and utilities from
``scripts/LVEval`` without modifying third-party code, then runs
inference via the OpenAI-compatible client and scores predictions
using the original metrics module.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

from loguru import logger
from tqdm import tqdm

from llm_bench.client import LLMClient
from llm_bench.reporter import ensure_dir


class _MockModule:
    """Minimal stand-in for ``torch`` so third-party imports succeed."""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return _MockModule()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return _MockModule()


class LVEvalRunner:
    """Execute the LVEval benchmark suite.

    Attributes:
        _client: :class:`LLMClient` instance for API calls.
        _max_length: Maximum token length for prompt truncation.
        _output_dir: Directory where ``.jsonl`` predictions are saved.
        _scripts_dir: Absolute path to ``scripts/LVEval``.
        _config: Loaded ``scripts.LVEval.config`` module.
        _utils: Loaded ``scripts.LVEval.utils`` module.
        _metrics: Loaded ``scripts.LVEval.metrics`` module.
    """

    def __init__(
        self,
        client: LLMClient,
        output_dir: str | Path,
        max_length: int = 32000,
        limit: int | None = None,
    ) -> None:
        """Prepare the runner.

        Args:
            client: Initialized LLM client.
            output_dir: Base output directory; predictions are written
                to ``output_dir/lveval/``.
            max_length: Token budget for prompt truncation.
            limit: If set, evaluate only the first *N* samples per
                dataset (useful for quick smoke tests).
        """
        self._client = client
        self._max_length = max_length
        self._limit = limit
        self._output_dir = Path(output_dir) / "lveval"
        ensure_dir(self._output_dir)

        repo_root = Path(__file__).resolve().parents[1]
        self._scripts_dir = repo_root / "scripts" / "LVEval"
        self._config, self._utils, self._metrics = self._load_third_party_modules()

    def _load_third_party_modules(self) -> tuple[Any, Any, Any]:
        """Load ``config``, ``utils``, and ``metrics`` from ``scripts/LVEval``.

        ``utils`` imports ``torch`` for local-model loading paths we do
        not use; we inject a lightweight mock so the import succeeds.

        Returns:
            A 3-tuple of loaded module objects.
        """
        # Mock heavy ML deps so scripts/LVEval/utils.py can import
        mocked: list[str] = []
        for mod in ("torch", "transformers"):
            if mod not in sys.modules:
                sys.modules[mod] = _MockModule()  # type: ignore[assignment]
                mocked.append(mod)
        sys.path.insert(0, str(self._scripts_dir))
        try:
            config = importlib.import_module("config")
            utils = importlib.import_module("utils")
            metrics = importlib.import_module("metrics")
        finally:
            sys.path.pop(0)
            for mod in mocked:
                sys.modules.pop(mod, None)
        logger.debug("Loaded third-party LVEval modules: config, utils, metrics")
        return config, utils, metrics

    def _get_datasets(
        self,
        selected: list[str] | None = None,
        lengths: list[str] | None = None,
    ) -> list[str]:
        """Build the list of dataset names to evaluate.

        Args:
            selected: Subset of dataset base names. ``None`` evaluates
                all datasets defined in the third-party config.
            lengths: Length levels to include. Defaults to ``["64k"]``.

        Returns:
            Fully qualified dataset names, e.g.
            ``["hotpotwikiqa_mixup_64k", ...]``.
        """
        if lengths is None:
            lengths = ["64k"]
        if selected is None:
            selected = list(self._config.DATASET_SELECTED)
        datasets: list[str] = []
        for name in selected:
            for length in lengths:
                datasets.append(f"{name}_{length}")
        return datasets

    def _predict_dataset(self, dataset_name: str) -> list[dict[str, Any]]:
        """Run inference on a single LVEval dataset.

        Args:
            dataset_name: Fully qualified dataset name with length
                suffix.

        Returns:
            List of prediction dictionaries compatible with the original
            evaluation script.
        """
        dataset_base = re.split(r"_.{1,3}k", dataset_name)[0]
        datas = self._utils.load_LVEval_dataset(
            dataset_name,
            data_path=f"data/lveval/{dataset_base}",
        )
        if self._limit is not None:
            datas = datas[: self._limit]
        logger.info("Predicting LVEval dataset {} with {} samples", dataset_name, len(datas))
        dataset_base = re.split(r"_.{1,3}k", dataset_name)[0]
        prompt_format = self._config.DATASET_PROMPT[dataset_base]
        max_gen = self._config.DATASET_MAXGEN[dataset_base]

        preds: list[dict[str, Any]] = []
        for json_obj in tqdm(datas, desc=dataset_name):
            prompt = prompt_format.format(**json_obj)
            prompt = self._client.truncate_prompt(
                prompt,
                self._max_length,
            )
            raw_pred = self._client.chat(
                prompt,
                max_tokens=max_gen,
                temperature=0.1,
            )
            pred = self._utils.post_process(raw_pred, self._client._model)
            preds.append(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "gold_ans": (
                        json_obj["answer_keywords"]
                        if "answer_keywords" in json_obj
                        else None
                    ),
                    "input": json_obj["input"],
                    "all_classes": (
                        json_obj["all_classes"] if "all_classes" in json_obj else None
                    ),
                    "length": json_obj["length"],
                },
            )
        return preds

    def _score_dataset(
        self,
        dataset_name: str,
        preds: list[dict[str, Any]],
    ) -> float:
        """Score predictions using the original LVEval metric.

        Args:
            dataset_name: Fully qualified dataset name.
            preds: Prediction dictionaries from
                :meth:`_predict_dataset`.

        Returns:
            Mean score scaled to 0-100.
        """
        dataset_base = re.split(r"_.{1,3}k", dataset_name)[0]
        metric_fn = self._config.DATASET_METRIC[dataset_base]
        logger.debug("Scoring LVEval dataset {} ({} predictions)", dataset_name, len(preds))
        total_score = 0.0
        total_sample = 0
        for item in preds:
            total_sample += 1
            score = 0.0
            for ground_truth in item["answers"]:
                score = max(
                    score,
                    metric_fn(
                        item["pred"],
                        ground_truth,
                        item.get("gold_ans"),
                    ),
                )
                break
            total_score += score
        return round(100 * total_score / total_sample, 2)

    def run(
        self,
        selected: list[str] | None = None,
        lengths: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Run the LVEval benchmark.

        Args:
            selected: Dataset base names to evaluate. ``None`` runs all.
            lengths: Length levels. ``None`` defaults to ``["64k"]``.

        Returns:
            Mapping ``dataset_base -> {length_level: score}``.
        """
        datasets = self._get_datasets(selected, lengths)
        results: dict[str, dict[str, float]] = {}

        for dataset_name in datasets:
            preds = self._predict_dataset(dataset_name)

            score = self._score_dataset(dataset_name, preds)
            dataset_base = re.split(r"_.{1,3}k", dataset_name)[0]
            length = dataset_name.split("_")[-1]
            if dataset_base not in results:
                results[dataset_base] = {}
            results[dataset_base][length] = score
            logger.info("{}: {:.2f}", dataset_name, score)

        return results
