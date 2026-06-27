# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""LVEval benchmark runner.

Dynamically imports evaluation configuration and utilities from
``scripts/LVEval`` without modifying third-party code, then runs
inference via the OpenAI-compatible client and scores predictions
using the original metrics module.
"""

import importlib.util
import re
import sys
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


class _MockModule:
    """Minimal stand-in for ``torch`` so third-party imports succeed."""

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Return a mock object for any attribute access."""
        return _MockModule()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Return a mock object for any call."""
        return _MockModule()


class LVEvalRunner(BaseRunner):
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
        max_tokens: int = 1024,
        temperature: float = 0.0,
        *,
        force: bool = False,
    ) -> None:
        """Prepare the runner.

        Args:
            client: Initialized LLM client.
            output_dir: Base output directory; predictions are written
                to ``output_dir/lveval/``.
            max_length: Token budget for prompt truncation.
            limit: If set, evaluate only the first *N* samples per
                dataset (useful for quick smoke tests).
            max_tokens: Maximum new tokens to generate.
            temperature: Sampling temperature.
            force: If ``True``, re-run even when cached JSONL exists.
        """
        super().__init__(client, output_dir, "lveval", limit, force=force)
        self._max_length = max_length
        self._max_tokens = max_tokens
        self._temperature = temperature

        repo_root = Path(__file__).resolve().parents[2]
        self._scripts_dir = repo_root / "scripts" / "LVEval"
        self._config, self._utils, self._metrics = self._load_third_party_modules()

    def _load_third_party_modules(self) -> tuple[Any, Any, Any]:
        """Load ``config``, ``utils``, and ``metrics`` from ``scripts/LVEval``.

        ``utils`` imports ``torch`` for local-model loading paths we do
        not use; we inject a lightweight mock so the import succeeds.

        Returns:
            A 3-tuple of loaded module objects.
        """
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

    def dry_run(
        self,
        *,
        selected: list[str] | None = None,
        lengths: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Load dataset and print metadata without API calls."""
        datasets = self._get_datasets(selected, lengths)
        for dataset_name in datasets:
            dataset_base = re.split(r"_.{1,3}k", dataset_name)[0]
            data = self._utils.load_LVEval_dataset(
                dataset_name,
                data_path=f"data/lveval/{dataset_base}",
            )
            data = self._apply_limit(data)
            logger.info(
                "LVEval {} ({}) — {} samples",
                dataset_name,
                dataset_base,
                len(data),
            )
            for item in data:
                logger.info("  Sample: {}", item.get("id", "unknown"))
                logger.info(
                    "    Input: {} chars",
                    len(item.get("input", "")),
                )
                logger.info(
                    "    Question: {}",
                    str(item.get("question", ""))[:200],
                )
                logger.info(
                    "    Answers: {}",
                    item.get("answers", []),
                )

    def _predict_dataset(
        self,
        dataset_name: str,
        skip: int = 0,
        writer: _JsonlWriter | None = None,
    ) -> list[dict[str, Any]]:
        """Run inference on a single LVEval dataset.

        Args:
            dataset_name: Fully qualified dataset name with length
                suffix.
            skip: Number of samples to skip (already cached).
            writer: Optional streaming JSONL writer.

        Returns:
            List of prediction dictionaries compatible with the original
            evaluation script.
        """
        dataset_base = re.split(r"_.{1,3}k", dataset_name)[0]
        datas = self._utils.load_LVEval_dataset(
            dataset_name,
            data_path=f"data/lveval/{dataset_base}",
        )
        datas = self._apply_limit(datas)
        if skip:
            datas = datas[skip:]
            logger.info("Skipping {} cached samples for {}", skip, dataset_name)
        logger.info(
            "Predicting LVEval dataset {} with {} samples", dataset_name, len(datas)
        )
        dataset_base = re.split(r"_.{1,3}k", dataset_name)[0]
        prompt_format = self._config.DATASET_PROMPT[dataset_base]

        preds: list[dict[str, Any]] = []
        for json_obj in self._progress(datas, desc=dataset_name):
            prompt = prompt_format.format(**json_obj)
            prompt = self._client.truncate_prompt(
                prompt,
                self._max_length,
            )
            response = self._chat(
                prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            pred = (
                self._utils.post_process(response.content, self._client._model)
                if response.valid
                else ""
            )
            record = {
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
                "valid": response.valid,
                "finish_reason": response.finish_reason,
            }
            preds.append(record)
            if writer is not None:
                writer.write(record)
        return preds

    def _score_dataset(
        self,
        dataset_name: str,
        preds: list[dict[str, Any]],
    ) -> float:
        """Score predictions using the original LVEval metric.

        Only valid predictions are counted toward the score.

        Args:
            dataset_name: Fully qualified dataset name.
            preds: Prediction dictionaries from
                :meth:`_predict_dataset`.

        Returns:
            Mean score scaled to 0-100.
        """
        dataset_base = re.split(r"_.{1,3}k", dataset_name)[0]
        metric_fn = self._config.DATASET_METRIC[dataset_base]
        logger.debug(
            "Scoring LVEval dataset {} ({} predictions)", dataset_name, len(preds)
        )
        total_score = 0.0
        total_sample = 0
        for item in preds:
            if not item.get("valid", True):
                continue
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
            total_score += score
        return self._accuracy(total_score, total_sample)

    def run(
        self,
        selected: list[str] | None = None,
        lengths: list[str] | None = None,
        **kwargs: Any,
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
            filename = f"{dataset_name}.jsonl"
            existing, writer = self._resume_jsonl(filename)
            try:
                new_preds = self._predict_dataset(
                    dataset_name, skip=len(existing), writer=writer
                )
            finally:
                writer.close()
            preds = existing + new_preds

            score = self._score_dataset(dataset_name, preds)
            dataset_base = re.split(r"_.{1,3}k", dataset_name)[0]
            length = dataset_name.split("_")[-1]
            if dataset_base not in results:
                results[dataset_base] = {}
            results[dataset_base][length] = score
            logger.info("{}: {:.2f}", dataset_name, score)

        return results


# ---- Registry configuration -------------------------------------------------

_LVEVAL_DATASETS = [
    "hotpotwikiqa_mixup",
    "loogle_SD_mixup",
    "loogle_CR_mixup",
    "loogle_MIR_mixup",
    "multifieldqa_en_mixup",
    "multifieldqa_zh_mixup",
    "factrecall_en",
    "factrecall_zh",
    "cmrc_mixup",
    "lic_mixup",
    "dureader_mixup",
]
_LVEVAL_LENGTHS = ["16k", "32k", "64k", "128k", "256k"]


class Metadata(RunnerMetadata):
    """Self-registration metadata for the LVEval runner."""

    name = "lveval"
    dataset = "lveval"
    runner_cls = LVEvalRunner
    cli_args = [
        ArgSpec(
            name="lveval",
            flag="--lveval",
            help="Run the LVEval benchmark.",
            is_flag=True,
        ),
        ArgSpec(
            name="lveval_datasets",
            flag="--lveval-datasets",
            help="LVEval dataset base names to evaluate (default: all).",
            nargs="+",
            choices=_LVEVAL_DATASETS,
            default=_LVEVAL_DATASETS,
        ),
        ArgSpec(
            name="lveval_lengths",
            flag="--lveval-lengths",
            help="LVEval length levels (default: 64k).",
            nargs="+",
            choices=_LVEVAL_LENGTHS,
            default=["64k"],
        ),
    ]
    persistence = PersistenceSpec(
        layout="multi",
        categories=_LVEVAL_DATASETS,
        filename="*.jsonl",
        id_key="sample_id",
        sample_id_factory=lambda stem, i, rec: f"{stem}_{i}",
    )

    @classmethod
    def build_runner(cls, client, output_dir, args):
        """Construct an LVEval runner from parsed CLI args."""
        return LVEvalRunner(
            client,  # type: ignore[arg-type]
            output_dir,
            max_length=args.max_length,
            limit=args.limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            force=args.force,
        )

    @classmethod
    def to_scores(cls, result):
        """Flatten nested ``dataset -> {length: score}`` plus overall."""
        scores: dict[str, dict[str, Any]] = {}
        all_vals: list[float] = []
        for ds, lengths in result.items():
            for length, score in lengths.items():
                key = f"{ds}_{length}"
                scores[key] = {
                    "accuracy": score,
                    "correct": None,
                    "total": None,
                }
                all_vals.append(score)
        scores["overall"] = {
            "accuracy": (
                sum(all_vals) / len(all_vals) if all_vals else 0.0
            ),
            "correct": None,
            "total": None,
        }
        return scores

    @classmethod
    def extract_run_kwargs(cls, args):
        """Extract LVEval run() kwargs from parsed CLI args."""
        return {
            "selected": args.lveval_datasets,
            "lengths": args.lveval_lengths,
        }
