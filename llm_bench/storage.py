# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""SQLite storage for benchmark results.

Provides persistent storage for aggregated scores and per-sample
predictions, enabling cross-model comparison in HTML reports.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from llm_bench.runners import BenchmarkResults

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    benchmark TEXT NOT NULL,
    dataset TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    config TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    benchmark TEXT NOT NULL,
    category TEXT NOT NULL,
    accuracy REAL,
    correct INTEGER,
    total INTEGER
);

CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    benchmark TEXT NOT NULL,
    sample_id TEXT NOT NULL,
    data TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scores_model ON scores(model);
CREATE INDEX IF NOT EXISTS idx_scores_benchmark ON scores(benchmark);
CREATE INDEX IF NOT EXISTS idx_samples_model ON samples(model);
CREATE INDEX IF NOT EXISTS idx_samples_benchmark ON samples(benchmark);
CREATE INDEX IF NOT EXISTS idx_samples_sample_id ON samples(sample_id);
"""


class BenchmarkDB:
    """SQLite-backed storage for benchmark results.

    Args:
        db_path: Path to the SQLite database file. Parent directories
            are created automatically.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        logger.debug("Opened benchmark DB at {}", self._path)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "BenchmarkDB":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def save_results(
        self,
        model: str,
        benchmark: str,
        dataset: str,
        scores: dict[str, dict[str, Any]],
        config: dict[str, Any] | None = None,
    ) -> int:
        """Save aggregated benchmark scores.

        Args:
            model: Model identifier.
            benchmark: Benchmark name (e.g. ``"matharena"``).
            dataset: Dataset name (e.g. ``"aime_2026"``).
            scores: Mapping ``category -> {accuracy, correct, total}``.
            config: Optional run configuration to store.

        Returns:
            The ``run_id`` of the inserted run record.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        config_json = json.dumps(config) if config else None
        cursor = self._conn.execute(
            "INSERT INTO runs (model, benchmark, dataset, timestamp, config) "
            "VALUES (?, ?, ?, ?, ?)",
            (model, benchmark, dataset, timestamp, config_json),
        )
        run_id = cursor.lastrowid
        assert run_id is not None

        for category, stats in scores.items():
            self._conn.execute(
                "INSERT INTO scores (run_id, model, benchmark, category, "
                "accuracy, correct, total) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    model,
                    benchmark,
                    category,
                    stats.get("accuracy"),
                    stats.get("correct"),
                    stats.get("total"),
                ),
            )

        self._conn.commit()
        logger.debug(
            "Saved {} score categories for {}/{} (run_id={})",
            len(scores),
            model,
            benchmark,
            run_id,
        )
        return run_id

    def clear_model_benchmark(self, model: str, benchmark: str) -> None:
        """Delete all data for a specific model + benchmark pair.

        Removes matching rows from ``runs``, ``scores``, and
        ``samples``. Used when ``--force`` is specified.

        Args:
            model: Model identifier.
            benchmark: Benchmark name.
        """
        # Find run IDs to delete
        cursor = self._conn.execute(
            "SELECT id FROM runs WHERE model = ? AND benchmark = ?",
            (model, benchmark),
        )
        run_ids = [row[0] for row in cursor.fetchall()]

        if not run_ids:
            return

        placeholders = ",".join("?" * len(run_ids))
        self._conn.execute(
            f"DELETE FROM samples WHERE run_id IN ({placeholders})", run_ids
        )
        self._conn.execute(
            f"DELETE FROM scores WHERE run_id IN ({placeholders})", run_ids
        )
        self._conn.execute(f"DELETE FROM runs WHERE id IN ({placeholders})", run_ids)
        self._conn.commit()
        logger.info("Cleared {} runs for {}/{}", len(run_ids), model, benchmark)

    def save_samples(
        self,
        run_id: int,
        model: str,
        benchmark: str,
        samples: list[dict[str, Any]],
        id_key: str = "sample_id",
    ) -> None:
        """Save per-sample prediction data.

        Each sample is stored as a JSON blob keyed by ``id_key``.

        Args:
            run_id: The run this data belongs to.
            model: Model identifier.
            benchmark: Benchmark name.
            samples: List of sample dictionaries.
            id_key: Key in each sample dict that serves as the unique
                sample identifier.
        """
        for sample in samples:
            sample_id = str(sample.get(id_key, ""))
            data = json.dumps(sample, ensure_ascii=False)
            self._conn.execute(
                "INSERT INTO samples (run_id, model, benchmark, sample_id, data) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, model, benchmark, sample_id, data),
            )

        self._conn.commit()
        logger.debug(
            "Saved {} samples for {}/{} (run_id={})",
            len(samples),
            model,
            benchmark,
            run_id,
        )

    def query_all_scores(
        self,
        model: str | None = None,
        benchmark: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query aggregated scores with optional filtering.

        Args:
            model: If set, filter to this model only.
            benchmark: If set, filter to this benchmark only.

        Returns:
            List of score dictionaries.
        """
        sql = "SELECT model, benchmark, category, accuracy, correct, total FROM scores"
        conditions: list[str] = []
        params: list[Any] = []
        if model is not None:
            conditions.append("model = ?")
            params.append(model)
        if benchmark is not None:
            conditions.append("benchmark = ?")
            params.append(benchmark)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY model, benchmark, category"

        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()]
        self._conn.row_factory = None
        return rows

    def query_samples(
        self,
        model: str,
        benchmark: str,
    ) -> list[dict[str, Any]]:
        """Query per-sample data for a model + benchmark.

        Args:
            model: Model identifier.
            benchmark: Benchmark name.

        Returns:
            List of sample dictionaries (``data`` field is parsed JSON).
        """
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.execute(
            "SELECT sample_id, data FROM samples "
            "WHERE model = ? AND benchmark = ? ORDER BY id",
            (model, benchmark),
        )
        results: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            sample = json.loads(row["data"])
            sample["sample_id"] = row["sample_id"]
            results.append(sample)
        self._conn.row_factory = None
        return results

    def query_models(self) -> list[str]:
        """Return all distinct model names in the database.

        Returns:
            Sorted list of model identifiers.
        """
        cursor = self._conn.execute("SELECT DISTINCT model FROM runs ORDER BY model")
        return [row[0] for row in cursor.fetchall()]

    def query_benchmarks(self, model: str | None = None) -> list[str]:
        """Return all distinct benchmark names.

        Args:
            model: If set, only return benchmarks for this model.

        Returns:
            Sorted list of benchmark names.
        """
        if model is not None:
            cursor = self._conn.execute(
                "SELECT DISTINCT benchmark FROM runs WHERE model = ? "
                "ORDER BY benchmark",
                (model,),
            )
        else:
            cursor = self._conn.execute(
                "SELECT DISTINCT benchmark FROM runs ORDER BY benchmark"
            )
        return [row[0] for row in cursor.fetchall()]

    def save_benchmark_results(
        self,
        results: BenchmarkResults,
        config: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        """Save a complete BenchmarkResults object to the database.

        Converts the in-memory results structure to database rows
        for each benchmark that has data.

        Args:
            results: Aggregated results from all runners.
            config: Optional run configuration.

        Returns:
            Mapping ``benchmark -> run_id`` for each saved benchmark.
        """
        run_ids: dict[str, int] = {}

        # LVEval
        if results.lveval:
            scores: dict[str, dict[str, Any]] = {}
            all_vals: list[float] = []
            for ds, lengths in results.lveval.items():
                for length, score in lengths.items():
                    key = f"{ds}_{length}"
                    scores[key] = {
                        "accuracy": score,
                        "correct": None,
                        "total": None,
                    }
                    all_vals.append(score)
            scores["overall"] = {
                "accuracy": sum(all_vals) / len(all_vals) if all_vals else 0.0,
                "correct": None,
                "total": None,
            }
            run_ids["lveval"] = self.save_results(
                results.model, "lveval", "lveval", scores, config
            )

        # LongBench
        if results.longbench:
            lb_scores: dict[str, dict[str, Any]] = {}
            for cat, acc in results.longbench.items():
                lb_scores[cat] = {
                    "accuracy": acc,
                    "correct": None,
                    "total": None,
                }
            run_ids["longbench"] = self.save_results(
                results.model, "longbench", "longbench_v2", lb_scores, config
            )

        # MathArena
        if results.matharena:
            run_ids["matharena"] = self.save_results(
                results.model,
                "matharena",
                "aime_2026",
                {"overall": results.matharena},
                config,
            )

        # BFCL
        if results.bfcl:
            bfcl_scores: dict[str, dict[str, Any]] = {}
            for cat, stats in results.bfcl.items():
                bfcl_scores[cat] = {
                    "accuracy": stats.get("accuracy", 0.0) * 100,
                    "correct": stats.get("correct_count", 0),
                    "total": stats.get("total_count", 0),
                }
            run_ids["bfcl"] = self.save_results(
                results.model, "bfcl", "bfcl_v4", bfcl_scores, config
            )

        # SimpleVQA
        if results.simplevqa:
            svqa_scores: dict[str, dict[str, Any]] = {}
            overall = results.simplevqa.get("overall", {})
            svqa_scores["overall"] = {
                "accuracy": overall.get("accuracy", 0.0),
                "correct": overall.get("correct", 0),
                "total": overall.get("total", 0),
            }
            for cat, stats in results.simplevqa.get("by_category", {}).items():
                svqa_scores[cat] = {
                    "accuracy": stats.get("accuracy", 0.0),
                    "correct": stats.get("correct", 0),
                    "total": stats.get("total", 0),
                }
            run_ids["simplevqa"] = self.save_results(
                results.model, "simplevqa", "simplevqa", svqa_scores, config
            )

        # CompareBench
        if results.comparebench:
            cb_scores: dict[str, dict[str, Any]] = {}
            cb_overall = results.comparebench.get("overall", {})
            cb_scores["overall"] = {
                "accuracy": cb_overall.get("accuracy", 0.0),
                "correct": cb_overall.get("correct", 0),
                "total": cb_overall.get("total", 0),
            }
            for split, stats in results.comparebench.get("by_split", {}).items():
                cb_scores[split] = {
                    "accuracy": stats.get("accuracy", 0.0),
                    "correct": stats.get("correct", 0),
                    "total": stats.get("total", 0),
                }
            run_ids["comparebench"] = self.save_results(
                results.model, "comparebench", "comparebench", cb_scores, config
            )

        # MMMU
        if results.mmmu:
            mmmu_scores: dict[str, dict[str, Any]] = {}
            mmmu_overall = results.mmmu.get("overall", {})
            mmmu_scores["overall"] = {
                "accuracy": mmmu_overall.get("accuracy", 0.0),
                "correct": mmmu_overall.get("correct", 0),
                "total": mmmu_overall.get("total", 0),
            }
            for domain, stats in results.mmmu.get("by_domain", {}).items():
                mmmu_scores[domain] = {
                    "accuracy": stats.get("accuracy", 0.0),
                    "correct": stats.get("correct", 0),
                    "total": stats.get("total", 0),
                }
            for subject, stats in results.mmmu.get("by_subject", {}).items():
                mmmu_scores[subject] = {
                    "accuracy": stats.get("accuracy", 0.0),
                    "correct": stats.get("correct", 0),
                    "total": stats.get("total", 0),
                }
            for qtype, stats in results.mmmu.get("by_question_type", {}).items():
                mmmu_scores[qtype] = {
                    "accuracy": stats.get("accuracy", 0.0),
                    "correct": stats.get("correct", 0),
                    "total": stats.get("total", 0),
                }
            for diff, stats in results.mmmu.get("by_difficulty", {}).items():
                mmmu_scores[diff] = {
                    "accuracy": stats.get("accuracy", 0.0),
                    "correct": stats.get("correct", 0),
                    "total": stats.get("total", 0),
                }
            run_ids["mmmu"] = self.save_results(
                results.model, "mmmu", "mmmu", mmmu_scores, config
            )

        return run_ids
