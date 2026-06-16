# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""SQLite storage for benchmark results.

Provides persistent storage for aggregated scores and per-sample
predictions, enabling cross-model comparison in HTML reports.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

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
