# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""SQLite storage for benchmark results.

Provides persistent storage for aggregated scores and per-sample
predictions, enabling cross-model comparison in HTML reports.
"""

import sqlite3
from pathlib import Path

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
