# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Tests for the SQLite storage module."""

import sqlite3
import tempfile
from pathlib import Path

from llm_bench.storage import BenchmarkDB


def test_creates_tables_on_init():
    """Verify that all expected tables are created on init."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        db = BenchmarkDB(db_path)
        db.close()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "runs" in tables
        assert "scores" in tables
        assert "samples" in tables


def test_save_results_inserts_scores():
    """Verify save_results inserts a run and its score rows."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with BenchmarkDB(db_path) as db:
            db.save_results(
                model="gpt-4",
                benchmark="matharena",
                dataset="aime_2026",
                scores={"overall": {"accuracy": 75.0, "correct": 15, "total": 20}},
            )

        # Reopen and verify
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM scores").fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0]["model"] == "gpt-4"
        assert rows[0]["benchmark"] == "matharena"
        assert rows[0]["category"] == "overall"
        assert rows[0]["accuracy"] == 75.0
        assert rows[0]["correct"] == 15
        assert rows[0]["total"] == 20


def test_save_results_multiple_categories():
    """Verify save_results handles multiple score categories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with BenchmarkDB(db_path) as db:
            db.save_results(
                model="deepseek",
                benchmark="longbench",
                dataset="longbench_v2",
                scores={
                    "overall": {"accuracy": 60.0, "correct": 120, "total": 200},
                    "easy": {"accuracy": 80.0, "correct": 80, "total": 100},
                    "hard": {"accuracy": 40.0, "correct": 40, "total": 100},
                },
            )

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM scores ORDER BY category"
        ).fetchall()
        conn.close()

        assert len(rows) == 3
        categories = [r["category"] for r in rows]
        assert "easy" in categories
        assert "hard" in categories
        assert "overall" in categories
