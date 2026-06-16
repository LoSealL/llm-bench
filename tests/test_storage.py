# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Tests for the SQLite storage module."""

import json
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
        rows = conn.execute("SELECT * FROM scores ORDER BY category").fetchall()
        conn.close()

        assert len(rows) == 3
        categories = [r["category"] for r in rows]
        assert "easy" in categories
        assert "hard" in categories
        assert "overall" in categories


def test_save_samples():
    """Verify save_samples stores per-sample JSON data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with BenchmarkDB(db_path) as db:
            run_id = db.save_results(
                model="gpt-4",
                benchmark="matharena",
                dataset="aime_2026",
                scores={"overall": {"accuracy": 75.0, "correct": 15, "total": 20}},
            )
            db.save_samples(
                run_id=run_id,
                model="gpt-4",
                benchmark="matharena",
                samples=[
                    {"sample_id": "p1", "pred": "42", "answer": "42", "correct": True},
                    {"sample_id": "p2", "pred": "7", "answer": "8", "correct": False},
                ],
            )

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM samples ORDER BY id").fetchall()
        conn.close()

        assert len(rows) == 2
        assert rows[0]["sample_id"] == "p1"
        data = json.loads(rows[0]["data"])
        assert data["correct"] is True
        assert rows[1]["sample_id"] == "p2"


def test_clear_model_benchmark():
    """Verify clear_model_benchmark removes target data only."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with BenchmarkDB(db_path) as db:
            run_id = db.save_results(
                model="gpt-4",
                benchmark="matharena",
                dataset="aime_2026",
                scores={"overall": {"accuracy": 75.0, "correct": 15, "total": 20}},
            )
            db.save_samples(
                run_id=run_id,
                model="gpt-4",
                benchmark="matharena",
                samples=[{"sample_id": "p1", "pred": "42"}],
            )
            # Also save a different model's data
            db.save_results(
                model="deepseek",
                benchmark="matharena",
                dataset="aime_2026",
                scores={"overall": {"accuracy": 60.0, "correct": 12, "total": 20}},
            )

            db.clear_model_benchmark("gpt-4", "matharena")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        scores = conn.execute("SELECT * FROM scores").fetchall()
        samples = conn.execute("SELECT * FROM samples").fetchall()
        runs = conn.execute("SELECT * FROM runs").fetchall()
        conn.close()

        # gpt-4 data cleared
        assert len(scores) == 1
        assert scores[0]["model"] == "deepseek"
        assert len(samples) == 0
        assert len(runs) == 1
        assert runs[0]["model"] == "deepseek"


def test_clear_model_benchmark_only_clears_specified():
    """Clearing model A + benchmark X should not affect model A + benchmark Y."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with BenchmarkDB(db_path) as db:
            db.save_results(
                model="gpt-4",
                benchmark="matharena",
                dataset="aime_2026",
                scores={"overall": {"accuracy": 75.0, "correct": 15, "total": 20}},
            )
            db.save_results(
                model="gpt-4",
                benchmark="longbench",
                dataset="longbench_v2",
                scores={"overall": {"accuracy": 80.0, "correct": 160, "total": 200}},
            )

            db.clear_model_benchmark("gpt-4", "matharena")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM scores").fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0]["benchmark"] == "longbench"


def test_query_all_scores():
    """Verify query_all_scores returns all rows and supports filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with BenchmarkDB(db_path) as db:
            db.save_results(
                model="gpt-4",
                benchmark="matharena",
                dataset="aime_2026",
                scores={
                    "overall": {"accuracy": 75.0, "correct": 15, "total": 20},
                },
            )
            db.save_results(
                model="deepseek",
                benchmark="matharena",
                dataset="aime_2026",
                scores={
                    "overall": {"accuracy": 60.0, "correct": 12, "total": 20},
                },
            )
            db.save_results(
                model="gpt-4",
                benchmark="longbench",
                dataset="longbench_v2",
                scores={
                    "overall": {"accuracy": 80.0, "correct": 160, "total": 200},
                },
            )

            all_scores = db.query_all_scores()

        # Should have 3 rows total
        assert len(all_scores) == 3
        # Each row should have the expected keys
        for row in all_scores:
            assert "model" in row
            assert "benchmark" in row
            assert "category" in row
            assert "accuracy" in row


def test_query_samples_for_benchmark():
    """Verify query_samples returns parsed JSON data with sample_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with BenchmarkDB(db_path) as db:
            run_id = db.save_results(
                model="gpt-4",
                benchmark="matharena",
                dataset="aime_2026",
                scores={"overall": {"accuracy": 75.0, "correct": 15, "total": 20}},
            )
            db.save_samples(
                run_id=run_id,
                model="gpt-4",
                benchmark="matharena",
                samples=[
                    {
                        "sample_id": "p1",
                        "pred": "42",
                        "answer": "42",
                        "correct": True,
                    },
                    {
                        "sample_id": "p2",
                        "pred": "7",
                        "answer": "8",
                        "correct": False,
                    },
                ],
            )

            samples = db.query_samples("gpt-4", "matharena")

        assert len(samples) == 2
        assert samples[0]["sample_id"] == "p1"
        assert samples[1]["sample_id"] == "p2"


def test_query_models():
    """Verify query_models returns sorted distinct model names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        with BenchmarkDB(db_path) as db:
            db.save_results(
                model="gpt-4",
                benchmark="matharena",
                dataset="aime_2026",
                scores={"overall": {"accuracy": 75.0, "correct": 15, "total": 20}},
            )
            db.save_results(
                model="deepseek",
                benchmark="longbench",
                dataset="longbench_v2",
                scores={
                    "overall": {"accuracy": 80.0, "correct": 160, "total": 200},
                },
            )

            models = db.query_models()

        assert sorted(models) == ["deepseek", "gpt-4"]


def test_save_benchmark_results():
    """Test the high-level helper that converts BenchmarkResults to DB rows."""
    from llm_bench.runners import BenchmarkResults

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        results = BenchmarkResults(
            model="gpt-4",
            matharena={"accuracy": 75.0, "correct": 15, "total": 20},
            longbench={
                "overall": 80.0,
                "easy": 90.0,
                "hard": 70.0,
                "short": 85.0,
                "medium": 75.0,
                "long": 65.0,
            },
        )

        with BenchmarkDB(db_path) as db:
            db.save_benchmark_results(results)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        scores = conn.execute(
            "SELECT * FROM scores ORDER BY benchmark, category"
        ).fetchall()
        conn.close()

        # matharena: 1 score row (overall)
        # longbench: 6 score rows (overall, easy, hard, short, medium, long)
        assert len(scores) == 7
        ma_scores = [s for s in scores if s["benchmark"] == "matharena"]
        lb_scores = [s for s in scores if s["benchmark"] == "longbench"]
        assert len(ma_scores) == 1
        assert len(lb_scores) == 6
