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
