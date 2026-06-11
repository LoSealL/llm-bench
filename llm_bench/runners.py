# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Package-level shared types and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkResults:
    """Aggregate results from all benchmark runners.

    Attributes:
        model: The evaluated model identifier.
        lveval: Mapping ``dataset_name -> {length_level: score}``.
        longbench: Mapping of category names to accuracy percentages.
        matharena: Mapping with keys ``accuracy``, ``correct``, ``total``.
        bfcl: Mapping ``category -> {accuracy, correct_count, total_count}``.
    """

    model: str = ""
    lveval: dict[str, dict[str, float]] = field(default_factory=dict)
    longbench: dict[str, float] = field(default_factory=dict)
    matharena: dict[str, Any] = field(default_factory=dict)
    bfcl: dict[str, Any] = field(default_factory=dict)
