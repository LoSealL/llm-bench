# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""BFCL v4 constants and category definitions."""

from pathlib import Path


VERSION_PREFIX = "BFCL_v4"

NON_LIVE_CATEGORY = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
]

LIVE_CATEGORY = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
    "live_relevance",
]

SINGLE_TURN_CATEGORY = NON_LIVE_CATEGORY + LIVE_CATEGORY
ALL_CATEGORIES = SINGLE_TURN_CATEGORY

TEST_COLLECTION_MAPPING = {
    "all": ALL_CATEGORIES,
    "single_turn": SINGLE_TURN_CATEGORY,
    "live": LIVE_CATEGORY,
    "non_live": NON_LIVE_CATEGORY,
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "scripts" / "BFCL"
PROMPT_PATH = DATA_ROOT / "prompts"
POSSIBLE_ANSWER_PATH = DATA_ROOT / "ground_truth"
