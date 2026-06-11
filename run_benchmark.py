# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Main CLI entry point for the LLM benchmark suite.

Opt-in orchestration for LVEval, LongBench-v2, MathArena, and BFCL v4
evaluations; generates a consolidated report for selected benchmarks.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from loguru import logger

from llm_bench.bfcl_constants import ALL_CATEGORIES, TEST_COLLECTION_MAPPING
from llm_bench.bfcl_runner import BFCLRunner
from llm_bench.client import LLMClient
from llm_bench.config import load_config
from llm_bench.longbench_runner import LongBenchRunner
from llm_bench.lveval_runner import LVEvalRunner
from llm_bench.matharena_runner import MathArenaRunner
from llm_bench.reporter import generate_html_report, generate_raw_csvs
from llm_bench.runners import BenchmarkResults


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with user-supplied or default values.
    """
    parser = argparse.ArgumentParser(
        description="Run LLM benchmarks via OpenAI-compatible API.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL (overrides OPENAI_BASE_URL from .env).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the endpoint (overrides OPENAI_API_KEY from .env).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model identifier (overrides OPENAI_MODEL from .env).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory for predictions and reports.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=32000,
        help="Maximum token length for prompt truncation.",
    )
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
    _BFCL_CATEGORIES = ALL_CATEGORIES + list(TEST_COLLECTION_MAPPING.keys())

    parser.add_argument(
        "--lveval-datasets",
        nargs="+",
        choices=_LVEVAL_DATASETS,
        default=None,
        metavar="DATASET",
        help="LVEval dataset base names to evaluate (default: all).",
    )
    parser.add_argument(
        "--lveval-lengths",
        nargs="+",
        choices=_LVEVAL_LENGTHS,
        default=["64k"],
        metavar="LENGTH",
        help="LVEval length levels (default: 64k).",
    )
    parser.add_argument(
        "--lveval",
        action="store_true",
        help="Run the LVEval benchmark.",
    )
    parser.add_argument(
        "--longbench",
        action="store_true",
        help="Run the LongBench-v2 benchmark.",
    )
    parser.add_argument(
        "--matharena",
        action="store_true",
        help="Run the MathArena benchmark.",
    )
    parser.add_argument(
        "--bfcl",
        action="store_true",
        help="Run the BFCL v4 benchmark.",
    )
    parser.add_argument(
        "--bfcl-categories",
        nargs="+",
        choices=_BFCL_CATEGORIES,
        default=None,
        metavar="CATEGORY",
        help="BFCL categories to evaluate (default: simple_python multiple).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit each dataset to the first N samples (for testing).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the benchmark pipeline."""
    args = parse_args()
    config = load_config()

    # Allow CLI overrides of config values
    if args.base_url is not None:
        config = replace(config, base_url=args.base_url)
    if args.api_key is not None:
        config = replace(config, api_key=args.api_key)
    if args.model is not None:
        config = replace(config, model=args.model)

    # Initialise shared client
    client = LLMClient(config)

    results = BenchmarkResults(model=config.model)

    logger.info("Running benchmarks for model {}", config.model)
    logger.debug(
        "Active benchmarks: lveval=%s longbench=%s matharena=%s bfcl=%s",
        args.lveval,
        args.longbench,
        args.matharena,
        args.bfcl,
    )

    if not any([args.lveval, args.longbench, args.matharena, args.bfcl]):
        logger.warning(
            "No benchmark selected. Use --lveval, --longbench, --matharena, "
            "and/or --bfcl to choose which benchmarks to run."
        )
        return

    if args.lveval:
        logger.info("Running LVEval")
        lveval = LVEvalRunner(
            client,
            args.output_dir,
            max_length=args.max_length,
            limit=args.limit,
        )
        results.lveval = lveval.run(
            selected=args.lveval_datasets,
            lengths=args.lveval_lengths,
        )

    if args.longbench:
        logger.info("Running LongBench-v2")
        longbench = LongBenchRunner(
            client,
            args.output_dir,
            limit=args.limit,
        )
        results.longbench = longbench.run()

    if args.matharena:
        logger.info("Running MathArena")
        matharena = MathArenaRunner(
            client,
            args.output_dir,
            limit=args.limit,
        )
        results.matharena = matharena.run()

    if args.bfcl:
        logger.info("Running BFCL v4")
        bfcl = BFCLRunner(
            client,
            args.output_dir,
            categories=args.bfcl_categories,
            limit=args.limit,
            max_tokens=args.max_length,
        )
        results.bfcl = bfcl.run()

    logger.info("Generating reports")
    out_dir = Path(args.output_dir)
    generate_raw_csvs(results, out_dir)
    generate_html_report(results, out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
