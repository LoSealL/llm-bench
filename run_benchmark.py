# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Main CLI entry point for the LLM benchmark suite.

Orchestrates LVEval, LongBench-v2, and MathArena evaluations and
generates a consolidated report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument(
        "--lveval-datasets",
        nargs="+",
        default=None,
        help="LVEval dataset base names to evaluate (default: all).",
    )
    parser.add_argument(
        "--lveval-lengths",
        nargs="+",
        default=["64k"],
        help="LVEval length levels (default: 64k).",
    )
    parser.add_argument(
        "--skip-lveval",
        action="store_true",
        help="Skip the LVEval benchmark.",
    )
    parser.add_argument(
        "--skip-longbench",
        action="store_true",
        help="Skip the LongBench-v2 benchmark.",
    )
    parser.add_argument(
        "--skip-matharena",
        action="store_true",
        help="Skip the MathArena benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the benchmark pipeline."""
    args = parse_args()
    config = load_config()

    # Allow CLI override of the model name
    model_name = args.model or config.model

    # Initialise shared client with the (possibly overridden) model
    client = LLMClient(config)
    client._model = model_name

    results = BenchmarkResults(model=model_name)

    if not args.skip_lveval:
        print("=" * 60)
        print("Running LVEval")
        print("=" * 60)
        lveval = LVEvalRunner(
            client,
            args.output_dir,
            max_length=args.max_length,
        )
        results.lveval = lveval.run(
            selected=args.lveval_datasets,
            lengths=args.lveval_lengths,
        )

    if not args.skip_longbench:
        print("=" * 60)
        print("Running LongBench-v2")
        print("=" * 60)
        longbench = LongBenchRunner(client, args.output_dir)
        results.longbench = longbench.run()

    if not args.skip_matharena:
        print("=" * 60)
        print("Running MathArena")
        print("=" * 60)
        matharena = MathArenaRunner(client, args.output_dir)
        results.matharena = matharena.run()

    print("=" * 60)
    print("Generating reports")
    print("=" * 60)
    out_dir = Path(args.output_dir)
    generate_raw_csvs(results, out_dir)
    generate_html_report(results, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
