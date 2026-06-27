# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Main CLI entry point for the LLM benchmark suite.

Opt-in orchestration for LVEval, LongBench-v2, MathArena, BFCL v4,
SimpleVQA, CompareBench, MMMU, OCRBench v2, and Omni AI OCR
evaluations; generates a consolidated report for selected benchmarks.

All benchmark dispatch is driven by the registry in
:mod:`llm_bench.registry` — adding a benchmark requires editing only the
registry, not this file.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger

from llm_bench.client import LLMClient
from llm_bench.config import load_config
from llm_bench.registry import (
    BENCHMARKS,
    build_argparser,
    selected_benchmarks,
)
from llm_bench.reporter import generate_html_report
from llm_bench.runners import BenchmarkResults
from llm_bench.storage import BenchmarkDB


def _run_dry_run(args: argparse.Namespace) -> None:
    """Instantiate selected runners with a dummy client and call dry_run.

    Iterates the registry; no per-benchmark conditional branches.
    """
    for descriptor in selected_benchmarks(args):
        runner = descriptor.build_runner(None, args.output_dir, args)
        runner.dry_run(**descriptor.extract_run_kwargs(args))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments via the registry-generated parser.

    Returns:
        Parsed namespace with user-supplied or default values.
    """
    parser = build_argparser()
    args = parser.parse_args()
    logger.remove()
    if args.verbose >= 2:
        logger.add(sys.stderr, level="TRACE")
    elif args.verbose == 1:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")
    return args


def _save_samples_to_db(
    db: BenchmarkDB,
    model: str,
    args: argparse.Namespace,
) -> None:
    """Load per-sample JSONL data and save to the database.

    Iterates the registry, branching only on the descriptor's
    ``persistence.category`` data field — never on the benchmark identity.

    Args:
        db: Open database connection.
        model: Model identifier.
        args: Parsed CLI arguments (to know which benchmarks ran).
    """
    out_dir = Path(args.output_dir)
    model_dir = out_dir / model.replace("/", "_")

    def _get_run_id(benchmark: str) -> int | None:
        """Return latest run_id for model+benchmark, or None."""
        cursor = db._conn.execute(
            "SELECT id FROM runs WHERE model = ? AND benchmark = ? "
            "ORDER BY id DESC LIMIT 1",
            (model, benchmark),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        """Load records from a JSONL file, skipping bad lines."""
        records: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load {} for DB: {}", path, exc)
        return records

    for descriptor in selected_benchmarks(args):
        bench_dir = model_dir / descriptor.name
        if not bench_dir.exists():
            continue
        run_id = _get_run_id(descriptor.name)
        if run_id is None:
            continue

        pspec = descriptor.persistence
        if pspec.layout == "single":
            jsonl_path = bench_dir / pspec.filename
            if not jsonl_path.exists():
                continue
            records = _load_jsonl(jsonl_path)
            if records:
                db.save_samples(
                    run_id=run_id,
                    model=model,
                    benchmark=descriptor.name,
                    samples=records,
                    id_key=pspec.id_key,
                )
        elif pspec.layout == "multi":
            for jsonl_file in bench_dir.glob(pspec.filename):
                stem = jsonl_file.stem
                records = _load_jsonl(jsonl_file)
                if not records:
                    continue
                if pspec.sample_id_factory is not None:
                    for i, rec in enumerate(records):
                        if pspec.id_key not in rec:
                            rec[pspec.id_key] = pspec.sample_id_factory(
                                stem, i, rec
                            )
                db.save_samples(
                    run_id=run_id,
                    model=model,
                    benchmark=descriptor.name,
                    samples=records,
                    id_key=pspec.id_key,
                )


def main() -> None:
    """Execute the benchmark pipeline."""
    args = parse_args()

    selected = selected_benchmarks(args)
    if not selected:
        flag_names = ", ".join(
            d.cli_args[0].flag for d in BENCHMARKS
        )
        logger.warning(
            "No benchmark selected. Use {} to choose which "
            "benchmarks to run.",
            flag_names,
        )
        return

    # Dry-run mode: inspect datasets without loading config or API calls
    if args.dry_run:
        logger.info("Dry-run mode: inspecting datasets without API calls")
        _run_dry_run(args)
        return

    config = load_config()

    if args.base_url is not None:
        config = replace(config, base_url=args.base_url)
    if args.api_key is not None:
        config = replace(config, api_key=args.api_key)
    if args.model is not None:
        config = replace(config, model=args.model)
    if args.no_thinking:
        config = replace(config, enable_thinking=False)

    client = LLMClient(config)

    results = BenchmarkResults(model=config.model)

    logger.info("Running benchmarks for model {}", config.model)
    logger.debug(
        "Active benchmarks: {}",
        ", ".join(d.name for d in selected),
    )

    # Open database for historical storage
    out_dir = Path(args.output_dir)
    db_path = out_dir / "benchmarks.db"
    db = BenchmarkDB(db_path)

    # Build config dict for storage
    run_config = {
        "max_length": args.max_length,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "limit": args.limit,
    }

    # Clear model+benchmark data when --force is used
    if args.force:
        for descriptor in selected:
            db.clear_model_benchmark(config.model, descriptor.name)

    interrupted = False
    try:
        for descriptor in selected:
            logger.info("Running {}", descriptor.name)
            runner = descriptor.build_runner(
                client, args.output_dir, args
            )
            results.results[descriptor.name] = runner.run(
                **descriptor.extract_run_kwargs(args)
            )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Interrupted — saving completed results")

    # Save aggregated results to SQLite
    logger.info("Saving results to database")
    db.save_benchmark_results(results, config=run_config)

    # Save per-sample data from JSONL files to SQLite
    _save_samples_to_db(db, config.model, args)

    logger.info("Generating reports")
    generate_html_report(db, out_dir)
    db.close()
    if interrupted:
        logger.warning("Partial results saved — re-run to complete")
    else:
        logger.info("Done.")


if __name__ == "__main__":
    main()
