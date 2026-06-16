# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Main CLI entry point for the LLM benchmark suite.

Opt-in orchestration for LVEval, LongBench-v2, MathArena, and BFCL v4
evaluations; generates a consolidated report for selected benchmarks.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger

from llm_bench.bfcl_constants import ALL_CATEGORIES, TEST_COLLECTION_MAPPING
from llm_bench.client import LLMClient
from llm_bench.config import load_config
from llm_bench.reporter import generate_html_report, generate_raw_csvs
from llm_bench.runner import (
    BFCLRunner,
    CompareBenchRunner,
    LVEvalRunner,
    LongBenchRunner,
    MathArenaRunner,
    SimpleVQARunner,
)
from llm_bench.runners import BenchmarkResults
from llm_bench.storage import BenchmarkDB


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
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Override max output tokens for all runners (default: 1024).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Override sampling temperature for all runners.",
    )
    lveval_datasets = [
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
    lveval_lengths = ["16k", "32k", "64k", "128k", "256k"]
    bfcl_categories = ALL_CATEGORIES + list(TEST_COLLECTION_MAPPING.keys())
    comparebench_splits = [
        "CompareTallyBench",
        "CompareGeometryBench",
        "CompareSpatialBench",
        "CompareHistBench",
        "CompareCelebrityBench",
        "CompareLandmarkBench",
    ]

    parser.add_argument(
        "--lveval-datasets",
        nargs="+",
        choices=lveval_datasets,
        default=None,
        metavar="DATASET",
        help="LVEval dataset base names to evaluate (default: all).",
    )
    parser.add_argument(
        "--lveval-lengths",
        nargs="+",
        choices=lveval_lengths,
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
        choices=bfcl_categories,
        default=None,
        metavar="CATEGORY",
        help="BFCL categories to evaluate (default: simple_python multiple).",
    )
    parser.add_argument(
        "--comparebench",
        action="store_true",
        help="Run the CompareBench benchmark.",
    )
    parser.add_argument(
        "--comparebench-splits",
        nargs="+",
        choices=comparebench_splits,
        default=None,
        metavar="SPLIT",
        help="CompareBench splits to evaluate (default: all).",
    )
    parser.add_argument(
        "--simplevqa",
        action="store_true",
        help="Run the SimpleVQA benchmark.",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=None,
        help="Resize VQA images to this width before sending.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=None,
        help="Resize VQA images to this height before sending.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit each dataset to the first N samples (for testing).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for DEBUG, -vv for TRACE).",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        default=False,
        help="Re-run even when cached JSONL already exists.",
    )
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

    Reads the JSONL files produced by each runner and persists
    them to SQLite for historical querying.

    Args:
        db: Open database connection.
        model: Model identifier.
        args: Parsed CLI arguments (to know which benchmarks ran).
    """
    out_dir = Path(args.output_dir)

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

    def _load_and_save(
        benchmark: str,
        filename: str,
        id_key: str = "sample_id",
    ) -> None:
        jsonl_path = out_dir / benchmark / filename
        if not jsonl_path.exists():
            return
        records = _load_jsonl(jsonl_path)
        if records:
            run_id = _get_run_id(benchmark)
            if run_id is not None:
                db.save_samples(
                    run_id=run_id,
                    model=model,
                    benchmark=benchmark,
                    samples=records,
                    id_key=id_key,
                )

    # LVEval: multiple JSONL files per dataset
    if args.lveval:
        lveval_dir = out_dir / "lveval"
        if lveval_dir.exists():
            run_id = _get_run_id("lveval")
            if run_id is not None:
                for jsonl_file in lveval_dir.glob("*.jsonl"):
                    dataset_name = jsonl_file.stem
                    records = _load_jsonl(jsonl_file)
                    if records:
                        for i, rec in enumerate(records):
                            if "sample_id" not in rec:
                                rec["sample_id"] = f"{dataset_name}_{i}"
                        db.save_samples(
                            run_id=run_id,
                            model=model,
                            benchmark="lveval",
                            samples=records,
                            id_key="sample_id",
                        )

    # LongBench
    if args.longbench:
        _load_and_save("longbench", "predictions.jsonl", id_key="_id")

    # MathArena
    if args.matharena:
        _load_and_save("matharena", "predictions.jsonl", id_key="problem_idx")

    # BFCL: per-category JSONL files
    if args.bfcl:
        bfcl_dir = out_dir / "bfcl"
        if bfcl_dir.exists():
            run_id = _get_run_id("bfcl")
            if run_id is not None:
                for jsonl_file in bfcl_dir.glob("*.jsonl"):
                    records = _load_jsonl(jsonl_file)
                    if records:
                        for rec in records:
                            if "sample_id" not in rec:
                                rec["sample_id"] = rec.get("id", "")
                        db.save_samples(
                            run_id=run_id,
                            model=model,
                            benchmark="bfcl",
                            samples=records,
                            id_key="sample_id",
                        )

    # SimpleVQA
    if args.simplevqa:
        _load_and_save("simplevqa", "predictions.jsonl", id_key="data_id")

    # CompareBench
    if args.comparebench:
        _load_and_save("comparebench", "predictions.jsonl", id_key="data_id")


def main() -> None:
    """Execute the benchmark pipeline."""
    args = parse_args()
    config = load_config()

    if args.base_url is not None:
        config = replace(config, base_url=args.base_url)
    if args.api_key is not None:
        config = replace(config, api_key=args.api_key)
    if args.model is not None:
        config = replace(config, model=args.model)

    client = LLMClient(config)

    results = BenchmarkResults(model=config.model)

    logger.info("Running benchmarks for model {}", config.model)
    logger.debug(
        "Active benchmarks: lveval={} longbench={} matharena={} bfcl={} "
        "simplevqa={} comparebench={}",
        args.lveval,
        args.longbench,
        args.matharena,
        args.bfcl,
        args.simplevqa,
        args.comparebench,
    )

    if not any(
        [
            args.lveval,
            args.longbench,
            args.matharena,
            args.bfcl,
            args.simplevqa,
            args.comparebench,
        ]
    ):
        logger.warning(
            "No benchmark selected. Use --lveval, --longbench, --matharena, "
            "--bfcl, --simplevqa, and/or --comparebench to choose which "
            "benchmarks to run."
        )
        return

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

    # Collect which benchmarks will run for --force clearing
    benchmarks_to_run: list[str] = []
    if args.lveval:
        benchmarks_to_run.append("lveval")
    if args.longbench:
        benchmarks_to_run.append("longbench")
    if args.matharena:
        benchmarks_to_run.append("matharena")
    if args.bfcl:
        benchmarks_to_run.append("bfcl")
    if args.simplevqa:
        benchmarks_to_run.append("simplevqa")
    if args.comparebench:
        benchmarks_to_run.append("comparebench")

    # Clear model+benchmark data when --force is used
    if args.force:
        for bench in benchmarks_to_run:
            db.clear_model_benchmark(config.model, bench)

    if args.lveval:
        logger.info("Running LVEval")
        lveval = LVEvalRunner(
            client,
            args.output_dir,
            max_length=args.max_length,
            limit=args.limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            force=args.force,
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
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            force=args.force,
        )
        results.longbench = longbench.run()

    if args.matharena:
        logger.info("Running MathArena")
        matharena = MathArenaRunner(
            client,
            args.output_dir,
            limit=args.limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            force=args.force,
        )
        results.matharena = matharena.run()

    if args.bfcl:
        logger.info("Running BFCL v4")
        bfcl = BFCLRunner(
            client,
            args.output_dir,
            categories=args.bfcl_categories,
            limit=args.limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            force=args.force,
        )
        results.bfcl = bfcl.run()

    if args.simplevqa:
        logger.info("Running SimpleVQA")
        simplevqa = SimpleVQARunner(
            client,
            args.output_dir,
            limit=args.limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            image_width=args.image_width,
            image_height=args.image_height,
            force=args.force,
        )
        results.simplevqa = simplevqa.run()

    if args.comparebench:
        logger.info("Running CompareBench")
        comparebench = CompareBenchRunner(
            client,
            args.output_dir,
            limit=args.limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            image_width=args.image_width,
            image_height=args.image_height,
            force=args.force,
        )
        results.comparebench = comparebench.run(
            selected_splits=args.comparebench_splits,
        )

    # Save aggregated results to SQLite
    logger.info("Saving results to database")
    db.save_benchmark_results(results, config=run_config)

    # Save per-sample data from JSONL files to SQLite
    _save_samples_to_db(db, config.model, args)

    logger.info("Generating reports")
    generate_raw_csvs(results, out_dir)
    generate_html_report(db, out_dir)
    db.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
