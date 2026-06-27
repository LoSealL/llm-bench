# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Benchmark registry — collects runner metadata classes.

Each runner module defines a ``Metadata`` subclass (inheriting from
:class:`~llm_bench.runners.RunnerMetadata`) with class-level attributes
that describe the benchmark's CLI, persistence, and score transformation.
This module imports those classes and assembles them into
:data:`BENCHMARKS`.
"""

import argparse
from typing import Any

from llm_bench.runner import (
    BFCLRunner,
    CompareBenchRunner,
    LVEvalRunner,
    LongBenchRunner,
    MMMURunner,
    MathArenaRunner,
    OmniOCRBenchRunner,
    OCRBenchV2Runner,
    SimpleVQARunner,
)
from llm_bench.runner.bfcl import Metadata as _BFCLMetadata
from llm_bench.runner.comparebench import Metadata as _CompareBenchMetadata
from llm_bench.runner.longbench import Metadata as _LongBenchMetadata
from llm_bench.runner.lveval import Metadata as _LVEvalMetadata
from llm_bench.runner.matharena import Metadata as _MathArenaMetadata
from llm_bench.runner.mmmu import Metadata as _MMMUMetadata
from llm_bench.runner.ocrbench_omni import Metadata as _OmniMetadata
from llm_bench.runner.ocrbench_v2 import Metadata as _OCRV2Metadata
from llm_bench.runner.simplevqa import Metadata as _SimpleVQAMetadata
from llm_bench.runners import RunnerMetadata

# Re-export the runner classes so ``from llm_bench.registry import …``
# works the same as before.
__all__ = [
    "BFCLRunner",
    "BENCHMARKS",
    "CompareBenchRunner",
    "LVEvalRunner",
    "LongBenchRunner",
    "MMMURunner",
    "MathArenaRunner",
    "OmniOCRBenchRunner",
    "OCRBenchV2Runner",
    "RunnerMetadata",
    "SimpleVQARunner",
    "build_argparser",
    "get_descriptor",
    "selected_benchmarks",
]

# ---------------------------------------------------------------------------
# BENCHMARKS registry
# ---------------------------------------------------------------------------

#: Tuple of all registered runner-metadata classes, in dispatch order.
BENCHMARKS: tuple[type[RunnerMetadata], ...] = (
    _LVEvalMetadata,
    _LongBenchMetadata,
    _MathArenaMetadata,
    _BFCLMetadata,
    _SimpleVQAMetadata,
    _CompareBenchMetadata,
    _MMMUMetadata,
    _OCRV2Metadata,
    _OmniMetadata,
)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_descriptor(name: str) -> type[RunnerMetadata]:
    """Fetch a runner-metadata class by benchmark name.

    Args:
        name: Canonical benchmark identifier.

    Returns:
        The matching ``RunnerMetadata`` subclass.

    Raises:
        KeyError: If no benchmark with *name* is registered.
    """
    for m in BENCHMARKS:
        if m.name == name:
            return m
    raise KeyError(f"No benchmark registered with name '{name}'")


def selected_benchmarks(
    args: argparse.Namespace,
) -> list[type[RunnerMetadata]]:
    """Filter the registry to benchmarks whose selection flag is set.

    Args:
        args: Parsed CLI namespace.

    Returns:
        List of metadata classes whose on/off flag is truthy.
    """
    return [
        m
        for m in BENCHMARKS
        if getattr(args, m.cli_args[0].name, False)
    ]


# ---------------------------------------------------------------------------
# Argument-parser builder
# ---------------------------------------------------------------------------


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Add the non-benchmark CLI arguments to *parser*.

    Args:
        parser: The argument parser to configure.
    """
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL "
        "(overrides OPENAI_BASE_URL from .env).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the endpoint "
        "(overrides OPENAI_API_KEY from .env).",
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
    parser.add_argument(
        "--no-thinking",
        "-nt",
        action="store_true",
        default=False,
        help="Disable extended thinking (sets enable_thinking: false).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Load datasets and print info without calling the API.",
    )


def _add_benchmark_arg(
    parser: argparse.ArgumentParser,
    spec: Any,
) -> None:
    """Add a single benchmark-specific argument to *parser*.

    Args:
        parser: The argument parser to configure.
        spec: An :class:`~llm_bench.runners.ArgSpec` instance.
    """
    if spec.is_flag:
        parser.add_argument(
            spec.flag,
            dest=spec.name,
            action="store_true",
            default=False,
            help=spec.help,
        )
    else:
        kwargs: dict[str, Any] = {
            "dest": spec.name,
            "help": spec.help,
            "default": spec.default,
        }
        if spec.nargs is not None:
            kwargs["nargs"] = spec.nargs
        if spec.choices is not None:
            kwargs["choices"] = spec.choices
        parser.add_argument(spec.flag, **kwargs)


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI argument parser from the registry.

    Returns:
        A fully-configured ``ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description="Run LLM benchmarks via OpenAI-compatible API.",
    )
    _add_shared_args(parser)
    for metadata in BENCHMARKS:
        for spec in metadata.cli_args:
            _add_benchmark_arg(parser, spec)
    return parser
