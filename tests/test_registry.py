# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Tests for the benchmark registry."""

import argparse

import pytest

from llm_bench.registry import (
    BENCHMARKS,
    build_argparser,
    get_descriptor,
    selected_benchmarks,
)
from llm_bench.runners import ArgSpec, PersistenceSpec, RunnerMetadata


def test_get_descriptor_returns_correct_entry():
    """get_descriptor returns the metadata class for a known benchmark."""
    meta = get_descriptor("matharena")
    assert meta.name == "matharena"
    assert meta.dataset == "aime_2026"


def test_get_descriptor_raises_for_unknown():
    """get_descriptor raises KeyError for an unknown benchmark."""
    with pytest.raises(KeyError, match="nonexistent"):
        get_descriptor("nonexistent")


def test_selected_benchmarks_filters_correctly():
    """selected_benchmarks returns only benchmarks whose flag is True."""
    args = argparse.Namespace(
        lveval=True,
        longbench=False,
        matharena=True,
        bfcl=False,
        simplevqa=False,
        comparebench=False,
        mmmu=False,
        ocrbench_v2=False,
        ocrbench_omni=False,
    )
    selected = selected_benchmarks(args)
    names = [m.name for m in selected]
    assert "lveval" in names
    assert "matharena" in names
    assert "longbench" not in names
    assert "bfcl" not in names


def test_selected_benchmarks_empty():
    """selected_benchmarks returns empty list when no flags are set."""
    args = argparse.Namespace(
        lveval=False,
        longbench=False,
        matharena=False,
        bfcl=False,
        simplevqa=False,
        comparebench=False,
        mmmu=False,
        ocrbench_v2=False,
        ocrbench_omni=False,
    )
    assert selected_benchmarks(args) == []


def test_all_descriptors_have_selection_flag():
    """Every metadata class's first cli_arg must be the is_flag selection flag."""
    for m in BENCHMARKS:
        assert m.cli_args[0].is_flag, (
            f"{m.name}: cli_args[0] must be is_flag=True"
        )


def test_all_persistence_layouts_valid():
    """Every persistence.layout must be single or multi."""
    for m in BENCHMARKS:
        assert m.persistence.layout in ("single", "multi"), (
            f"{m.name}: invalid persistence layout "
            f"'{m.persistence.layout}'"
        )


def test_all_names_unique():
    """All benchmark names in the registry must be unique."""
    names = [m.name for m in BENCHMARKS]
    assert len(names) == len(set(names)), "Duplicate benchmark names"


def test_build_argparser_accepts_each_flag():
    """The generated parser accepts each benchmark's selection flag."""
    parser = build_argparser()
    for m in BENCHMARKS:
        flag = m.cli_args[0].flag
        args = parser.parse_args([flag])
        assert getattr(args, m.cli_args[0].name) is True


def test_build_argparser_help_has_no_errors():
    """The generated parser can produce help text without crashing."""
    parser = build_argparser()
    try:
        parser.format_help()
    except Exception:
        pytest.fail("parser.format_help() raised an exception")


def test_argspec_rejects_flag_with_choices():
    """ArgSpec raises ValueError when is_flag and choices are both set."""
    with pytest.raises(ValueError, match="incompatible"):
        ArgSpec(
            name="bad",
            flag="--bad",
            help="bad",
            is_flag=True,
            choices=["a", "b"],
        )


def test_runner_metadata_build_runner_not_implemented():
    """RunnerMetadata.build_runner raises NotImplementedError by default."""

    class _Stub(RunnerMetadata):
        name = "stub"
        dataset = "stub"
        runner_cls = BENCHMARKS[0].runner_cls
        cli_args = [
            ArgSpec(
                name="stub",
                flag="--stub",
                help="stub",
                is_flag=True,
            ),
        ]
        persistence = PersistenceSpec(
            layout="single",
            categories=[],
            filename="predictions.jsonl",
            id_key="id",
        )

    with pytest.raises(NotImplementedError):
        _Stub.build_runner(None, "out", argparse.Namespace())


def test_nine_benchmarks_registered():
    """The registry contains exactly nine benchmarks."""
    assert len(BENCHMARKS) == 9
