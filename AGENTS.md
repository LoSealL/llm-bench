# Repository Guide for Agents

## Project Overview

LLM benchmark suite that evaluates models via OpenAI-compatible APIs against nine datasets:
- **LVEval** (`scripts/LVEval/`) — long-context QA benchmark
- **LongBench-v2** (`scripts/LongBench/`) — multiple-choice long-context benchmark
- **MathArena** (HuggingFace `MathArena/aime_2026`) — math reasoning benchmark
- **BFCL v4** (`scripts/BFCL/`) — function-calling benchmark
- **SimpleVQA** (HuggingFace `lmms-lab/SimpleVQA`) — visual question answering
- **CompareBench** (HuggingFace `qiuzhangTiTi/CompareBench`) — visual comparison
- **MMMU** (HuggingFace `MMMU/MMMU`) — multimodal understanding
- **OCRBench v2** (HuggingFace `akalen/ocrbench_v2`) — optical character recognition
- **Omni AI OCR** (HuggingFace `omni-ai/ocrbench-omni-1`) — OCR benchmark

## Critical Architecture Rule

**Never modify files under `scripts/`.** Both `scripts/LVEval/` and `scripts/LongBench/` are third-party benchmark code treated as read-only.

All custom logic lives in the `llm_bench/` package and `run_benchmark.py`.

## Project Structure

```
llm-bench/
├── run_benchmark.py           # CLI entry point (registry-driven dispatch)
├── llm_bench/                 # Custom code only
│   ├── config.py              # .env loader
│   ├── client.py              # OpenAI client wrapper
│   ├── registry.py            # Benchmark registry (imports Metadata classes)
│   ├── runners.py             # ArgSpec, PersistenceSpec, RunnerMetadata, BaseRunner
│   ├── runner/                # One module per benchmark
│   │   ├── bfcl.py            # BFCL v4
│   │   ├── lveval.py          # LVEval
│   │   ├── longbench.py       # LongBench-v2
│   │   ├── matharena.py       # MathArena
│   │   ├── simplevqa.py       # SimpleVQA
│   │   ├── comparebench.py    # CompareBench
│   │   ├── mmmu.py            # MMMU
│   │   ├── ocrbench_v2.py     # OCRBench v2
│   │   └── ocrbench_omni.py   # Omni AI OCR
│   ├── bfcl_constants.py      # BFCL category definitions
│   ├── bfcl_eval.py           # BFCL evaluation logic
│   ├── bfcl_utils.py          # BFCL utilities
│   ├── reporter.py            # HTML report generation
│   └── storage.py             # SQLite persistence
├── scripts/                   # THIRD-PARTY — DO NOT MODIFY (except BFCL data)
│   ├── BFCL/                  # BFCL v4 prompts and ground truth
│   ├── LVEval/
│   └── LongBench/
├── .env                       # OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
└── pyproject.toml
```

## Registry Architecture

Every benchmark is self-registered via a `Metadata(RunnerMetadata)` class defined in its runner module. The registry in `llm_bench/registry.py` imports all Metadata classes into a `BENCHMARKS` tuple — no static globals, no hand-wired dispatch.

### Key types (defined in `llm_bench/runners.py`)

| Type | Purpose |
|------|---------|
| `ArgSpec` | Frozen dataclass describing one CLI argument (name, flag, help, nargs, choices, default, is_flag) |
| `PersistenceSpec` | Frozen dataclass describing how JSONL predictions are loaded (layout: single/multi, categories, filename, id_key, sample_id_factory) |
| `RunnerMetadata` | Base class with class-level attributes (`name`, `dataset`, `runner_cls`, `cli_args`, `persistence`) and classmethods (`build_runner`, `to_scores`, `extract_run_kwargs`) |

### Import chain (no circular dependencies)

```
runners.py  ←── defines ArgSpec, PersistenceSpec, RunnerMetadata, BaseRunner
    ↑
runner/*.py ←── imports from runners.py; defines Metadata(RunnerMetadata) subclass
    ↑
registry.py ←── imports Metadata classes; builds BENCHMARKS tuple
    ↑
run_benchmark.py, storage.py ←── use registry for dispatch
```

### Dispatch flows

All four dispatch sites in `run_benchmark.py` (dry-run, selection guard, execution loop, DB persistence) collapse to single loops iterating `BENCHMARKS`:
- `selected_benchmarks(args)` → filters Metadata classes by CLI flags
- `descriptor.build_runner(client, output_dir, args)` → constructs runner
- `descriptor.to_scores(raw_result)` → transforms output for storage
- `descriptor.extract_run_kwargs(args)` → extra kwargs for `runner.run()`

## Toolchain & Commands

Package manager is `uv` (lockfile: `uv.lock`).

```bash
# Install dependencies
uv sync

# Lint (88 column limit)
uv run ruff check llm_bench/ run_benchmark.py

# Auto-fix
uv run ruff check --fix llm_bench/ run_benchmark.py

# Format
uv run ruff format llm_bench/ run_benchmark.py

# Type check
uv run pyright llm_bench/ run_benchmark.py

# Run tests
uv run pytest tests/ -v

# Run all benchmarks
uv run python run_benchmark.py --lveval --longbench --matharena --bfcl

# Run selected benchmarks
uv run python run_benchmark.py --bfcl --bfcl-categories simple_python multiple

# Override endpoint and model
uv run python run_benchmark.py --base-url https://api.example.com --api-key sk-xxx --model gpt-4 --lveval --bfcl

# Run with options
uv run python run_benchmark.py --model deepseek-chat --lveval --lveval-lengths 32k 64k

# Dry-run (inspect datasets without API calls)
uv run python run_benchmark.py --bfcl --dry-run --limit 1

# Generate HTML report from existing DB
uv run python -m llm_bench.reporter
```

## Code Style Requirements

- **Line length**: 88 columns (enforced by ruff)
- **Type annotations**: Full coverage required; run `pyright` to verify
- **Import grouping**: stdlib → third-party → local; no function-local imports
- **Copyright header**: Every new file must start with:
  ```python
  # Copyright (c) 2026 llm-bench authors
  # SPDX-License-Identifier: MIT
  """Module docstring."""
  ```
- **Docstrings**: Google style for every module, class, and function
- **No** `from __future__ import annotations` — removed from all files

## Design Patterns

### Reusing Third-Party Code Without Modifying It

LVEval config/utils/metrics are imported dynamically at runtime:
```python
import sys
import importlib
sys.path.insert(0, str(scripts_dir))
config = importlib.import_module("config")
sys.path.pop(0)
```

LongBench prompt templates are read as text files:
```python
template = (repo_root / "scripts/LongBench/prompts/0shot.txt").read_text()
```

### Adding a New Benchmark

1. Create `llm_bench/runner/<name>.py` with a `BaseRunner` subclass
2. Define a `Metadata(RunnerMetadata)` class at the bottom with:
   - Class attributes: `name`, `dataset`, `runner_cls`, `cli_args`, `persistence`
   - Classmethods: `build_runner`, `to_scores` (optionally `extract_run_kwargs`)
3. Import the `Metadata` class in `llm_bench/registry.py` and add to `BENCHMARKS` tuple
4. No edits to `run_benchmark.py` or `storage.py` needed — registry-driven dispatch handles the rest

Example Metadata class:
```python
from llm_bench.runners import ArgSpec, BaseRunner, PersistenceSpec, RunnerMetadata

class MyRunner(BaseRunner):
    ...

class Metadata(RunnerMetadata):
    name = "mybench"
    dataset = "my_dataset"
    runner_cls = MyRunner
    cli_args = [ArgSpec(name="mybench", flag="--mybench", help="...", is_flag=True)]
    persistence = PersistenceSpec(layout="single", categories=[], filename="predictions.jsonl", id_key="id")

    @classmethod
    def build_runner(cls, client, output_dir, args):
        return MyRunner(client, output_dir)

    @classmethod
    def to_scores(cls, result):
        return {"overall": result}
```

## Environment Configuration

`.env` file (already present, never commit it):
```
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-...
OPENAI_MODEL=deepseek-v4-flash
```

Loaded via `llm_bench.config.load_config()`.

## Output Artifacts

Running benchmarks produces:
- `results/lveval/*.jsonl` — raw predictions
- `results/longbench/*.jsonl` — raw predictions
- `results/matharena/*.jsonl` — raw predictions
- `results/bfcl/*.jsonl` — raw predictions
- `results/simplevqa/*.jsonl` — raw predictions
- `results/comparebench/*.jsonl` — raw predictions
- `results/mmmu/*.jsonl` — raw predictions
- `results/ocrbench_v2/*.jsonl` — raw predictions
- `results/ocrbench_omni/*.jsonl` — raw predictions
- `results/benchmarks.db` — SQLite database with historical scores
- `results/benchmark_report.html` — Chart.js dashboard (no raw data shown)

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan:
`specs/002-registry-refactor/plan.md`
<!-- SPECKIT END -->
