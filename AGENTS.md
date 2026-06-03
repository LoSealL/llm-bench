# Repository Guide for Agents

## Project Overview

LLM benchmark suite that evaluates models via OpenAI-compatible APIs against three datasets:
- **LVEval** (`scripts/LVEval/`) — long-context QA benchmark
- **LongBench-v2** (`scripts/LongBench/`) — multiple-choice long-context benchmark
- **MathArena/aime_2026** (HuggingFace) — math reasoning benchmark

## Critical Architecture Rule

**Never modify files under `scripts/`.** Both `scripts/LVEval/` and `scripts/LongBench/` are third-party benchmark code treated as read-only.

All custom logic lives in the `llm_bench/` package and `run_benchmark.py`.

## Project Structure

```
llm-bench/
├── run_benchmark.py           # CLI entry point
├── llm_bench/                 # Custom code only
│   ├── config.py              # .env loader
│   ├── client.py              # OpenAI client wrapper
│   ├── lveval_runner.py       # LVEval evaluation (imports scripts/LVEval/)
│   ├── longbench_runner.py    # LongBench-v2 evaluation (reads scripts/LongBench/prompts/)
│   ├── matharena_runner.py    # MathArena evaluation (new)
│   ├── reporter.py            # CSV + HTML report generation
│   └── runners.py             # Shared types
├── scripts/                   # THIRD-PARTY — DO NOT MODIFY
│   ├── LVEval/
│   └── LongBench/
├── .env                       # OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
└── pyproject.toml
```

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

# Run benchmarks
uv run python run_benchmark.py

# Run with options
uv run python run_benchmark.py --model deepseek-chat --lveval-lengths 32k 64k --skip-matharena
```

## Code Style Requirements

- **Line length**: 88 columns (enforced by ruff)
- **Type annotations**: Full coverage required; run `pyright` to verify
- **Import grouping**: stdlib → third-party → local
- **Copyright header**: Every new file must start with:
  ```python
  # Copyright (c) 2026 llm-bench authors
  # SPDX-License-Identifier: MIT
  """Module docstring."""
  ```
- **Docstrings**: Google style for every module, class, and function
- **Future annotations**: Start every file with `from __future__ import annotations`

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

1. Create `llm_bench/<name>_runner.py`
2. Inherit nothing — just expose a `run() -> dict` method
3. Use `llm_bench.client.LLMClient` for API calls
4. Cache predictions to `results/<name>/` as `.jsonl`
5. Wire into `run_benchmark.py`

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
- `results/matharena/results.jsonl` — raw predictions
- `results/raw/*.csv` — per-dataset CSVs
- `results/benchmark_report.html` — Chart.js dashboard (no raw data shown)
