# llm-bench

Benchmark suite for evaluating LLMs via OpenAI-compatible APIs against nine datasets.

## Benchmarks

| Benchmark | Dataset | Task |
|-----------|---------|------|
| LVEval | `LVEval` | Long-context QA |
| LongBench-v2 | `LongBench-v2` | Long-context multiple-choice |
| MathArena | `MathArena/aime_2026` | Math reasoning |
| BFCL v4 | `BFCL/v4` | Function calling |
| SimpleVQA | `lmms-lab/SimpleVQA` | Visual question answering |
| CompareBench | `qiuzhangTiTi/CompareBench` | Visual comparison |
| MMMU | `MMMU/MMMU` | Multimodal understanding |
| OCRBench v2 | `akalen/ocrbench_v2` | Optical character recognition |
| Omni AI OCR | `omni-ai/ocrbench-omni-1` | OCR |

## Quick Start

```bash
# Setup
uv sync
cp .env.example .env  # edit with your API credentials

# Run benchmarks
uv run python run_benchmark.py --bfcl --matharena

# Dry-run (inspect datasets without API calls)
uv run python run_benchmark.py --mmmu --dry-run --limit 1

# Generate HTML report
uv run python -m llm_bench.reporter
```

## Environment

Create a `.env` file:

```
OPENAI_BASE_URL=https://api.deepseek.com
OPENAI_API_KEY=sk-...
OPENAI_MODEL=deepseek-v4-flash
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--model` | Model name (overrides `.env`) |
| `--base-url` | API base URL |
| `--api-key` | API key |
| `--limit N` | Cap samples per benchmark |
| `--max-tokens N` | Max output tokens |
| `--temperature F` | Sampling temperature |
| `--force` | Re-run even when cached |
| `--dry-run` | Inspect datasets without API calls |
| `--output-dir DIR` | Output directory (default: `results`) |

### Benchmark-specific flags

| Flag | Benchmark |
|------|-----------|
| `--bfcl-categories` | BFCL categories |
| `--lveval-datasets` | LVEval datasets |
| `--lveval-lengths` | LVEval length levels |
| `--comparebench-splits` | CompareBench splits |
| `--mmmu-split` | MMMU split |

## Output

```
results/
├── bfcl/*.jsonl
├── lveval/*.jsonl
├── benchmarks.db              # SQLite history
└── benchmark_report.html      # Chart.js dashboard
```

## Adding a Benchmark

1. Create `llm_bench/runner/<name>.py` with a `BaseRunner` subclass
2. Define a `Metadata(RunnerMetadata)` class
3. Import and register in `llm_bench/registry.py`

See `AGENTS.md` for full architecture details.

## License

MIT
