# Copyright (c) 2026 llm-bench authors
# SPDX-License-Identifier: MIT
"""Result reporting utilities.

Generates raw CSV files and a summarised HTML dashboard for benchmark
runs. The HTML page uses Chart.js for bar charts and contains **no**
raw per-sample data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from llm_bench.runners import BenchmarkResults


def ensure_dir(path: Path) -> None:
    """Create a directory and all parents if they do not exist.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    """Write a CSV file using the standard library.

    Args:
        path: Output file path.
        headers: Column names.
        rows: Data rows.
    """
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)


def generate_raw_csvs(results: BenchmarkResults, out_dir: Path) -> None:
    """Persist per-dataset results as CSV files.

    Args:
        results: Aggregated results from all runners.
        out_dir: Directory where ``raw/`` will be created.
    """
    raw_dir = out_dir / "raw"
    ensure_dir(raw_dir)

    # LVEval: dataset, length, score
    lveval_rows: list[list[str]] = []
    for ds, lengths in results.lveval.items():
        for length, score in lengths.items():
            lveval_rows.append([ds, length, str(score)])
    _write_csv(
        raw_dir / "lveval_results.csv",
        ["dataset", "length", "score"],
        lveval_rows,
    )

    # LongBench: difficulty, length, accuracy
    lb = results.longbench
    lb_rows: list[list[str]] = [
        ["overall", str(lb.get("overall", 0.0))],
        ["easy", str(lb.get("easy", 0.0))],
        ["hard", str(lb.get("hard", 0.0))],
        ["short", str(lb.get("short", 0.0))],
        ["medium", str(lb.get("medium", 0.0))],
        ["long", str(lb.get("long", 0.0))],
    ]
    _write_csv(
        raw_dir / "longbench_results.csv",
        ["category", "accuracy"],
        lb_rows,
    )

    # MathArena
    ma = results.matharena
    ma_rows: list[list[str]] = [
        ["accuracy", str(ma.get("accuracy", 0.0))],
        ["correct", str(ma.get("correct", 0))],
        ["total", str(ma.get("total", 0))],
    ]
    _write_csv(
        raw_dir / "matharena_results.csv",
        ["metric", "value"],
        ma_rows,
    )

    # BFCL
    bfcl_rows: list[list[str]] = []
    for category, stats in results.bfcl.items():
        bfcl_rows.append(
            [
                category,
                str(round(stats.get("accuracy", 0.0) * 100, 2)),
                str(stats.get("correct_count", 0)),
                str(stats.get("total_count", 0)),
            ]
        )
    _write_csv(
        raw_dir / "bfcl_results.csv",
        ["category", "accuracy", "correct", "total"],
        bfcl_rows,
    )


def generate_html_report(results: BenchmarkResults, out_dir: Path) -> None:
    """Render a summarised HTML report with Chart.js bar charts.

    The report contains **no raw data tables**; only aggregated scores
    are visualised.

    Args:
        results: Aggregated results from all runners.
        out_dir: Directory where ``benchmark_report.html`` will be
            written.
    """
    ensure_dir(out_dir)

    # Compute summary scores
    lveval_scores = [
        sum(lengths.values()) / len(lengths) if lengths else 0.0
        for lengths in results.lveval.values()
    ]
    lveval_avg = sum(lveval_scores) / len(lveval_scores) if lveval_scores else 0.0
    lb_overall = results.longbench.get("overall", 0.0)
    ma_acc = results.matharena.get("accuracy", 0.0)
    bfcl_stats = results.bfcl
    bfcl_avg = 0.0
    if bfcl_stats:
        bfcl_avg = sum(s.get("accuracy", 0.0) * 100 for s in bfcl_stats.values()) / len(
            bfcl_stats
        )

    labels = ["LVEval", "LongBench-v2", "MathArena", "BFCL v4"]
    values = [
        round(lveval_avg, 2),
        round(lb_overall, 2),
        round(ma_acc, 2),
        round(bfcl_avg, 2),
    ]
    chart_data = json.dumps({"labels": labels, "values": values})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {{--bg:#fff;--fg:#1a1a2e;--accent:#4f46e5;--muted:#6b7280;}}
  * {{box-sizing:border-box;margin:0;padding:0;}}
  body {{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--fg);line-height:1.6;padding:2rem;}}
  .container {{max-width:960px;margin:0 auto;}}
  h1 {{font-size:1.75rem;margin-bottom:0.5rem;}}
  .subtitle {{color:var(--muted);margin-bottom:2rem;}}
  .card {{background:#f8fafc;border-radius:0.75rem;padding:1.5rem;margin-bottom:1.5rem;box-shadow:0 1px 3px rgba(0,0,0,0.05);}}
  .card h2 {{font-size:1.125rem;margin-bottom:1rem;color:var(--accent);}}
  .metric {{display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid #e5e7eb;}}
  .metric:last-child {{border-bottom:none;}}
  .metric span:first-child {{color:var(--muted);}}
  .metric span:last-child {{font-weight:600;}}
  canvas {{max-height:320px;}}
</style>
</head>
<body>
<div class="container">
  <h1>LLM Benchmark Report</h1>
  <p class="subtitle">Generated by llm-bench</p>

  <div class="card">
    <h2>Model</h2>
    <div class="metric"><span>Model</span><span>{results.model}</span></div>
  </div>

  <div class="card">
    <h2>Overall Scores</h2>
    <canvas id="overallChart"></canvas>
  </div>

  <div class="card">
    <h2>LVEval Summary</h2>
    <div class="metric"><span>Average Score</span><span>{lveval_avg:.2f}</span></div>
  </div>

  <div class="card">
    <h2>LongBench-v2 Summary</h2>
    <div class="metric"><span>Overall</span><span>{lb_overall:.1f}%</span></div>
    <div class="metric"><span>Easy</span><span>{
        results.longbench.get("easy", 0.0):.1f}%</span></div>
    <div class="metric"><span>Hard</span><span>{
        results.longbench.get("hard", 0.0):.1f}%</span></div>
    <div class="metric"><span>Short</span><span>{
        results.longbench.get("short", 0.0):.1f}%</span></div>
    <div class="metric"><span>Medium</span><span>{
        results.longbench.get("medium", 0.0):.1f}%</span></div>
    <div class="metric"><span>Long</span><span>{
        results.longbench.get("long", 0.0):.1f}%</span></div>
  </div>

  <div class="card">
    <h2>MathArena Summary</h2>
    <div class="metric"><span>Accuracy</span><span>{ma_acc:.1f}%</span></div>
    <div class="metric"><span>Correct</span><span>{
        results.matharena.get("correct", 0)
    }/{results.matharena.get("total", 0)}</span></div>
  </div>

  <div class="card">
    <h2>BFCL v4 Summary</h2>
    <div class="metric"><span>Average Accuracy</span><span>{bfcl_avg:.1f}%</span></div>
    {
        "".join(
            f'<div class="metric"><span>{cat}</span><span>{stats.get("accuracy", 0.0) * 100:.1f}% '
            f"({stats.get('correct_count', 0)}/{stats.get('total_count', 0)})</span></div>"
            for cat, stats in results.bfcl.items()
        )
    }
  </div>
</div>

<script>
  const data = {chart_data};
  new Chart(document.getElementById('overallChart'),{{
    type:'bar',
    data:{{
      labels:data.labels,
      datasets:[{{
        label:'Score',
        data:data.values,
        backgroundColor:['#4f46e5','#06b6d4','#10b981','#f59e0b'],
        borderRadius:6,
      }}]
    }},
    options:{{
      responsive:true,
      maintainAspectRatio:false,
      scales:{{y:{{beginAtZero:true,max:100,title:{{display:true,text:'Score / Accuracy (%)'}}}}}},
      plugins:{{legend:{{display:false}}}}
    }}
  }});
</script>
</body>
</html>"""

    report_path = out_dir / "benchmark_report.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info("Report saved to {}", report_path)
