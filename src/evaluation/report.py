"""Export clean Markdown and self-contained HTML comparison reports."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.evaluation.metrics import TASK_METRICS


def build_comparison_markdown(
    df: pd.DataFrame,
    task: str,
    split_label: str,
    primary_metric: str,
    narrative_md: str,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cols = [
        c for c in (
            ["display_name", "family", "variant", "feature_space", "source"]
            + TASK_METRICS.get(task, [])
            + ["train_time_s", "n_train_rows"]
        )
        if c in df.columns
    ]
    view = df[cols].copy()
    for c in view.select_dtypes(include=["float", "float64"]).columns:
        view[c] = view[c].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")

    headers = " | ".join(cols)
    sep = " | ".join(["---"] * len(cols))
    body_lines = [
        " | ".join(str(row[c]) for c in cols)
        for _, row in view.iterrows()
    ]
    table_md = "\n".join([f"| {headers} |", f"| {sep} |", *[f"| {line} |" for line in body_lines]])

    return (
        f"# City Hotel Booking Demand — {task.title()} Model Comparison\n\n"
        f"- **Generated:** {now}\n"
        f"- **Evaluation Split:** `{split_label}`\n"
        f"- **Primary Selection Metric:** `{primary_metric}`\n\n"
        f"## Analytical Summary\n\n{narrative_md}\n\n"
        f"## Leaderboard Table\n\n{table_md}\n"
    )


def build_comparison_html(
    df: pd.DataFrame,
    task: str,
    split_label: str,
    primary_metric: str,
    narrative_md: str,
) -> str:
    cols = [
        c for c in (
            ["display_name", "family", "variant", "feature_space", "source"]
            + TASK_METRICS.get(task, [])
            + ["train_time_s", "n_train_rows"]
        )
        if c in df.columns
    ]
    view = df[cols].copy()
    table_html = view.to_html(index=False, float_format=lambda x: f"{x:.4f}", classes="leaderboard", border=0)
    narrative_items = "".join(
        f"<li>{line.lstrip('- ').strip()}</li>"
        for line in narrative_md.splitlines()
        if line.strip()
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Model Comparison — {task.title()}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&family=Manrope:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
  body {{ font-family: 'Manrope', system-ui, -apple-system, sans-serif; background: #F4F0E9; max-width: 1080px; margin: 36px auto; padding: 0 24px; color: #23211E; }}
  h1, h2 {{ font-family: 'Plus Jakarta Sans', 'Manrope', sans-serif; font-weight: 600; color: #1D1C1A; letter-spacing: -0.02em; }}
  h1 {{ font-size: 1.75rem; border-bottom: 1px solid #DDD5C7; padding-bottom: 12px; }}
  h2 {{ font-size: 1.25rem; margin-top: 28px; }}
  .meta {{ font-family: 'IBM Plex Mono', monospace; color: #6E695F; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 22px; }}
  .summary {{ background: #FBF9F5; border: 1px solid #DDD5C7; border-left: 3.5px solid #8A6A3B; padding: 16px 20px; border-radius: 12px; line-height: 1.6; }}
  table.leaderboard {{ background: #FCFBF8; border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 0.90rem; border-radius: 12px; overflow: hidden; border: 1px solid #DDD5C7; }}
  table.leaderboard th, table.leaderboard td {{ border-bottom: 1px solid #E4DCCE; padding: 10px 12px; text-align: left; }}
  table.leaderboard th {{ background: #EDE6DA; color: #3A3733; font-weight: 600; font-size: 0.82rem; letter-spacing: 0.02em; }}
  table.leaderboard tr:hover {{ background: #F6F1E6; }}
</style>
</head>
<body>
  <h1>City Hotel (H2) — {task.title()} Model Comparison</h1>
  <div class="meta">Split: <b>{split_label}</b> &nbsp;|&nbsp; Primary Metric: <b>{primary_metric}</b></div>
  <h2>Executive Summary</h2>
  <div class="summary"><ul>{narrative_items}</ul></div>
  <h2>Leaderboard</h2>
  {table_html}
</body>
</html>"""
