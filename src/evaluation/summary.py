"""Deterministic, data-driven comparison narrative generator (no hard-coded text)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import METRIC_META, format_metric


def generate_comparison_summary(
    df: pd.DataFrame,
    task: str,
    primary_metric: str,
    split_label: str = "val",
) -> str:
    """Produce a concise, analytical Markdown summary directly from the leaderboard DataFrame."""
    if df.empty or primary_metric not in df.columns:
        return "_No evaluated models match the current filter selection._"

    work = df.dropna(subset=[primary_metric]).copy()
    if work.empty:
        return f"_Metric `{primary_metric}` is unavailable for the selected runs._"

    meta = METRIC_META.get(primary_metric, {"higher_is_better": True})
    hib = bool(meta.get("higher_is_better", True))
    work = work.sort_values(primary_metric, ascending=not hib).reset_index(drop=True)

    best = work.iloc[0]
    best_val = float(best[primary_metric])
    best_name = f"**{best['display_name']}** (`{best['variant']}`, `{best['feature_space']}`)"

    bullets: list[str] = []
    bullets.append(
        f"- **Top Performer ({split_label.upper()} split):** {best_name} leads all {len(work)} evaluated configurations "
        f"with **{primary_metric} = {format_metric(primary_metric, best_val)}** "
        f"(trained in `{float(best.get('train_time_s', 0.0)):.2f}s` on `{int(best.get('n_train_rows', 0)):,}` rows)."
    )

    if len(work) >= 2:
        runner = work.iloc[1]
        r_val = float(runner[primary_metric])
        gap = abs(best_val - r_val)
        rel = (gap / abs(r_val) * 100.0) if r_val != 0 else 0.0
        bullets.append(
            f"- **Margin over Runner-Up:** Ahead of **{runner['display_name']}** (`{runner['variant']}`, "
            f"{format_metric(primary_metric, r_val)}) by **{format_metric(primary_metric, gap)}** ({rel:.1f}% relative difference)."
        )

    # Baseline vs Selected / Tuned delta analysis
    pair_rows = []
    for m_name, grp in work.groupby("display_name"):
        by_v = {str(r["variant"]): float(r[primary_metric]) for _, r in grp.dropna(subset=[primary_metric]).iterrows()}
        b_v = by_v.get("baseline", by_v.get("baseline_nopca"))
        t_v = by_v.get("tuned", by_v.get("variant_selected", by_v.get("baseline_pca90")))
        if b_v is not None and t_v is not None and len(by_v) >= 2:
            gain_v = (t_v - b_v) if hib else (b_v - t_v)
            rel_v = (gain_v / abs(b_v) * 100.0) if b_v != 0 else 0.0
            pair_rows.append({"model": str(m_name), "baseline": b_v, "tuned": t_v, "gain": gain_v, "rel_gain": rel_v})

    if pair_rows:
        pair_df = pd.DataFrame(pair_rows).sort_values("gain", ascending=False).reset_index(drop=True)
        top_tune = pair_df.iloc[0]
        improved_cnt = int((pair_df["gain"] > 0).sum())
        bullets.append(
            f"- **Configuration / Tuning Impact:** Improved **{improved_cnt}/{len(pair_df)}** paired model(s); "
            f"**{top_tune['model']}** moved from `{format_metric(primary_metric, top_tune['baseline'])}` to "
            f"`{format_metric(primary_metric, top_tune['tuned'])}` (**{top_tune['rel_gain']:+.2f}%**)."
        )

    # Efficiency / Pareto observation
    if "train_time_s" in work.columns and len(work) >= 2:
        fastest = work.sort_values("train_time_s", ascending=True).iloc[0]
        bullets.append(
            f"- **Speed vs Accuracy Trade-off:** **{fastest['display_name']}** (`{fastest['variant']}`) trained fastest "
            f"(`{float(fastest['train_time_s']):.2f}s`, {primary_metric} = `{format_metric(primary_metric, fastest[primary_metric])}`)."
        )

    deployed_rows = work[work.get("is_deployed", False) == True]  # noqa: E712
    if not deployed_rows.empty:
        dep = deployed_rows.iloc[0]
        bullets.append(
            f"- **Active Deployed Artifact:** **{dep['display_name']}** (`{dep['bundle_id']}`) is currently active for inference "
            f"({primary_metric} = `{format_metric(primary_metric, dep[primary_metric])}`)."
        )

    return "\n".join(bullets)
