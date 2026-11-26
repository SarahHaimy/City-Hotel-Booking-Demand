"""Compare Models page — cross-family leaderboard, auto-generated analytical summary, Bklit visual comparisons & exports."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.cache import get_cached_store, get_experiments_df
from app.components.bklit_chart import render_bklit_chart
from app.components.style import (
    format_table_columns,
    glass_container,
    page_header,
    pill_html,
    render_empty_artifacts_banner,
    run_page_safely,
    section_header,
)
from app.state import cache_token
from src.config import selection
from src.evaluation.figures import (
    fig_calibration_curves,
    fig_classification_radar,
    fig_cv_boxplots,
    fig_grouped_bars,
    fig_pareto_time_vs_score,
    fig_pr_curves,
    fig_residual_overlay,
    fig_roc_curves,
    fig_tuning_dumbbell,
)
from src.evaluation.metrics import METRIC_META, TASK_METRICS
from src.evaluation.report import build_comparison_html, build_comparison_markdown
from src.evaluation.summary import generate_comparison_summary


def _highlight_best_dark(s: pd.Series, higher_is_better: bool) -> list[str]:
    valid = pd.to_numeric(s, errors="coerce")
    if valid.dropna().empty:
        return [""] * len(s)
    target = valid.max() if higher_is_better else valid.min()
    return [
        "background-color: #F3EADB; color: #6E5229; font-weight: 600;"
        if pd.notna(v) and np.isclose(v, target)
        else ""
        for v in valid
    ]


def _render() -> None:
    page_header(
        "Model Comparison Leaderboard & Visual Analytics",
        "Side-by-side comparison across implemented models, baseline vs hyperparameter-tuned variants, feature spaces, and user-trained runs",
        pill_html("Text Narrative + Interactive Bklit Charts", "m1"),
    )

    store = get_cached_store()
    df_all = get_experiments_df(cache_token())
    if df_all.empty:
        render_empty_artifacts_banner()
        return

    with st.expander("Cross-Task Summary — All Implemented Models & Variants", icon=":material/view_list:", expanded=False):
        cross_rows = []
        for t_key in ("regression", "classification", "clustering"):
            for b in store.list_bundles(task=t_key):
                m_name, m_val = b.card.primary_metric()
                cross_rows.append(
                    {
                        "Task": t_key.title(),
                        "Model": b.card.display_name,
                        "Variant": b.card.variant,
                        "Feature Space": b.card.feature_space,
                        "Source": b.card.source,
                        "Deployed": "Active" if b.card.is_deployed else "",
                        "Primary Metric": m_name,
                        "Score": round(float(m_val), 4) if m_val is not None else None,
                        "Train Time (s)": round(b.card.train_time_s, 3),
                        "Train Rows": b.card.n_train_rows,
                    }
                )
        if cross_rows:
            st.dataframe(pd.DataFrame(cross_rows), width="stretch", hide_index=True)

    with glass_container("reeded"):
        st.markdown('<div class="section-kicker">Comparison Scope & Filters</div>', unsafe_allow_html=True)
        f1, f2, f3, f4, f5 = st.columns([1.3, 1.2, 1.2, 1.2, 1.2])
        with f1:
            task = st.selectbox(
                "Task",
                options=["regression", "classification", "clustering"],
                format_func=lambda t: t.title(),
                key="cmp_task",
            )
        df_task = df_all[df_all["task"] == task].copy()
        if df_task.empty:
            render_empty_artifacts_banner(task)
            return

        avail_splits = [s for s in ("val", "test", "train", "cv") if s in df_task["split"].unique()]
        default_split = selection(task)["split"]
        with f2:
            split_sel = st.selectbox(
                "Evaluation Split",
                options=avail_splits,
                index=avail_splits.index(default_split) if default_split in avail_splits else 0,
                key="cmp_split",
            )
        metrics_avail = [m for m in TASK_METRICS[task] if m in df_task.columns]
        default_metric = selection(task)["metric"]
        with f3:
            primary_metric = st.selectbox(
                "Primary Metric",
                options=metrics_avail,
                index=metrics_avail.index(default_metric) if default_metric in metrics_avail else 0,
                key="cmp_metric",
            )
        with f4:
            src_filter = st.selectbox("Source", options=["All (Project + User)", "Project Only", "User Only"], key="cmp_src")
        with f5:
            fs_opts = ["All Spaces", *sorted(df_task["feature_space"].dropna().unique().tolist())]
            fs_filter = st.selectbox("Feature Space", options=fs_opts, key="cmp_fs")

        work = df_task[df_task["split"] == split_sel].copy()
        if src_filter == "Project Only":
            work = work[work["source"] == "project"]
        elif src_filter == "User Only":
            work = work[work["source"] == "user"]
        if fs_filter != "All Spaces":
            work = work[work["feature_space"] == fs_filter]

        all_models = sorted(work["display_name"].unique().tolist())
        picked_models = st.multiselect("Filter Model Families", options=all_models, default=all_models, key="cmp_models")

    work = work[work["display_name"].isin(picked_models)]

    if work.empty:
        st.info("No runs match the active filters.", icon=":material/info:")
        return

    hib = bool(METRIC_META.get(primary_metric, {}).get("higher_is_better", True))
    work = work.sort_values(primary_metric, ascending=not hib).reset_index(drop=True)
    work.insert(0, "Rank", np.arange(1, len(work) + 1))

    section_header(
        "Executive Analytical Summary",
        "Auto-synthesized performance delta narrative across ranked variants.",
        kicker="1 · Synthesis",
    )
    narrative_md = generate_comparison_summary(work, task=task, primary_metric=primary_metric, split_label=split_sel)
    with glass_container("solid"):
        st.markdown(narrative_md)

    section_header(
        f"Multi-Metric Leaderboard ({split_sel.upper()} Split)",
        "Best value in every metric column is highlighted in Champagne Bronze.",
        kicker="2 · Quantitative Benchmark",
    )
    display_cols = [
        c for c in (
            ["Rank", "display_name", "family", "variant", "feature_space", "source", "is_deployed"]
            + metrics_avail
            + ["train_time_s", "n_train_rows"]
        )
        if c in work.columns
    ]
    table_df = work[display_cols].copy()

    styler = table_df.style
    for m_col in metrics_avail:
        if m_col in table_df.columns:
            m_hib = bool(METRIC_META.get(m_col, {}).get("higher_is_better", True))
            styler = styler.apply(_highlight_best_dark, higher_is_better=m_hib, subset=[m_col])
    st.dataframe(styler, column_config=format_table_columns(table_df), width="stretch", hide_index=True)

    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button(
            "Export Leaderboard (.csv)",
            data=table_df.to_csv(index=False).encode("utf-8"),
            file_name=f"comparison_{task}_{split_sel}.csv",
            mime="text/csv",
            icon=":material/download:",
            width="stretch",
        )
    with e2:
        md_report = build_comparison_markdown(work, task, split_sel, primary_metric, narrative_md)
        st.download_button(
            "Export Comparison Report (.md)",
            data=md_report.encode("utf-8"),
            file_name=f"comparison_report_{task}.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
        )
    with e3:
        html_report = build_comparison_html(work, task, split_sel, primary_metric, narrative_md)
        st.download_button(
            "Export Styled Report (.html)",
            data=html_report.encode("utf-8"),
            file_name=f"comparison_report_{task}.html",
            mime="text/html",
            icon=":material/download:",
            width="stretch",
        )

    section_header(
        "Visual Comparison Analytics",
        "Interactive Bklit comparisons across variants, runtime trade-offs, CV fold stability, and diagnostic curves.",
        kicker="3 · Diagnostics",
    )
    tabs = st.tabs(
        [
            ":material/bar_chart: Grouped Bars & Tuning Gain",
            ":material/timer: Accuracy vs Training Time",
            ":material/candlestick_chart: Cross-Validation Stability",
            ":material/stacked_line_chart: Task-Specific Curve Overlays",
        ]
    )

    with tabs[0]:
        render_bklit_chart(fig_grouped_bars(work, primary_metric, hib), key="cmp_bars")
        render_bklit_chart(fig_tuning_dumbbell(work, primary_metric, hib), key="cmp_dumb")

    with tabs[1]:
        render_bklit_chart(fig_pareto_time_vs_score(work, primary_metric, hib), key="cmp_pareto")

    with tabs[2]:
        cv_map: dict[str, list[float]] = {}
        cv_metric_lbl = primary_metric
        for bid in work["bundle_id"]:
            try:
                b = store.get_bundle(str(bid))
            except KeyError:
                continue
            cv_j = b.read_json("cv_scores.json")
            if cv_j and "fold_scores" in cv_j:
                cv_map[f"{b.card.display_name} ({b.card.variant})"] = cv_j["fold_scores"]
                cv_metric_lbl = cv_j.get("metric", primary_metric)
        if cv_map:
            render_bklit_chart(fig_cv_boxplots(cv_map, cv_metric_lbl), key="cmp_cvbox")
        else:
            st.info("Cross-validation fold distributions apply to supervised tasks.", icon=":material/info:")

    with tabs[3]:
        pred_file = "predictions_test.csv" if split_sel == "test" else "predictions_val.csv"
        if task == "regression":
            residuals_map: dict[str, np.ndarray] = {}
            for bid in work["bundle_id"].head(6):
                try:
                    b = store.get_bundle(str(bid))
                except KeyError:
                    continue
                pdf = b.read_csv(pred_file)
                if pdf is not None and {"y_true", "y_pred"}.issubset(pdf.columns):
                    residuals_map[f"{b.card.display_name} ({b.card.variant})"] = (pdf["y_true"] - pdf["y_pred"]).to_numpy()
            if residuals_map:
                render_bklit_chart(fig_residual_overlay(residuals_map), key="cmp_res_ov")

        elif task == "classification":
            curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for bid in work["bundle_id"].head(8):
                try:
                    b = store.get_bundle(str(bid))
                except KeyError:
                    continue
                pdf = b.read_csv(pred_file)
                if pdf is not None and {"y_true", "y_proba"}.issubset(pdf.columns):
                    curves[f"{b.card.display_name} ({b.card.variant})"] = (
                        pdf["y_true"].to_numpy(),
                        pdf["y_proba"].to_numpy(),
                    )
            if curves:
                c1, c2 = st.columns(2)
                with c1:
                    render_bklit_chart(fig_roc_curves(curves), key="cmp_roc")
                with c2:
                    render_bklit_chart(fig_pr_curves(curves), key="cmp_pr")
                r1, r2 = st.columns(2)
                with r1:
                    render_bklit_chart(fig_calibration_curves(curves), key="cmp_cal")
                with r2:
                    radar_df = work.head(6).copy()
                    radar_df["label"] = radar_df["display_name"] + " (" + radar_df["variant"] + ")"
                    render_bklit_chart(fig_classification_radar(radar_df), key="cmp_radar")

        else:
            for m_other in ("davies_bouldin", "calinski_harabasz"):
                if m_other in work.columns:
                    render_bklit_chart(
                        fig_grouped_bars(work, m_other, METRIC_META[m_other]["higher_is_better"]),
                        key=f"cmp_clu_{m_other}",
                    )


run_page_safely(_render)
