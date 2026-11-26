"""Tuning Lab page — inspect hyperparameter search spaces, CV optimization trials, and baseline-vs-tuned gains (emoji-free)."""
from __future__ import annotations

import streamlit as st

from app.cache import get_cached_store, get_experiments_df
from app.components.bklit_chart import render_bklit_chart
from app.components.style import glass_container, page_header, pill_html, render_empty_artifacts_banner, run_page_safely, section_header
from app.state import cache_token
from src.config import selection
from src.evaluation.figures import fig_parallel_trials, fig_param_slice, fig_tuning_dumbbell
from src.evaluation.metrics import METRIC_META
from src.registry.registry import list_specs


def _render() -> None:
    page_header(
        "Hyperparameter Optimization Lab",
        "Inspect documented search spaces, optimization criteria, CV trial landscapes, and baseline-to-tuned gains (Spec §11)",
        pill_html("Validation / Train-Fold CV Only", "deployed"),
    )

    store = get_cached_store()
    with glass_container("reeded"):
        c_sel, c_link = st.columns([2.2, 1.3], vertical_alignment="bottom")
        with c_sel:
            task = st.selectbox(
                "Analytical Task",
                options=["regression", "classification", "clustering"],
                format_func=lambda t: t.title(),
                key="tl_task",
            )
        with c_link:
            st.page_link("views/train_studio.py", label="Launch New Search in Train Studio", icon=":material/model_training:", width="stretch")

    with st.expander("Documented Search Spaces & Optimization Metrics", icon=":material/table_chart:", expanded=False):
        space_rows = []
        for s in list_specs(task):
            space_rows.append(
                {
                    "Model": s.display_name,
                    "Family": s.family,
                    "Optimization Metric": s.default_scoring,
                    "Baseline Config": str(s.baseline_params),
                    "Search Space": str(s.search_space),
                }
            )
        st.dataframe(space_rows, width="stretch", hide_index=True)

    df_exp = get_experiments_df(cache_token(), task=task, split=selection(task)["split"])
    if df_exp.empty:
        render_empty_artifacts_banner(task)
        return

    section_header(
        "Baseline vs Selected / Tuned Gain",
        "Paired comparison between default untuned configuration and optimized configuration.",
        kicker="1 · Optimization Delta",
    )
    prim = selection(task)["metric"]
    hib = METRIC_META[prim]["higher_is_better"]
    render_bklit_chart(fig_tuning_dumbbell(df_exp, prim, hib), key="tl_dumbbell")

    tuned_bundles = [
        b for b in store.list_bundles(task=task)
        if b.read_csv("tuning_results.csv") is not None or b.card.tuning_summary is not None
    ]
    if not tuned_bundles:
        st.info(
            "No CV search trial tables (`tuning_results.csv`) found for this task yet. "
            "Launch a **Tune** run in **Train Studio** or run `python main.py tune`.",
            icon=":material/info:",
        )
        return

    section_header(
        "Trial-Level Search Landscape Inspector",
        "Explore cross-validation candidate scores across individual hyperparameter dimensions.",
        kicker="2 · Candidate Trials",
    )
    b_map = {f"{b.card.display_name} [{b.card.bundle_id}]": b for b in tuned_bundles}
    picked_lbl = st.selectbox("Select Tuned Run", options=list(b_map.keys()))
    b_sel = b_map[picked_lbl]
    tsum = b_sel.card.tuning_summary or {}

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Search Strategy", str(tsum.get("method", "grid")).title())
    c2.metric("Optimization Criterion", str(tsum.get("scoring", "—")))
    c3.metric("Evaluated Candidates", f"{tsum.get('n_candidates', '—')}")
    best_sc = tsum.get("best_cv_score")
    c4.metric("Best CV Score", f"{best_sc:.4f}" if isinstance(best_sc, (int, float)) else "—")

    best_params = tsum.get("best_params", b_sel.card.params) or {}
    if best_params:
        with glass_container("solid"):
            st.markdown('<div class="section-kicker">Winning Hyperparameter Set</div>', unsafe_allow_html=True)
            st.dataframe(
                [{"Hyperparameter": str(k), "Selected Value": str(v)} for k, v in best_params.items()],
                width="stretch",
                hide_index=True,
            )

    trials_df = b_sel.read_csv("tuning_results.csv")
    if trials_df is not None and not trials_df.empty:
        param_cols = [c for c in trials_df.columns if c.startswith("param_")]
        score_col = "mean_test_score" if "mean_test_score" in trials_df.columns else trials_df.columns[-1]

        if len(param_cols) >= 2 and len(trials_df) >= 2:
            render_bklit_chart(fig_parallel_trials(trials_df, param_cols, score_col=score_col), key="tl_parcoords")

        if param_cols:
            chosen_p = st.selectbox("Inspect Single Hyperparameter Slice", options=param_cols)
            render_bklit_chart(fig_param_slice(trials_df, chosen_p, score_col=score_col), key="tl_slice")

        st.markdown("##### All Evaluated Search Candidates")
        st.dataframe(trials_df, width="stretch", hide_index=True)


run_page_safely(_render)
