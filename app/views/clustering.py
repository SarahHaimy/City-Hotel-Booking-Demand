"""Clustering Studio page — inspect K-Means variants, k-selection diagnostics, 2D PCA structure, and guest personas (emoji-free)."""
from __future__ import annotations

import streamlit as st

from app.cache import get_cached_store, get_experiments_df
from app.components.bklit_chart import render_bklit_chart
from app.components.metric_tiles import render_metric_tiles
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
from src.evaluation.figures import (
    fig_cluster_profile_heatmap,
    fig_cluster_scatter_2d,
    fig_dendrogram_from_coords,
    fig_grouped_bars,
    fig_k_selection,
)
from src.evaluation.metrics import TASK_METRICS


def _render() -> None:
    page_header(
        "Unsupervised Guest Segmentation & Cluster Profiling",
        "Evaluate internal clustering quality, k-selection elbow & Silhouette curves, 2D PCA projections, and behavioral guest personas",
        pill_html("Strictly Unsupervised · Zero Target Leakage", "deployed"),
    )

    store = get_cached_store()
    bundles = store.list_bundles(task="clustering")
    if not bundles:
        render_empty_artifacts_banner("clustering")
        return

    section_header(
        "Variant & Space Comparison",
        "Internal unsupervised metrics comparing original feature space against PCA-90% variance retention.",
        kicker="1 · Clustering Benchmark",
    )
    df_clu = get_experiments_df(cache_token(), task="clustering", split="train")
    if not df_clu.empty:
        cols = [
            c for c in (
                ["display_name", "variant", "feature_space", "source", "n_clusters", "silhouette", "davies_bouldin", "calinski_harabasz", "noise_pct", "min_cluster_pct", "train_time_s", "n_train_rows"]
            )
            if c in df_clu.columns
        ]
        sub_clu = df_clu[cols].sort_values("silhouette", ascending=False)
        st.dataframe(sub_clu, column_config=format_table_columns(sub_clu), width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            render_bklit_chart(fig_grouped_bars(df_clu, "silhouette", True, "Silhouette Score Across Clustering Runs (higher)"), key="clu_sil_bars")
        with c2:
            render_bklit_chart(fig_grouped_bars(df_clu, "davies_bouldin", False, "Davies-Bouldin Index Across Clustering Runs (lower)"), key="clu_db_bars")

    section_header(
        "Interactive Cluster Structure & Guest Personas",
        "Inspect 2D principal-component separation, standardized centroid profiles, and behavioral cohort summaries.",
        kicker="2 · Cohort Profiling",
    )
    b_map = {
        f"{'[Deployed] ' if b.card.is_deployed else ''}{b.card.display_name} ({b.card.variant}, {b.card.feature_space}) [{b.card.source}]": b
        for b in bundles
    }
    chosen_lbl = st.selectbox("Select Clustering Run to Inspect", options=list(b_map.keys()))
    b_sel = b_map[chosen_lbl]
    tr_m = b_sel.card.metrics.get("train", {})

    render_metric_tiles(tr_m, TASK_METRICS["clustering"])
    if b_sel.card.sampling_info:
        si = b_sel.card.sampling_info
        st.info(
            f"**Documented Representative Sampling (Spec §10):** `{si['strategy']}` — "
            f"`{si['sample_rows']:,}` of `{si['full_rows']:,}` cleaned bookings (`seed={si['seed']}`).",
            icon=":material/info:",
        )

    tuning_df = b_sel.read_csv("tuning_results.csv")
    if tuning_df is None:
        sib_tuned = next((x for x in bundles if x.card.model_id == b_sel.card.model_id and x.read_csv("tuning_results.csv") is not None), None)
        if sib_tuned:
            tuning_df = sib_tuned.read_csv("tuning_results.csv")
    if tuning_df is not None and "k" in tuning_df.columns and tuning_df["k"].nunique() >= 2:
        k_agg = tuning_df.groupby("k", as_index=False).first()
        render_bklit_chart(fig_k_selection(k_agg), key="clu_ksel")

    v1, v2 = st.columns([1.3, 1.2])
    with v1:
        proj = b_sel.read_csv("pca_2d.csv")
        ev2 = b_sel.read_json("pca_2d_variance")
        if proj is not None and not proj.empty:
            render_bklit_chart(
                fig_cluster_scatter_2d(proj, tuple(ev2) if ev2 else None, f"2D PCA Structure — {b_sel.card.display_name} ({b_sel.card.variant})"),
                key="clu_2d_main",
            )
    with v2:
        prof = b_sel.read_csv("cluster_profile.csv")
        if prof is not None and not prof.empty:
            render_bklit_chart(
                fig_cluster_profile_heatmap(
                    prof,
                    ["LeadTime", "TotalNights", "TotalGuests", "ADR", "TotalOfSpecialRequests", "HasChildren", "HasParking", "IsRepeatedGuest"],
                ),
                key="clu_prof_main",
            )

    dend = b_sel.read_json("dendrogram.json")
    if dend:
        render_bklit_chart(fig_dendrogram_from_coords(dend["icoord"], dend["dcoord"]), key="clu_dend_main")

    personas = b_sel.read_json("personas.json") or []
    if personas:
        section_header(
            "Discovered Behavioral Guest Personas",
            "Business-interpretable segment profiles derived from cluster centroids and modal attributes.",
            kicker="3 · Segment Cards",
        )
        pcols = st.columns(min(4, len(personas)), gap="medium")
        for idx, p in enumerate(personas):
            with pcols[idx % len(pcols)]:
                with glass_container("frosted"):
                    st.markdown(f'<div class="section-kicker">Cluster {p["cluster"]}</div>', unsafe_allow_html=True)
                    st.markdown(f"**{p['title']}**")
                    st.caption(p["summary"])
                    if p.get("posthoc_cancel_rate") is not None:
                        st.caption(f"Post-hoc cancel rate *(check only)*: **{p['posthoc_cancel_rate']:.1%}**")

    if prof is not None and not prof.empty:
        with st.expander("Full Cluster Centroid & Mode Table", icon=":material/table_chart:"):
            st.dataframe(prof, width="stretch", hide_index=True)

    cancel_df = b_sel.read_csv("cancel_rate.csv")
    if cancel_df is not None and not cancel_df.empty:
        with st.expander("Post-Hoc Cancellation Propensity Audit (Strictly Post-Hoc — Never Supplied to Clustering)", icon=":material/verified_user:"):
            st.caption(
                "`IsCanceled` is completely excluded from clustering inputs (`FORBIDDEN` assertion). "
                "Inspecting post-hoc cancellation rates verifies whether behavioral guest segments naturally correlate with business risk."
            )
            st.dataframe(cancel_df, width="stretch", hide_index=True)


run_page_safely(_render)
