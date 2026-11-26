"""PCA & LDA Dimensionality Reduction page — scree variance, loadings, with/without PCA, and Original vs PCA vs LDA."""
from __future__ import annotations

import streamlit as st

from app.cache import get_cached_store, get_experiments_df, get_pca_lda_data
from app.components.bklit_chart import render_bklit_chart
from app.components.style import format_table_columns, glass_container, page_header, pill_html, run_page_safely, section_header
from app.state import cache_token
from src.evaluation.figures import (
    fig_grouped_bars,
    fig_lda_projection,
    fig_pca_loadings,
    fig_pca_scree,
    fig_roc_curves,
)


def _render() -> None:
    page_header(
        "PCA & LDA Dimensionality Reduction Analysis",
        "Explained-variance retention justification, principal component loadings, with-vs-without PCA benchmarks, and Original vs PCA vs LDA classifier comparisons (Spec §9)",
        pill_html("Fitted Strictly Inside Training Pipelines", "space"),
    )

    diag = get_pca_lda_data()
    clf_d = diag["classification"]
    clu_d = diag["clustering"]

    t_scree, t_sup_pca, t_lda = st.tabs(
        [
            ":material/ssid_chart: PCA Scree & Component Loadings",
            ":material/compare: Models With vs Without PCA",
            ":material/scatter_plot: Classification: Original vs PCA vs LDA",
        ]
    )

    with t_scree:
        with glass_container("reeded"):
            rep_choice = st.radio(
                "Prepared Feature Matrix to Inspect",
                options=["Classification Feature Matrix (Standardized + One-Hot)", "Clustering Feature Matrix (Log1p + Standardized + One-Hot)"],
                horizontal=True,
            )
        active = clf_d if rep_choice.startswith("Classification") else clu_d
        k1, k2, k3 = st.columns(3)
        k1.metric("Encoded Input Dimensions", f"{active['n_raw_encoded']}")
        k2.metric("Components for ≥90% Variance", f"{active['k90']} PCs")
        k3.metric("Exact Variance Retained", f"{active['variance_at_k90']:.2%}")

        st.info(
            f"**Component Count Justification:** Retaining **90% cumulative variance** compresses "
            f"`{active['n_raw_encoded']}` standardized features down to **{active['k90']} orthogonal principal components** "
            f"(`{active['variance_at_k90']:.2%}` variance kept), eliminating multicollinearity across one-hot categories while preserving dominant booking structure.",
            icon=":material/verified:",
        )
        c1, c2 = st.columns(2)
        with c1:
            render_bklit_chart(fig_pca_scree(active["explained_variance_ratio"], chosen_k=active["k90"]), key="pca_scree")
        with c2:
            render_bklit_chart(fig_pca_loadings(active["loadings"]), key="pca_loadings")

    with t_sup_pca:
        df_clu = get_experiments_df(cache_token(), task="clustering", split="train")
        if not df_clu.empty and set(df_clu["feature_space"]) >= {"original", "pca"}:
            section_header(
                "Implemented K-Means Clustering: Original Space vs PCA 90%",
                "Comparison between `baseline_nopca` (all standardized engineered features) and `baseline_pca90` (7 principal components).",
                kicker="Unsupervised PCA Impact",
            )
            clu_sub = df_clu[df_clu["variant"].isin(["baseline_nopca", "baseline_pca90"])].copy()
            clu_sub["variant"] = clu_sub["feature_space"].str.upper()
            render_bklit_chart(
                fig_grouped_bars(clu_sub, "silhouette", True, "K-Means Silhouette: Original Space vs PCA (90% -> 7 PCs)"),
                key="pca_clu_bars",
            )
            cols = [c for c in ["display_name", "feature_space", "pca_components", "pca_variance_kept", "silhouette", "davies_bouldin", "calinski_harabasz", "train_time_s"] if c in clu_sub.columns]
            t_view = clu_sub[cols]
            st.dataframe(t_view, column_config=format_table_columns(t_view), width="stretch", hide_index=True)

        any_sup_pca = False
        for tname, metric, hib in (("regression", "RMSE", False), ("classification", "F1", True)):
            df_t = get_experiments_df(cache_token(), task=tname, split="val")
            if df_t.empty:
                continue
            pca_models = set(df_t[df_t["feature_space"] == "pca"]["model_id"].unique())
            sub = df_t[
                df_t["model_id"].isin(pca_models)
                & (df_t["feature_space"].isin(["original", "pca"]))
            ].copy()
            if not sub.empty:
                any_sup_pca = True
                section_header(f"{tname.title()} ({metric})", kicker="Supervised PCA Impact")
                sub["variant"] = sub["feature_space"].str.upper()
                render_bklit_chart(fig_grouped_bars(sub, metric, hib, f"{tname.title()}: Original vs PCA (90% Variance)"), key=f"pca_sup_{tname}")
                cols = [c for c in ["display_name", "feature_space", "pca_components", "pca_variance_kept", metric, "train_time_s"] if c in sub.columns]
                s_view = sub[cols]
                st.dataframe(s_view, column_config=format_table_columns(s_view), width="stretch", hide_index=True)
        if not any_sup_pca:
            st.info(
                "Supervised PCA runs have not been committed in Milestone 1 yet (only K-Means PCA-90% is in the codebase). "
                "You can train any supervised model with `PCA Reduced Space` in **Train Studio** and it will appear here automatically.",
                icon=":material/info:",
            )

    with t_lda:
        section_header(
            "Original Features vs PCA (90%) vs Supervised LDA (n_components = 1)",
            "For binary classification (C = 2), Fisher's Linear Discriminant Analysis projects features onto the single most discriminative 1D axis (LD1).",
            kicker="Discriminant Subspace",
        )
        render_bklit_chart(fig_lda_projection(clf_d["lda_val_z"], clf_d["lda_val_y"]), key="lda_proj")

        df_clf = get_experiments_df(cache_token(), task="classification", split="val")
        has_lda = not df_clf.empty and ("lda" in set(df_clf["feature_space"]))
        if has_lda:
            lda_models = set(df_clf[df_clf["feature_space"] == "lda"]["model_id"].unique())
            comp = df_clf[df_clf["model_id"].isin(lda_models)].copy()
            if not comp.empty:
                comp["variant"] = comp["feature_space"].str.upper()
                c1, c2 = st.columns(2)
                with c1:
                    render_bklit_chart(fig_grouped_bars(comp, "F1", True, "F1-Score: Original vs PCA vs LDA"), key="lda_f1_bars")
                with c2:
                    store = get_cached_store()
                    roc_map = {}
                    for _, r in comp.iterrows():
                        try:
                            b = store.get_bundle(str(r["bundle_id"]))
                            pdf = b.read_csv("predictions_val.csv")
                            if pdf is not None and {"y_true", "y_proba"}.issubset(pdf.columns):
                                roc_map[f"{b.card.display_name} [{b.card.feature_space.upper()}]"] = (
                                    pdf["y_true"].to_numpy(),
                                    pdf["y_proba"].to_numpy(),
                                )
                        except Exception:
                            pass
                    if roc_map:
                        render_bklit_chart(fig_roc_curves(roc_map), key="lda_roc")

                cols = [c for c in ["display_name", "feature_space", "n_features_out", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "train_time_s"] if c in comp.columns]
                l_view = comp[cols]
                st.dataframe(l_view, column_config=format_table_columns(l_view), width="stretch", hide_index=True)
        else:
            st.info(
                "No LDA-transformed classifier bundle is pre-loaded yet (LDA experiments are scheduled for Milestone 2). "
                "Select `LDA Supervised Projection` in **Train Studio** to train and compare an LDA classifier live.",
                icon=":material/info:",
            )


run_page_safely(_render)
