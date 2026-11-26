"""Home / Executive Overview page — deployed models, track completion matrix, and quick actions (emoji-free)."""
from __future__ import annotations

import streamlit as st

from app.cache import get_cached_store
from app.components.style import family_pill_html, glass_container, page_header, pill_html, run_page_safely, section_header
from src.evaluation.metrics import format_metric
from src.registry.registry import list_specs


def _render() -> None:
    page_header(
        "Hospitality Demand & Pricing Intelligence",
        "End-to-end machine learning system on the Lisbon City Hotel (H2) PMS dataset: ADR regression, cancellation risk classification, and guest segmentation",
        pill_html("79,330 Bookings · 31 Source Variables", "m1"),
    )

    store = get_cached_store()
    all_bundles = store.list_bundles()
    proj_bundles = [b for b in all_bundles if b.card.source == "project"]
    user_bundles = [b for b in all_bundles if b.card.source == "user"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("H2 Source Bookings", "79,330", help="City Hotel bookings arriving Jul 2015 – Aug 2017.")
    k2.metric("Implemented Models", f"{len(list_specs())}", help="Across Regression, Classification, and Clustering.")
    k3.metric("Saved Project Variants", f"{len(proj_bundles)}", help="Serialized sklearn pipelines + model_card.json metadata.")
    k4.metric("User-Trained Bundles", f"{len(user_bundles)}", help="Custom models persisted in artifacts/user/ across sessions.")

    section_header(
        "Active Deployed Inference Pipelines",
        "Serialized preprocessing, dimensionality-reduction, and estimator pipelines loaded directly from disk with zero startup retraining.",
        kicker="Production Routing",
    )

    dep_cols = st.columns(3, gap="medium")
    task_titles = {
        "regression": ("Task A — ADR Regression", "Forecasts room Average Daily Rate (EUR/night) at reservation entry"),
        "classification": ("Task B — Cancellation Risk", "Predicts binary IsCanceled probability at reservation entry"),
        "clustering": ("Task C — Guest Segmentation", "Unsupervised behavioral cohort discovery (zero outcome labels)"),
    }
    for col, (task_key, (title, subtitle)) in zip(dep_cols, task_titles.items()):
        dep = store.get_deployed(task_key)
        with col:
            with glass_container("frosted"):
                st.markdown(f'<div class="section-kicker">{title}</div>', unsafe_allow_html=True)
                st.caption(subtitle)
                if dep is None:
                    st.warning("No deployed bundle yet — run `build-artifacts` or open **Artifacts & Jobs**.", icon=":material/warning:")
                else:
                    c = dep.card
                    m_name, m_val = c.primary_metric()
                    test_val = c.metrics.get("test", {}).get(m_name)
                    st.markdown(
                        f"**{c.display_name}** (`{c.variant}`) &nbsp;{family_pill_html(c.family)}",
                        unsafe_allow_html=True,
                    )
                    test_txt = f" · **Test {m_name}:** `{format_metric(m_name, test_val)}`" if test_val is not None else ""
                    st.markdown(f"**Val {m_name}:** `{format_metric(m_name, m_val, with_arrow=True)}`{test_txt}")
                    st.caption(f"Criterion: {c.selection_criterion} · `{c.bundle_id}`")

    section_header(
        "Track Readiness & Experimental Coverage",
        "Summary of implemented model families, paired baseline-vs-optimized variants, and dimensionality-reduced pipelines.",
        kicker="Repository Audit",
    )
    track_rows = []
    for task_key in ("regression", "classification", "clustering"):
        specs = list_specs(task_key)
        tbundles = [b for b in proj_bundles if b.card.task == task_key]
        models_with_base = {b.card.model_id for b in tbundles if b.card.variant in ("baseline", "baseline_nopca")}
        models_with_opt = {
            b.card.model_id
            for b in tbundles
            if b.card.variant in ("tuned", "variant_selected", "baseline_pca90")
        }
        pca_lda_cnt = sum(1 for b in tbundles if b.card.feature_space in ("pca", "lda"))
        paired = len(models_with_base & models_with_opt)
        track_rows.append(
            {
                "Analytical Track": task_key.title(),
                "Implemented Models": ", ".join(s.display_name for s in specs),
                "Baseline + Optimized Pairs": f"{paired} / {len(specs)}",
                "PCA / LDA Variants": pca_lda_cnt,
                "Saved Project Bundles": len(tbundles),
                "Milestone Status": "3+ Complete" if paired >= 3 else f"Milestone 1 Verified ({paired} family)",
            }
        )
    st.dataframe(track_rows, width="stretch", hide_index=True)

    section_header(
        "Quick Workflow Navigation",
        "Jump directly into single-reservation scoring, batch CSV evaluation, custom model training, or side-by-side comparison.",
        kicker="Workspaces",
    )
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.page_link("views/predict.py", label="Predict Single Booking", icon=":material/online_prediction:", width="stretch")
    with q2:
        st.page_link("views/batch_evaluate.py", label="Test Model on Uploaded CSV", icon=":material/dataset:", width="stretch")
    with q3:
        st.page_link("views/train_studio.py", label="Train Model on Your Data", icon=":material/model_training:", width="stretch")
    with q4:
        st.page_link("views/compare.py", label="Compare Models & Variants", icon=":material/leaderboard:", width="stretch")


run_page_safely(_render)
