"""Model Catalog page — browse all registered models, filter by task/family/source, and inspect specs + diagnostics."""
from __future__ import annotations

import streamlit as st

from app.cache import get_cached_store
from app.components.model_card import render_bundle_detail_panel, render_spec_summary_card
from app.components.style import glass_container, page_header, pill_html, render_empty_artifacts_banner, run_page_safely, section_header
from src.registry.registry import get_spec, list_specs


def _render() -> None:
    page_header(
        "Model Catalog — Specifications & Empirical Performance",
        "Inspect every implemented model architecture, hyperparameter configuration, validation/CV/test metrics, and diagnostic plots",
        pill_html("Project & User-Trained Models", "m1"),
    )

    store = get_cached_store()

    with glass_container("reeded"):
        st.markdown('<div class="section-kicker">Catalog Filters</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1:
            task_sel = st.selectbox(
                "Analytical Task",
                options=["regression", "classification", "clustering"],
                format_func=lambda t: {
                    "regression": "Task A — ADR Regression",
                    "classification": "Task B — Cancellation Classification",
                    "clustering": "Task C — Guest Clustering",
                }[t],
                key="cat_task_filter",
            )
        specs = list_specs(task_sel)
        families = ["All Families", *sorted({s.family for s in specs})]
        with f2:
            fam_sel = st.selectbox("Model Family", options=families, key="cat_fam_filter")
        with f3:
            src_sel = st.selectbox(
                "Bundle Source",
                options=["All Sources", "Official Project Models", "User-Trained Models Only"],
                key="cat_src_filter",
            )

    task_bundles = store.list_bundles(task=task_sel)
    if not task_bundles:
        render_empty_artifacts_banner(task_sel)

    if src_sel == "User-Trained Models Only":
        user_bundles = [b for b in task_bundles if b.card.source == "user"]
        if not user_bundles:
            st.info("No user-trained bundles saved for this task yet. Train a model in **Train Studio** to see it here.", icon=":material/info:")
            return
        opts = {f"{b.card.run_name or b.card.display_name} [{b.card.bundle_id}]": b for b in user_bundles}
        chosen_u = st.selectbox("Select User-Trained Bundle", options=list(opts.keys()))
        b_sel = opts[chosen_u]
        render_bundle_detail_panel(b_sel, get_spec(b_sel.card.task, b_sel.card.model_id), [b_sel], store)
        return

    filtered_specs = [s for s in specs if fam_sel == "All Families" or s.family == fam_sel]

    active_id = st.session_state.get("catalog_selected_bundle")
    target_bundle = None
    if active_id:
        try:
            cand = store.get_bundle(active_id)
            if cand.card.task == task_sel:
                target_bundle = cand
        except KeyError:
            target_bundle = None
    if target_bundle is None:
        target_bundle = store.get_deployed(task_sel) or (task_bundles[0] if task_bundles else None)

    section_header(
        "Registered Model Families",
        "Summary of saved variants and primary selection metrics for this analytical track.",
        kicker="Architecture Overview",
    )
    grid_cols = st.columns(2, gap="medium")
    for idx, spec in enumerate(filtered_specs):
        sbundles = [
            b for b in task_bundles
            if b.card.model_id == spec.model_id
            and (src_sel != "Official Project Models" or b.card.source == "project")
        ]
        is_open = bool(target_bundle and target_bundle.card.model_id == spec.model_id)
        with grid_cols[idx % 2]:
            picked = render_spec_summary_card(
                spec,
                sbundles,
                on_select_key=f"cat_pick_{spec.task}_{spec.model_id}",
                is_currently_open=is_open,
            )
            if picked is not None:
                target_bundle = picked
                st.session_state["catalog_selected_bundle"] = picked.card.bundle_id
                st.rerun()

    if target_bundle is not None:
        section_header(
            "Specification & Empirical Diagnostics Inspector",
            "Switch variants and inspect pipeline stages, split metrics, interactive curves, and provenance.",
            kicker="Deep Inspection",
        )
        siblings = [b for b in task_bundles if b.card.model_id == target_bundle.card.model_id]
        try:
            spec_obj = get_spec(target_bundle.card.task, target_bundle.card.model_id)
        except KeyError:
            spec_obj = None
        render_bundle_detail_panel(target_bundle, spec_obj, siblings, store)


run_page_safely(_render)
