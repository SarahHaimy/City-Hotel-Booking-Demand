"""Catalog summary cards and full multi-tab bundle detail inspector (emoji-free, Material icons)."""
from __future__ import annotations

import streamlit as st

from app.components.bklit_chart import render_bklit_chart
from app.components.metric_tiles import render_metric_tiles
from app.components.style import family_pill_html, glass_container, pill_html
from app.state import add_to_compare, bump_cache_token
from src.evaluation.figures import (
    fig_calibration_curves,
    fig_cluster_profile_heatmap,
    fig_cluster_scatter_2d,
    fig_confusion_matrix,
    fig_cv_boxplots,
    fig_dendrogram_from_coords,
    fig_feature_weights,
    fig_parity,
    fig_pr_curves,
    fig_residuals,
    fig_roc_curves,
    fig_segment_errors,
    fig_threshold_sweep,
)
from src.evaluation.metrics import (
    TASK_METRICS,
    compute_classification_metrics,
    format_metric,
)
from src.registry.artifacts import ArtifactBundle, ArtifactStore
from src.registry.specs import ModelSpec


def render_spec_summary_card(
    spec: ModelSpec,
    bundles: list[ArtifactBundle],
    on_select_key: str,
    is_currently_open: bool = False,
) -> ArtifactBundle | None:
    by_var = {b.card.variant: b for b in bundles if b.card.source == "project"}
    best_b = (
        by_var.get("tuned")
        or by_var.get("variant_selected")
        or by_var.get("baseline_pca90")
        or by_var.get("baseline")
        or (bundles[0] if bundles else None)
    )
    base_b = by_var.get("baseline") or by_var.get("baseline_nopca")

    with glass_container("frosted"):
        badges = family_pill_html(spec.family)
        if spec.track_origin == "milestone_1":
            badges += pill_html("Implemented", "m1")
        if any(b.card.is_deployed for b in bundles):
            badges += pill_html("Deployed", "deployed")
        if not bundles:
            badges += pill_html("Untrained", "muted")

        st.markdown(f"**{spec.display_name}** &nbsp; {badges}", unsafe_allow_html=True)
        st.caption(spec.description)

        if best_b is not None:
            m_name, m_val = best_b.card.primary_metric()
            delta_txt = ""
            if base_b is not None and base_b.card.bundle_id != best_b.card.bundle_id:
                _, b_val = base_b.card.primary_metric()
                if b_val is not None and m_val is not None:
                    diff = m_val - b_val
                    delta_txt = f"  *(baseline `{format_metric(m_name, b_val)}` -> `{best_b.card.variant}` `{diff:+.3f}`)*"
            st.markdown(
                f"**{m_name}:** `{format_metric(m_name, m_val, with_arrow=True)}`{delta_txt} "
                f"· `{len(bundles)}` variant(s) · `{best_b.card.train_time_s:.2f}s`"
            )
            if is_currently_open:
                st.caption(":material/check_circle: Active in the specification & diagnostics inspector below.")
            elif st.button("Inspect Specifications & Diagnostics", key=on_select_key, icon=":material/analytics:", width="stretch"):
                return best_b
        else:
            st.caption(f"Cost: `{spec.cost_hint}` · Ready to train in **Train Studio**.")
    return None


def render_bundle_detail_panel(
    bundle: ArtifactBundle,
    spec: ModelSpec | None,
    sibling_bundles: list[ArtifactBundle],
    store: ArtifactStore,
) -> None:
    card = bundle.card
    badges = family_pill_html(card.family)
    if card.is_deployed:
        badges += pill_html("Active Deployed", "deployed")
    if card.source == "user":
        badges += pill_html("User-Trained", "user")
    if card.feature_space != "original":
        badges += pill_html(f"Space: {card.feature_space.upper()}", "space")

    st.markdown(f"### {card.display_name} — `{card.variant}` {badges}", unsafe_allow_html=True)
    if card.run_name and card.source == "user":
        st.caption(f"Run label: **{card.run_name}** · Bundle ID: `{card.bundle_id}`")

    def _var_label(b: ArtifactBundle) -> str:
        if b.card.source == "user":
            return f"{b.card.variant} ({b.card.feature_space}, user:{b.card.bundle_id[-6:]})"
        return f"{b.card.variant} ({b.card.feature_space}, {b.card.source})"

    if len(sibling_bundles) > 1:
        options = {_var_label(b): b for b in sibling_bundles}
        current_lbl = _var_label(bundle)
        chosen_lbl = st.segmented_control(
            "Saved Variants for this Model Family",
            options=list(options.keys()),
            default=current_lbl if current_lbl in options else list(options.keys())[0],
            key=f"var_seg_{card.model_id}_{card.task}",
        )
        if chosen_lbl and chosen_lbl in options:
            bundle = options[chosen_lbl]
            card = bundle.card

    baseline_ref = next(
        (b for b in sibling_bundles if b.card.variant == "baseline" and b.card.feature_space == "original"),
        None,
    )

    t_spec, t_perf, t_diag, t_prov = st.tabs(
        [
            ":material/account_tree: Architecture & Specs",
            ":material/speed: Evaluation Metrics",
            ":material/monitoring: Visual Diagnostics",
            ":material/verified: Provenance & Actions",
        ]
    )

    with t_spec:
        if spec:
            st.markdown(f"**Model Summary:** {spec.description}")
            st.markdown(f"**Assumptions & Trade-offs:** {spec.assumptions_and_pros}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Pipeline Stages")
            steps_flow = " → ".join(f"`{s}`" for s in card.preprocessing_steps)
            st.info(f"**Raw Booking Frame** → {steps_flow} → **Prediction**", icon=":material/route:")
            if card.pca_info and card.pca_info.get("components_kept"):
                vk = card.pca_info.get("variance_kept")
                vk_str = f" ({vk:.1%} variance retained)" if vk is not None else ""
                st.caption(f"Dimensionality reduction: **{card.pca_info['components_kept']} component(s)**{vk_str}")
            st.markdown(f"**Required Raw Inputs ({len(card.inputs)}):** " + ", ".join(f"`{c}`" for c in card.inputs))

        with c2:
            st.markdown("##### Hyperparameter Configuration")
            base_params = baseline_ref.card.params if baseline_ref else (spec.baseline_params if spec else {})
            rows = []
            all_keys = sorted(set(card.params.keys()) | set(base_params.keys()))
            for k in all_keys:
                cur_v = card.params.get(k, "—")
                base_v = base_params.get(k, "—")
                rows.append(
                    {
                        "Parameter": k,
                        "Active Value": str(cur_v),
                        "Baseline Default": str(base_v),
                        "Modified": "Modified" if str(cur_v) != str(base_v) else "",
                    }
                )
            st.dataframe(rows, width="stretch", hide_index=True)

        st.caption(
            "**Verified Excluded Leakage Columns:** "
            + ", ".join(f"`{c}`" for c in card.forbidden_checked)
        )

    with t_perf:
        metric_keys = TASK_METRICS.get(card.task, [])
        if card.task in ("regression", "classification"):
            st.markdown("##### Validation Split Performance *(Model Selection Split)*")
            ref_val = (
                baseline_ref.card.metrics.get("val")
                if baseline_ref and baseline_ref.card.bundle_id != card.bundle_id
                else None
            )
            render_metric_tiles(card.metrics.get("val", {}), metric_keys, reference=ref_val)

            cv_json = bundle.read_json("cv_scores.json")
            if cv_json:
                st.markdown(
                    f"##### {cv_json.get('folds', 5)}-Fold Training Cross-Validation: "
                    f"`{cv_json['metric']} = {cv_json['mean']:.4f} ± {cv_json['std']:.4f}`"
                )
                render_bklit_chart(
                    fig_cv_boxplots({f"{card.display_name} ({card.variant})": cv_json["fold_scores"]}, cv_json["metric"]),
                    key=f"cv_box_{card.bundle_id}",
                )

            if "test" in card.metrics and card.metrics["test"]:
                st.markdown("##### Unseen Holdout Test Split *(Final Read-Only Evaluation)*")
                ref_test = (
                    baseline_ref.card.metrics.get("test")
                    if baseline_ref and baseline_ref.card.bundle_id != card.bundle_id
                    else None
                )
                render_metric_tiles(card.metrics["test"], metric_keys, reference=ref_test)
        else:
            st.markdown("##### Internal Unsupervised Clustering Metrics")
            ref_tr = (
                baseline_ref.card.metrics.get("train")
                if baseline_ref and baseline_ref.card.bundle_id != card.bundle_id
                else None
            )
            render_metric_tiles(card.metrics.get("train", {}), metric_keys, reference=ref_tr)
            tr_m = card.metrics.get("train", {})
            st.caption(
                f"Discovered clusters: **{tr_m.get('n_clusters', '—')}** · "
                f"Rows clustered: **{card.n_train_rows:,}** · "
                f"Noise share: **{tr_m.get('noise_pct', 0.0):.1%}** · "
                f"Smallest cluster share: **{tr_m.get('min_cluster_pct', 0.0):.1%}**"
            )

    with t_diag:
        if card.task == "regression":
            preds_df = bundle.read_csv("predictions_val.csv")
            if preds_df is not None and {"y_true", "y_pred"}.issubset(preds_df.columns):
                d1, d2 = st.columns(2)
                with d1:
                    render_bklit_chart(
                        fig_parity(preds_df["y_true"], preds_df["y_pred"], f"{card.display_name} ({card.variant})"),
                        key=f"par_{card.bundle_id}",
                    )
                with d2:
                    render_bklit_chart(
                        fig_residuals(preds_df["y_true"], preds_df["y_pred"], f"{card.display_name} ({card.variant})"),
                        key=f"res_{card.bundle_id}",
                    )
            fw = bundle.read_csv("feature_weights.csv")
            if fw is not None and not fw.empty:
                render_bklit_chart(
                    fig_feature_weights(fw, title=f"Top Feature Drivers — {card.display_name}"),
                    key=f"fw_{card.bundle_id}",
                )

        elif card.task == "classification":
            preds_df = bundle.read_csv("predictions_val.csv")
            if preds_df is not None and {"y_true", "y_proba"}.issubset(preds_df.columns):
                default_thr = float(card.threshold or 0.50)
                live_thr = st.slider(
                    "Interactive Decision Threshold (recomputes Confusion Matrix & F1 on validation probabilities)",
                    min_value=0.10, max_value=0.90, value=default_thr, step=0.05,
                    key=f"thr_sl_{card.bundle_id}",
                )
                live_m = compute_classification_metrics(
                    preds_df["y_true"], None, y_proba=preds_df["y_proba"], threshold=live_thr
                )
                render_metric_tiles(live_m, ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"])

                c1, c2 = st.columns(2)
                with c1:
                    render_bklit_chart(
                        fig_confusion_matrix(
                            live_m["TN"], live_m["FP"], live_m["FN"], live_m["TP"],
                            title=f"Validation Confusion Matrix @ Threshold {live_thr:.2f}",
                        ),
                        key=f"cm_{card.bundle_id}",
                    )
                with c2:
                    sweep_df = bundle.read_csv("threshold_sweep.csv")
                    if sweep_df is not None:
                        render_bklit_chart(
                            fig_threshold_sweep(sweep_df, active_threshold=live_thr),
                            key=f"sw_{card.bundle_id}",
                        )

                label_key = f"{card.display_name} ({card.variant})"
                curve_map = {label_key: (preds_df["y_true"].to_numpy(), preds_df["y_proba"].to_numpy())}
                r1, r2, r3 = st.columns(3)
                with r1:
                    render_bklit_chart(fig_roc_curves(curve_map), key=f"roc_{card.bundle_id}")
                with r2:
                    render_bklit_chart(fig_pr_curves(curve_map), key=f"pr_{card.bundle_id}")
                with r3:
                    render_bklit_chart(fig_calibration_curves(curve_map), key=f"cal_{card.bundle_id}")

            seg_df = bundle.read_csv("segment_errors.csv")
            if seg_df is not None and not seg_df.empty:
                render_bklit_chart(fig_segment_errors(seg_df), key=f"seg_{card.bundle_id}")
                with st.expander("View False-Alarm Audit Table by MarketSegment", icon=":material/table_chart:"):
                    st.dataframe(seg_df, width="stretch", hide_index=True)

            fw = bundle.read_csv("feature_weights.csv")
            if fw is not None and not fw.empty:
                render_bklit_chart(
                    fig_feature_weights(fw, title=f"Top Cancellation Risk Factors — {card.display_name}"),
                    key=f"fw_clf_{card.bundle_id}",
                )

        else:
            proj = bundle.read_csv("pca_2d.csv")
            ev2 = bundle.read_json("pca_2d_variance")
            if proj is not None and not proj.empty:
                render_bklit_chart(
                    fig_cluster_scatter_2d(proj, tuple(ev2) if ev2 else None, f"2D PCA Segment Projection — {card.display_name}"),
                    key=f"clu2d_{card.bundle_id}",
                )
            prof = bundle.read_csv("cluster_profile.csv")
            if prof is not None and not prof.empty:
                render_bklit_chart(
                    fig_cluster_profile_heatmap(
                        prof, ["LeadTime", "TotalNights", "TotalGuests", "ADR", "TotalOfSpecialRequests", "HasChildren", "HasParking", "IsRepeatedGuest"]
                    ),
                    key=f"cluprof_{card.bundle_id}",
                )
            dend = bundle.read_json("dendrogram.json")
            if dend:
                render_bklit_chart(
                    fig_dendrogram_from_coords(dend["icoord"], dend["dcoord"]),
                    key=f"dend_{card.bundle_id}",
                )

    with t_prov:
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            if st.button("Set as Deployed Model", key=f"dep_btn_{card.bundle_id}", icon=":material/check_circle:", disabled=card.is_deployed, width="stretch"):
                store.set_deployed(card.task, card.bundle_id)
                bump_cache_token()
                st.success(f"`{card.bundle_id}` is now the active deployed {card.task} model.", icon=":material/check_circle:")
                st.rerun()
        with a2:
            if st.button("Add to Comparison", key=f"cmp_btn_{card.bundle_id}", icon=":material/add:", width="stretch"):
                add_to_compare(card.bundle_id)
                st.toast(f"Added {card.display_name} ({card.variant}) to comparison basket.")
        with a3:
            zip_bytes = store.export_bundle_zip(card.bundle_id)
            st.download_button(
                "Download Bundle (.zip)",
                data=zip_bytes,
                file_name=f"{card.bundle_id.replace('/', '_')}.zip",
                mime="application/zip",
                icon=":material/download:",
                width="stretch",
                key=f"dl_zip_{card.bundle_id}",
            )
        with a4:
            if card.source == "user":
                if st.button("Delete User Bundle", key=f"del_btn_{card.bundle_id}", icon=":material/delete:", width="stretch"):
                    store.delete_bundle(card.bundle_id)
                    bump_cache_token()
                    st.rerun()

        p1, p2 = st.columns(2)
        with p1:
            st.markdown(
                f"- **Bundle ID:** `{card.bundle_id}`\n"
                f"- **Created:** `{card.created_at}`\n"
                f"- **Training Time:** `{card.train_time_s:.3f}s` on `{card.n_train_rows:,}` rows\n"
                f"- **Selection Criterion:** {card.selection_criterion}\n"
                f"- **Random Seed:** `{card.seed}`"
            )
        with p2:
            samp_txt = (
                f"`{card.sampling_info['strategy']}` (`{card.sampling_info['sample_rows']:,}` rows)"
                if card.sampling_info
                else "Full split (no subsampling)"
            )
            st.markdown(
                f"- **scikit-learn:** `{card.sklearn_version}` · **Python:** `{card.python_version}`\n"
                f"- **Git Commit:** `{card.git_commit}`\n"
                f"- **Dataset:** `{card.data_version.get('dataset_name', 'H2')}` (`sha256={card.data_version.get('sha256_train', '—')}`)\n"
                f"- **Sampling:** {samp_txt}"
            )
        with st.expander("Raw `model_card.json`", icon=":material/code:"):
            st.json(card.to_dict())
