"""Batch Prediction & Holdout Evaluation — test any model(s) on user-uploaded CSV/XLSX or project splits (emoji-free)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.cache import get_cached_store
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
from app.components.uploader import render_data_uploader
from src.data.validation import ensure_task_splits
from src.evaluation.figures import (
    fig_confusion_matrix,
    fig_parity,
    fig_residuals,
    fig_roc_curves,
    fig_segment_errors,
)
from src.evaluation.metrics import TASK_METRICS
from src.inference.predict import predict_batch_dataframe
from src.training.service import evaluate_bundle_on_data


def _render() -> None:
    page_header(
        "Batch Prediction & Custom Dataset Evaluation",
        "Pick any project or user-trained model to score unlabeled bookings OR evaluate accuracy and diagnostics on labeled datasets",
        pill_html("CSV / XLSX + Shift Detection", "m1"),
    )

    store = get_cached_store()
    if not store.list_bundles():
        render_empty_artifacts_banner()
        return

    with glass_container("reeded"):
        mode = st.radio(
            "Testing Mode",
            options=[
                "1. Upload Custom Dataset (Batch Predict or Labeled Evaluation)",
                "2. Evaluate Models on Project Validation / Final Test Splits",
            ],
            horizontal=True,
        )

    if mode.startswith("1."):
        with glass_container("frosted"):
            st.markdown('<div class="section-kicker">Model & Task Selection</div>', unsafe_allow_html=True)
            c_task, c_mods = st.columns([1.2, 2.2])
            with c_task:
                task = st.selectbox(
                    "Primary Task to Validate & Test",
                    options=["regression", "classification", "clustering"],
                    format_func=lambda t: {
                        "regression": "Task A — ADR Regression",
                        "classification": "Task B — Cancellation Classification",
                        "clustering": "Task C — Guest Clustering",
                    }[t],
                )
            candidates = [
                b for b in store.list_bundles(task=task)
                if (task != "clustering" or b.card.supports_predict_new)
            ]
            if not candidates:
                st.warning(f"No `{task}` bundles available.", icon=":material/warning:")
                return

            label_map = {
                f"{'[Deployed] ' if b.card.is_deployed else ''}{b.card.display_name} [{b.card.variant}, {b.card.feature_space}, {b.card.source}]": b
                for b in candidates
            }
            with c_mods:
                selected_labels = st.multiselect(
                    "Select Model(s) to Test",
                    options=list(label_map.keys()),
                    default=[list(label_map.keys())[0]],
                )
        if not selected_labels:
            st.info("Select at least one model above.", icon=":material/info:")
            return

        chosen_bundles = [label_map[k] for k in selected_labels]
        req_union = sorted({col for b in chosen_bundles for col in b.card.inputs})
        forb_union = sorted({col for b in chosen_bundles for col in b.card.forbidden_checked})

        section_header("Upload or Select Validation Dataset", kicker="Step 2 · Data Input")
        df_up, report, ds_label = render_data_uploader(
            key_prefix=f"batch_{task}",
            task=task,
            required_inputs=req_union,
            forbidden_inputs=forb_union,
            require_target=False,
        )
        if df_up is None or report is None or not report.ok:
            return

        target_col = "ADR" if task == "regression" else ("IsCanceled" if task == "classification" else None)
        has_ground_truth = bool(target_col and target_col in df_up.columns and df_up[target_col].notna().all())

        section_header(
            f"Batch Predictions on {ds_label}",
            f"Showing first 25 of {len(df_up):,} scored reservations.",
            kicker="Step 3 · Scored Output",
        )
        primary_b = chosen_bundles[0]
        scored_df = predict_batch_dataframe(
            df_up,
            reg_bundle_id=primary_b.card.bundle_id if task == "regression" else None,
            clf_bundle_id=primary_b.card.bundle_id if task == "classification" else None,
            clu_bundle_id=primary_b.card.bundle_id if task == "clustering" else None,
            store=store,
        )
        st.dataframe(scored_df.head(25), width="stretch")
        st.download_button(
            "Download Full Predictions (.csv)",
            data=scored_df.to_csv(index=False).encode("utf-8"),
            file_name=f"predictions_{task}_{ds_label.replace(' ', '_')}.csv",
            mime="text/csv",
            icon=":material/download:",
        )

        if has_ground_truth and task in ("regression", "classification"):
            section_header(
                f"Ground-Truth Evaluation on {ds_label}",
                f"Target column `{target_col}` detected — computing full evaluation metrics and shift diagnostics.",
                kicker="Step 4 · Accuracy & Shift Audit",
            )
            comp_rows = []
            roc_curves = {}
            eval_outputs = {}
            for b in chosen_bundles:
                ev = evaluate_bundle_on_data(b, df_up)
                eval_outputs[b.card.bundle_id] = (b, ev)
                row = {
                    "Model": b.card.display_name,
                    "Variant": b.card.variant,
                    "Feature Space": b.card.feature_space,
                    "Source": b.card.source,
                    **{k: ev["metrics"].get(k) for k in TASK_METRICS[task]},
                    f"Shift vs Val {ev['ref_metric']} (%)": round(ev["shift_pct"], 2) if ev["shift_pct"] is not None else None,
                }
                comp_rows.append(row)
                if task == "classification" and "y_proba" in ev["predictions"]:
                    roc_curves[f"{b.card.display_name} ({b.card.variant})"] = (
                        ev["predictions"]["y_true"].to_numpy(),
                        ev["predictions"]["y_proba"].to_numpy(),
                    )

            comp_df = pd.DataFrame(comp_rows)
            st.dataframe(comp_df, column_config=format_table_columns(comp_df), width="stretch", hide_index=True)

            b0, ev0 = eval_outputs[primary_b.card.bundle_id]
            if ev0.get("shift_pct") is not None and abs(ev0["shift_pct"]) > 20.0:
                st.warning(
                    f"**Distribution Shift Alert:** `{b0.card.display_name}` achieved `{ev0['ref_metric']}` "
                    f"`{ev0['metrics'][ev0['ref_metric']]:.3f}` vs `{ev0['ref_val']:.3f}` on H2 validation "
                    f"(`{ev0['shift_pct']:+.1f}%` shift).",
                    icon=":material/warning:",
                )

            render_metric_tiles(ev0["metrics"], TASK_METRICS[task], reference=b0.card.metrics.get("val"), ref_label="vs H2 val")
            p_df = ev0["predictions"]
            if task == "regression":
                c1, c2 = st.columns(2)
                with c1:
                    render_bklit_chart(fig_parity(p_df["y_true"], p_df["y_pred"], b0.card.display_name), key="be_par")
                with c2:
                    render_bklit_chart(fig_residuals(p_df["y_true"], p_df["y_pred"], b0.card.display_name), key="be_res")
            else:
                c1, c2 = st.columns(2)
                m0 = ev0["metrics"]
                with c1:
                    render_bklit_chart(
                        fig_confusion_matrix(m0["TN"], m0["FP"], m0["FN"], m0["TP"], f"Confusion Matrix — {b0.card.display_name}"),
                        key="be_cm",
                    )
                with c2:
                    if roc_curves:
                        render_bklit_chart(fig_roc_curves(roc_curves), key="be_roc")
                if not ev0["segment_errors"].empty:
                    render_bklit_chart(fig_segment_errors(ev0["segment_errors"]), key="be_seg")

    else:
        with glass_container("reeded"):
            c1, c2 = st.columns(2)
            with c1:
                task = st.selectbox("Supervised Task", options=["regression", "classification"], format_func=lambda t: t.title())
            with c2:
                split_choice = st.selectbox(
                    "Evaluation Split",
                    options=["val", "test"],
                    format_func=lambda s: "Validation Split (15% — Model Selection)" if s == "val" else "Holdout Test Split (15% — Final Read-Only Evaluation)",
                )
        if split_choice == "test":
            st.info(
                "**Experimental Integrity Notice (Spec §21):** Test split results are strictly read-only for final reporting — never use test scores to iterate on features or hyperparameters.",
                icon=":material/lock:",
            )

        _, val_df, test_df = ensure_task_splits(task)
        eval_df = val_df if split_choice == "val" else test_df
        bundles = store.list_bundles(task=task)
        rows = []
        for b in bundles:
            ev = evaluate_bundle_on_data(b, eval_df)
            rows.append(
                {
                    "Bundle": b.card.bundle_id,
                    "Model": b.card.display_name,
                    "Variant": b.card.variant,
                    "Feature Space": b.card.feature_space,
                    "Source": b.card.source,
                    "Deployed": "Active" if b.card.is_deployed else "",
                    **{k: ev["metrics"].get(k) for k in TASK_METRICS[task]},
                }
            )
        out_df = pd.DataFrame(rows)
        st.dataframe(out_df, column_config=format_table_columns(out_df), width="stretch", hide_index=True)


run_page_safely(_render)
