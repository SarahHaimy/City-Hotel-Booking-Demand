"""Predict page — single-booking ADR regression, IsCanceled classification (with Bklit Gauge), cluster assignment & what-if sweep."""
from __future__ import annotations

import streamlit as st

from app.cache import get_cached_store
from app.components.bklit_chart import render_bklit_chart
from app.components.booking_form import render_booking_form
from app.components.style import glass_container, page_header, pill_html, render_empty_artifacts_banner, run_page_safely, section_header
from src.data.schema import H2_COLUMNS, MONTHS
from src.evaluation.figures import SERIES, fig_cancel_gauge, fig_what_if
from src.inference.predict import (
    predict_across_all_models,
    predict_single_booking,
    what_if_sweep,
)


def _bundle_selector(label: str, task: str, key: str, only_predict_new: bool = False) -> str | None:
    store = get_cached_store()
    bundles = [b for b in store.list_bundles(task=task) if (not only_predict_new or b.card.supports_predict_new)]
    if not bundles:
        st.caption(f"No `{task}` bundles available.")
        return None
    dep = store.get_deployed(task)
    opts = {}
    default_idx = 0
    for idx, b in enumerate(bundles):
        dep_tag = "[Deployed] " if b.card.is_deployed else ""
        src_tag = " [User]" if b.card.source == "user" else ""
        lbl = f"{dep_tag}{b.card.display_name} ({b.card.variant}, {b.card.feature_space}){src_tag}"
        opts[lbl] = b.card.bundle_id
        if dep and b.card.bundle_id == dep.card.bundle_id:
            default_idx = idx
    chosen = st.selectbox(label, options=list(opts.keys()), index=default_idx, key=key)
    return opts[chosen]


def _render() -> None:
    page_header(
        "Interactive Booking Prediction & What-If Simulator",
        "Predict Average Daily Rate (ADR), Cancellation Risk (IsCanceled), and Guest Segment from saved pipeline artifacts",
        pill_html("Saved Artifacts · Zero Retraining", "deployed"),
    )

    store = get_cached_store()
    if not store.list_bundles():
        render_empty_artifacts_banner()
        return

    section_header(
        "Select Active Inference Pipelines",
        "Choose which saved Regression, Classification, and Clustering bundles score the reservation below.",
        kicker="Step 1 · Model Routing",
    )
    with glass_container("reeded"):
        m1, m2, m3 = st.columns(3)
        with m1:
            reg_id = _bundle_selector("ADR Regression Model", "regression", "pred_reg_sel")
        with m2:
            clf_id = _bundle_selector("Cancellation Classifier", "classification", "pred_clf_sel")
        with m3:
            clu_id = _bundle_selector("Customer Segmenter", "clustering", "pred_clu_sel", only_predict_new=True)

        t1, t2 = st.columns([1.2, 2.8], vertical_alignment="center")
        with t1:
            use_custom_thr = st.checkbox("Override Decision Threshold", value=False, key="pred_use_thr")
        with t2:
            thr_val = st.slider(
                "Custom Classification Decision Threshold",
                0.15, 0.85, 0.50, 0.05,
                disabled=not use_custom_thr,
                key="pred_thr_val",
            )

    section_header(
        "Enter Reservation Attributes",
        "Populate booking parameters manually, apply a typical guest archetype preset, sample a real validation row, or paste JSON/CSV.",
        kicker="Step 2 · Booking Input",
    )
    booking = render_booking_form("single_pred")

    res = predict_single_booking(
        booking,
        reg_bundle_id=reg_id,
        clf_bundle_id=clf_id,
        clu_bundle_id=clu_id,
        threshold_override=thr_val if use_custom_thr else None,
        store=store,
    )

    section_header(
        "Live Multi-Task Predictions",
        "Real-time inference across pricing, cancellation probability, and behavioral cohort assignment.",
        kicker="Step 3 · Model Output",
    )
    o1, o2, o3 = st.columns(3, gap="medium")

    with o1:
        with glass_container("frosted"):
            st.markdown('<div class="section-kicker">Task A — Predicted ADR</div>', unsafe_allow_html=True)
            r = res.get("regression")
            if r:
                total_nights = max(1, int(booking["StaysInWeekNights"]) + int(booking["StaysInWeekendNights"]))
                st.metric("Predicted Nightly ADR", f"EUR {r['pred_adr']:.2f}")
                st.metric("Est. Stay Room Revenue", f"EUR {r['pred_adr'] * total_nights:,.2f}", help=f"Across {total_nights} total night(s).")
                if r["val_rmse"] is not None:
                    st.caption(
                        f"±1 Val-RMSE Band: `EUR {r['interval_low']:.2f} – EUR {r['interval_high']:.2f}` · `{r['model_name']}`"
                    )
            else:
                st.info("No regression artifact selected.", icon=":material/info:")

    with o2:
        with glass_container("frosted"):
            st.markdown('<div class="section-kicker">Task B — Cancellation Risk</div>', unsafe_allow_html=True)
            c = res.get("classification")
            if c:
                prob = c["cancel_proba"]
                pill_kind = "user" if c["pred_canceled"] == 1 else "deployed"
                render_bklit_chart(fig_cancel_gauge(prob, label="Cancel Risk %"), key="pred_gauge")
                st.markdown(
                    f"{pill_html(c['decision_label'], pill_kind)} "
                    f"<span style='font-size:0.80rem;color:var(--fg-ink-500);'>threshold {c['threshold']:.2f} · {c['model_name']}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No classification artifact selected.", icon=":material/info:")

    with o3:
        with glass_container("frosted"):
            st.markdown('<div class="section-kicker">Task C — Guest Cohort</div>', unsafe_allow_html=True)
            k = res.get("clustering")
            if k:
                persona = k.get("persona") or {}
                st.metric("Assigned Cluster", f"Cluster {k['cluster']}")
                if persona:
                    st.markdown(f"**{persona.get('title', '')}**")
                    st.caption(persona.get("summary", ""))
                st.caption(f"Segmenter: `{k['model_name']}`")
            else:
                st.info("Select K-Means to assign out-of-sample guest segments.", icon=":material/info:")

    t_consensus, t_whatif = st.tabs(
        [
            ":material/compare_arrows: Cross-Variant Consensus",
            ":material/ssid_chart: What-If Sensitivity Sweep",
        ]
    )

    with t_consensus:
        c_reg, c_clf = st.columns(2)
        with c_reg:
            st.markdown("##### All Regression Variants on This Booking")
            st.dataframe(predict_across_all_models(booking, "regression", store=store), width="stretch", hide_index=True)
        with c_clf:
            st.markdown("##### All Classification Variants on This Booking")
            st.dataframe(
                predict_across_all_models(
                    booking, "classification", threshold_override=thr_val if use_custom_thr else None, store=store
                ),
                width="stretch",
                hide_index=True,
            )

    with t_whatif:
        st.caption("Sweep one booking attribute while holding all other inputs fixed to inspect model sensitivity:")
        sweep_feature = st.selectbox(
            "Feature to Sweep",
            options=[
                "LeadTime",
                "StaysInWeekNights",
                "StaysInWeekendNights",
                "Adults",
                "PreviousCancellations",
                "ArrivalDateMonth",
                "ReservedRoomType",
                "MarketSegment",
                "Meal",
            ],
        )
        if sweep_feature == "LeadTime":
            grid = list(range(0, 365, 15))
        elif sweep_feature in ("StaysInWeekNights", "StaysInWeekendNights"):
            grid = list(range(0, 15))
        elif sweep_feature == "Adults":
            grid = [1, 2, 3, 4]
        elif sweep_feature == "PreviousCancellations":
            grid = list(range(0, 8))
        elif sweep_feature == "ArrivalDateMonth":
            grid = list(MONTHS)
        else:
            grid = list(H2_COLUMNS[sweep_feature].categories)

        df_sw = what_if_sweep(booking, sweep_feature, grid, reg_bundle_id=reg_id, clf_bundle_id=clf_id, store=store)
        w1, w2 = st.columns(2)
        if "Predicted ADR (€)" in df_sw.columns:
            with w1:
                render_bklit_chart(
                    fig_what_if(
                        df_sw, sweep_feature, "Predicted ADR (€)",
                        title=f"Predicted ADR vs {sweep_feature}",
                        y_title="ADR (EUR)", color=SERIES["slate"],
                    ),
                    key="whatif_adr",
                )
        if "Cancel Probability" in df_sw.columns:
            with w2:
                render_bklit_chart(
                    fig_what_if(
                        df_sw, sweep_feature, "Cancel Probability",
                        title=f"Cancellation Probability vs {sweep_feature}",
                        y_title="Cancel Probability", color=SERIES["bronze"],
                        y_domain=[0.0, 1.0],
                    ),
                    key="whatif_clf",
                )


run_page_safely(_render)
