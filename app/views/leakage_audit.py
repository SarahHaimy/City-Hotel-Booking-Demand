"""Leakage Audit page — interactive 31-variable matrix, stage separation protocol, and live assertions (emoji-free)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.cache import get_cached_store
from app.components.style import format_table_columns, glass_container, page_header, pill_html, run_page_safely, section_header
from src.data.schema import LEAKAGE_AUDIT_MATRIX, TASK_FORBIDDEN_COLUMNS
from src.data.validation import ensure_task_splits


def _render() -> None:
    page_header(
        "Anti-Leakage Audit & Experimental Integrity",
        "Complete feature-timing justification across all 31 H2 variables, stage isolation protocol, and live pipeline verification (Spec §16)",
        pill_html("Prediction Point: Reservation Entry", "deployed"),
    )

    st.info(
        "**Assumed Lifecycle Prediction Point:** All supervised predictions (**ADR** and **IsCanceled**) are evaluated at the "
        "**moment a booking first enters the Property Management System (PMS)**, prior to waitlist resolution, room assignment at check-in, "
        "mid-lifecycle booking amendments, or final check-out/cancellation.",
        icon=":material/verified_user:",
    )

    section_header(
        "End-to-End Stage Separation Protocol",
        "Six-stage pipeline barrier guaranteeing zero validation or holdout test contamination.",
        kicker="1 · Methodology",
    )
    with glass_container("solid"):
        st.markdown(
            "1. **Raw Ingestion (`H2.csv`)** — whitespace trimming and `'NULL'` token normalization only.\n"
            "2. **Stateless Row Cleaning (`clean_data`)** — deduplication, 0-guest removal, `Babies -> HasBabies` (uses **zero** dataset statistics).\n"
            "3. **Stratified 70 / 15 / 15 Split (`seed=42`)** — strictly disjoint Train, Validation, and Holdout Test partitions.\n"
            "4. **Fit Only on Train (70%)** — `SimpleImputer`, `StandardScaler`, `OneHotEncoder`, `PCA`, `LDA`, and `Grid/RandomizedSearchCV` folds never see Validation or Test rows.\n"
            "5. **Validation (15%)** — model selection and classification decision-threshold optimization.\n"
            "6. **Holdout Test (15%)** — single read-only final evaluation."
        )

    section_header(
        "Live Automated Leakage & Split Assertions",
        "Runtime verification across active split indices and every saved model bundle's input schema.",
        kicker="2 · Live Assertions",
    )
    store = get_cached_store()
    checks: list[dict[str, str]] = []

    for tname in ("regression", "classification"):
        try:
            tr, va, te = ensure_task_splits(tname)
            overlap = (set(tr.index) & set(va.index)) | (set(tr.index) & set(te.index)) | (set(va.index) & set(te.index))
            checks.append(
                {
                    "Audit Assertion": f"Disjoint row indices across {tname} train / val / test splits",
                    "Status": "PASS" if not overlap else "FAIL",
                    "Details": f"Train={len(tr):,}, Val={len(va):,}, Test={len(te):,}, Index overlap={len(overlap)}",
                }
            )
        except Exception as exc:
            checks.append({"Audit Assertion": f"Split check ({tname})", "Status": "SKIP", "Details": str(exc)})

    for b in store.list_bundles():
        forb = set(TASK_FORBIDDEN_COLUMNS.get(b.card.task, [])) - ({b.card.target} if b.card.target else set())
        bad = set(b.card.inputs) & forb
        checks.append(
            {
                "Audit Assertion": f"Bundle `{b.card.bundle_id}` excludes {b.card.task} forbidden columns",
                "Status": "PASS" if not bad else "FAIL",
                "Details": f"{len(b.card.inputs)} safe inputs; 0/{len(forb)} forbidden columns present" if not bad else f"Leaked: {bad}",
            }
        )
    st.dataframe(checks, width="stretch", hide_index=True)

    section_header(
        "Source-Variable Timing & Leakage Matrix (All 31 H2 Variables)",
        "Per-variable inclusion/exclusion status across ADR Regression, Cancellation Classification, and Guest Clustering.",
        kicker="3 · Variable Audit",
    )
    df_mat = pd.DataFrame(LEAKAGE_AUDIT_MATRIX)
    filter_status = st.selectbox(
        "Filter Variables by Role",
        options=["All 31 Variables", "Excluded Leakage / Proxies Only", "Used Predictors Only"],
    )
    if filter_status == "Excluded Leakage / Proxies Only":
        mask = (
            df_mat["regression"].str.startswith("excluded_o")
            | df_mat["regression"].str.startswith("excluded_p")
            | df_mat["classification"].str.startswith("excluded_p")
        )
        df_mat = df_mat[mask]
    elif filter_status == "Used Predictors Only":
        mask = df_mat[["regression", "classification", "clustering"]].isin(["used", "engineered"]).any(axis=1)
        df_mat = df_mat[mask]

    st.dataframe(df_mat, column_config=format_table_columns(df_mat), width="stretch", hide_index=True)


run_page_safely(_render)
