"""Reusable CSV/XLSX upload widget with schema validation report and interactive column mapping (emoji-free)."""
from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

from app.components.style import glass_container, pill_html
from src.config import path as cfg_path
from src.data.validation import ValidationReport, load_uploaded_table, validate_dataframe


def render_data_uploader(
    key_prefix: str,
    task: Literal["regression", "classification", "clustering"],
    required_inputs: list[str],
    forbidden_inputs: list[str] | None = None,
    require_target: bool = False,
    min_rows: int = 1,
    allow_sample_buttons: bool = True,
) -> tuple[pd.DataFrame | None, ValidationReport | None, str]:
    raw_df: pd.DataFrame | None = None
    dataset_label = "Uploaded file"

    u_col, s_col = st.columns([2.6, 1.4], vertical_alignment="bottom") if allow_sample_buttons else (st.container(), None)
    with u_col:
        uploaded = st.file_uploader(
            "Upload Booking Dataset (CSV or Excel .xlsx)",
            type=["csv", "xlsx", "xls"],
            key=f"{key_prefix}_file",
            help="Columns matching H2 source labels or snake_case equivalents are mapped automatically.",
        )

    sample_state_key = f"{key_prefix}_sample_choice"
    if s_col is not None:
        with s_col:
            with glass_container("reeded"):
                st.markdown('<div class="section-kicker">Quick Validation Slices</div>', unsafe_allow_html=True)
                st.caption("Load a 250-row sample slice immediately:")
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Labeled (250)", key=f"{key_prefix}_smp_lbl", icon=":material/description:", width="stretch"):
                        st.session_state[sample_state_key] = "labeled"
                with b2:
                    if st.button("Unlabeled (250)", key=f"{key_prefix}_smp_unlbl", icon=":material/description:", width="stretch"):
                        st.session_state[sample_state_key] = "unlabeled"

    if uploaded is not None:
        st.session_state.pop(sample_state_key, None)
        try:
            raw_df = load_uploaded_table(uploaded, filename=uploaded.name)
            dataset_label = uploaded.name
        except Exception as exc:
            st.error(f"Failed to read uploaded file `{uploaded.name}`: {exc}", icon=":material/error:")
            return None, None, ""
    elif st.session_state.get(sample_state_key):
        kind = st.session_state[sample_state_key]
        fname = "sample_bookings_labeled.csv" if kind == "labeled" else "sample_bookings_unlabeled.csv"
        fpath = cfg_path("samples") / fname
        if fpath.exists():
            raw_df = load_uploaded_table(fpath, filename=fname)
            dataset_label = f"Sample ({fname})"
        else:
            st.warning(f"Sample file `{fpath}` not generated yet.", icon=":material/warning:")

    if raw_df is None:
        return None, None, ""

    custom_map_key = f"{key_prefix}_col_map"
    custom_mapping: dict[str, str] = dict(st.session_state.get(custom_map_key, {}))
    norm_df, report = validate_dataframe(
        raw_df,
        task=task,
        required_inputs=required_inputs,
        forbidden_inputs=forbidden_inputs,
        custom_mapping=custom_mapping,
        require_target=require_target,
        min_rows=min_rows,
        check_test_overlap=require_target,
    )

    target_col = "ADR" if task == "regression" else ("IsCanceled" if task == "classification" else None)
    missing_targets = (
        [target_col] if (require_target and target_col and target_col not in norm_df.columns) else []
    )
    need_cols = report.missing_required + missing_targets
    if need_cols and report.unmatched_columns:
        with st.expander("Manual Column Mapping (map unmatched upload columns to H2 fields)", icon=":material/build:", expanded=True):
            updated_map = dict(custom_mapping)
            mcols = st.columns(min(3, len(report.unmatched_columns)))
            options = ["(Ignore)", *need_cols]
            for idx, ucol in enumerate(report.unmatched_columns):
                with mcols[idx % len(mcols)]:
                    cur_sel = updated_map.get(ucol, "(Ignore)")
                    choice = st.selectbox(
                        f"Upload column `{ucol}` ->",
                        options=options,
                        index=options.index(cur_sel) if cur_sel in options else 0,
                        key=f"{key_prefix}_map_{ucol}",
                    )
                    if choice != "(Ignore)":
                        updated_map[ucol] = choice
                    else:
                        updated_map.pop(ucol, None)
            if updated_map != custom_mapping:
                st.session_state[custom_map_key] = updated_map
                norm_df, report = validate_dataframe(
                    raw_df,
                    task=task,
                    required_inputs=required_inputs,
                    forbidden_inputs=forbidden_inputs,
                    custom_mapping=updated_map,
                    require_target=require_target,
                    min_rows=min_rows,
                    check_test_overlap=require_target,
                )

    with glass_container("solid"):
        status_pill = pill_html("VALID", "deployed") if report.ok else pill_html("BLOCKED", "user")
        targets_txt = ", ".join(report.present_targets) if report.present_targets else "None (unlabeled)"
        st.markdown(
            f"**Dataset Validation — `{dataset_label}`** &nbsp;{status_pill} &nbsp;|&nbsp; "
            f"**Rows:** `{report.n_rows:,}` · **Columns:** `{report.n_cols}` · "
            f"**Detected Targets:** `{targets_txt}`",
            unsafe_allow_html=True,
        )
        for issue in report.errors:
            st.error(f"**{issue.code}:** {issue.message}", icon=":material/error:")
        for issue in report.warnings:
            st.warning(f"**{issue.code}:** {issue.message}", icon=":material/warning:")
        for issue in report.infos:
            st.info(f"**{issue.code}:** {issue.message}", icon=":material/info:")

        with st.expander("Preview Normalized Data (first 15 rows)", icon=":material/table_rows:"):
            st.dataframe(norm_df.head(15), width="stretch")

    return norm_df, report, dataset_label
