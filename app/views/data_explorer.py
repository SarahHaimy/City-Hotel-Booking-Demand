"""Data Explorer page — H2 dataset inspection, cleaning audit, interactive distributions, and EDA gallery."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.cache import get_cleaned_h2, get_eda_table, get_raw_h2
from app.components.bklit_chart import render_bklit_chart
from app.components.style import glass_container, page_header, pill_html, run_page_safely, section_header
from src.config import path as cfg_path, split_dir
from src.data.data_split import summarize_splits
from src.data.schema import DOCUMENTED_RENAMES, H2_COLUMNS
from src.data.validation import ensure_task_splits, load_uploaded_table
from src.evaluation.figures import fig_categorical_bars, fig_numeric_distribution


def _render() -> None:
    page_header(
        "Data Explorer & Preprocessing Audit",
        "Inspect H2 City Hotel dimensions, variable semantics, missingness, cleaning rules, stratified splits, and interactive distributions",
        pill_html("79,330 Raw Rows · Zero-Leakage Split Protocol", "m1"),
    )

    with glass_container("reeded"):
        src_mode = st.radio(
            "Dataset Source",
            options=["City Hotel H2.csv (Project Dataset)", "Upload Custom CSV/XLSX to Profile"],
            horizontal=True,
        )
        if src_mode.startswith("Upload"):
            up = st.file_uploader("Upload CSV or Excel file to profile", type=["csv", "xlsx"])
            if up is None:
                st.info("Upload a CSV/XLSX file above to inspect its schema, missing values, and feature distributions.", icon=":material/info:")
                return
            df = load_uploaded_table(up, filename=up.name)
            is_official = False
        else:
            df = get_raw_h2()
            is_official = True

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Exact Duplicates", f"{int(df.duplicated().sum()):,}")
    if "IsCanceled" in df.columns:
        c4.metric("Raw Cancellation Rate", f"{pd.to_numeric(df['IsCanceled'], errors='coerce').mean():.1%}")

    t_overview, t_clean, t_dist, t_gallery = st.tabs(
        [
            ":material/table_chart: Schema & Summary Tables",
            ":material/filter_alt: Structural Cleaning & Splits",
            ":material/bar_chart: Interactive Distributions",
            ":material/photo_library: Saved EDA Figures",
        ]
    )

    with t_overview:
        section_header(
            "Verified Source Variables & Empirical Quality",
            "Data types, cardinality, missingness, and domain semantics across every column.",
            kicker="Schema Profile",
        )
        schema_rows = []
        for col in df.columns:
            spec = H2_COLUMNS.get(str(col))
            na_cnt = int(df[col].isna().sum())
            schema_rows.append(
                {
                    "Column": str(col),
                    "Dtype": str(df[col].dtype),
                    "Unique": int(df[col].nunique(dropna=True)),
                    "Missing": na_cnt,
                    "Missing %": round(na_cnt / max(len(df), 1) * 100.0, 2),
                    "Role / Type": spec.kind if spec else "custom",
                    "Verified Meaning (Spec §4)": spec.meaning if spec else "User-supplied column",
                }
            )
        st.dataframe(schema_rows, width="stretch", hide_index=True)

        if is_official:
            e1, e2 = st.columns(2, gap="medium")
            with e1:
                num_sum = get_eda_table("numeric_summary.csv")
                if num_sum is not None:
                    st.markdown("##### Numeric Summary & IQR Outliers")
                    st.dataframe(num_sum, width="stretch", hide_index=True)
            with e2:
                unusual = get_eda_table("unusual_values.csv")
                if unusual is not None:
                    st.markdown("##### Structural Anomalies Detected in Raw H2")
                    st.dataframe(unusual, width="stretch", hide_index=True)
                cat_card = get_eda_table("categorical_cardinality.csv")
                if cat_card is not None:
                    st.markdown("##### Categorical Cardinality & Mode Share")
                    st.dataframe(cat_card, width="stretch", hide_index=True)

    with t_clean:
        section_header(
            "Stateless Row-Level Cleaning (src/data/data_cleaning.py)",
            f"Executed prior to splitting because every rule is purely structural (no learned statistics). Documented rename: {DOCUMENTED_RENAMES}.",
            kicker="Cleaning & Partitioning",
        )
        if is_official:
            with glass_container("reeded"):
                fc1, fc2, fc3 = st.columns(3)
                drop_z = fc1.checkbox("Drop ADR == 0 (`drop_zero_adr`)", value=True, help="True for ADR regression & clustering; False for cancellation classification.")
                drop_c = fc2.checkbox("Drop sparse Company (`~95% NULL`)", value=True)
                keep_zn = fc3.checkbox("Keep 0-night non-canceled stays", value=False)
            cleaned_df, rep_df = get_cleaned_h2(drop_zero_adr=drop_z, drop_company=drop_c, keep_zero_nights=keep_zn)

            r1, r2 = st.columns([1.4, 1.6], gap="medium")
            with r1:
                st.markdown("##### Step-by-Step Row Removal Audit")
                st.dataframe(rep_df, width="stretch", hide_index=True)
                st.caption(f"Retained rows: **{len(cleaned_df):,}** / {len(df):,} ({len(cleaned_df)/len(df):.1%})")
            with r2:
                st.markdown("##### Disjoint Stratified 70 / 15 / 15 Splits (`seed=42`)")
                for tname in ("regression", "classification"):
                    sdir = split_dir(tname)
                    tr, va, te = ensure_task_splits(tname)
                    sum_df = summarize_splits(tr, va, te)
                    sum_df.insert(0, "Task Track", tname.title())
                    sum_df["Split Folder"] = str(sdir.name)
                    st.dataframe(sum_df, width="stretch", hide_index=True)
        else:
            st.dataframe(df.describe(include="all").T.reset_index(), width="stretch", hide_index=True)

    with t_dist:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if c not in num_cols]
        split_opts = ["(None)", *[c for c in ("IsCanceled", "MarketSegment", "CustomerType", "Meal") if c in df.columns]]

        d1, d2 = st.columns(2, gap="medium")
        with d1:
            with glass_container("frosted"):
                st.markdown('<div class="section-kicker">Continuous & Count Variables</div>', unsafe_allow_html=True)
                if num_cols:
                    nc1, nc2 = st.columns(2)
                    nc = nc1.selectbox("Numeric Column", options=num_cols, index=num_cols.index("ADR") if "ADR" in num_cols else 0)
                    sp_n = nc2.selectbox("Group / Color By", options=split_opts, key="eda_num_split")
                    render_bklit_chart(
                        fig_numeric_distribution(df, nc, None if sp_n == "(None)" else sp_n),
                        key="eda_num_chart",
                    )
        with d2:
            with glass_container("frosted"):
                st.markdown('<div class="section-kicker">Categorical Variables</div>', unsafe_allow_html=True)
                if cat_cols:
                    cc1, cc2 = st.columns(2)
                    cc = cc1.selectbox(
                        "Categorical Column",
                        options=cat_cols,
                        index=cat_cols.index("MarketSegment") if "MarketSegment" in cat_cols else 0,
                    )
                    sp_c = cc2.selectbox("Group / Color By", options=split_opts, index=1 if "IsCanceled" in split_opts else 0, key="eda_cat_split")
                    render_bklit_chart(
                        fig_categorical_bars(df, cc, None if sp_c == "(None)" else sp_c),
                        key="eda_cat_chart",
                    )

    with t_gallery:
        fig_dir: Path = cfg_path("outputs") / "eda" / "figures"
        pngs = sorted(fig_dir.glob("*.png")) if fig_dir.exists() else []
        if not pngs:
            st.info("No static EDA figures found in `outputs/eda/figures/`. Run `python main.py analyze` to generate them.", icon=":material/info:")
        else:
            gcols = st.columns(2, gap="medium")
            for idx, p in enumerate(pngs):
                with gcols[idx % 2]:
                    with glass_container("solid"):
                        st.image(str(p), caption=p.stem.replace("_", " ").title(), width="stretch")


run_page_safely(_render)
