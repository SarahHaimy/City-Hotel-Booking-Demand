"""Streamlit resource & data caches so large datasets and pipelines load once."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import path as cfg_path
from src.data.data_cleaning import clean_data
from src.data.loader import load_h2
from src.registry.artifacts import ArtifactStore
from src.registry.experiments import load_experiments_log
from src.training.jobs import JobRunner, get_job_runner
from src.training.service import compute_pca_lda_diagnostics


@st.cache_resource(show_spinner=False)
def get_cached_store() -> ArtifactStore:
    return ArtifactStore()


@st.cache_resource(show_spinner=False)
def get_cached_runner() -> JobRunner:
    return get_job_runner()


@st.cache_resource(show_spinner="Loading saved pipeline artifact…")
def load_cached_pipeline(bundle_id: str, created_at: str) -> Any:
    store = get_cached_store()
    return store.get_bundle(bundle_id).load_pipeline()


@st.cache_data(show_spinner="Loading City Hotel (H2) dataset…")
def get_raw_h2() -> pd.DataFrame:
    return load_h2(cfg_path("raw_data"))


@st.cache_data(show_spinner="Running stateless structural cleaning…")
def get_cleaned_h2(
    drop_zero_adr: bool = True,
    drop_company: bool = True,
    keep_zero_nights: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = get_raw_h2()
    return clean_data(
        raw,
        drop_zero_adr=drop_zero_adr,
        drop_company=drop_company,
        keep_zero_nights=keep_zero_nights,
    )


@st.cache_data(show_spinner=False)
def get_eda_table(filename: str) -> pd.DataFrame | None:
    p: Path = cfg_path("outputs") / "eda" / "tables" / filename
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def get_experiments_df(token: int, task: str | None = None, split: str | None = None) -> pd.DataFrame:
    _ = token  # cache-busting token incremented on artifact changes
    return load_experiments_log(task=task, split=split, store=get_cached_store())


@st.cache_data(show_spinner="Computing PCA variance, loadings, and LDA projection…")
def get_pca_lda_data() -> dict[str, Any]:
    return compute_pca_lda_diagnostics()
