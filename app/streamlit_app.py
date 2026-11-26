"""Main entry point for the City Hotel Booking Demand Streamlit ML Studio."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.cache import get_cached_store  # noqa: E402
from app.components.job_monitor import (  # noqa: E402
    open_job_inspector_dialog,
    render_sidebar_jobs_summary,
)
from app.components.style import (  # noqa: E402
    ASSETS_DIR,
    inject_global_css,
    render_sidebar_brand,
    render_sidebar_status,
)
from app.state import init_state  # noqa: E402
from src.registry.model_card import current_git_commit  # noqa: E402

ICON_PATH = ASSETS_DIR / "hotel_mark.svg"

st.set_page_config(
    page_title="City Hotel H2 — Demand Intelligence",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else ":material/apartment:",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()
init_state()

pages = {
    "Overview & Data": [
        st.Page("views/home.py", title="Executive Overview", icon=":material/dashboard:", default=True),
        st.Page("views/data_explorer.py", title="Data Explorer & EDA", icon=":material/query_stats:"),
        st.Page("views/leakage_audit.py", title="Anti-Leakage Audit", icon=":material/verified_user:"),
    ],
    "Models & Comparisons": [
        st.Page("views/model_catalog.py", title="Model Catalog & Specs", icon=":material/menu_book:"),
        st.Page("views/compare.py", title="Compare Models", icon=":material/leaderboard:"),
        st.Page("views/tuning_lab.py", title="Tuning Lab", icon=":material/tune:"),
    ],
    "Predict, Test & Train": [
        st.Page("views/predict.py", title="Predict Booking", icon=":material/online_prediction:"),
        st.Page("views/batch_evaluate.py", title="Batch Predict & Evaluate", icon=":material/dataset:"),
        st.Page("views/train_studio.py", title="Train Studio", icon=":material/model_training:"),
    ],
    "Segmentation & Reduction": [
        st.Page("views/clustering.py", title="Guest Segmentation", icon=":material/hub:"),
        st.Page("views/pca_lda.py", title="PCA & LDA Analysis", icon=":material/scatter_plot:"),
    ],
    "System": [
        st.Page("views/artifacts.py", title="Artifacts & Jobs", icon=":material/inventory_2:"),
    ],
}

nav = st.navigation(pages)

with st.sidebar:
    render_sidebar_brand()
    store = get_cached_store()
    dep_count = sum(1 for t in ("regression", "classification", "clustering") if store.get_deployed(t) is not None)
    total_bundles = len(store.list_bundles())
    render_sidebar_status(dep_count, total_bundles)
    render_sidebar_jobs_summary()
    st.caption(f"CSE382 ML System · git `{current_git_commit()}` · seed `42`")

dialog_job_id = st.session_state.get("inspect_dialog_job_id")
if dialog_job_id:
    open_job_inspector_dialog(dialog_job_id)

nav.run()
