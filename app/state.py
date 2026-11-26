"""Session-state keys and cross-page navigation helpers."""
from __future__ import annotations

from typing import Any

import streamlit as st

from src.data.schema import BOOKING_PRESETS


def init_state() -> None:
    defaults: dict[str, Any] = {
        "active_task": "regression",
        "catalog_selected_bundle": None,
        "compare_basket": [],
        "active_job_id": None,
        "artifact_cache_token": 0,
        "predict_booking": dict(next(iter(BOOKING_PRESETS.values()))),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def bump_cache_token() -> None:
    st.session_state["artifact_cache_token"] = int(st.session_state.get("artifact_cache_token", 0)) + 1


def cache_token() -> int:
    return int(st.session_state.get("artifact_cache_token", 0))


def add_to_compare(bundle_id: str) -> None:
    basket: list[str] = list(st.session_state.get("compare_basket", []))
    if bundle_id not in basket:
        basket.append(bundle_id)
        st.session_state["compare_basket"] = basket
