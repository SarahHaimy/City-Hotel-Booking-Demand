"""KPI metric tile rows with automatic direction-aware deltas and tooltips."""
from __future__ import annotations

from typing import Any

import streamlit as st

from src.evaluation.metrics import METRIC_META, format_metric


def render_metric_tiles(
    metrics: dict[str, Any],
    keys: list[str],
    reference: dict[str, Any] | None = None,
    ref_label: str = "vs baseline",
) -> None:
    present_keys = [k for k in keys if k in metrics and metrics[k] is not None]
    if not present_keys:
        st.caption("No evaluation metrics recorded for this split.")
        return

    cols = st.columns(len(present_keys))
    for col, key in zip(cols, present_keys):
        meta = METRIC_META.get(key, {"label": key, "higher_is_better": True, "help": ""})
        val = metrics[key]
        val_str = format_metric(key, val, with_arrow=True)

        delta_str = None
        delta_color = "normal" if meta.get("higher_is_better", True) else "inverse"
        if reference and key in reference and reference[key] is not None and val is not None:
            diff = float(val) - float(reference[key])
            if abs(diff) > 1e-6:
                delta_str = f"{diff:+.4f} ({ref_label})" if abs(diff) < 1 else f"{diff:+.2f} ({ref_label})"

        with col:
            st.metric(
                label=meta.get("label", key),
                value=val_str,
                delta=delta_str,
                delta_color=delta_color,
                help=meta.get("help", ""),
            )
