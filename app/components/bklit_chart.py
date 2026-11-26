"""Streamlit v2 Custom Component wrapper mounting pre-compiled @bklitui/ui React charts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import streamlit as st

ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
_JS_PATH = ASSETS_DIR / "bklit_bundle.js"
_CSS_PATH = ASSETS_DIR / "bklit_bundle.css"

_BKLIT_COMPONENT = st.components.v2.component(
    "bklit_ui_chart",
    css=_CSS_PATH.read_text(encoding="utf-8"),
    js=_JS_PATH.read_text(encoding="utf-8"),
    isolate_styles=True,
)


def render_bklit_chart(spec: dict[str, Any] | list[dict[str, Any]], key: str | None = None) -> None:
    """Render one (or a side-by-side pair of) Bklit UI chart specification(s)."""
    if isinstance(spec, list):
        cols = st.columns(len(spec))
        for idx, sub_spec in enumerate(spec):
            sub_key = f"{key}_{idx}" if key else None
            with cols[idx]:
                render_bklit_chart(sub_spec, key=sub_key)
        return

    if not spec:
        return

    # Deterministic content hash ensures reactive updates when sliders/filters change
    sig = hashlib.md5(json.dumps(spec, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8]
    eff_key = f"{key}_{sig}" if key else f"bklit_{sig}_{id(spec)}"
    _BKLIT_COMPONENT(data=spec, key=eff_key)
