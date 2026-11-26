"""Dynamic hyperparameter widgets and search-space editors generated from ModelSpec."""
from __future__ import annotations

from typing import Any

import streamlit as st

from src.registry.specs import ModelSpec


def render_param_editor(
    spec: ModelSpec,
    key_prefix: str = "pe",
    preset_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render appropriate Streamlit widgets for every ParamSpec on a ModelSpec."""
    values: dict[str, Any] = dict(spec.default_param_dict())
    if preset_values:
        values.update(preset_values)

    if not spec.params:
        st.caption("This model has no editable hyperparameters.")
        return values

    cols = st.columns(min(3, len(spec.params)))
    for idx, p in enumerate(spec.params):
        cur = values.get(p.name, p.default)
        wkey = f"{key_prefix}_{spec.task}_{spec.model_id}_{p.name}"
        with cols[idx % len(cols)]:
            if p.kind == "bool":
                values[p.name] = st.checkbox(p.label, value=bool(cur), help=p.help, key=wkey)
            elif p.kind == "choice":
                opts = list(p.choices)
                values[p.name] = st.selectbox(
                    p.label,
                    options=opts,
                    index=opts.index(cur) if cur in opts else 0,
                    help=p.help,
                    key=wkey,
                )
            elif p.kind == "int":
                low_i = int(p.low) if p.low is not None else 0
                high_i = int(p.high) if p.high is not None else 1000
                step_i = int(p.step) if p.step is not None else 1
                cur_i = max(low_i, min(high_i, int(cur if cur is not None else p.default)))
                values[p.name] = int(
                    st.number_input(
                        p.label, min_value=low_i, max_value=high_i, value=cur_i, step=step_i, help=p.help, key=wkey
                    )
                )
            else:
                low_f = float(p.low) if p.low is not None else 0.0
                high_f = float(p.high) if p.high is not None else 100.0
                cur_f = max(low_f, min(high_f, float(cur if cur is not None else p.default)))
                fmt = "%.6g" if p.kind == "log_float" else "%.3f"
                step_f = float(p.step) if p.step is not None else (cur_f * 0.1 if cur_f > 0 else 0.01)
                values[p.name] = float(
                    st.number_input(
                        p.label,
                        min_value=low_f,
                        max_value=high_f,
                        value=cur_f,
                        step=step_f,
                        format=fmt,
                        help=p.help,
                        key=wkey,
                    )
                )
    return values


def render_search_space_editor(spec: ModelSpec, key_prefix: str = "sse") -> dict[str, list[Any]]:
    """Display and optionally customize comma-separated candidate grids for hyperparameter tuning."""
    edited: dict[str, list[Any]] = {}
    st.caption("Edit comma-separated candidate values per parameter (use `None` for unbounded depth/weight):")
    cols = st.columns(min(2, max(1, len(spec.search_space))))
    for idx, (param_key, default_list) in enumerate(spec.search_space.items()):
        default_str = ", ".join("None" if v is None else str(v) for v in default_list)
        with cols[idx % len(cols)]:
            raw = st.text_input(
                f"`{param_key}`",
                value=default_str,
                key=f"{key_prefix}_{spec.task}_{spec.model_id}_{param_key}",
            )
            parsed: list[Any] = []
            for tok in raw.split(","):
                t = tok.strip()
                if not t:
                    continue
                if t.lower() == "none":
                    parsed.append(None)
                elif t.lower() in ("true", "false"):
                    parsed.append(t.lower() == "true")
                else:
                    try:
                        parsed.append(int(t) if t.isdigit() else float(t))
                    except ValueError:
                        parsed.append(t)
            edited[param_key] = parsed or list(default_list)
    return edited
