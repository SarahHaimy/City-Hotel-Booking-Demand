"""Live background job monitor (using st.fragment) and sidebar jobs history drawer (emoji-free)."""
from __future__ import annotations

import streamlit as st

from app.cache import get_cached_runner, get_cached_store
from app.components.model_card import render_bundle_detail_panel
from app.components.style import glass_container, pill_html
from app.state import bump_cache_token
from src.registry.registry import get_spec


def _status_pill(status: str) -> str:
    mapping = {
        "queued": pill_html("QUEUED", "muted"),
        "running": pill_html("RUNNING", "m1"),
        "succeeded": pill_html("SUCCEEDED", "deployed"),
        "failed": pill_html("FAILED", "user"),
        "cancelled": pill_html("CANCELLED", "muted"),
    }
    return mapping.get(status, pill_html(status.upper(), "muted"))


def _dismiss_job_dialog() -> None:
    st.session_state.pop("inspect_dialog_job_id", None)


@st.dialog("Background Job Inspector", width="large", icon=":material/terminal:", on_dismiss=_dismiss_job_dialog)
def open_job_inspector_dialog(job_id: str) -> None:
    runner = get_cached_runner()
    rec = runner.get_job(job_id)
    if rec is None:
        st.warning(f"Job `{job_id}` was not found in `runs/`.", icon=":material/warning:")
        return

    st.markdown(f"#### {rec.title} &nbsp; {_status_pill(rec.status)}", unsafe_allow_html=True)
    ts_parts = [f"Job ID: `{rec.job_id}`", f"Type: `{rec.kind}`"]
    if rec.created_at:
        ts_parts.append(f"Created: `{rec.created_at[:19].replace('T', ' ')} UTC`")
    if rec.finished_at:
        ts_parts.append(f"Finished: `{rec.finished_at[:19].replace('T', ' ')} UTC`")
    st.caption(" · ".join(ts_parts))
    st.caption(f"Status: {rec.message}")
    st.progress(float(rec.progress))

    if rec.status in ("queued", "running"):
        c1, c2 = st.columns(2)
        if c1.button("Refresh Status", key=f"dlg_ref_{job_id}", icon=":material/refresh:", width="stretch"):
            st.rerun()
        if c2.button("Cancel Job", key=f"dlg_cancel_{job_id}", icon=":material/stop_circle:", width="stretch"):
            runner.cancel(job_id)
            st.toast("Cancellation signal sent.")
            st.rerun()

    logs = runner.read_logs(job_id, tail_lines=80)
    st.markdown("**Execution Log Stream**")
    if logs:
        st.code(logs, language="text")
    else:
        st.caption("No log lines emitted yet.")

    if rec.status == "succeeded":
        bump_cache_token()
        if rec.result_bundle_id:
            st.success(
                f"Saved bundle **`{rec.result_bundle_id}`** — "
                f"{rec.result_summary.get('primary_metric', 'Score')}: "
                f"`{rec.result_summary.get('primary_value')}` "
                f"(`{rec.result_summary.get('train_time_s', 0):.2f}s`).",
                icon=":material/check_circle:",
            )
            store = get_cached_store()
            try:
                bundle = store.get_bundle(rec.result_bundle_id)
            except KeyError:
                bundle = None

            b1, b2 = st.columns(2)
            if bundle is not None:
                if b1.button(
                    "Open Bundle in Model Catalog",
                    key=f"dlg_cat_{job_id}",
                    type="primary",
                    icon=":material/menu_book:",
                    width="stretch",
                ):
                    st.session_state["catalog_selected_bundle"] = bundle.card.bundle_id
                    st.session_state["cat_task_filter"] = bundle.card.task
                    st.session_state["cat_fam_filter"] = "All Families"
                    st.session_state["cat_src_filter"] = "All Sources"
                    st.session_state.pop(f"var_seg_{bundle.card.model_id}_{bundle.card.task}", None)
                    st.session_state.pop("inspect_dialog_job_id", None)
                    st.switch_page("views/model_catalog.py")
            else:
                b1.caption("Note: Output bundle is no longer present on disk.")

            if b2.button(
                "Open Artifacts & Jobs Page",
                key=f"dlg_art_{job_id}",
                icon=":material/inventory_2:",
                width="stretch",
            ):
                st.session_state["active_job_id"] = job_id
                st.session_state.pop("inspect_dialog_job_id", None)
                st.switch_page("views/artifacts.py")
        elif rec.result_summary:
            st.success(
                f"Completed: built {rec.result_summary.get('built_count', 0)} artifact bundles.",
                icon=":material/check_circle:",
            )
            if st.button(
                "Open Artifacts & Jobs Page",
                key=f"dlg_art_reb_{job_id}",
                type="primary",
                icon=":material/inventory_2:",
                width="stretch",
            ):
                st.session_state["active_job_id"] = job_id
                st.session_state.pop("inspect_dialog_job_id", None)
                st.switch_page("views/artifacts.py")
    elif rec.status == "failed" and rec.error:
        st.error(f"Job failed: `{rec.error}`", icon=":material/error:")


@st.fragment(run_every=1.5)
def render_live_job_monitor(job_id: str) -> None:
    runner = get_cached_runner()
    rec = runner.get_job(job_id)
    if rec is None:
        st.warning(f"Job `{job_id}` not found.", icon=":material/warning:")
        return

    with glass_container("frosted"):
        h1, h2 = st.columns([3.5, 1.0])
        with h1:
            st.markdown(
                f"**{rec.title}** &nbsp; {_status_pill(rec.status)}",
                unsafe_allow_html=True,
            )
            st.caption(f"Job ID: `{rec.job_id}` · Status: {rec.message}")
        with h2:
            if rec.status in ("queued", "running"):
                if st.button("Cancel Job", key=f"cancel_{job_id}", icon=":material/stop_circle:", width="stretch"):
                    runner.cancel(job_id)
                    st.toast("Cancellation signal sent.")

        st.progress(float(rec.progress))

        logs = runner.read_logs(job_id, tail_lines=40)
        if logs:
            with st.expander("Execution Log Stream", icon=":material/terminal:", expanded=(rec.status in ("running", "failed"))):
                st.code(logs, language="text")

        if rec.status == "succeeded":
            bump_cache_token()
            if rec.result_bundle_id:
                st.success(
                    f"Saved bundle **`{rec.result_bundle_id}`** — "
                    f"{rec.result_summary.get('primary_metric', 'Score')}: "
                    f"`{rec.result_summary.get('primary_value')}` "
                    f"(`{rec.result_summary.get('train_time_s', 0):.2f}s`).",
                    icon=":material/check_circle:",
                )
                store = get_cached_store()
                try:
                    bundle = store.get_bundle(rec.result_bundle_id)
                    spec = get_spec(bundle.card.task, bundle.card.model_id)
                    render_bundle_detail_panel(bundle, spec, [bundle], store)
                except KeyError:
                    st.caption("Note: Output bundle is no longer present on disk.")
            elif rec.result_summary:
                st.success(
                    f"Completed: built {rec.result_summary.get('built_count', 0)} artifact bundles.",
                    icon=":material/check_circle:",
                )

        elif rec.status == "failed" and rec.error:
            st.error(f"Job failed: `{rec.error}`", icon=":material/error:")


def render_sidebar_jobs_summary() -> None:
    runner = get_cached_runner()
    jobs = runner.list_jobs(limit=6)
    if not jobs:
        return
    running = [j for j in jobs if j.status in ("queued", "running")]
    label = f"Background Jobs ({len(running)} active)" if running else f"Recent Jobs ({len(jobs)})"
    with st.sidebar.expander(label, icon=":material/history:", expanded=bool(running)):
        for j in jobs:
            st.caption(f"[{j.status.upper()}] **{j.title}** (`{j.progress * 100:.0f}%`)")
            if st.button("Inspect", key=f"sb_job_{j.job_id}", icon=":material/visibility:", width="stretch"):
                st.session_state["active_job_id"] = j.job_id
                st.session_state["inspect_dialog_job_id"] = j.job_id
