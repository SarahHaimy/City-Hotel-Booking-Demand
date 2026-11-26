"""Artifacts & Jobs Management page — verify bundle health, switch deployed models, import/export ZIPs, rebuild official bundles (emoji-free)."""
from __future__ import annotations

import streamlit as st

from app.cache import get_cached_runner, get_cached_store
from app.components.job_monitor import render_live_job_monitor
from app.components.style import glass_container, page_header, pill_html, run_page_safely, section_header
from app.state import bump_cache_token
from src.training.service import build_official_artifacts


def _render() -> None:
    page_header(
        "Artifact Store, Deployment Control & Job History",
        "Verify serialized pipeline health, select active deployed models per task, import/export .zip bundles, or regenerate official artifacts",
        pill_html("Zero Startup Retraining", "deployed"),
    )

    store = get_cached_store()
    bundles = store.list_bundles()

    section_header(
        "Active Deployed Models per Task",
        "Choose which saved artifact serves as the default production pipeline for each analytical track.",
        kicker="1 · Deployment Routing",
    )
    dcols = st.columns(3, gap="medium")
    for col, tname in zip(dcols, ("regression", "classification", "clustering")):
        tb = [b for b in bundles if b.card.task == tname and (tname != "clustering" or b.card.supports_predict_new)]
        with col:
            with glass_container("frosted"):
                st.markdown(f'<div class="section-kicker">{tname}</div>', unsafe_allow_html=True)
                if not tb:
                    st.caption("No eligible bundles.")
                else:
                    cur_dep = store.get_deployed(tname)
                    opts = {f"{b.card.display_name} ({b.card.variant}, {b.card.source})": b.card.bundle_id for b in tb}
                    cur_idx = 0
                    for idx, bid in enumerate(opts.values()):
                        if cur_dep and bid == cur_dep.card.bundle_id:
                            cur_idx = idx
                    sel_lbl = st.selectbox("Active Bundle", options=list(opts.keys()), index=cur_idx, key=f"art_dep_{tname}")
                    if cur_dep is None or opts[sel_lbl] != cur_dep.card.bundle_id:
                        if st.button(f"Activate for {tname.title()}", key=f"art_set_{tname}", type="primary", icon=":material/check_circle:", width="stretch"):
                            store.set_deployed(tname, opts[sel_lbl])
                            bump_cache_token()
                            st.success(f"Updated deployed {tname} bundle.", icon=":material/check_circle:")
                            st.rerun()
                    else:
                        st.caption(":material/verified: Currently active in production routing.")

    section_header(
        "Saved Bundle Inventory & Integrity Verification",
        "Inspect serialized artifact sizes, scikit-learn versions, primary metrics, and pipeline health.",
        kicker="2 · Artifact Registry",
    )
    run_full_verify = st.checkbox("Run live `joblib.load()` deserialization health check on every bundle", value=False)
    rows = []
    for b in bundles:
        if run_full_verify:
            h = store.verify_bundle(b)
            status_str = {"ok": "Healthy", "warning": f"Warning: {h.message}", "error": f"Error: {h.message}"}[h.status]
            sz_kb = round(h.size_bytes / 1024.0, 1)
        else:
            exists = b.pipeline_path.exists()
            status_str = "Present" if exists else "Missing pipeline.joblib"
            sz_kb = round(b.size_bytes() / 1024.0, 1)
        m_name, m_val = b.card.primary_metric()
        rows.append(
            {
                "Bundle ID": b.card.bundle_id,
                "Task": b.card.task.title(),
                "Model": b.card.display_name,
                "Variant": b.card.variant,
                "Space": b.card.feature_space,
                "Source": b.card.source,
                "Deployed": "Active" if b.card.is_deployed else "",
                "Primary Metric": f"{m_name} = {m_val:.4f}" if m_val is not None else "—",
                "Size (KB)": sz_kb,
                "sklearn": b.card.sklearn_version,
                "Health": status_str,
            }
        )
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.warning("No bundles found on disk yet.", icon=":material/warning:")

    section_header(
        "Import, Export & Rebuild Utilities",
        "Import external `.zip` model archives into the user sandbox or regenerate all official project bundles.",
        kicker="3 · Maintenance",
    )
    u1, u2 = st.columns(2, gap="medium")
    with u1:
        with glass_container("frosted"):
            st.markdown('<div class="section-kicker">Archive Import</div>', unsafe_allow_html=True)
            st.markdown("##### :material/upload_file: Import Saved Bundle Archive (`.zip`)")
            up_zip = st.file_uploader("Upload exported bundle .zip", type=["zip"], key="art_zip_up")
            if up_zip is not None and st.button("Import Bundle into User Store", icon=":material/archive:", type="primary", width="stretch"):
                try:
                    imp = store.import_bundle_zip(up_zip.read(), force_user_namespace=True)
                    bump_cache_token()
                    st.success(f"Imported `{imp.card.bundle_id}` successfully.", icon=":material/check_circle:")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Import failed: {exc}", icon=":material/error:")

    with u2:
        with glass_container("frosted"):
            st.markdown('<div class="section-kicker">Clean Reproduction</div>', unsafe_allow_html=True)
            st.markdown("##### :material/refresh: Rebuild Official Project Artifacts")
            st.caption("Submission ZIPs exclude binary `.joblib` files (Spec §17). Use this on a fresh clone to regenerate all official repository bundles.")
            fast_mode = st.checkbox("Fast preset mode (use validated tuned presets instead of full CV search)", value=False)
            if st.button("Rebuild Official Artifacts in Background", type="primary", icon=":material/play_arrow:", width="stretch"):
                runner = get_cached_runner()
                job = runner.submit(
                    title="Rebuild Official Project Artifacts",
                    kind="rebuild",
                    fn=build_official_artifacts,
                    kwargs=dict(core_only=False, run_grid_search=not fast_mode),
                )
                st.session_state["active_job_id"] = job.job_id

    active_job = st.session_state.get("active_job_id")
    if active_job:
        render_live_job_monitor(active_job)

    section_header(
        "Background Job Execution History",
        "Inspect execution logs, timestamps, and output artifacts for recent background runs.",
        kicker="4 · Job Ledger",
    )
    jobs = get_cached_runner().list_jobs(limit=25)
    if jobs:
        j_opts = {f"[{j.status.upper()}] {j.title} ({j.job_id})": j.job_id for j in jobs}
        jc1, jc2 = st.columns([3.2, 1.0], vertical_alignment="bottom")
        sel_job_lbl = jc1.selectbox("Select Recorded Job", options=list(j_opts.keys()), key="art_hist_job_sel")
        if jc2.button("Inspect Logs & Output", key="art_hist_inspect", icon=":material/visibility:", width="stretch"):
            st.session_state["active_job_id"] = j_opts[sel_job_lbl]
            st.session_state["inspect_dialog_job_id"] = j_opts[sel_job_lbl]
            st.rerun()
        st.dataframe([j.to_dict() for j in jobs], width="stretch", hide_index=True)
    else:
        st.caption("No background jobs recorded in `runs/` yet.")


run_page_safely(_render)
