"""Headless Streamlit AppTest smoke suite verifying every GUI page renders without exceptions."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_ENTRY = ROOT / "app" / "streamlit_app.py"

PAGE_PATHS = [
    "views/home.py",
    "views/data_explorer.py",
    "views/leakage_audit.py",
    "views/model_catalog.py",
    "views/compare.py",
    "views/tuning_lab.py",
    "views/predict.py",
    "views/batch_evaluate.py",
    "views/train_studio.py",
    "views/clustering.py",
    "views/pca_lda.py",
    "views/artifacts.py",
]


@pytest.mark.parametrize("page_rel", PAGE_PATHS)
def test_page_renders_without_exception(page_rel: str):
    at = AppTest.from_file(str(APP_ENTRY), default_timeout=45)
    at.run()
    assert not at.exception, f"Entry page raised exception: {at.exception}"

    at.switch_page(page_rel)
    at.run()
    assert not at.exception, f"Page {page_rel} raised exception: {at.exception}"
    # Also ensure our safe_page boundary did not catch a hidden runtime error
    err_msgs = [e.value for e in at.error]
    assert not any("An unexpected error occurred while rendering this view" in m for m in err_msgs), (
        f"Page {page_rel} rendered error boundary: {err_msgs}"
    )


def test_sidebar_recent_job_inspect_opens_dialog():
    import shutil
    from src.training.jobs import JobRecord, get_job_runner

    runner = get_job_runner()
    test_job_id = "99991231_235959_testjb"
    rec = JobRecord(
        job_id=test_job_id,
        title="Unit Test Regression Run",
        kind="train",
        status="succeeded",
        progress=1.0,
        message="Completed successfully",
        result_bundle_id="regression/linear_regression/variant_selected",
        result_summary={"primary_metric": "RMSE", "primary_value": 20.23, "train_time_s": 0.42},
    )
    runner._write_record(rec)
    runner._append_log(test_job_id, "Synthetic log line for job inspector dialog test")
    try:
        at = AppTest.from_file(str(APP_ENTRY), default_timeout=45)
        at.run()
        assert not at.exception

        btn = at.button(key=f"sb_job_{test_job_id}")
        btn.click()
        at.run()
        assert not at.exception
        assert at.session_state["inspect_dialog_job_id"] == test_job_id
        assert any("Synthetic log line for job inspector dialog test" in c.value for c in at.code)

        # Clicking "Open Bundle in Model Catalog" inside the dialog navigates cleanly
        open_cat_btn = at.button(key=f"dlg_cat_{test_job_id}")
        open_cat_btn.click()
        at.run()
        assert not at.exception
        assert at.session_state["catalog_selected_bundle"] == "regression/linear_regression/variant_selected"
    finally:
        shutil.rmtree(runner.runs_dir / test_job_id, ignore_errors=True)
