"""Integration tests for registry, artifact store, schema validation, training, inference, jobs, and CLI."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from main import main as cli_main
from src.data.schema import BOOKING_PRESETS
from src.data.validation import normalize_columns, validate_dataframe
from src.inference.predict import (
    predict_batch_dataframe,
    predict_single_booking,
    what_if_sweep,
)
from src.registry.artifacts import ArtifactStore
from src.registry.registry import get_spec, list_specs
from src.training.jobs import JobRunner
from src.training.service import (
    evaluate_bundle_on_data,
    prepare_user_splits,
    train_clustering_model,
    train_supervised_model,
)


def _synthetic_bookings(n: int = 160, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    presets = list(BOOKING_PRESETS.values())
    rows = []
    for i in range(n):
        base = dict(presets[i % len(presets)])
        base["LeadTime"] = int(rng.integers(0, 320))
        base["StaysInWeekNights"] = int(rng.integers(1, 7))
        base["StaysInWeekendNights"] = int(rng.integers(0, 3))
        base["Adults"] = int(rng.integers(1, 4))
        base["ADR"] = float(round(65.0 + 18.0 * base["Adults"] + 0.12 * base["LeadTime"] + rng.normal(0, 8), 2))
        cancel_logit = -1.2 + 0.008 * base["LeadTime"] + (0.9 if base["PreviousCancellations"] > 0 else 0.0)
        p = 1.0 / (1.0 + np.exp(-cancel_logit))
        base["IsCanceled"] = int(rng.random() < p)
        rows.append(base)
    return pd.DataFrame(rows)


def test_registry_has_all_three_tracks():
    assert [s.model_id for s in list_specs("regression")] == ["linear_regression"]
    assert [s.model_id for s in list_specs("classification")] == ["gaussian_nb"]
    assert [s.model_id for s in list_specs("clustering")] == ["kmeans"]
    assert get_spec("regression", "linear_regression").track_origin == "milestone_1"
    assert get_spec("classification", "gaussian_nb").track_origin == "milestone_1"
    assert get_spec("clustering", "kmeans").track_origin == "milestone_1"


def test_column_normalization_and_validation():
    df = _synthetic_bookings(40)
    snake = df.rename(columns={"LeadTime": "lead_time", "IsCanceled": "is_canceled", "ADR": "adr"})
    snake["ReservationStatus"] = np.where(snake["is_canceled"] == 1, "Canceled", "Check-Out")

    spec = get_spec("classification", "gaussian_nb")
    norm, rep = validate_dataframe(
        snake,
        task="classification",
        required_inputs=spec.required_inputs,
        forbidden_inputs=spec.forbidden_inputs,
        require_target=True,
    )
    assert rep.ok
    assert "LeadTime" in norm.columns and "IsCanceled" in norm.columns
    assert "ReservationStatus" in rep.dropped_leakage

    # Missing target should block training validation
    no_target = norm.drop(columns=["IsCanceled"])
    _, bad_rep = validate_dataframe(
        no_target, task="classification", required_inputs=spec.required_inputs, require_target=True
    )
    assert not bad_rep.ok
    assert any(i.code == "MISSING_TARGET" for i in bad_rep.errors)


def test_end_to_end_train_save_verify_zip_and_infer(tmp_path: Path):
    store = ArtifactStore(root_dir=tmp_path / "artifacts")
    df = _synthetic_bookings(180)
    tr, va, te, _ = prepare_user_splits(df, task="regression", apply_cleaning=False)

    reg_b = train_supervised_model(
        task="regression",
        model_id="linear_regression",
        mode="baseline",
        splits=(tr, va, te),
        source="project",
        evaluate_test=True,
        store=store,
    )
    clf_b = train_supervised_model(
        task="classification",
        model_id="gaussian_nb",
        mode="tune",
        cv_folds=3,
        n_iter=4,
        tune_threshold=True,
        splits=(tr, va, te),
        source="project",
        evaluate_test=True,
        store=store,
    )
    clu_b = train_clustering_model(
        model_id="kmeans",
        mode="baseline",
        df=df,
        source="project",
        store=store,
    )

    assert store.verify_bundle(reg_b).status == "ok"
    assert store.verify_bundle(clf_b).status == "ok"
    assert store.verify_bundle(clu_b).status == "ok"

    # Single booking prediction + what-if sweep + batch prediction
    sample_booking = next(iter(BOOKING_PRESETS.values()))
    single = predict_single_booking(sample_booking, store=store)
    assert single["regression"]["pred_adr"] > 0
    assert 0.0 <= single["classification"]["cancel_proba"] <= 1.0
    assert single["clustering"]["cluster"] in (0, 1, 2)

    sw = what_if_sweep(sample_booking, "LeadTime", [5, 50, 200], store=store)
    assert len(sw) == 3 and "Predicted ADR (€)" in sw.columns

    batch = predict_batch_dataframe(
        va.head(15),
        reg_bundle_id=reg_b.card.bundle_id,
        clf_bundle_id=clf_b.card.bundle_id,
        clu_bundle_id=clu_b.card.bundle_id,
        store=store,
    )
    assert {"pred_ADR", "p_IsCanceled", "pred_IsCanceled", "pred_Cluster"}.issubset(batch.columns)

    ev = evaluate_bundle_on_data(clf_b, va)
    assert "F1" in ev["metrics"] and "ROC_AUC" in ev["metrics"]

    # ZIP export & import into user namespace
    zbytes = store.export_bundle_zip(reg_b.card.bundle_id)
    imp = store.import_bundle_zip(zbytes, force_user_namespace=True)
    assert imp.card.source == "user"
    assert store.verify_bundle(imp).status == "ok"


def test_job_runner_background_execution(tmp_path: Path):
    runner = JobRunner(runs_dir=tmp_path / "runs")

    def _dummy_job(x: int, progress_cb=None):
        if progress_cb:
            progress_cb(0.5, "Halfway")
        return [f"item_{i}" for i in range(x)]

    rec = runner.submit("Dummy Test Job", "train", _dummy_job, kwargs={"x": 3})
    for _ in range(30):
        cur = runner.get_job(rec.job_id)
        if cur and cur.status in ("succeeded", "failed"):
            break
        time.sleep(0.05)
    final = runner.get_job(rec.job_id)
    assert final is not None and final.status == "succeeded"
    assert final.result_summary["built_count"] == 3
    assert "Halfway" in runner.read_logs(rec.job_id)


def test_cli_list_models(capsys: pytest.CaptureFixture[str]):
    rc = cli_main(["list-models", "--task", "regression"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "linear_regression" in out
