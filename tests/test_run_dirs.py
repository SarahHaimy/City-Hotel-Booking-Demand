from pathlib import Path
import os
import pytest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.regression.linear_regression import test as test_module
from src.regression.linear_regression import train as train_module
from src.regression.linear_regression.features import RegressionConfig
from src.regression.linear_regression.linear_regression import (
    get_artifact_dir,
    get_latest_run_name,
    get_next_run_dir,
    get_run_dir,
    load_model,
    save_model,
)


def test_get_next_run_dir_empty_base(tmp_path: Path):
    next_id, run_dir = get_next_run_dir(base_dir=tmp_path, min_run=2)
    assert next_id == 2
    assert run_dir == tmp_path / "2"
    assert run_dir.is_dir()


def test_get_next_run_dir_with_folder_1(tmp_path: Path):
    (tmp_path / "1").mkdir()
    next_id, run_dir = get_next_run_dir(base_dir=tmp_path, min_run=2)
    assert next_id == 2
    assert run_dir == tmp_path / "2"
    assert run_dir.is_dir()


def test_get_next_run_dir_increments_consecutively(tmp_path: Path):
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()
    (tmp_path / "random_file.txt").write_text("ignore")
    (tmp_path / "non_numeric_dir").mkdir()

    next_id, run_dir = get_next_run_dir(base_dir=tmp_path, min_run=2)
    assert next_id == 3
    assert run_dir == tmp_path / "3"
    assert run_dir.is_dir()


def test_get_run_dir(tmp_path: Path):
    (tmp_path / "1").mkdir()
    (tmp_path / "2").mkdir()

    assert get_run_dir(1, base_dir=tmp_path) == tmp_path / "1"
    assert get_run_dir(2, base_dir=tmp_path) == tmp_path / "2"
    assert get_run_dir(None, base_dir=tmp_path) == tmp_path / "2"


def test_get_artifact_dir_and_persistence(tmp_path: Path):
    artifact_base = tmp_path / "artifacts"
    artifact_base.mkdir()

    with patch("src.regression.linear_regression.linear_regression.ARTIFACT_DIR", artifact_base), \
         patch("src.regression.linear_regression.linear_regression.MODEL_PATH", artifact_base / "linear_regression.joblib"), \
         patch("src.regression.linear_regression.linear_regression.SUMMARY_PATH", artifact_base / "training_summary.json"), \
         patch("src.regression.linear_regression.linear_regression.SELECTED_CONFIG_PATH", artifact_base / "selected_config.json"):

        assert get_artifact_dir() == artifact_base
        assert get_artifact_dir(1) == artifact_base / "1"

        dummy_model_1 = LinearRegression()
        dummy_model_1.fit([[1]], [1])
        save_model(dummy_model_1, run_id=1)

        dummy_model_2 = LinearRegression()
        dummy_model_2.fit([[2]], [2])
        save_model(dummy_model_2, run_id=2)

        # Root has latest (run 2, fits [[2]], [2])
        loaded_latest = load_model()
        assert loaded_latest.predict([[1]])[0] == pytest.approx(2.0)
        assert (artifact_base / "1" / "linear_regression.joblib").exists()
        assert (artifact_base / "2" / "linear_regression.joblib").exists()

        # Loading run 1 specifically (fits [[1]], [1])
        loaded_1 = load_model(run_id=1)
        assert loaded_1.predict([[1]])[0] == pytest.approx(1.0)


def test_named_run_directories_and_model_persistence(tmp_path: Path):
    output_base = tmp_path / "outputs"
    artifact_base = tmp_path / "artifacts"
    model_path = artifact_base / "linear_regression.joblib"

    with patch("src.regression.linear_regression.linear_regression.ARTIFACT_DIR", artifact_base), \
         patch("src.regression.linear_regression.linear_regression.MODEL_PATH", model_path):
        assert get_run_dir(base_dir=output_base, run_name="base") == output_base / "base"
        assert get_artifact_dir(run_name="base") == artifact_base / "base"

        model = LinearRegression().fit([[1], [2]], [1, 2])
        saved_path = save_model(model, run_name="base")

        assert saved_path == artifact_base / "base" / "linear_regression.joblib"
        assert saved_path.exists()
        assert not model_path.exists()
        loaded = load_model(run_name="base")
        assert loaded.predict([[3]])[0] == pytest.approx(3.0)


def test_get_latest_run_name(tmp_path: Path):
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    older.mkdir()
    newer.mkdir()
    (older / "marker").write_text("")
    (newer / "marker").write_text("")
    os.utime(older, (1, 1))

    assert get_latest_run_name(tmp_path) == "newer"


def test_get_run_dir_rejects_path_traversal(tmp_path: Path):
    with pytest.raises(ValueError, match="run_name"):
        get_run_dir(base_dir=tmp_path, run_name="..\\outside")


def test_train_base_bypasses_selected_config(tmp_path: Path):
    output_dir = tmp_path / "outputs" / "base"
    with patch.object(train_module, "load_split", return_value=pd.DataFrame()), \
         patch.object(train_module, "get_run_dir", return_value=output_dir), \
         patch.object(train_module, "train_final") as train_final, \
         patch.object(train_module, "load_selected_config", side_effect=AssertionError):
        train_module.main(["--base", "--run-name", "base"])

    cfg = train_final.call_args.args[0]
    assert isinstance(cfg, RegressionConfig)
    assert cfg == RegressionConfig()
    assert train_final.call_args.kwargs["save_selected_config"] is False
    assert train_final.call_args.kwargs["run_name"] == "base"


def test_test_entrypoint_selects_and_writes_named_run(tmp_path: Path):
    output_dir = tmp_path / "outputs" / "base"
    summary = {"iqr_limits": None, "config": {"log_target": False}}
    with patch.object(test_module, "load_summary", return_value=summary) as load_summary, \
         patch.object(test_module, "load_model") as load_model, \
         patch.object(test_module, "get_run_dir", return_value=output_dir), \
         patch.object(test_module, "load_split", return_value=pd.DataFrame({"ADR": [1.0]})), \
         patch.object(test_module, "evaluate", return_value=({"MAE": 0.0}, np.array([1.0]))), \
         patch.object(test_module, "save_diagnostic_plots"):
        test_module.main(["--run-name", "base"])

    load_summary.assert_called_once_with(run_id=None, run_name="base")
    load_model.assert_called_once_with(run_id=None, run_name="base")
    assert (output_dir / "test_metrics.csv").exists()
    assert (output_dir / "test_predictions.csv").exists()
