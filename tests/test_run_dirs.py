from pathlib import Path
import pytest
from unittest.mock import patch
from sklearn.linear_model import LinearRegression

from src.regression.linear_regression.linear_regression import (
    get_artifact_dir,
    get_next_run_dir,
    get_run_dir,
    load_model,
    load_selected_config,
    load_summary,
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

