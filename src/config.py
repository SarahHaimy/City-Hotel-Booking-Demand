"""Project configuration loaded from config.yaml (single source of truth for paths/seeds)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

_DEFAULTS = {
    "seed": 42,
    "paths": {
        "raw_data": "data/H2.csv",
        "artifacts": "artifacts",
        "user_artifacts": "artifacts/user",
        "outputs": "outputs",
        "experiments_log": "outputs/experiments/results.csv",
        "runs": "runs",
        "samples": "data/samples",
    },
    "splits": {"regression": "data/preprocessed", "classification": "data/bayes"},
    "cleaning": {
        "regression": {"drop_zero_adr": True, "drop_company": True, "keep_zero_nights": False},
        "classification": {"drop_zero_adr": False, "drop_company": True, "keep_zero_nights": True},
        "clustering": {"drop_zero_adr": True, "drop_company": True, "keep_zero_nights": False},
    },
    "selection": {
        "regression": {"metric": "RMSE", "split": "val", "higher_is_better": False},
        "classification": {"metric": "F1", "split": "val", "higher_is_better": True},
        "clustering": {"metric": "silhouette", "split": "train", "higher_is_better": True},
    },
    "jobs": {"max_parallel": 1, "default_cv_folds": 5, "default_n_iter": 20, "default_timeout_s": 1800},
    "sampling": {
        "slow_model_rows": 15000,
        "agglomerative_rows": 6000,
        "dbscan_rows": 12000,
        "silhouette_rows": 10000,
        "scatter_points": 8000,
    },
    "app": {"title": "City Hotel Booking Demand — ML Studio", "max_upload_mb": 200},
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@lru_cache(maxsize=1)
def get_config() -> dict:
    data = {}
    if CONFIG_PATH.exists():
        data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return _merge(_DEFAULTS, data)


def path(key: str) -> Path:
    """Absolute path for a key under `paths:` in config.yaml."""
    return ROOT / get_config()["paths"][key]


def split_dir(task: str) -> Path:
    return ROOT / get_config()["splits"][task]


def cleaning_flags(task: str) -> dict:
    return dict(get_config()["cleaning"][task])


def selection(task: str) -> dict:
    return dict(get_config()["selection"][task])


SEED: int = int(get_config()["seed"])
