"""
linear_regression.py

The linear regression ADR baseline: model builder, metrics, diagnostic plots,
paths, and save/load helpers. train.py and test.py are the entry points.

Pipeline (one object, saved with joblib):

    TransformedTargetRegressor            (raw ADR, or log1p/expm1, clipped at 0)
      Pipeline
        engineer_features                 raw booking fields -> derived columns
        ColumnTransformer                 impute, scale, one hot (fit on train only)
        LinearRegression                  sklearn, untuned

Because feature engineering is inside the pipeline, model.predict() accepts a
dataframe of raw booking fields.

Do not run this file directly. The target transform helpers below are pickled
with the model, so they must always be imported as
src.regression.linear_regression.linear_regression, never run as __main__.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regression.linear_regression.features import (  # noqa: E402
    TARGET,
    RegressionConfig,
    engineer_features,
)
from src.regression.linear_regression.preprocessing import (  # noqa: E402
    build_preprocessor,
    within_limits,
)

SEED = 42

DATA_DIR = ROOT / "data" / "preprocessed"
BASE_OUTPUT_DIR = ROOT / "outputs" / "regression" / "linear_regression"
OUTPUT_DIR = BASE_OUTPUT_DIR
ARTIFACT_DIR = ROOT / "artifacts" / "regression" / "linear_regression"
MODEL_PATH = ARTIFACT_DIR / "linear_regression.joblib"
SUMMARY_PATH = ARTIFACT_DIR / "training_summary.json"
SELECTED_CONFIG_PATH = ARTIFACT_DIR / "selected_config.json"


# ----------------------------------------------------------------------------
# Output directories & run management
# ----------------------------------------------------------------------------
def get_next_run_dir(base_dir: Optional[Path] = None, min_run: int = 2) -> Tuple[int, Path]:
    """Find the next incremental run number (starting at min_run) and create its directory.

    Inspects base_dir for integer-named subdirectories, finds the current maximum,
    and returns (next_run_id, base_dir / str(next_run_id)).
    """
    base = Path(base_dir) if base_dir is not None else BASE_OUTPUT_DIR
    base.mkdir(parents=True, exist_ok=True)
    existing_runs = [
        int(p.name) for p in base.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    next_id = max(existing_runs + [min_run - 1]) + 1
    run_dir = base / str(next_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return next_id, run_dir


def new_run_name() -> str:
    """Return a timestamp suitable for a unique run directory name."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _validate_run_name(run_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError(
            "run_name must start with a letter or digit and contain only "
            "letters, digits, periods, underscores, or hyphens"
        )
    return run_name


def get_latest_run_name(base_dir: Optional[Path] = None) -> Optional[str]:
    """Return the most recently modified run directory name, if any."""
    base = Path(base_dir) if base_dir is not None else BASE_OUTPUT_DIR
    if not base.exists():
        return None
    run_dirs = [path for path in base.iterdir() if path.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda path: path.stat().st_mtime_ns).name


def get_run_dir(
    run_id: Optional[int] = None,
    base_dir: Optional[Path] = None,
    run_name: Optional[str] = None,
) -> Path:
    """Return a named/numeric run directory, or the latest legacy numeric run."""
    if run_id is not None and run_name is not None:
        raise ValueError("specify either run_id or run_name, not both")
    base = Path(base_dir) if base_dir is not None else BASE_OUTPUT_DIR
    base.mkdir(parents=True, exist_ok=True)
    if run_name is not None:
        run_dir = base / _validate_run_name(run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    if run_id is not None:
        run_dir = base / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    existing_runs = [
        int(p.name) for p in base.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    if existing_runs:
        return base / str(max(existing_runs))
    return base


# ----------------------------------------------------------------------------
# Data access
# ----------------------------------------------------------------------------
def load_split(name: str) -> pd.DataFrame:
    """Read data/preprocessed/<name>.csv (name is train, val or test).

    Swap this for the project loader if you prefer. The split files are
    written once and shared by every model, so nothing here re-splits.
    """
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Create the shared split first.")
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Features are the raw columns. engineer_features() picks what it needs."""
    return df.drop(columns=[TARGET]), df[TARGET]


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def _identity(y):
    return y


def _clip_nonnegative(y):
    return np.clip(y, 0, None)


def _expm1_clip(z):
    return np.clip(np.expm1(z), 0, None)


def build_model(cfg: Optional[RegressionConfig] = None) -> TransformedTargetRegressor:
    """Unfitted model for a config. Negative predictions are clipped to 0."""
    cfg = cfg or RegressionConfig()
    pipeline = Pipeline(
        [
            ("engineer", FunctionTransformer(engineer_features, validate=False)),
            ("preprocess", build_preprocessor(cfg)),
            ("regressor", LinearRegression()),
        ]
    )
    if cfg.log_target:
        func, inverse = np.log1p, _expm1_clip
    else:
        func, inverse = _identity, _clip_nonnegative
    return TransformedTargetRegressor(
        regressor=pipeline, func=func, inverse_func=inverse, check_inverse=False
    )


def feature_names(model) -> list:
    """Names of the columns the linear model actually sees (after fitting)."""
    return list(model.regressor_.named_steps["preprocess"].get_feature_names_out())


def coefficient_table(model) -> pd.DataFrame:
    """Coefficients on the transformed target scale, largest magnitude first."""
    lr = model.regressor_.named_steps["regressor"]
    table = pd.DataFrame({"feature": feature_names(model), "coefficient": lr.coef_})
    table.loc[len(table)] = ["(intercept)", float(lr.intercept_)]
    order = table["coefficient"].abs().sort_values(ascending=False).index
    return table.loc[order].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Metrics (always on the original ADR scale)
# ----------------------------------------------------------------------------
def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate(model, df: pd.DataFrame, limits=None) -> Tuple[dict, np.ndarray]:
    """Metrics on every row of df, plus secondary metrics on rows inside the
    training IQR limits when limits are given. Returns (metrics, predictions)."""
    X, y = split_xy(df)
    pred = model.predict(X)
    metrics = regression_metrics(y, pred)
    metrics["n_rows"] = int(len(df))
    if limits is not None:
        mask = within_limits(y, limits)
        sub = regression_metrics(y[mask], pred[mask])
        metrics.update({f"{k}_in_limits": v for k, v in sub.items()})
        metrics["n_rows_in_limits"] = int(mask.sum())
    return metrics, pred


def cv_rmse(cfg: RegressionConfig, train_df: pd.DataFrame, n_splits: int = 5) -> float:
    """Mean 5 fold RMSE on the training rows, on the original ADR scale."""
    X, y = split_xy(train_df)
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = cross_val_score(
        build_model(cfg), X, y, cv=folds, scoring="neg_root_mean_squared_error"
    )
    return float(-scores.mean())


# ----------------------------------------------------------------------------
# Diagnostic plots
# ----------------------------------------------------------------------------
def save_diagnostic_plots(y_true, y_pred, out_dir: Path, prefix: str) -> None:
    """Residuals vs predicted, and predicted vs actual. One accent color,
    thin marks, recessive grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accent, ink, muted, grid = "#2a6fdb", "#1f2328", "#6b7280", "#e5e7eb"
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    out_dir.mkdir(parents=True, exist_ok=True)

    def style(ax, xlabel, ylabel, title):
        ax.set_xlabel(xlabel, color=muted)
        ax.set_ylabel(ylabel, color=muted)
        ax.set_title(title, loc="left", color=ink, fontsize=11)
        ax.grid(True, color=grid, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted)

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=150)
    ax.scatter(y_pred, residual, s=4, alpha=0.25, color=accent, linewidths=0)
    ax.axhline(0, color=ink, linewidth=1)
    style(ax, "Predicted ADR", "Residual (actual minus predicted)",
          f"Residuals vs predicted ({prefix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_residuals_vs_predicted.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=150)
    ax.scatter(y_true, y_pred, s=4, alpha=0.25, color=accent, linewidths=0)
    top = float(max(y_true.max(), y_pred.max()))
    ax.plot([0, top], [0, top], color=ink, linewidth=1)
    ax.text(top * 0.98, top * 0.9, "perfect fit", color=muted, ha="right", fontsize=8)
    style(ax, "Actual ADR", "Predicted ADR", f"Predicted vs actual ({prefix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_predicted_vs_actual.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------------
def get_artifact_dir(run_id: Optional[int] = None, run_name: Optional[str] = None) -> Path:
    """Return the artifact directory for a specific run or root ARTIFACT_DIR."""
    if run_id is not None and run_name is not None:
        raise ValueError("specify either run_id or run_name, not both")
    if run_name is not None:
        path = ARTIFACT_DIR / _validate_run_name(run_name)
        path.mkdir(parents=True, exist_ok=True)
        return path
    if run_id is not None:
        path = ARTIFACT_DIR / str(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR


def save_model(model, run_id: Optional[int] = None, run_name: Optional[str] = None) -> Path:
    """Save a model to a run directory or to the legacy root artifact path."""
    if run_id is not None and run_name is not None:
        raise ValueError("specify either run_id or run_name, not both")
    if run_name is not None:
        model_path = get_artifact_dir(run_name=run_name) / "linear_regression.joblib"
        joblib.dump(model, model_path)
        return model_path
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    if run_id is not None:
        run_artifact_dir = get_artifact_dir(run_id)
        joblib.dump(model, run_artifact_dir / "linear_regression.joblib")
    return MODEL_PATH


def load_model(run_id: Optional[int] = None, run_name: Optional[str] = None):
    """Load a run-specific model if specified, else the legacy root MODEL_PATH."""
    if run_id is not None and run_name is not None:
        raise ValueError("specify either run_id or run_name, not both")
    if run_name is not None:
        target = get_artifact_dir(run_name=run_name) / "linear_regression.joblib"
    else:
        target = (get_artifact_dir(run_id) / "linear_regression.joblib") if run_id is not None else MODEL_PATH
    if not target.exists():
        raise FileNotFoundError(f"{target} not found. Run train.py first.")
    return joblib.load(target)


def load_summary(run_id: Optional[int] = None, run_name: Optional[str] = None) -> dict:
    """Load a run-specific summary if specified, else the legacy root summary."""
    if run_id is not None and run_name is not None:
        raise ValueError("specify either run_id or run_name, not both")
    if run_name is not None:
        target = get_artifact_dir(run_name=run_name) / "training_summary.json"
    else:
        target = (get_artifact_dir(run_id) / "training_summary.json") if run_id is not None else SUMMARY_PATH
    if not target.exists():
        raise FileNotFoundError(f"{target} not found. Run train.py first.")
    return json.loads(target.read_text())


def load_selected_config(run_id: Optional[int] = None, run_name: Optional[str] = None) -> RegressionConfig:
    """Config chosen by the variant tests, or defaults if none was saved."""
    if run_id is not None and run_name is not None:
        raise ValueError("specify either run_id or run_name, not both")
    if run_name is not None:
        target = get_artifact_dir(run_name=run_name) / "selected_config.json"
    else:
        target = (get_artifact_dir(run_id) / "selected_config.json") if run_id is not None else SELECTED_CONFIG_PATH
    if target.exists():
        return RegressionConfig.from_dict(json.loads(target.read_text()))
    return RegressionConfig()


# ----------------------------------------------------------------------------
# Entry point for main.py
# ----------------------------------------------------------------------------
def run(stage: str = "train") -> None:
    """stage is 'train' (train on train.csv) or 'test' (final test.csv report)."""
    if stage == "train":
        from src.regression.linear_regression import train

        train.main([])
    elif stage == "test":
        from src.regression.linear_regression import test

        test.main([])
    else:
        raise ValueError("stage must be 'train' or 'test'")
