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
import sys
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
OUTPUT_DIR = ROOT / "outputs" / "regression" / "linear_regresssion"
ARTIFACT_DIR = ROOT / "artifacts" / "regression" / "linear_regresssion"
MODEL_PATH = ARTIFACT_DIR / "linear_regression.joblib"
SUMMARY_PATH = ARTIFACT_DIR / "training_summary.json"
SELECTED_CONFIG_PATH = ARTIFACT_DIR / "selected_config.json"


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
def save_model(model) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return MODEL_PATH


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run train.py first.")
    return joblib.load(MODEL_PATH)


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"{SUMMARY_PATH} not found. Run train.py first.")
    return json.loads(SUMMARY_PATH.read_text())


def load_selected_config() -> RegressionConfig:
    """Config chosen by the variant tests, or the defaults if none was saved."""
    if SELECTED_CONFIG_PATH.exists():
        return RegressionConfig.from_dict(json.loads(SELECTED_CONFIG_PATH.read_text()))
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
