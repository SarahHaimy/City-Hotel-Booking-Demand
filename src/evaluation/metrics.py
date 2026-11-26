"""Shared evaluation metrics and metadata across Regression, Classification, and Clustering."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)

from src.config import SEED

# Display metadata used by metric tiles, leaderboards, and chart tooltips.
METRIC_META: dict[str, dict[str, Any]] = {
    # Regression
    "MAE": {
        "label": "MAE (€)",
        "unit": "€",
        "fmt": "{:.2f}",
        "higher_is_better": False,
        "help": "Mean Absolute Error on original ADR scale (EUR/night). Lower is better.",
    },
    "MSE": {
        "label": "MSE (€²)",
        "unit": "€²",
        "fmt": "{:.2f}",
        "higher_is_better": False,
        "help": "Mean Squared Error on original ADR scale. Lower is better.",
    },
    "RMSE": {
        "label": "RMSE (€)",
        "unit": "€",
        "fmt": "{:.2f}",
        "higher_is_better": False,
        "help": "Root Mean Squared Error on original ADR scale (EUR/night); penalizes large pricing misses. Lower is better.",
    },
    "MedAE": {
        "label": "Median AE (€)",
        "unit": "€",
        "fmt": "{:.2f}",
        "higher_is_better": False,
        "help": "Median Absolute Error (robust to extreme suite/event pricing outliers). Lower is better.",
    },
    "R2": {
        "label": "R²",
        "unit": "",
        "fmt": "{:.4f}",
        "higher_is_better": True,
        "help": "Proportion of ADR variance explained (1.0 = perfect, 0.0 = mean predictor). Higher is better.",
    },
    # Classification
    "Accuracy": {
        "label": "Accuracy",
        "unit": "",
        "fmt": "{:.4f}",
        "higher_is_better": True,
        "help": "Overall fraction of bookings classified correctly. Higher is better.",
    },
    "Precision": {
        "label": "Precision",
        "unit": "",
        "fmt": "{:.4f}",
        "higher_is_better": True,
        "help": "Among bookings predicted to cancel, fraction that actually canceled (controls false alarms). Higher is better.",
    },
    "Recall": {
        "label": "Recall",
        "unit": "",
        "fmt": "{:.4f}",
        "higher_is_better": True,
        "help": "Among bookings that actually canceled, fraction caught by the model. Higher is better.",
    },
    "F1": {
        "label": "F1-Score",
        "unit": "",
        "fmt": "{:.4f}",
        "higher_is_better": True,
        "help": "Harmonic mean of Precision and Recall on the canceled class. Higher is better.",
    },
    "ROC_AUC": {
        "label": "ROC-AUC",
        "unit": "",
        "fmt": "{:.4f}",
        "higher_is_better": True,
        "help": "Area Under the Receiver Operating Characteristic curve (threshold-independent ranking). Higher is better.",
    },
    "PR_AUC": {
        "label": "PR-AUC",
        "unit": "",
        "fmt": "{:.4f}",
        "higher_is_better": True,
        "help": "Area Under the Precision-Recall curve for the canceled class. Higher is better.",
    },
    "Brier": {
        "label": "Brier Score",
        "unit": "",
        "fmt": "{:.4f}",
        "higher_is_better": False,
        "help": "Mean squared error of predicted cancellation probabilities (calibration quality). Lower is better.",
    },
    # Clustering
    "silhouette": {
        "label": "Silhouette",
        "unit": "",
        "fmt": "{:.4f}",
        "higher_is_better": True,
        "help": "Mean intra-cluster cohesion vs nearest-cluster separation in [-1, 1]. Higher is better.",
    },
    "davies_bouldin": {
        "label": "Davies–Bouldin",
        "unit": "",
        "fmt": "{:.4f}",
        "higher_is_better": False,
        "help": "Average similarity ratio of each cluster with its most similar neighbor. Lower is better.",
    },
    "calinski_harabasz": {
        "label": "Calinski–Harabasz",
        "unit": "",
        "fmt": "{:.1f}",
        "higher_is_better": True,
        "help": "Ratio of between-cluster dispersion to within-cluster dispersion. Higher is better.",
    },
}

TASK_METRICS: dict[str, list[str]] = {
    "regression": ["RMSE", "MAE", "MSE", "MedAE", "R2"],
    "classification": ["F1", "ROC_AUC", "PR_AUC", "Accuracy", "Precision", "Recall", "Brier"],
    "clustering": ["silhouette", "davies_bouldin", "calinski_harabasz"],
}


def format_metric(metric_name: str, value: Any, with_arrow: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    meta = METRIC_META.get(metric_name)
    if not meta:
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    txt = meta["fmt"].format(float(value))
    if with_arrow:
        arrow = "↑" if meta["higher_is_better"] else "↓"
        return f"{txt} {arrow}"
    return txt


def compute_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(yt, yp))
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MedAE": float(median_absolute_error(yt, yp)),
        "R2": float(r2_score(yt, yp)),
        "n_rows": int(len(yt)),
    }


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=int)
    if y_proba is not None:
        prob = np.asarray(y_proba, dtype=float)
        yp = (prob >= float(threshold)).astype(int)
    else:
        prob = None
        yp = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "Accuracy": float(accuracy_score(yt, yp)),
        "Precision": float(precision_score(yt, yp, zero_division=0)),
        "Recall": float(recall_score(yt, yp, zero_division=0)),
        "F1": float(f1_score(yt, yp, zero_division=0)),
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "n_rows": int(len(yt)),
    }
    if prob is not None and len(np.unique(yt)) == 2:
        out["ROC_AUC"] = float(roc_auc_score(yt, prob))
        out["PR_AUC"] = float(average_precision_score(yt, prob))
        out["Brier"] = float(brier_score_loss(yt, np.clip(prob, 0.0, 1.0)))
    else:
        out["ROC_AUC"] = None
        out["PR_AUC"] = None
        out["Brier"] = None
    return out


def sweep_thresholds(
    y_true: Any,
    y_proba: Any,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    yt = np.asarray(y_true, dtype=int)
    prob = np.asarray(y_proba, dtype=float)
    grid = thresholds or [round(t, 2) for t in np.linspace(0.15, 0.85, 15)]
    rows = []
    neg_total = max(int((yt == 0).sum()), 1)
    for thr in grid:
        yp = (prob >= thr).astype(int)
        fp = int(((yp == 1) & (yt == 0)).sum())
        rows.append(
            {
                "threshold": float(thr),
                "Accuracy": float(accuracy_score(yt, yp)),
                "Precision": float(precision_score(yt, yp, zero_division=0)),
                "Recall": float(recall_score(yt, yp, zero_division=0)),
                "F1": float(f1_score(yt, yp, zero_division=0)),
                "FalseAlarmRate": float(fp / neg_total),
            }
        )
    return pd.DataFrame(rows)


def segment_error_table(
    df: pd.DataFrame,
    y_true: Any,
    y_pred: Any,
    group_col: str = "MarketSegment",
) -> pd.DataFrame:
    """Segment-level cancellation & false-alarm breakdown (generalizes final_gaussian_nb.py)."""
    if group_col not in df.columns:
        return pd.DataFrame()
    work = pd.DataFrame(
        {
            group_col: df[group_col].astype(str).values,
            "Actual": np.asarray(y_true, dtype=int),
            "Predicted": np.asarray(y_pred, dtype=int),
        }
    )
    work["FalseAlarm"] = ((work["Actual"] == 0) & (work["Predicted"] == 1)).astype(int)
    work["MissedCancel"] = ((work["Actual"] == 1) & (work["Predicted"] == 0)).astype(int)
    work["NotCanceled"] = (work["Actual"] == 0).astype(int)

    grp = work.groupby(group_col, dropna=False).agg(
        Bookings=("Actual", "count"),
        ActualCancelRate=("Actual", "mean"),
        PredCancelRate=("Predicted", "mean"),
        NotCanceledBookings=("NotCanceled", "sum"),
        FalseAlarms=("FalseAlarm", "sum"),
        MissedCancels=("MissedCancel", "sum"),
    )
    grp["FalseAlarmRate"] = np.where(
        grp["NotCanceledBookings"] > 0,
        grp["FalseAlarms"] / grp["NotCanceledBookings"],
        0.0,
    )
    return grp.reset_index().sort_values("Bookings", ascending=False).reset_index(drop=True)


def compute_clustering_metrics(
    X_transformed: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 10_000,
    seed: int = SEED,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Internal clustering metrics; handles DBSCAN noise points (label == -1) cleanly."""
    lbl = np.asarray(labels, dtype=int)
    n_total = len(lbl)
    noise_mask = lbl == -1
    noise_cnt = int(noise_mask.sum())
    core_mask = ~noise_mask
    unique_clusters = sorted(c for c in np.unique(lbl) if c >= 0)
    n_clusters = len(unique_clusters)

    sizes = {str(c): int((lbl == c).sum()) for c in unique_clusters}
    if noise_cnt > 0:
        sizes["noise (-1)"] = noise_cnt

    out: dict[str, Any] = {
        "n_clusters": int(n_clusters),
        "n_rows": int(n_total),
        "noise_count": noise_cnt,
        "noise_pct": float(noise_cnt / max(n_total, 1)),
        "cluster_sizes": [int((lbl == c).sum()) for c in unique_clusters],
        "cluster_size_map": sizes,
        "min_cluster_pct": (
            float(min((lbl == c).sum() for c in unique_clusters) / max(n_total, 1))
            if n_clusters > 0
            else 0.0
        ),
    }

    X_eval = np.asarray(X_transformed, dtype=float)[core_mask]
    lbl_eval = lbl[core_mask]
    if n_clusters >= 2 and len(lbl_eval) > n_clusters:
        samp = min(sample_size, len(lbl_eval))
        out["silhouette"] = float(
            silhouette_score(X_eval, lbl_eval, sample_size=samp, random_state=seed)
        )
        out["davies_bouldin"] = float(davies_bouldin_score(X_eval, lbl_eval))
        out["calinski_harabasz"] = float(calinski_harabasz_score(X_eval, lbl_eval))
    else:
        out["silhouette"] = None
        out["davies_bouldin"] = None
        out["calinski_harabasz"] = None

    if extra:
        out.update(extra)
    return out
