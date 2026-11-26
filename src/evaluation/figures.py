"""Bklit UI chart specification factories styled in the "Fluted Glass" Travertine & Smoked Bronze palette.

Every factory returns a JSON-serializable Bklit chart specification mounted by
`app/components/bklit_chart.py` (`@bklitui/ui` React + D3 + Motion bundle).

The series palette below is the single source of truth for chart colours; the UI
design tokens (`app/components/tokens.py`) re-export it as `--fg-chart-*` variables.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import SEED

# Named, low-chroma architectural series hues (legible on warm ivory glass)
SERIES: dict[str, str] = {
    "slate": "#2F4550",     # smoked slate — primary data ink
    "bronze": "#A57A3F",    # bronze — accent / selected / tuned
    "sage": "#6F8A6A",      # muted sage
    "clay": "#B06A50",      # fired clay — negative / alarm
    "smoke": "#6E8BA0",     # smoked blue glass
    "taupe": "#8C7B6B",     # warm taupe
    "plum": "#7B5E73",      # dusk plum-grey
    "sand": "#C9AE7C",      # champagne sand
    "stone": "#B9AE9C",     # baseline / untuned stone
    "reference": "#A89F90", # dashed reference & ideal lines
    "deep_sage": "#4F6B57", # deep sage for secondary emphasis
}

FAMILY_COLORS: dict[str, str] = {
    "linear": SERIES["slate"],
    "ensemble": SERIES["sage"],
    "boosting": SERIES["bronze"],
    "tree": SERIES["clay"],
    "bayes": SERIES["smoke"],
    "neighbors": SERIES["deep_sage"],
    "svm": SERIES["plum"],
    "neural_net": SERIES["clay"],
    "centroid": SERIES["slate"],
    "mixture": SERIES["sage"],
    "hierarchical": SERIES["taupe"],
    "density": SERIES["bronze"],
    "other": SERIES["taupe"],
}

PALETTE = [
    SERIES["slate"],   # --chart-1
    SERIES["bronze"],  # --chart-2
    SERIES["sage"],    # --chart-3
    SERIES["clay"],    # --chart-4
    SERIES["smoke"],   # --chart-5
    SERIES["taupe"],   # --chart-6
    SERIES["plum"],
    SERIES["sand"],
]


def family_color(family: str) -> str:
    return FAMILY_COLORS.get(str(family).lower(), FAMILY_COLORS["other"])


def _subsample_idx(n: int, max_points: int = 1800, seed: int = SEED) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=max_points, replace=False)


def _downsample_curve(x: np.ndarray, y: np.ndarray, max_pts: int = 180) -> tuple[list[float], list[float]]:
    if len(x) <= max_pts:
        return [round(float(v), 5) for v in x], [round(float(v), 5) for v in y]
    idx = np.linspace(0, len(x) - 1, max_pts).astype(int)
    return [round(float(x[i]), 5) for i in idx], [round(float(y[i]), 5) for i in idx]


# ---------------------------------------------------------------------------
# Bklit Gauge & What-If Helpers (Predict Page)
# ---------------------------------------------------------------------------
def fig_cancel_gauge(prob_fraction: float, label: str = "Cancel Risk %") -> dict[str, Any]:
    return {
        "kind": "bklit_gauge",
        "value": round(float(prob_fraction) * 100.0, 2),
        "unitLabel": label,
        "height": 200,
    }


def fig_what_if(
    df_sw: pd.DataFrame,
    feature_name: str,
    y_col: str,
    title: str,
    y_title: str,
    color: str = SERIES["slate"],
    y_domain: list[float] | None = None,
) -> dict[str, Any]:
    x_raw = df_sw[feature_name]
    y_vals = [round(float(v), 4) for v in pd.to_numeric(df_sw[y_col], errors="coerce").fillna(0)]
    if pd.api.types.is_numeric_dtype(x_raw):
        return {
            "kind": "xy_curves",
            "title": title,
            "xTitle": feature_name,
            "yTitle": y_title,
            "yDomain": y_domain,
            "height": 310,
            "lines": [
                {
                    "name": y_title,
                    "x": [float(v) for v in x_raw],
                    "y": y_vals,
                    "color": color,
                }
            ],
        }
    rows = [{feature_name: str(xv), "value": yv} for xv, yv in zip(x_raw, y_vals)]
    return {
        "kind": "bklit_bar",
        "title": title,
        "xKey": feature_name,
        "xTitle": feature_name,
        "height": 310,
        "data": rows,
        "series": [{"key": "value", "label": y_title, "color": color}],
    }


# ---------------------------------------------------------------------------
# Regression Diagnostics
# ---------------------------------------------------------------------------
def fig_parity(y_true: Any, y_pred: Any, label: str = "Model", max_points: int = 1600) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    idx = _subsample_idx(len(yt), max_points=max_points)
    return {
        "kind": "xy_scatter",
        "title": f"Predicted vs Actual ADR — {label}",
        "subtitle": "Dashed diagonal indicates ideal parity (y = x)",
        "xTitle": "Actual ADR (EUR)",
        "yTitle": "Predicted ADR (EUR)",
        "diagonal": True,
        "includeZeroY": True,
        "height": 360,
        "groups": [
            {
                "name": label,
                "x": [round(float(v), 2) for v in yt[idx]],
                "y": [round(float(v), 2) for v in yp[idx]],
                "color": SERIES["slate"],
            }
        ],
    }


def fig_residuals(y_true: Any, y_pred: Any, label: str = "Model", max_points: int = 1600) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    res = yt - yp
    idx = _subsample_idx(len(yt), max_points=max_points)
    return {
        "kind": "xy_scatter",
        "title": f"Residuals vs Predicted ADR — {label}",
        "subtitle": "Zero-error horizontal baseline shown in graphite",
        "xTitle": "Predicted ADR (EUR)",
        "yTitle": "Residual: Actual - Predicted (EUR)",
        "zeroHline": True,
        "includeZeroY": True,
        "height": 360,
        "groups": [
            {
                "name": label,
                "x": [round(float(v), 2) for v in yp[idx]],
                "y": [round(float(v), 2) for v in res[idx]],
                "color": SERIES["bronze"],
            }
        ],
    }


def fig_residual_overlay(residuals_by_model: dict[str, np.ndarray], n_bins: int = 24) -> dict[str, Any]:
    if not residuals_by_model:
        return {}
    all_vals = np.concatenate([np.asarray(v, dtype=float) for v in residuals_by_model.values()])
    lo, hi = np.percentile(all_vals, [1, 99])
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = [f"{0.5 * (edges[i] + edges[i + 1]):.0f}" for i in range(n_bins)]

    rows = [{"bin": c} for c in centers]
    series = []
    for idx, (m_name, arr) in enumerate(residuals_by_model.items()):
        key = f"s_{idx}"
        hist, _ = np.histogram(np.clip(arr, lo, hi), bins=edges, density=True)
        for b_i in range(n_bins):
            rows[b_i][key] = round(float(hist[b_i]), 5)
        series.append({"key": key, "label": m_name, "color": PALETTE[idx % len(PALETTE)]})

    return {
        "kind": "bklit_bar",
        "title": "Residual Error Density Distribution Overlay",
        "subtitle": "Binned residual density (EUR) across models",
        "xKey": "bin",
        "xTitle": "Residual Bin Center (EUR)",
        "height": 340,
        "data": rows,
        "series": series,
    }


def fig_feature_weights(
    df_weights: pd.DataFrame,
    feature_col: str = "feature",
    value_col: str = "importance",
    title: str = "Top Feature Attributions",
    top_n: int = 16,
) -> dict[str, Any]:
    work = df_weights[df_weights[feature_col] != "(intercept)"].copy()
    work["_abs"] = work[value_col].abs()
    work = work.sort_values("_abs", ascending=False).head(top_n)
    return {
        "kind": "signed_hbar",
        "title": title,
        "subtitle": "Slate = positive effect · Clay = negative effect",
        "data": [
            {"feature": str(r[feature_col]), "value": round(float(r[value_col]), 5)}
            for _, r in work.iterrows()
        ],
    }


# ---------------------------------------------------------------------------
# Classification Diagnostics
# ---------------------------------------------------------------------------
def fig_confusion_matrix(tn: int, fp: int, fn: int, tp: int, title: str = "Confusion Matrix") -> dict[str, Any]:
    mat = [[int(tn), int(fp)], [int(fn), int(tp)]]
    tot = max(tn + fp + fn + tp, 1)
    text = [[f"{mat[r][c]:,}\n({mat[r][c] / tot:.1%})" for c in range(2)] for r in range(2)]
    return {
        "kind": "matrix_heatmap",
        "title": title,
        "subtitle": "Counts and share of validation observations",
        "xLabels": ["Pred: Stay (0)", "Pred: Cancel (1)"],
        "yLabels": ["Actual: Stay (0)", "Actual: Cancel (1)"],
        "z": mat,
        "text": text,
        "diverging": False,
        "height": 300,
    }


def fig_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    lines: list[dict[str, Any]] = []
    for i, (label, (yt, yp)) in enumerate(curves.items()):
        fpr, tpr, _ = roc_curve(yt, yp)
        auc_val = float(np.trapezoid(tpr, fpr))
        xs, ys = _downsample_curve(fpr, tpr)
        lines.append(
            {
                "name": f"{label} (AUC={auc_val:.3f})",
                "x": xs,
                "y": ys,
                "color": PALETTE[i % len(PALETTE)],
            }
        )
    lines.append({"name": "Random (0.500)", "x": [0.0, 1.0], "y": [0.0, 1.0], "dashed": True, "color": SERIES["reference"]})
    return {
        "kind": "xy_curves",
        "title": "ROC Curve Comparison",
        "xTitle": "False Positive Rate",
        "yTitle": "True Positive Rate (Recall)",
        "xDomain": [0, 1],
        "yDomain": [0, 1.02],
        "height": 340,
        "lines": lines,
    }


def fig_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    lines: list[dict[str, Any]] = []
    base_rate = None
    for i, (label, (yt, yp)) in enumerate(curves.items()):
        prec, rec, _ = precision_recall_curve(yt, yp)
        order = np.argsort(rec)
        xs, ys = _downsample_curve(rec[order], prec[order])
        base_rate = float(np.mean(yt))
        lines.append({"name": label, "x": xs, "y": ys, "color": PALETTE[i % len(PALETTE)]})
    return {
        "kind": "xy_curves",
        "title": "Precision-Recall Curve Comparison",
        "xTitle": "Recall",
        "yTitle": "Precision",
        "xDomain": [0, 1],
        "yDomain": [0, 1.02],
        "hline": round(base_rate, 4) if base_rate is not None else None,
        "height": 340,
        "lines": lines,
    }


def fig_calibration_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], n_bins: int = 10) -> dict[str, Any]:
    lines: list[dict[str, Any]] = [
        {"name": "Ideal Calibration", "x": [0.0, 1.0], "y": [0.0, 1.0], "dashed": True, "color": SERIES["reference"]}
    ]
    for i, (label, (yt, yp)) in enumerate(curves.items()):
        prob_true, prob_pred = calibration_curve(yt, np.clip(yp, 0.0, 1.0), n_bins=n_bins)
        lines.append(
            {
                "name": label,
                "x": [round(float(v), 4) for v in prob_pred],
                "y": [round(float(v), 4) for v in prob_true],
                "color": PALETTE[i % len(PALETTE)],
            }
        )
    return {
        "kind": "xy_curves",
        "title": "Probability Calibration (Reliability Diagram)",
        "xTitle": "Mean Predicted Probability",
        "yTitle": "Empirical Cancellation Rate",
        "xDomain": [0, 1],
        "yDomain": [0, 1.02],
        "height": 340,
        "lines": lines,
    }


def fig_threshold_sweep(df_sweep: pd.DataFrame, active_threshold: float = 0.5) -> dict[str, Any]:
    xs = [round(float(v), 3) for v in df_sweep["threshold"]]
    lines = []
    for col, color in (
        ("Precision", SERIES["slate"]),
        ("Recall", SERIES["stone"]),
        ("F1", SERIES["bronze"]),
        ("FalseAlarmRate", SERIES["deep_sage"]),
    ):
        if col in df_sweep.columns:
            lines.append({"name": col, "x": xs, "y": [round(float(v), 4) for v in df_sweep[col]], "color": color})
    return {
        "kind": "xy_curves",
        "title": "Decision Threshold Operating-Point Sweep",
        "xTitle": "Probability Threshold",
        "yTitle": "Metric Value",
        "yDomain": [0, 1.02],
        "vline": float(active_threshold),
        "vlineLabel": f"thr={active_threshold:.2f}",
        "height": 300,
        "lines": lines,
    }


def fig_classification_radar(df_models: pd.DataFrame, label_col: str = "label") -> dict[str, Any]:
    axes = [c for c in ("Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC") if c in df_models.columns]
    metrics = [{"key": a, "label": a} for a in axes]
    rows = []
    for idx, (_, row) in enumerate(df_models.iterrows()):
        item: dict[str, Any] = {"id": f"m_{idx}", "label": str(row[label_col])}
        for a in axes:
            # Bklit RadarChart scales metrics on 0..100
            item[a] = round(float(row[a]) * 100.0, 2) if pd.notna(row[a]) else 0.0
        rows.append(item)
    return {
        "kind": "bklit_radar",
        "title": "Multi-Metric Classification Profile (0–100 Scale)",
        "metrics": metrics,
        "data": rows,
        "height": 370,
    }


def fig_segment_errors(df_seg: pd.DataFrame, group_col: str = "MarketSegment") -> dict[str, Any]:
    rows = []
    for _, r in df_seg.iterrows():
        rows.append(
            {
                group_col: str(r[group_col]),
                "ActualCancelRate": round(float(r["ActualCancelRate"]), 4),
                "PredCancelRate": round(float(r["PredCancelRate"]), 4),
                "FalseAlarmRate": round(float(r["FalseAlarmRate"]), 4),
            }
        )
    return {
        "kind": "bklit_bar",
        "title": f"Cancellation & False-Alarm Rates by {group_col}",
        "xKey": group_col,
        "xTitle": group_col,
        "height": 330,
        "data": rows,
        "series": [
            {"key": "ActualCancelRate", "label": "Actual Cancel Rate", "color": SERIES["slate"]},
            {"key": "PredCancelRate", "label": "Predicted Cancel Rate", "color": SERIES["stone"]},
            {"key": "FalseAlarmRate", "label": "False Alarm Rate", "color": SERIES["bronze"]},
        ],
    }


# ---------------------------------------------------------------------------
# Model Comparison & Hyperparameter Tuning
# ---------------------------------------------------------------------------
def fig_grouped_bars(
    df: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
    title: str | None = None,
) -> dict[str, Any]:
    work = df.dropna(subset=[metric]).copy()
    work["model_label"] = work["display_name"] + np.where(
        work["feature_space"] != "original", " (" + work["feature_space"].str.upper() + ")", ""
    )
    variants = list(dict.fromkeys(work["variant"].astype(str).tolist()))
    piv = work.pivot_table(index="model_label", columns="variant", values=metric, aggfunc="first").reset_index()

    rows = []
    for _, r in piv.iterrows():
        item: dict[str, Any] = {"model": str(r["model_label"])}
        for v in variants:
            item[v] = round(float(r[v]), 4) if v in r and pd.notna(r[v]) else 0.0
        rows.append(item)

    variant_colors = {
        "baseline": SERIES["stone"],
        "baseline_nopca": SERIES["stone"],
        "baseline_pca90": SERIES["bronze"],
        "variant_selected": SERIES["bronze"],
        "tuned": SERIES["bronze"],
        "calibrated": SERIES["slate"],
        "pca": SERIES["slate"],
        "pca90": SERIES["slate"],
        "lda": SERIES["deep_sage"],
        "ORIGINAL": SERIES["stone"],
        "PCA": SERIES["bronze"],
        "LDA": SERIES["deep_sage"],
        "custom": SERIES["smoke"],
    }
    series = [
        {"key": v, "label": v.replace("_", " ").title(), "color": variant_colors.get(v, PALETTE[i % len(PALETTE)])}
        for i, v in enumerate(variants)
    ]
    direction = "higher is better" if higher_is_better else "lower is better"
    return {
        "kind": "bklit_bar",
        "title": title or f"{metric} Across Models & Variants ({direction})",
        "xKey": "model",
        "height": 330,
        "data": rows,
        "series": series,
    }


def fig_tuning_dumbbell(df: pd.DataFrame, metric: str, higher_is_better: bool) -> dict[str, Any]:
    rows_out = []
    for model_name, grp in df.groupby("display_name"):
        by_v = {str(r["variant"]): float(r[metric]) for _, r in grp.dropna(subset=[metric]).iterrows()}
        b_val = by_v.get("baseline", by_v.get("baseline_nopca"))
        t_val = by_v.get("tuned", by_v.get("variant_selected", by_v.get("baseline_pca90")))
        if b_val is not None and t_val is not None and (len(by_v) >= 2):
            rows_out.append({"model": str(model_name), "baseline": round(b_val, 4), "tuned": round(t_val, 4)})
    rows_out.sort(key=lambda r: r["tuned"], reverse=higher_is_better)
    direction = "higher is better" if higher_is_better else "lower is better"
    return {
        "kind": "dumbbell",
        "title": f"Baseline vs Selected / Tuned Configuration — {metric} ({direction})",
        "subtitle": "Connects initial baseline to variant-selected / tuned configuration",
        "data": rows_out,
    }


def fig_pareto_time_vs_score(df: pd.DataFrame, metric: str, higher_is_better: bool) -> dict[str, Any]:
    work = df.dropna(subset=[metric, "train_time_s"]).copy()
    work["label"] = work["display_name"] + " [" + work["variant"] + "]"
    groups = []
    for fam, grp in work.groupby("family"):
        groups.append(
            {
                "name": str(fam).title(),
                "x": [round(float(v), 3) for v in grp["train_time_s"]],
                "y": [round(float(v), 4) for v in grp[metric]],
                "labels": grp["label"].astype(str).tolist(),
                "color": family_color(str(fam)),
                "size": 7,
            }
        )
    direction = "top-left is ideal" if higher_is_better else "bottom-left is ideal"
    return {
        "kind": "xy_scatter",
        "title": f"Performance vs Training Time Trade-off ({direction})",
        "xTitle": "Training Time (seconds)",
        "yTitle": metric,
        "showLabels": True,
        "height": 360,
        "groups": groups,
    }


def fig_cv_boxplots(cv_by_model: dict[str, list[float]], metric_label: str) -> dict[str, Any]:
    return {
        "kind": "boxplot",
        "title": f"Cross-Validation Fold Score Distribution — {metric_label}",
        "subtitle": "IQR box, median bar, and individual training fold scores",
        "height": 320,
        "groups": [
            {"label": label, "values": [round(float(v), 5) for v in vals]}
            for label, vals in cv_by_model.items()
        ],
    }


def fig_param_slice(df_trials: pd.DataFrame, param_col: str, score_col: str = "mean_test_score") -> dict[str, Any]:
    work = df_trials[[param_col, score_col]].dropna().copy()
    work[param_col] = work[param_col].astype(str)
    groups = [
        {"label": str(k), "values": [round(float(v), 5) for v in grp[score_col]]}
        for k, grp in work.groupby(param_col, sort=False)
    ]
    return {
        "kind": "boxplot",
        "title": f"CV Score Distribution vs {param_col}",
        "height": 320,
        "groups": groups,
    }


def fig_parallel_trials(df_trials: pd.DataFrame, param_cols: list[str], score_col: str = "mean_test_score") -> dict[str, Any]:
    dims = []
    work = df_trials.copy()
    for col in param_cols[:6]:
        s = work[col]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().all():
            dims.append({"label": col.replace("param_", ""), "values": [float(v) for v in num]})
        else:
            cats = s.astype(str).unique().tolist()
            mapping = {c: idx for idx, c in enumerate(cats)}
            dims.append(
                {
                    "label": col.replace("param_", ""),
                    "values": [float(mapping[v]) for v in s.astype(str)],
                    "tickvals": list(mapping.values()),
                    "ticktext": cats,
                }
            )
    dims.append({"label": "CV Score", "values": [float(v) for v in pd.to_numeric(work[score_col], errors="coerce").fillna(0)]})
    return {
        "kind": "parcoords",
        "title": "Hyperparameter Search Parallel-Coordinates Landscape",
        "subtitle": "Bronze lines indicate the highest-performing CV candidates",
        "height": 350,
        "dimensions": dims,
    }


# ---------------------------------------------------------------------------
# Clustering & Dimensionality Reduction (PCA / LDA)
# ---------------------------------------------------------------------------
def fig_cluster_scatter_2d(
    df_proj: pd.DataFrame,
    var_explained: tuple[float, float] | None = None,
    title: str = "2D PCA Projection of Discovered Guest Segments",
) -> dict[str, Any]:
    work = df_proj.copy()
    if len(work) > 2000:
        work = work.sample(n=2000, random_state=SEED)
    groups = []
    for i, (cid, grp) in enumerate(work.groupby("cluster")):
        hover = [
            f"Cluster {cid} · ADR EUR {r.get('ADR', 0):.0f} · Lead {r.get('LeadTime', 0):.0f}d"
            for _, r in grp.iterrows()
        ]
        groups.append(
            {
                "name": f"Cluster {cid}",
                "x": [round(float(v), 3) for v in grp["PC1"]],
                "y": [round(float(v), 3) for v in grp["PC2"]],
                "labels": hover,
                "color": PALETTE[i % len(PALETTE)],
                "size": 3.2,
            }
        )
    xlab = f"PC1 ({var_explained[0]:.1%} var)" if var_explained else "PC1"
    ylab = f"PC2 ({var_explained[1]:.1%} var)" if var_explained else "PC2"
    return {
        "kind": "xy_scatter",
        "title": title,
        "xTitle": xlab,
        "yTitle": ylab,
        "height": 390,
        "groups": groups,
    }


def fig_k_selection(df_k: pd.DataFrame) -> list[dict[str, Any]]:
    ks = [float(v) for v in df_k["k"]]
    left = {
        "kind": "xy_curves",
        "title": "Silhouette & Davies-Bouldin vs Cluster Count (k)",
        "xTitle": "Number of Clusters (k)",
        "yTitle": "Score",
        "height": 320,
        "lines": [
            {"name": "Silhouette (higher)", "x": ks, "y": [round(float(v), 4) for v in df_k["silhouette"]], "color": SERIES["slate"]},
            {"name": "Davies-Bouldin (lower)", "x": ks, "y": [round(float(v), 4) for v in df_k["davies_bouldin"]], "color": SERIES["bronze"]},
        ],
    }
    right_col = "inertia" if ("inertia" in df_k.columns and df_k["inertia"].notna().any()) else "calinski_harabasz"
    right = {
        "kind": "xy_curves",
        "title": f"{right_col.replace('_', ' ').title()} vs Cluster Count (k)",
        "xTitle": "Number of Clusters (k)",
        "yTitle": right_col.title(),
        "height": 320,
        "lines": [
            {"name": right_col.title(), "x": ks, "y": [round(float(v), 2) for v in df_k[right_col]], "color": SERIES["deep_sage"]},
        ],
    }
    return [left, right]


def fig_cluster_profile_heatmap(df_profile: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    cols = [c for c in numeric_cols if c in df_profile.columns]
    mat = df_profile[cols].to_numpy(dtype=float)
    means = mat.mean(axis=0, keepdims=True)
    stds = np.where(mat.std(axis=0, keepdims=True) == 0, 1.0, mat.std(axis=0, keepdims=True))
    z = ((mat - means) / stds).round(3).tolist()
    raw_txt = [[f"{mat[r, c]:.2f}" for c in range(len(cols))] for r in range(len(df_profile))]
    ylabels = [f"Cluster {c}" for c in df_profile["cluster"]] if "cluster" in df_profile.columns else [f"Cluster {i}" for i in df_profile.index]
    return {
        "kind": "matrix_heatmap",
        "title": "Standardized Cluster Feature Profile",
        "subtitle": "Cell text = raw centroid mean · Color = standardized Z-score",
        "xLabels": cols,
        "yLabels": ylabels,
        "z": z,
        "text": raw_txt,
        "diverging": True,
        "height": 320,
    }


def fig_dendrogram_from_coords(icoord: list[list[float]], dcoord: list[list[float]], title: str = "Hierarchical Clustering Ward Dendrogram (Truncated)") -> dict[str, Any]:
    return {
        "kind": "dendrogram",
        "title": title,
        "icoord": icoord,
        "dcoord": dcoord,
        "height": 350,
    }


def fig_pca_scree(explained_ratio: list[float] | np.ndarray, chosen_k: int | None = None) -> dict[str, Any]:
    ev = np.asarray(explained_ratio, dtype=float)
    cum = np.cumsum(ev)
    rows = [
        {
            "pc": f"PC{i + 1}",
            "individual": round(float(ev[i]), 4),
            "cumulative": round(float(cum[i]), 4),
        }
        for i in range(min(len(ev), 24))
    ]
    note = f"90% cumulative variance threshold reached at {chosen_k} principal components." if chosen_k else ""
    return {
        "kind": "bklit_scree",
        "title": "PCA Explained Variance (Scree Bars & Cumulative Curve)",
        "subtitle": "Rendered with @bklitui/ui ComposedChart (SeriesBar + Line)",
        "subtitleNote": note,
        "height": 340,
        "data": rows,
    }


def fig_pca_loadings(df_loadings: pd.DataFrame, top_features: int = 14) -> dict[str, Any]:
    work = df_loadings.copy()
    pc_cols = [c for c in work.columns if c.startswith("PC")]
    work["_norm"] = np.sqrt((work[pc_cols] ** 2).sum(axis=1))
    work = work.sort_values("_norm", ascending=False).head(top_features)
    return {
        "kind": "matrix_heatmap",
        "title": "Top Feature Loadings on Retained Principal Components",
        "subtitle": "Bronze = positive loading · Slate = negative loading",
        "xLabels": pc_cols,
        "yLabels": work["feature"].astype(str).tolist(),
        "z": work[pc_cols].round(3).to_numpy().tolist(),
        "diverging": True,
        "height": 380,
    }


def fig_lda_projection(z_1d: np.ndarray, y_true: np.ndarray, n_bins: int = 24) -> dict[str, Any]:
    z = np.asarray(z_1d, dtype=float).ravel()
    y = np.asarray(y_true, dtype=int).ravel()
    lo, hi = np.percentile(z, [1, 99])
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = [f"{0.5 * (edges[i] + edges[i + 1]):.1f}" for i in range(n_bins)]
    h0, _ = np.histogram(np.clip(z[y == 0], lo, hi), bins=edges, density=True)
    h1, _ = np.histogram(np.clip(z[y == 1], lo, hi), bins=edges, density=True)
    rows = [
        {"ld1": centers[i], "stay": round(float(h0[i]), 4), "cancel": round(float(h1[i]), 4)}
        for i in range(n_bins)
    ]
    return {
        "kind": "bklit_bar",
        "title": "1D Linear Discriminant (LDA) Class Separation",
        "subtitle": "Binned density along Fisher discriminant axis LD1",
        "xKey": "ld1",
        "xTitle": "LD1 Projection Value",
        "height": 330,
        "data": rows,
        "series": [
            {"key": "stay", "label": "Not Canceled (0)", "color": SERIES["slate"]},
            {"key": "cancel", "label": "Canceled (1)", "color": SERIES["bronze"]},
        ],
    }


# ---------------------------------------------------------------------------
# Exploratory Data Analysis (Data Explorer)
# ---------------------------------------------------------------------------
def fig_numeric_distribution(df: pd.DataFrame, col: str, split_by: str | None = None, n_bins: int = 22) -> dict[str, Any]:
    s_num = pd.to_numeric(df[col], errors="coerce").dropna()
    if s_num.empty:
        return {}
    lo, hi = float(s_num.min()), float(np.percentile(s_num, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    labels = [f"{0.5 * (edges[i] + edges[i + 1]):.0f}" for i in range(n_bins)]

    if split_by and split_by in df.columns:
        top_groups = df[split_by].astype(str).value_counts().head(4).index.tolist()
        rows = [{"bin": lbl} for lbl in labels]
        series = []
        for idx, gval in enumerate(top_groups):
            sub_vals = pd.to_numeric(df.loc[df[split_by].astype(str) == gval, col], errors="coerce").dropna()
            cnts, _ = np.histogram(np.clip(sub_vals, lo, hi), bins=edges)
            key = f"g_{idx}"
            for i in range(n_bins):
                rows[i][key] = int(cnts[i])
            series.append({"key": key, "label": f"{split_by}={gval}", "color": PALETTE[idx % len(PALETTE)]})
    else:
        cnts, _ = np.histogram(np.clip(s_num, lo, hi), bins=edges)
        rows = [{"bin": labels[i], "count": int(cnts[i])} for i in range(n_bins)]
        series = [{"key": "count", "label": col, "color": SERIES["slate"]}]

    q25, q50, q75 = np.percentile(s_num, [25, 50, 75])
    return {
        "kind": "bklit_bar",
        "title": f"Distribution of {col}" + (f" by {split_by}" if split_by else ""),
        "subtitle": f"Median = {q50:.2f} · IQR = [{q25:.2f}, {q75:.2f}]",
        "xKey": "bin",
        "xTitle": col,
        "height": 340,
        "data": rows,
        "series": series,
    }


def fig_categorical_bars(df: pd.DataFrame, col: str, split_by: str | None = None, top_k: int = 12) -> dict[str, Any]:
    top_vals = df[col].astype(str).value_counts().head(top_k).index.tolist()
    work = df[df[col].astype(str).isin(top_vals)].copy()
    work[col] = work[col].astype(str)

    if split_by and split_by in work.columns and split_by != col:
        work[split_by] = work[split_by].astype(str)
        sub_groups = work[split_by].value_counts().head(4).index.tolist()
        piv = work.groupby([col, split_by]).size().unstack(fill_value=0)
        rows = []
        for cat in top_vals:
            item: dict[str, Any] = {"category": cat}
            for idx, sg in enumerate(sub_groups):
                item[f"g_{idx}"] = int(piv.loc[cat, sg]) if (cat in piv.index and sg in piv.columns) else 0
            rows.append(item)
        series = [
            {"key": f"g_{idx}", "label": f"{split_by}={sg}", "color": PALETTE[idx % len(PALETTE)]}
            for idx, sg in enumerate(sub_groups)
        ]
    else:
        vc = work[col].value_counts()
        rows = [{"category": cat, "count": int(vc.get(cat, 0))} for cat in top_vals]
        series = [{"key": "count", "label": "Bookings", "color": SERIES["bronze"]}]

    return {
        "kind": "bklit_bar",
        "title": f"Category Breakdown — {col}",
        "xKey": "category",
        "xTitle": col,
        "height": 340,
        "data": rows,
        "series": series,
    }
