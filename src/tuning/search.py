"""Hyperparameter optimization routines for supervised (Grid/Randomized CV) and unsupervised tasks."""
from __future__ import annotations

import itertools
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold

from src.config import SEED
from src.evaluation.metrics import compute_clustering_metrics
from src.registry.specs import FeatureSpace, ModelSpec


def _is_pipeline_param_grid(search_space: dict[str, list[Any]]) -> bool:
    return any("__" in k for k in search_space)


def run_supervised_search(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    base_params: dict[str, Any] | None = None,
    search_space: dict[str, list[Any]] | None = None,
    method: str = "random",
    cv_folds: int = 5,
    n_iter: int = 12,
    scoring: str | None = None,
    feature_space: FeatureSpace = "original",
    n_components: float | int | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[Any, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Tune a supervised model strictly on training folds (never touching the test split)."""
    base = dict(spec.default_param_dict())
    if base_params:
        base.update(base_params)
    space = dict(search_space or spec.search_space)
    score_name = scoring or spec.default_scoring

    cv = (
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
        if spec.task == "classification"
        else KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    )

    # Case A: config-level search space (e.g. LinearRegression variant parameters)
    if not _is_pipeline_param_grid(space):
        keys = list(space.keys())
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]
        if method == "random" and len(combos) > n_iter:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(combos), size=n_iter, replace=False)
            combos = [combos[int(i)] for i in idx]

        from sklearn.model_selection import cross_val_score

        rows = []
        best_score = -np.inf
        best_cfg: dict[str, Any] = dict(base)
        for idx, override in enumerate(combos):
            cand_params = {**base, **override}
            if progress_cb:
                progress_cb((idx + 0.2) / max(len(combos), 1), f"Evaluating candidate {idx + 1}/{len(combos)}: {override}")
            est = spec.build_pipeline(cand_params, feature_space, n_components)
            fold_scores = cross_val_score(est, X_train, y_train, cv=cv, scoring=score_name, n_jobs=2)
            mean_s = float(np.mean(fold_scores))
            std_s = float(np.std(fold_scores))
            row = {f"param_{k}": v for k, v in override.items()}
            row["mean_test_score"] = mean_s
            row["std_test_score"] = std_s
            rows.append(row)
            if mean_s > best_score:
                best_score = mean_s
                best_cfg = cand_params

        best_model = spec.build_pipeline(best_cfg, feature_space, n_components)
        best_model.fit(X_train, y_train)
        trials_df = pd.DataFrame(rows).sort_values("mean_test_score", ascending=False).reset_index(drop=True)
        trials_df["rank_test_score"] = np.arange(1, len(trials_df) + 1)
        summary = {
            "method": method,
            "scoring": score_name,
            "cv_folds": cv_folds,
            "n_candidates": len(combos),
            "best_cv_score": float(best_score),
            "best_params": best_cfg,
            "search_space": space,
        }
        return best_model, best_cfg, trials_df, summary

    # Case B: estimator parameter grid via sklearn GridSearchCV / RandomizedSearchCV
    est = spec.build_pipeline(base, feature_space, n_components)
    total_grid = int(np.prod([len(v) for v in space.values()])) if space else 1
    if method == "grid" or total_grid <= n_iter:
        searcher: Any = GridSearchCV(
            est, param_grid=space, scoring=score_name, cv=cv, n_jobs=2, refit=True
        )
        effective_method = "grid"
        n_cand = total_grid
    else:
        searcher = RandomizedSearchCV(
            est, param_distributions=space, n_iter=n_iter, scoring=score_name,
            cv=cv, random_state=SEED, n_jobs=2, refit=True,
        )
        effective_method = "random"
        n_cand = n_iter

    if progress_cb:
        progress_cb(0.25, f"Running {effective_method} search ({n_cand} candidates × {cv_folds} folds)…")
    searcher.fit(X_train, y_train)

    cv_df = pd.DataFrame(searcher.cv_results_)
    keep_cols = [
        c for c in cv_df.columns
        if c.startswith("param_") or c in ("mean_test_score", "std_test_score", "mean_fit_time", "rank_test_score")
    ]
    trials_df = cv_df[keep_cols].sort_values("rank_test_score").reset_index(drop=True)

    clean_best = dict(base)
    for k, v in searcher.best_params_.items():
        short_k = k.split("__")[-1]
        clean_best[short_k] = v

    summary = {
        "method": effective_method,
        "scoring": score_name,
        "cv_folds": cv_folds,
        "n_candidates": int(len(trials_df)),
        "best_cv_score": float(searcher.best_score_),
        "best_params": clean_best,
        "search_space": {k: [str(x) if x is None else x for x in v] for k, v in space.items()},
    }
    return searcher.best_estimator_, clean_best, trials_df, summary


def run_clustering_search(
    spec: ModelSpec,
    X_raw: pd.DataFrame,
    base_params: dict[str, Any] | None = None,
    search_space: dict[str, list[Any]] | None = None,
    feature_space: FeatureSpace = "pca",
    n_components: float | int | None = 0.90,
    progress_cb: Callable[[float, str], None] | None = None,
) -> tuple[Any, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Optimize unsupervised clustering parameters strictly using internal criteria (Silhouette/DB/BIC)."""
    base = dict(spec.default_param_dict())
    if base_params:
        base.update(base_params)
    space = dict(search_space or spec.search_space)
    keys = list(space.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]

    rows = []
    best_sil = -np.inf
    best_params = dict(base)
    best_pipe = None

    for idx, override in enumerate(combos):
        cand = {**base, **override}
        if progress_cb:
            progress_cb((idx + 0.15) / max(len(combos), 1), f"Clustering candidate {idx + 1}/{len(combos)}: {override}")
        pipe = spec.build_pipeline(cand, feature_space, n_components)
        est = pipe.named_steps["clusterer"]
        X_trans = pipe[:-1].fit_transform(X_raw)
        if hasattr(est, "fit_predict"):
            labels = est.fit_predict(X_trans)
        else:
            est.fit(X_trans)
            labels = est.predict(X_trans)

        m = compute_clustering_metrics(X_trans, labels)
        row: dict[str, Any] = {f"param_{k}": v for k, v in override.items()}
        k_val = override.get("n_clusters") or override.get("n_components") or m["n_clusters"]
        row["k"] = int(k_val) if k_val is not None else m["n_clusters"]
        row["silhouette"] = m["silhouette"]
        row["davies_bouldin"] = m["davies_bouldin"]
        row["calinski_harabasz"] = m["calinski_harabasz"]
        row["noise_pct"] = m["noise_pct"]
        if hasattr(est, "inertia_"):
            row["inertia"] = float(est.inertia_)
        if hasattr(est, "bic"):
            row["bic"] = float(est.bic(X_trans))
            row["aic"] = float(est.aic(X_trans))
        row["mean_test_score"] = m["silhouette"] if m["silhouette"] is not None else -1.0
        rows.append(row)

        # Require >= 2 clusters and < 50% noise for DBSCAN to count as a sensible clustering
        sil = m["silhouette"]
        valid = sil is not None and m["n_clusters"] >= 2 and m["noise_pct"] < 0.50
        if valid and float(sil) > best_sil:
            best_sil = float(sil)
            best_params = cand
            best_pipe = pipe

    if best_pipe is None:
        best_pipe = spec.build_pipeline(best_params, feature_space, n_components)
        best_pipe.fit(X_raw)

    trials_df = pd.DataFrame(rows)
    summary = {
        "method": "grid_unsupervised",
        "scoring": "silhouette (internal, zero target labels)",
        "n_candidates": len(combos),
        "best_cv_score": float(best_sil) if best_sil > -np.inf else None,
        "best_params": best_params,
        "search_space": space,
    }
    return best_pipe, best_params, trials_df, summary
