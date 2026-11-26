"""Unified training, hyperparameter tuning, clustering, PCA/LDA analysis, and evaluation service.

Called identically by both the Streamlit GUI (`app/`) and the CLI (`main.py`).
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from src.classification.models import build_clf_preprocessor, sanitize_classification_inputs
from src.clustering.k_means.features import FEATURE_SET
from src.clustering.k_means.learn import cluster_profile, posthoc_cancel_rate
from src.clustering.models import describe_cluster_personas, engineer_cluster_features
from src.config import SEED, cleaning_flags, get_config, path as cfg_path, selection, split_dir
from src.data.data_cleaning import clean_data
from src.data.data_split import split_data
from src.data.validation import (
    dataframe_sha256,
    ensure_task_splits,
    load_clustering_frame,
    normalize_columns,
)
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_clustering_metrics,
    compute_regression_metrics,
    segment_error_table,
    sweep_thresholds,
)
from src.inference.predict import predict_with_bundle
from src.regression.linear_regression.preprocessing import paying_only
from src.registry.artifacts import ArtifactBundle, ArtifactStore, get_store
from src.registry.experiments import sync_experiments_log
from src.registry.model_card import ModelCard
from src.registry.registry import get_spec
from src.registry.specs import FeatureSpace
from src.tuning.search import run_clustering_search, run_supervised_search

ProgressCallback = Callable[[float, str], None]


def _noop_progress(frac: float, msg: str) -> None:
    return None


def _unwrap_pipeline(fitted_model: Any) -> Any:
    return getattr(fitted_model, "regressor_", fitted_model)


def _extract_pca_info(fitted_model: Any, feature_space: str, requested: Any) -> tuple[dict[str, Any], int | None]:
    pipe = _unwrap_pipeline(fitted_model)
    info: dict[str, Any] = {"requested_components": requested}
    n_feat_out: int | None = None
    if hasattr(pipe, "named_steps"):
        if "preprocess" in pipe.named_steps:
            try:
                n_feat_out = len(pipe.named_steps["preprocess"].get_feature_names_out())
            except Exception:
                pass
        elif "prep" in pipe.named_steps:
            try:
                n_feat_out = len(pipe.named_steps["prep"].get_feature_names_out())
            except Exception:
                pass
        if "pca" in pipe.named_steps:
            pca = pipe.named_steps["pca"]
            ev = pca.explained_variance_ratio_
            info["components_kept"] = int(pca.n_components_)
            info["variance_kept"] = float(np.sum(ev))
            info["explained_variance_ratio"] = [float(x) for x in ev[:20]]
            n_feat_out = int(pca.n_components_)
        elif "lda" in pipe.named_steps:
            lda = pipe.named_steps["lda"]
            ev = getattr(lda, "explained_variance_ratio_", [1.0])
            info["components_kept"] = 1
            info["variance_kept"] = float(np.sum(ev))
            n_feat_out = 1
    return info, n_feat_out


def extract_feature_weights(
    fitted_model: Any,
    val_X: pd.DataFrame | None = None,
    val_y: pd.Series | None = None,
    scoring: str | None = None,
) -> pd.DataFrame | None:
    """Extract linear coefficients, tree importances, or fast permutation importances."""
    pipe = _unwrap_pipeline(fitted_model)
    if not hasattr(pipe, "named_steps"):
        return None

    prep_step = pipe.named_steps.get("preprocess") or pipe.named_steps.get("prep")
    feat_names: list[str] | None = None
    if "pca" in pipe.named_steps:
        k = int(pipe.named_steps["pca"].n_components_)
        feat_names = [f"PC{i + 1}" for i in range(k)]
    elif "lda" in pipe.named_steps:
        feat_names = ["LD1"]
    elif prep_step is not None and hasattr(prep_step, "get_feature_names_out"):
        try:
            feat_names = [str(f) for f in prep_step.get_feature_names_out()]
        except Exception:
            feat_names = None

    est = list(pipe.named_steps.values())[-1]
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        if coef.ndim > 1:
            coef = coef[0]
        names = feat_names if feat_names and len(feat_names) == len(coef) else [f"f_{i}" for i in range(len(coef))]
        df_w = pd.DataFrame({"feature": names, "importance": coef})
        df_w = df_w.reindex(df_w["importance"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        return df_w

    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        names = feat_names if feat_names and len(feat_names) == len(imp) else [f"f_{i}" for i in range(len(imp))]
        return (
            pd.DataFrame({"feature": names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    if val_X is not None and val_y is not None and len(val_X) >= 30:
        sub_n = min(1000, len(val_X))
        idx = np.random.default_rng(SEED).choice(len(val_X), size=sub_n, replace=False)
        Xs = val_X.iloc[idx]
        ys = val_y.iloc[idx]
        try:
            perm = permutation_importance(
                fitted_model, Xs, ys, scoring=scoring, n_repeats=3, random_state=SEED, n_jobs=1
            )
            return (
                pd.DataFrame({"feature": list(Xs.columns), "importance": perm.importances_mean})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True)
            )
        except Exception:
            return None
    return None


def _subsample_supervised(
    df: pd.DataFrame,
    task: str,
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if not max_rows or len(df) <= max_rows:
        return df, None
    rng = np.random.default_rng(SEED)
    if task == "classification" and "IsCanceled" in df.columns:
        parts = []
        for _, grp in df.groupby("IsCanceled"):
            take = max(1, int(round(len(grp) / len(df) * max_rows)))
            idx = rng.choice(len(grp), size=min(take, len(grp)), replace=False)
            parts.append(grp.iloc[idx])
        sub = pd.concat(parts).sample(frac=1.0, random_state=SEED)
    else:
        idx = rng.choice(len(df), size=max_rows, replace=False)
        sub = df.iloc[idx]
    return sub, {
        "strategy": "stratified_random" if task == "classification" else "uniform_random",
        "sample_rows": int(len(sub)),
        "full_rows": int(len(df)),
        "seed": SEED,
    }


def prepare_user_splits(
    df: pd.DataFrame,
    task: Literal["regression", "classification"],
    apply_cleaning: bool = True,
    drop_zero_adr: bool | None = None,
    keep_zero_nights: bool | None = None,
    drop_company: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Normalize, optionally run clean_data(), and split an uploaded DataFrame 70/15/15."""
    work, _, _ = normalize_columns(df)
    clean_rep = None
    if apply_cleaning:
        defaults = cleaning_flags(task)
        dz = defaults["drop_zero_adr"] if drop_zero_adr is None else drop_zero_adr
        kz = defaults["keep_zero_nights"] if keep_zero_nights is None else keep_zero_nights
        # clean_data expects ADR & IsCanceled; fill dummy column only if missing for pure regression uploads
        added_dummy_cancel = False
        if "IsCanceled" not in work.columns:
            work["IsCanceled"] = 0
            added_dummy_cancel = True
        if "ADR" not in work.columns:
            work["ADR"] = 100.0
        if "StaysInWeekendNights" in work.columns and "StaysInWeekNights" in work.columns and "Adults" in work.columns:
            if "Children" not in work.columns:
                work["Children"] = 0
            if "Babies" not in work.columns:
                work["Babies"] = 0
            work, clean_rep = clean_data(
                work, drop_zero_adr=dz, drop_company=drop_company and ("Company" in work.columns), keep_zero_nights=kz
            )
        if added_dummy_cancel:
            work = work.drop(columns=["IsCanceled"])

    strat_col = "IsCanceled" if "IsCanceled" in work.columns and work["IsCanceled"].nunique() >= 2 else None
    if strat_col:
        tr, va, te = split_data(work, stratify_col=strat_col)
    else:
        from sklearn.model_selection import train_test_split

        tr_va, te = train_test_split(work, test_size=0.15, random_state=SEED)
        tr, va = train_test_split(tr_va, test_size=0.15 / 0.85, random_state=SEED)
    return tr, va, te, clean_rep


def train_supervised_model(
    task: Literal["regression", "classification"],
    model_id: str,
    mode: Literal["baseline", "tuned_preset", "custom", "tune"] = "baseline",
    custom_params: dict[str, Any] | None = None,
    search_space: dict[str, list[Any]] | None = None,
    search_method: Literal["grid", "random"] = "random",
    cv_folds: int = 5,
    n_iter: int = 12,
    feature_space: FeatureSpace = "original",
    pca_components: float | int | None = 0.90,
    tune_threshold: bool = False,
    evaluate_test: bool = False,
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None] | None = None,
    dataset_name: str = "H2 City Hotel (official splits)",
    source: Literal["project", "user"] = "project",
    variant_override: str | None = None,
    run_name: str = "",
    sample_max_rows: int | None = None,
    set_as_deployed: bool = False,
    progress_cb: ProgressCallback | None = None,
    store: ArtifactStore | None = None,
) -> ArtifactBundle:
    """Train or tune any registered supervised model and persist its complete ArtifactBundle."""
    cb = progress_cb or _noop_progress
    store = store or get_store()
    spec = get_spec(task, model_id)
    target_col = "ADR" if task == "regression" else "IsCanceled"

    cb(0.05, f"Loading {task} splits for {spec.display_name}…")
    if splits is None:
        train_df, val_df, test_df = ensure_task_splits(task)
        sdir_str = str(split_dir(task).relative_to(cfg_path("raw_data").parents[1]))
    else:
        train_df, val_df, test_df = splits
        train_df, _, _ = normalize_columns(train_df)
        val_df, _, _ = normalize_columns(val_df)
        if test_df is not None:
            test_df, _, _ = normalize_columns(test_df)
        sdir_str = "uploaded"

    if task == "regression":
        train_df = paying_only(train_df)
        val_df = paying_only(val_df)
        if test_df is not None and target_col in test_df.columns:
            test_df = paying_only(test_df)

    eff_cap = sample_max_rows or spec.recommended_max_rows
    train_fit, sampling_info = _subsample_supervised(train_df, task, eff_cap)

    X_train, y_train = train_fit.drop(columns=[target_col]), train_fit[target_col]
    X_val, y_val = val_df.drop(columns=[target_col]), val_df[target_col]

    # Assert forbidden columns aren't in spec's required inputs
    leaked = set(spec.required_inputs) & (set(spec.forbidden_inputs) - {target_col})
    if leaked:
        raise AssertionError(f"Leakage guard tripped: {leaked}")

    tuning_df: pd.DataFrame | None = None
    tuning_summary: dict[str, Any] | None = None

    t0 = time.perf_counter()
    if mode == "tune":
        cb(0.15, f"Tuning {spec.display_name} via {search_method} CV…")
        fitted_model, chosen_params, tuning_df, tuning_summary = run_supervised_search(
            spec=spec,
            X_train=X_train,
            y_train=y_train,
            base_params=custom_params,
            search_space=search_space,
            method=search_method,
            cv_folds=cv_folds,
            n_iter=n_iter,
            feature_space=feature_space,
            n_components=pca_components if feature_space == "pca" else None,
            progress_cb=cb,
        )
        variant_tag = variant_override or ("tuned" if feature_space == "original" else f"tuned_{feature_space}")
    else:
        if mode == "baseline":
            chosen_params = dict(spec.baseline_params)
            base_tag = "baseline"
        elif mode == "tuned_preset":
            chosen_params = dict(spec.tuned_params)
            base_tag = "tuned"
        else:
            chosen_params = {**spec.default_param_dict(), **(custom_params or {})}
            base_tag = "custom"
        variant_tag = variant_override or (base_tag if feature_space == "original" else f"{base_tag}_{feature_space}")

        cb(0.25, f"Fitting {spec.display_name} ({variant_tag}) on {len(X_train):,} rows…")
        fitted_model = spec.build_pipeline(
            chosen_params,
            feature_space,
            pca_components if feature_space == "pca" else None,
        )
        fitted_model.fit(X_train, y_train)
    train_secs = time.perf_counter() - t0

    # Cross-validation fold scores on training split for stability boxplots
    cb(0.68, f"Computing {cv_folds}-fold cross-validation scores…")
    eff_folds = min(cv_folds, 3) if spec.cost_hint == "slow" else cv_folds
    cv_splitter = (
        StratifiedKFold(n_splits=eff_folds, shuffle=True, random_state=SEED)
        if task == "classification"
        else KFold(n_splits=eff_folds, shuffle=True, random_state=SEED)
    )
    cv_est = spec.build_pipeline(
        chosen_params,
        feature_space,
        pca_components if feature_space == "pca" else None,
    )
    raw_cv = cross_val_score(cv_est, X_train, y_train, cv=cv_splitter, scoring=spec.default_scoring, n_jobs=2)
    if spec.default_scoring.startswith("neg_"):
        fold_vals = (-raw_cv).tolist()
        cv_metric_name = "RMSE"
    else:
        fold_vals = raw_cv.tolist()
        cv_metric_name = "F1"

    cv_payload = {
        "metric": cv_metric_name,
        "folds": eff_folds,
        "fold_scores": [float(v) for v in fold_vals],
        "mean": float(np.mean(fold_vals)),
        "std": float(np.std(fold_vals)),
    }

    # Validation evaluation
    cb(0.84, "Scoring validation split & computing diagnostics…")
    extra_tables: dict[str, pd.DataFrame] = {}
    threshold_used: float | None = None

    if task == "regression":
        val_pred = np.asarray(fitted_model.predict(X_val), dtype=float)
        val_metrics = compute_regression_metrics(y_val, val_pred)
        pred_val_df = pd.DataFrame({"y_true": y_val.to_numpy(dtype=float), "y_pred": val_pred})
        cv_summary_metrics = {f"{cv_metric_name}_mean": cv_payload["mean"], f"{cv_metric_name}_std": cv_payload["std"], "folds": eff_folds}
    else:
        val_proba = np.asarray(fitted_model.predict_proba(X_val)[:, 1], dtype=float)
        sweep_df = sweep_thresholds(y_val, val_proba)
        extra_tables["threshold_sweep"] = sweep_df
        if tune_threshold or mode in ("tune", "tuned_preset"):
            best_row = sweep_df.sort_values("F1", ascending=False).iloc[0]
            threshold_used = float(best_row["threshold"])
        else:
            threshold_used = float(chosen_params.get("threshold", 0.50))
        val_pred = (val_proba >= threshold_used).astype(int)
        val_metrics = compute_classification_metrics(y_val, val_pred, y_proba=val_proba, threshold=threshold_used)
        pred_val_df = pd.DataFrame(
            {"y_true": y_val.to_numpy(dtype=int), "y_pred": val_pred, "y_proba": val_proba}
        )
        seg_df = segment_error_table(val_df, y_val, val_pred, group_col="MarketSegment")
        if not seg_df.empty:
            extra_tables["segment_errors"] = seg_df
        cv_summary_metrics = {f"{cv_metric_name}_mean": cv_payload["mean"], f"{cv_metric_name}_std": cv_payload["std"], "folds": eff_folds}

    # Feature weights / importances
    weights_df = extract_feature_weights(fitted_model, X_val, y_val, scoring=spec.default_scoring)
    if weights_df is not None and not weights_df.empty:
        extra_tables["feature_weights"] = weights_df

    # Optional final test-set evaluation
    test_metrics: dict[str, Any] = {}
    pred_test_df: pd.DataFrame | None = None
    if evaluate_test and test_df is not None and target_col in test_df.columns:
        X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]
        if task == "regression":
            t_pred = np.asarray(fitted_model.predict(X_test), dtype=float)
            test_metrics = compute_regression_metrics(y_test, t_pred)
            pred_test_df = pd.DataFrame({"y_true": y_test.to_numpy(dtype=float), "y_pred": t_pred})
        else:
            t_proba = np.asarray(fitted_model.predict_proba(X_test)[:, 1], dtype=float)
            t_pred = (t_proba >= (threshold_used or 0.5)).astype(int)
            test_metrics = compute_classification_metrics(y_test, t_pred, y_proba=t_proba, threshold=threshold_used or 0.5)
            pred_test_df = pd.DataFrame({"y_true": y_test.to_numpy(dtype=int), "y_pred": t_pred, "y_proba": t_proba})

    pca_info, n_feat_out = _extract_pca_info(fitted_model, feature_space, pca_components if feature_space == "pca" else None)

    if source == "user":
        short_id = uuid.uuid4().hex[:8]
        bundle_id = f"user/{task}_{model_id}_{variant_tag}_{short_id}"
    else:
        bundle_id = f"{task}/{model_id}/{variant_tag}"

    pipe_obj = _unwrap_pipeline(fitted_model)
    prep_steps = list(pipe_obj.named_steps.keys()) if hasattr(pipe_obj, "named_steps") else [type(fitted_model).__name__]
    sel = selection(task)

    card = ModelCard(
        bundle_id=bundle_id,
        model_id=model_id,
        task=task,
        variant=variant_tag,
        display_name=spec.display_name,
        family=spec.family,
        source=source,
        run_name=run_name or f"{spec.display_name} ({variant_tag})",
        track_origin=spec.track_origin if source == "project" else "user",
        selection_criterion=f"{'Highest' if sel['higher_is_better'] else 'Lowest'} {sel['split']} {sel['metric']}",
        params=chosen_params,
        feature_space=feature_space,
        pca_info=pca_info,
        inputs=list(spec.required_inputs),
        forbidden_checked=list(spec.forbidden_inputs),
        preprocessing_steps=prep_steps,
        target=target_col,
        threshold=threshold_used,
        supports_proba=spec.supports_proba,
        supports_predict_new=True,
        metrics={"val": val_metrics, "cv": cv_summary_metrics, **({"test": test_metrics} if test_metrics else {})},
        tuning_summary=tuning_summary,
        train_time_s=round(train_secs, 4),
        n_train_rows=int(len(X_train)),
        n_features_out=n_feat_out,
        sampling_info=sampling_info,
        data_version={
            "dataset_name": dataset_name,
            "split_dir": sdir_str,
            "sha256_train": dataframe_sha256(train_fit),
            "cleaning_flags": cleaning_flags(task),
        },
    )

    cb(0.95, f"Saving bundle {bundle_id}…")
    bundle = store.save_bundle(
        card=card,
        pipeline=fitted_model,
        predictions_val=pred_val_df,
        predictions_test=pred_test_df,
        cv_scores=cv_payload,
        tuning_results=tuning_df,
        extra_tables=extra_tables,
        set_as_deployed=set_as_deployed,
    )
    sync_experiments_log(store=store)
    cb(1.0, "Completed.")
    return bundle


def train_clustering_model(
    model_id: str = "kmeans",
    mode: Literal["baseline", "tuned_preset", "custom", "tune"] = "baseline",
    custom_params: dict[str, Any] | None = None,
    search_space: dict[str, list[Any]] | None = None,
    feature_space: FeatureSpace = "pca",
    pca_components: float | int | None = 0.90,
    df: pd.DataFrame | None = None,
    dataset_name: str = "H2 City Hotel (cleaned)",
    source: Literal["project", "user"] = "project",
    variant_override: str | None = None,
    run_name: str = "",
    sample_max_rows: int | None = None,
    set_as_deployed: bool = False,
    progress_cb: ProgressCallback | None = None,
    store: ArtifactStore | None = None,
) -> ArtifactBundle:
    """Fit or optimize an unsupervised clustering algorithm and save profiles, 2D PCA, and dendrograms."""
    cb = progress_cb or _noop_progress
    store = store or get_store()
    spec = get_spec("clustering", model_id)

    cb(0.08, f"Preparing clustering data for {spec.display_name}…")
    if df is None:
        df_clean, _ = load_clustering_frame()
    else:
        df_clean, _, _ = normalize_columns(df)

    eff_cap = sample_max_rows or spec.recommended_max_rows
    sampling_info = None
    if eff_cap and len(df_clean) > eff_cap:
        idx = np.random.default_rng(SEED).choice(len(df_clean), size=eff_cap, replace=False)
        df_work = df_clean.iloc[np.sort(idx)].copy()
        sampling_info = {
            "strategy": "uniform_random_subset (documented per PDF §10 for O(n²) distance matrices)",
            "sample_rows": int(len(df_work)),
            "full_rows": int(len(df_clean)),
            "seed": SEED,
        }
    else:
        df_work = df_clean

    tuning_df: pd.DataFrame | None = None
    tuning_summary: dict[str, Any] | None = None
    nc = pca_components if feature_space == "pca" else None

    t0 = time.perf_counter()
    if mode == "tune":
        cb(0.18, f"Optimizing {spec.display_name} over unsupervised Silhouette criterion…")
        fitted_pipe, chosen_params, tuning_df, tuning_summary = run_clustering_search(
            spec=spec,
            X_raw=df_work,
            base_params=custom_params,
            search_space=search_space,
            feature_space=feature_space,
            n_components=nc,
            progress_cb=cb,
        )
        base_tag = "tuned"
    else:
        if mode == "baseline":
            chosen_params = dict(spec.baseline_params)
            base_tag = "baseline"
        elif mode == "tuned_preset":
            chosen_params = dict(spec.tuned_params)
            base_tag = "tuned"
        else:
            chosen_params = {**spec.default_param_dict(), **(custom_params or {})}
            base_tag = "custom"
        cb(0.30, f"Fitting {spec.display_name} ({base_tag}) on {len(df_work):,} bookings…")
        fitted_pipe = spec.build_pipeline(chosen_params, feature_space, nc)
        fitted_pipe.fit(df_work)
    train_secs = time.perf_counter() - t0

    variant_tag = variant_override or (base_tag if feature_space == "pca" else f"{base_tag}_nopca")

    cb(0.72, "Computing internal cluster metrics, profiles, and 2D projection…")
    est = fitted_pipe.named_steps["clusterer"]
    X_trans = fitted_pipe[:-1].transform(df_work)
    if hasattr(est, "labels_"):
        labels = np.asarray(est.labels_, dtype=int)
    else:
        labels = np.asarray(est.predict(X_trans), dtype=int)

    extra_m: dict[str, Any] = {}
    if hasattr(est, "inertia_"):
        extra_m["inertia"] = float(est.inertia_)
    if hasattr(est, "bic"):
        extra_m["bic"] = float(est.bic(X_trans))
        extra_m["aic"] = float(est.aic(X_trans))

    clu_metrics = compute_clustering_metrics(X_trans, labels, extra=extra_m)
    X_eng = engineer_cluster_features(df_work)

    # Profile & post-hoc cancel check (excluding noise label -1 for clean centroids)
    valid_mask = labels >= 0
    extra_tables: dict[str, pd.DataFrame] = {}
    extra_json: dict[str, Any] = {}

    if valid_mask.sum() > 0:
        prof = cluster_profile(X_eng.loc[valid_mask], labels[valid_mask], FEATURE_SET).reset_index()
        extra_tables["cluster_profile"] = prof
        cancel_df = None
        if "IsCanceled" in df_work.columns:
            cancel_df = posthoc_cancel_rate(df_work.loc[valid_mask], X_eng.loc[valid_mask], labels[valid_mask]).reset_index()
            extra_tables["cancel_rate"] = cancel_df
        extra_json["personas"] = describe_cluster_personas(prof, cancel_df)

    # 2D PCA projection for interactive visualization
    pca2 = PCA(n_components=2, random_state=SEED)
    coords = pca2.fit_transform(X_trans)
    ev2 = [float(x) for x in pca2.explained_variance_ratio_]
    max_pts = int(get_config()["sampling"]["scatter_points"])
    s_idx = np.arange(len(coords)) if len(coords) <= max_pts else np.random.default_rng(SEED).choice(len(coords), size=max_pts, replace=False)
    proj_df = pd.DataFrame(
        {
            "PC1": coords[s_idx, 0],
            "PC2": coords[s_idx, 1],
            "cluster": labels[s_idx],
            "LeadTime": X_eng["LeadTime"].iloc[s_idx].to_numpy(),
            "ADR": X_eng["ADR"].iloc[s_idx].to_numpy(),
            "TotalNights": X_eng["TotalNights"].iloc[s_idx].to_numpy(),
            "TotalGuests": X_eng["TotalGuests"].iloc[s_idx].to_numpy(),
            "MarketSegment": X_eng["MarketSegment"].iloc[s_idx].astype(str).to_numpy(),
        }
    )
    extra_tables["pca_2d"] = proj_df
    extra_json["pca_2d_variance"] = ev2

    # Ward dendrogram coordinates for Agglomerative Hierarchical clustering
    if model_id == "agglomerative":
        dend_sub = X_trans if len(X_trans) <= 2000 else X_trans[np.random.default_rng(SEED).choice(len(X_trans), size=2000, replace=False)]
        Z_link = linkage(dend_sub, method="ward")
        ddata = dendrogram(Z_link, truncate_mode="lastp", p=28, no_plot=True)
        extra_json["dendrogram"] = {"icoord": ddata["icoord"], "dcoord": ddata["dcoord"]}

    pca_info, n_feat_out = _extract_pca_info(fitted_pipe, feature_space, nc)
    if source == "user":
        bundle_id = f"user/clustering_{model_id}_{variant_tag}_{uuid.uuid4().hex[:8]}"
    else:
        bundle_id = f"clustering/{model_id}/{variant_tag}"

    card = ModelCard(
        bundle_id=bundle_id,
        model_id=model_id,
        task="clustering",
        variant=variant_tag,
        display_name=spec.display_name,
        family=spec.family,
        source=source,
        run_name=run_name or f"{spec.display_name} ({variant_tag})",
        track_origin=spec.track_origin if source == "project" else "user",
        selection_criterion="Highest internal Silhouette score (unsupervised)",
        params=chosen_params,
        feature_space=feature_space,
        pca_info=pca_info,
        inputs=list(spec.required_inputs),
        forbidden_checked=list(spec.forbidden_inputs),
        preprocessing_steps=list(fitted_pipe.named_steps.keys()),
        target=None,
        supports_proba=False,
        supports_predict_new=spec.supports_predict_new,
        metrics={"train": clu_metrics},
        tuning_summary=tuning_summary,
        train_time_s=round(train_secs, 4),
        n_train_rows=int(len(df_work)),
        n_features_out=n_feat_out,
        sampling_info=sampling_info,
        data_version={
            "dataset_name": dataset_name,
            "sha256_train": dataframe_sha256(df_work),
            "cleaning_flags": cleaning_flags("clustering"),
        },
    )

    cb(0.95, f"Saving clustering bundle {bundle_id}…")
    bundle = store.save_bundle(
        card=card,
        pipeline=fitted_pipe,
        tuning_results=tuning_df,
        extra_tables=extra_tables,
        extra_json=extra_json,
        set_as_deployed=set_as_deployed,
    )
    sync_experiments_log(store=store)
    cb(1.0, "Completed.")
    return bundle


def evaluate_bundle_on_data(
    bundle: ArtifactBundle,
    df: pd.DataFrame,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    """Evaluate a saved bundle on a labeled DataFrame and check for distribution shift vs validation."""
    norm, _, _ = normalize_columns(df)
    task = bundle.card.task

    if task == "regression":
        work = paying_only(norm) if "ADR" in norm.columns else norm
        res = predict_with_bundle(bundle, work)
        metrics = compute_regression_metrics(work["ADR"], res["pred_adr"])
        val_ref = bundle.card.metrics.get("val", {}).get("RMSE")
        shift_pct = ((metrics["RMSE"] - val_ref) / val_ref * 100.0) if val_ref else None
        preds_df = pd.DataFrame({"y_true": work["ADR"].to_numpy(dtype=float), "y_pred": res["pred_adr"]})
        return {"metrics": metrics, "predictions": preds_df, "shift_pct": shift_pct, "ref_metric": "RMSE", "ref_val": val_ref}

    if task == "classification":
        res = predict_with_bundle(bundle, norm, threshold_override=threshold_override)
        metrics = compute_classification_metrics(
            norm["IsCanceled"], res["pred_canceled"], y_proba=res["cancel_proba"], threshold=res["threshold"]
        )
        val_ref = bundle.card.metrics.get("val", {}).get("F1")
        shift_pct = ((metrics["F1"] - val_ref) / val_ref * 100.0) if val_ref else None
        preds_df = pd.DataFrame(
            {
                "y_true": norm["IsCanceled"].to_numpy(dtype=int),
                "y_pred": res["pred_canceled"],
                "y_proba": res["cancel_proba"],
            }
        )
        seg = segment_error_table(norm, norm["IsCanceled"], res["pred_canceled"])
        return {
            "metrics": metrics,
            "predictions": preds_df,
            "segment_errors": seg,
            "shift_pct": shift_pct,
            "ref_metric": "F1",
            "ref_val": val_ref,
        }

    # Unsupervised evaluation on new frame
    pipe = bundle.load_pipeline()
    X_trans = pipe[:-1].transform(norm)
    labels = pipe.predict(norm) if bundle.card.supports_predict_new else pipe.named_steps["clusterer"].fit_predict(X_trans)
    metrics = compute_clustering_metrics(X_trans, labels)
    return {"metrics": metrics, "labels": labels}


def compute_pca_lda_diagnostics() -> dict[str, Any]:
    """Fit diagnostic PCA & LDA transformers on training splits only to report variance, loadings, and LD1."""
    tr_clf, va_clf, _ = ensure_task_splits("classification")
    X_tr = sanitize_classification_inputs(tr_clf.drop(columns=["IsCanceled"]))
    y_tr = tr_clf["IsCanceled"].to_numpy(dtype=int)
    X_va = sanitize_classification_inputs(va_clf.drop(columns=["IsCanceled"]))
    y_va = va_clf["IsCanceled"].to_numpy(dtype=int)

    prep = build_clf_preprocessor(scale_numeric=True)
    Z_tr = prep.fit_transform(X_tr)
    Z_va = prep.transform(X_va)
    feat_names = [str(f) for f in prep.get_feature_names_out()]

    pca_full = PCA(random_state=SEED).fit(Z_tr)
    ev = pca_full.explained_variance_ratio_
    k90 = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)

    top_k = min(8, len(ev))
    loadings_df = pd.DataFrame(
        pca_full.components_[:top_k].T,
        columns=[f"PC{i + 1}" for i in range(top_k)],
    )
    loadings_df.insert(0, "feature", feat_names)

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(Z_tr, y_tr)
    ld1_va = lda.transform(Z_va).ravel()

    # Also compute clustering feature PCA scree (matches Milestone 1 90% -> 7 PCs)
    df_clu, _ = load_clustering_frame()
    from src.clustering.k_means.preprocessing import build_preprocessor as build_clu_prep

    X_clu = engineer_cluster_features(df_clu)
    prep_clu = build_clu_prep(FEATURE_SET)
    Z_clu = prep_clu.fit_transform(X_clu)
    pca_clu = PCA(random_state=SEED).fit(Z_clu)
    ev_clu = pca_clu.explained_variance_ratio_
    k90_clu = int(np.searchsorted(np.cumsum(ev_clu), 0.90) + 1)
    clu_names = [str(f) for f in prep_clu.get_feature_names_out()]
    clu_top = min(7, len(ev_clu))
    clu_loadings = pd.DataFrame(
        pca_clu.components_[:clu_top].T,
        columns=[f"PC{i + 1}" for i in range(clu_top)],
    )
    clu_loadings.insert(0, "feature", clu_names)

    return {
        "classification": {
            "explained_variance_ratio": ev.tolist(),
            "k90": k90,
            "variance_at_k90": float(np.sum(ev[:k90])),
            "n_raw_encoded": len(feat_names),
            "loadings": loadings_df,
            "lda_val_z": ld1_va,
            "lda_val_y": y_va,
        },
        "clustering": {
            "explained_variance_ratio": ev_clu.tolist(),
            "k90": k90_clu,
            "variance_at_k90": float(np.sum(ev_clu[:k90_clu])),
            "n_raw_encoded": len(clu_names),
            "loadings": clu_loadings,
        },
    }


def generate_sample_csvs() -> None:
    """Create ready-to-upload sample CSVs in data/samples/ drawn from validation rows."""
    samples_dir = cfg_path("samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    _, val_clf, _ = ensure_task_splits("classification")
    sub = val_clf.sample(n=min(250, len(val_clf)), random_state=SEED).copy()
    sub.to_csv(samples_dir / "sample_bookings_labeled.csv", index=False)
    unlabeled = sub.drop(
        columns=[c for c in ("ADR", "IsCanceled", "ReservationStatus", "ReservationStatusDate") if c in sub.columns]
    )
    unlabeled.to_csv(samples_dir / "sample_bookings_unlabeled.csv", index=False)


def build_official_artifacts(
    core_only: bool = False,
    run_grid_search: bool = True,
    skip_existing: bool = False,
    progress_cb: ProgressCallback | None = None,
    store: ArtifactStore | None = None,
) -> list[str]:
    """Generate only the exact model configurations implemented in the repository."""
    _ = (core_only, run_grid_search)
    cb = progress_cb or _noop_progress
    store = store or get_store()
    built: list[str] = []
    existing_ids = {b.card.bundle_id for b in store.list_bundles()} if skip_existing else set()

    def _exists(bid: str) -> bool:
        if bid in existing_ids:
            built.append(bid)
            return True
        return False

    generate_sample_csvs()
    total_steps = 7
    step = 0

    def bump(msg: str) -> None:
        nonlocal step
        step += 1
        cb(min(0.98, step / total_steps), msg)

    # 1. Regression (src/regression/linear_regression/): baseline + 6-test variant_selected
    bump("Linear Regression: baseline (raw ADR, default config)")
    if not _exists("regression/linear_regression/baseline"):
        b_reg0 = train_supervised_model(
            "regression", "linear_regression", mode="baseline", evaluate_test=True, store=store
        )
        built.append(b_reg0.card.bundle_id)

    bump("Linear Regression: variant_selected (selected_config.json from train.py --variants)")
    if not _exists("regression/linear_regression/variant_selected"):
        reg_spec = get_spec("regression", "linear_regression")
        b_reg1 = train_supervised_model(
            "regression",
            "linear_regression",
            mode="custom",
            custom_params=dict(reg_spec.tuned_params),
            variant_override="variant_selected",
            evaluate_test=True,
            store=store,
        )
        built.append(b_reg1.card.bundle_id)

    # 2. Classification (src/classification/): baseline + tuned (var_smoothing=1e-5, thr=0.60) + calibrated
    bump("Gaussian Naive Bayes: baseline (var_smoothing=1e-9, thr=0.50)")
    if not _exists("classification/gaussian_nb/baseline"):
        c_gnb0 = train_supervised_model(
            "classification", "gaussian_nb", mode="baseline", evaluate_test=True, store=store
        )
        built.append(c_gnb0.card.bundle_id)

    bump("Gaussian Naive Bayes: tuned (var_smoothing search -> 1e-5, thr=0.60)")
    if not _exists("classification/gaussian_nb/tuned"):
        c_gnb1 = train_supervised_model(
            "classification",
            "gaussian_nb",
            mode="tune",
            custom_params={"threshold": 0.60},
            search_space={"classifier__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 3e-5, 1e-4]},
            search_method="grid",
            tune_threshold=False,
            variant_override="tuned",
            evaluate_test=True,
            store=store,
        )
        built.append(c_gnb1.card.bundle_id)

    bump("Gaussian Naive Bayes: calibrated (5-fold Sigmoid CalibratedClassifierCV)")
    if not _exists("classification/gaussian_nb/calibrated"):
        c_cal = train_supervised_model(
            "classification",
            "gaussian_nb",
            mode="custom",
            custom_params={"var_smoothing": 1e-5, "calibrate": True, "threshold": 0.50},
            tune_threshold=True,
            variant_override="calibrated",
            evaluate_test=True,
            store=store,
        )
        built.append(c_cal.card.bundle_id)

    # 3. Clustering (src/clustering/k_means/learn.py): baseline_nopca + baseline_pca90 (k=3)
    bump("K-Means Clustering: baseline_nopca (k=3, --pca none)")
    if not _exists("clustering/kmeans/baseline_nopca"):
        k_nopca = train_clustering_model(
            "kmeans", mode="baseline", feature_space="original", variant_override="baseline_nopca", store=store
        )
        built.append(k_nopca.card.bundle_id)

    bump("K-Means Clustering: baseline_pca90 (k=3, --pca 0.90 -> 7 PCs)")
    if not _exists("clustering/kmeans/baseline_pca90"):
        k_pca = train_clustering_model(
            "kmeans",
            mode="baseline",
            feature_space="pca",
            pca_components=0.90,
            variant_override="baseline_pca90",
            store=store,
        )
        built.append(k_pca.card.bundle_id)

    # Set the canonical repo-selected variant per task as deployed
    deployed_targets = {
        "regression": "regression/linear_regression/variant_selected",
        "classification": "classification/gaussian_nb/tuned",
        "clustering": "clustering/kmeans/baseline_pca90",
    }
    for task_name, bid in deployed_targets.items():
        try:
            store.set_deployed(task_name, bid)
        except KeyError:
            pass

    sync_experiments_log(store=store)
    cb(1.0, f"Built {len(built)} official repository bundles.")
    return built
