"""
train.py

Trains the linear regression ADR baseline on data/preprocessed/train.csv.
Validation (val.csv) is used to compare variants. test.csv is never read here.

Usage (from the repo root):

    python src/regression/linear_regression/train.py
        Train the final model with the saved selected config (defaults if none)
        and write validation metrics, plots and the joblib pipeline.

    python src/regression/linear_regression/train.py --base --run-name base
        Train the untuned default RegressionConfig without reading or writing
        selected_config.json.

    python src/regression/linear_regression/train.py --variants
        Exhaustively evaluate the Cartesian product of every option in Tests
        1 to 6, select the best configuration, and train the final model.
        Add --tests 3 4 to restrict the search to those dimensions; other
        dimensions retain their RegressionConfig defaults.

    python src/regression/linear_regression/train.py --claims
        Check the assumed drivers of ADR (season, room type) before the report.

Selection rule: lowest validation RMSE on the original ADR scale. If multiple
combinations are within --tol (relative RMSE, default 0.5%) of the best score,
the first combination in the preferred option order is kept.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd 
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regression.linear_regression.features import (  # noqa: E402
    HARMONIC_COLUMNS,
    MONTH_TO_NUM,
    RAW_CATEGORICAL,
    TARGET,
    RegressionConfig,
    assert_no_leakage,
    engineer_features,
)
from src.regression.linear_regression.linear_regression import (  # noqa: E402
    OUTPUT_DIR,
    SELECTED_CONFIG_PATH,
    SUMMARY_PATH,
    build_model,
    coefficient_table,
    cv_rmse,
    evaluate,
    feature_names,
    get_artifact_dir,
    get_run_dir,
    load_selected_config,
    load_split,
    new_run_name,
    save_diagnostic_plots,
    save_model,
    split_xy,
)
from src.regression.linear_regression.preprocessing import (  # noqa: E402
    fit_iqr_limits,
    paying_only,
    within_limits,
)

NOISE_TOL = 0.005

# Each test: (name, [(label, config overrides), ...]) with the preferred option first.
VARIANT_TESTS = {
    1: ("target_transform", [
        ("raw_adr", {"log_target": False}),
        ("log_adr", {"log_target": True}),
    ]),
    2: ("outlier_filter", [
        ("no_filter", {"iqr_factor": None}),
        ("iqr_1.5", {"iqr_factor": 1.5}),
        ("iqr_3.0", {"iqr_factor": 3.0}),
    ]),
    3: ("seasonality", [
        ("month_onehot", {"seasonality": "month"}),
        ("doy_harmonics", {"seasonality": "harmonics"}),
    ]),
    # Fewer columns first: a lower min_frequency has to earn its extra columns.
    4: ("agent_grouping", [
        (f"min_freq_{k}", {"agent_min_frequency": k}) for k in (200, 100, 50, 25, 10)
    ]),
    5: ("flags", [
        ("neither", {"use_repeated_guest": False, "use_prior_cancel": False}),
        ("repeated_guest_only", {"use_repeated_guest": True, "use_prior_cancel": False}),
        ("prior_cancel_only", {"use_repeated_guest": False, "use_prior_cancel": True}),
        ("both", {"use_repeated_guest": True, "use_prior_cancel": True}),
    ]),
    6: ("night_features", [
        ("log_total_plus_share", {"nights": "log_share"}),
        ("raw_weekend_week", {"nights": "raw"}),
    ]),
}


# ----------------------------------------------------------------------------
# Fitting and scoring one config
# ----------------------------------------------------------------------------
def prepare(cfg: RegressionConfig, train: pd.DataFrame, val: pd.DataFrame):
    """Apply the regression row filters. IQR limits come from the training split
    only and filter training rows only. Validation stays unfiltered by IQR."""
    train, val = paying_only(train), paying_only(val)
    limits = None
    if cfg.iqr_factor is not None:
        limits = fit_iqr_limits(train[TARGET], cfg.iqr_factor)
        train = train[within_limits(train[TARGET], limits)]
    return train, val, limits


def fit_and_score(
    cfg: RegressionConfig, train: pd.DataFrame, val: pd.DataFrame,
    include_cv: bool = True,
):
    """Fit on train, score on val. Returns (row, model, limits, val_rows, val_pred)."""
    train_fit, val_rows, limits = prepare(cfg, train, val)
    X, y = split_xy(train_fit)

    model = build_model(cfg)
    start = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - start

    metrics, pred = evaluate(model, val_rows, limits)
    row = {
        "val_MAE": metrics["MAE"],
        "val_RMSE": metrics["RMSE"],
        "val_R2": metrics["R2"],
        "train_time_s": fit_seconds,
        "n_train_rows": len(train_fit),
        "n_features": len(feature_names(model)),
    }
    if include_cv:
        row["cv_RMSE"] = cv_rmse(cfg, train_fit)
    for key in ("MAE", "RMSE", "R2"):
        if f"{key}_in_limits" in metrics:
            row[f"val_{key}_in_limits"] = metrics[f"{key}_in_limits"]
    return row, model, limits, val_rows, pred


def choose_winner(results, tol: float) -> int:
    """Choose the first preferred config within tol of the best validation RMSE."""
    best_rmse = min(row["val_RMSE"] for row in results)
    acceptable_rmse = best_rmse * (1 + tol)
    return next(
        i for i, row in enumerate(results)
        if row["val_RMSE"] <= acceptable_rmse
    )


def build_variant_combinations(tests, base_cfg: Optional[RegressionConfig] = None):
    """Build the Cartesian product of selected dimensions over config defaults."""
    cfg = base_cfg or RegressionConfig()
    dimensions = []
    for number in tests:
        name, options = VARIANT_TESTS[number]
        dimensions.append((name, options))

    combinations = []
    for selected in product(*(options for _, options in dimensions)):
        labels = {dimensions[i][0]: selected[i][0] for i in range(len(dimensions))}
        config = cfg
        for _, overrides in selected:
            config = replace(config, **overrides)
        combinations.append((labels, config))
    return combinations


# ----------------------------------------------------------------------------
# Variant tests
# ----------------------------------------------------------------------------
def run_variant_tests(
    train, val, tests, tol: float, output_dir: Optional[Path] = None,
    run_id: Optional[int] = None, run_name: Optional[str] = None,
) -> RegressionConfig:
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    if run_id is not None and run_name is not None:
        raise ValueError("specify either run_id or run_name, not both")
    out_dir.mkdir(parents=True, exist_ok=True)
    if run_name is not None:
        artifact_run_dir = get_artifact_dir(run_name=run_name)
    elif run_id is not None:
        artifact_run_dir = get_artifact_dir(run_id)
    else:
        artifact_run_dir = None

    combinations = build_variant_combinations(tests)
    print(f"Evaluating {len(combinations)} variant combinations.")
    rows = []
    configs = []
    for index, (labels, cfg) in enumerate(combinations, start=1):
        metrics, *_ = fit_and_score(cfg, train, val, include_cv=False)
        rows.append({"combination": index, **labels, **metrics})
        configs.append(cfg)
        print(
            f"  {index}/{len(combinations)}: validation RMSE "
            f"{metrics['val_RMSE']:.3f}"
        )

    winner = choose_winner(rows, tol)
    rows[winner]["selected"] = True
    for index, row in enumerate(rows):
        row.setdefault("selected", False)
        row["rank"] = 1 + sum(
            other["val_RMSE"] < row["val_RMSE"] for other in rows
        )
    table = pd.DataFrame(rows).sort_values(["rank", "combination"])
    table.to_csv(out_dir / "variant_summary.csv", index=False)

    cfg = configs[winner]
    config_json = json.dumps(cfg.to_dict(), indent=2)
    config_path = (
        artifact_run_dir / "selected_config.json"
        if artifact_run_dir is not None
        else SELECTED_CONFIG_PATH
    )
    config_path.write_text(config_json)
    print(
        f"Selected combination {rows[winner]['combination']} "
        f"(validation RMSE {rows[winner]['val_RMSE']:.3f}; "
        f"rank {rows[winner]['rank']})"
    )
    print(f"Selected config: {cfg.to_dict()}")
    return cfg


# ----------------------------------------------------------------------------
# Final model
# ----------------------------------------------------------------------------
def train_final(
    cfg: RegressionConfig, train, val, output_dir: Optional[Path] = None,
    run_id: Optional[int] = None, run_name: Optional[str] = None,
    save_selected_config: bool = True,
) -> None:
    if run_id is not None and run_name is not None:
        raise ValueError("specify either run_id or run_name, not both")
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    row, model, limits, val_rows, val_pred = fit_and_score(cfg, train, val)
    assert_no_leakage(feature_names(model))

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_dir / "final_model_validation_metrics.csv", index=False)
    coefficient_table(model).to_csv(out_dir / "coefficients.csv", index=False)
    save_diagnostic_plots(val_rows[TARGET], val_pred, out_dir, "validation")

    model_path = save_model(model, run_id=run_id, run_name=run_name)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "run_name": run_name,
        "output_dir": str(out_dir),
        "sklearn_version": sklearn.__version__,
        "config": cfg.to_dict(),
        "iqr_limits": list(limits) if limits is not None else None,
        "n_train_rows": row["n_train_rows"],
        "validation_metrics": row,
    }
    summary_json = json.dumps(summary, indent=2)
    if run_name is not None:
        artifact_run_dir = get_artifact_dir(run_name=run_name)
        (artifact_run_dir / "training_summary.json").write_text(summary_json)
        if save_selected_config:
            (artifact_run_dir / "selected_config.json").write_text(
                json.dumps(cfg.to_dict(), indent=2)
            )
    elif run_id is not None:
        artifact_run_dir = get_artifact_dir(run_id)
        (artifact_run_dir / "training_summary.json").write_text(summary_json)
        if save_selected_config:
            (artifact_run_dir / "selected_config.json").write_text(
                json.dumps(cfg.to_dict(), indent=2)
            )
    else:
        SUMMARY_PATH.write_text(summary_json)

    print("Final model, validation metrics (original ADR scale)")
    for key in ("val_MAE", "val_RMSE", "val_R2", "cv_RMSE", "train_time_s"):
        print(f"  {key}: {row[key]:.4f}")
    print(f"Config: {cfg.to_dict()}")
    print(f"Saved {model_path}")


# ----------------------------------------------------------------------------
# Claims to verify before the report
# ----------------------------------------------------------------------------
def check_claims(train, val, output_dir: Optional[Path] = None) -> None:
    """Check that season is the strongest driver and ReservedRoomType the second.

    Fits one-feature-group models on train and reports validation R2, and saves
    mean ADR by month and by room type (train rows)."""
    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    train, val = paying_only(train), paying_only(val)
    eng_train, eng_val = engineer_features(train), engineer_features(val)
    y_train, y_val = train[TARGET], val[TARGET]
    out_dir.mkdir(parents=True, exist_ok=True)

    for column, order, fname in (
        ("ArrivalMonth", list(MONTH_TO_NUM), "claims_mean_adr_by_month.csv"),
        ("ReservedRoomType", None, "claims_mean_adr_by_room_type.csv"),
    ):
        table = y_train.groupby(eng_train[column]).agg(["mean", "median", "count"])
        table = table.reindex(order) if order else table.sort_values("mean", ascending=False)
        table.to_csv(out_dir / fname)
        print(f"Mean ADR by {column}")
        print(table.to_string(float_format=lambda v: f"{v:.1f}"), "\n")

    groups = {
        "Season: ArrivalMonth (one hot)": (["ArrivalMonth"], "cat"),
        "Season: day of year harmonics": (HARMONIC_COLUMNS, "num"),
        "Trend: DaysSinceStart": (["DaysSinceStart"], "num"),
    }
    for col in [*RAW_CATEGORICAL, "ArrivalDayOfWeek", "Agent"]:
        groups[col] = ([col], "cat")
    for col in ["Adults", "Children", "LeadTimeLog", "TotalNightsLog", "WeekendShare",
                "IsRepeatedGuest", "HadPriorCancel"]:
        groups[col] = ([col], "num")

    results = []
    for label, (cols, kind) in groups.items():
        if kind == "cat":
            step = OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=20, sparse_output=False)
        else:
            step = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler())])
        model = Pipeline([("x", ColumnTransformer([("g", step, cols)])),
                          ("lr", LinearRegression())])
        model.fit(eng_train, y_train)
        pred = np.clip(model.predict(eng_val), 0, None)
        results.append({
            "feature": label,
            "val_R2": r2_score(y_val, pred),
            "val_RMSE": float(np.sqrt(mean_squared_error(y_val, pred))),
        })

    table = pd.DataFrame(results).sort_values("val_R2", ascending=False)
    table.to_csv(out_dir / "claims_single_feature_r2.csv", index=False)
    print("Validation R2 of one-feature models (higher means a stronger single driver)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ----------------------------------------------------------------------------
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", action="store_true",
                        help="exhaustively evaluate all selected variant combinations")
    parser.add_argument("--tests", type=int, nargs="+", choices=sorted(VARIANT_TESTS),
                        help="which dimensions to include in the grid (default: all 1 to 6)")
    parser.add_argument("--with-optional", action="store_true",
                        help="legacy alias; Test 6 is included by default")
    parser.add_argument("--tol", type=float, default=NOISE_TOL,
                        help="relative RMSE tolerance for preferring earlier grid options (default 0.005)")
    parser.add_argument("--claims", action="store_true", help="check the driver claims and exit")
    parser.add_argument("--run", type=int, default=None,
                        help="legacy numeric run ID")
    parser.add_argument("--run-name", default=None,
                        help="name for this run's output and artifact folders (default: UTC timestamp)")
    parser.add_argument("--base", action="store_true",
                        help="train the untuned default RegressionConfig without reading or writing selected_config.json")
    args = parser.parse_args(argv)

    if args.run is not None and args.run_name is not None:
        parser.error("--run and --run-name cannot be used together")
    if args.base and args.variants:
        parser.error("--base cannot be combined with --variants")

    train, val = load_split("train"), load_split("val")

    if args.run_name is not None:
        run_id = None
        run_name = args.run_name
        run_dir = get_run_dir(run_name=run_name)
    elif args.run is not None:
        run_id = args.run
        run_name = None
        run_dir = get_run_dir(run_id)
    else:
        run_id = None
        run_name = new_run_name()
        run_dir = get_run_dir(run_name=run_name)

    run_label = f"run {run_id}" if run_id is not None else f"run '{run_name}'"
    print(f"Target run directory: {run_dir} ({run_label})")

    if args.claims:
        check_claims(train, val, output_dir=run_dir)
        return

    if args.variants:
        tests = sorted(set(args.tests)) if args.tests else sorted(VARIANT_TESTS)
        if args.with_optional and 6 not in tests:
            tests.append(6)
            tests.sort()
        cfg = run_variant_tests(
            train, val, tests, args.tol, output_dir=run_dir,
            run_id=run_id, run_name=run_name,
        )
    elif args.base:
        cfg = RegressionConfig()
    elif run_id is not None:
        cfg = load_selected_config(run_id=run_id)
    elif run_name is not None:
        run_config_path = get_artifact_dir(run_name=run_name) / "selected_config.json"
        cfg = load_selected_config(run_name=run_name) if run_config_path.exists() else load_selected_config()
    else:
        cfg = load_selected_config()

    train_final(
        cfg, train, val, output_dir=run_dir, run_id=run_id,
        run_name=run_name, save_selected_config=not args.base,
    )


if __name__ == "__main__":
    main()
