"""
train.py

Trains the linear regression ADR baseline on data/preprocessed/train.csv.
Validation (val.csv) is used to compare variants. test.csv is never read here.

Usage (from the repo root):

    python src/regression/linear_regression/train.py
        Train the final model with the saved selected config (defaults if none)
        and write validation metrics, plots and the joblib pipeline.

    python src/regression/linear_regression/train.py --variants
        Run Tests 1 to 5 in order, one at a time, keeping each winner before the
        next test, then train the final model with the winners.
        Add --with-optional to include Test 6 (night features).
        Add --tests 3 4 to run only some tests. A run that does not start at
        Test 1 continues from the saved selected config.

    python src/regression/linear_regression/train.py --claims
        Check the assumed drivers of ADR (season, room type) before the report.

Selection rule: lowest validation RMSE on the original ADR scale. Options in
each test are listed from preferred to least preferred, and a later option only
wins if it beats the incumbent by more than --tol (relative RMSE, default 0.5%).
Within that margin the earlier, simpler option is kept.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

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
    ARTIFACT_DIR,
    OUTPUT_DIR,
    SELECTED_CONFIG_PATH,
    SUMMARY_PATH,
    build_model,
    coefficient_table,
    cv_rmse,
    evaluate,
    feature_names,
    load_selected_config,
    load_split,
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


def fit_and_score(cfg: RegressionConfig, train: pd.DataFrame, val: pd.DataFrame):
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
        "cv_RMSE": cv_rmse(cfg, train_fit),
        "train_time_s": fit_seconds,
        "n_train_rows": len(train_fit),
        "n_features": len(feature_names(model)),
    }
    for key in ("MAE", "RMSE", "R2"):
        if f"{key}_in_limits" in metrics:
            row[f"val_{key}_in_limits"] = metrics[f"{key}_in_limits"]
    return row, model, limits, val_rows, pred


def choose_winner(results, tol: float) -> int:
    """Index of the winner. Options come preferred first. A later option replaces
    the incumbent only if its validation RMSE is lower by more than tol."""
    best = 0
    for i in range(1, len(results)):
        if results[i]["val_RMSE"] < results[best]["val_RMSE"] * (1 - tol):
            best = i
    return best


# ----------------------------------------------------------------------------
# Variant tests
# ----------------------------------------------------------------------------
def run_variant_tests(train, val, tests, tol: float) -> RegressionConfig:
    if min(tests) == 1:
        cfg = RegressionConfig()
    elif SELECTED_CONFIG_PATH.exists():
        cfg = load_selected_config()
        print(f"Continuing from saved selected config: {cfg.to_dict()}")
    else:
        cfg = RegressionConfig()
        print("No saved selected config, starting from defaults.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for number in tests:
        name, options = VARIANT_TESTS[number]
        rows = []
        for label, overrides in options:
            row, *_ = fit_and_score(replace(cfg, **overrides), train, val)
            rows.append({"variant": label, **row})
            print(f"  test {number} {label}: val RMSE {row['val_RMSE']:.3f}")

        winner = choose_winner(rows, tol)
        for i, row in enumerate(rows):
            row["selected"] = i == winner
        table = pd.DataFrame(rows)
        table.to_csv(OUTPUT_DIR / f"test{number}_{name}.csv", index=False)

        cfg = replace(cfg, **options[winner][1])
        SELECTED_CONFIG_PATH.write_text(json.dumps(cfg.to_dict(), indent=2))
        summary_rows.append({"test": number, "name": name, "winner": options[winner][0],
                             "val_RMSE": rows[winner]["val_RMSE"]})
        print(f"\nTest {number}: {name}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"Winner: {options[winner][0]}\n")

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "variant_summary.csv", index=False)
    return cfg


# ----------------------------------------------------------------------------
# Final model
# ----------------------------------------------------------------------------
def train_final(cfg: RegressionConfig, train, val) -> None:
    row, model, limits, val_rows, val_pred = fit_and_score(cfg, train, val)
    assert_no_leakage(feature_names(model))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(OUTPUT_DIR / "final_model_validation_metrics.csv", index=False)
    coefficient_table(model).to_csv(OUTPUT_DIR / "coefficients.csv", index=False)
    save_diagnostic_plots(val_rows[TARGET], val_pred, OUTPUT_DIR, "validation")

    model_path = save_model(model)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "config": cfg.to_dict(),
        "iqr_limits": list(limits) if limits is not None else None,
        "n_train_rows": row["n_train_rows"],
        "validation_metrics": row,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("Final model, validation metrics (original ADR scale)")
    for key in ("val_MAE", "val_RMSE", "val_R2", "cv_RMSE", "train_time_s"):
        print(f"  {key}: {row[key]:.4f}")
    print(f"Config: {cfg.to_dict()}")
    print(f"Saved {model_path}")


# ----------------------------------------------------------------------------
# Claims to verify before the report
# ----------------------------------------------------------------------------
def check_claims(train, val) -> None:
    """Check that season is the strongest driver and ReservedRoomType the second.

    Fits one-feature-group models on train and reports validation R2, and saves
    mean ADR by month and by room type (train rows)."""
    train, val = paying_only(train), paying_only(val)
    eng_train, eng_val = engineer_features(train), engineer_features(val)
    y_train, y_val = train[TARGET], val[TARGET]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for column, order, fname in (
        ("ArrivalMonth", list(MONTH_TO_NUM), "claims_mean_adr_by_month.csv"),
        ("ReservedRoomType", None, "claims_mean_adr_by_room_type.csv"),
    ):
        table = y_train.groupby(eng_train[column]).agg(["mean", "median", "count"])
        table = table.reindex(order) if order else table.sort_values("mean", ascending=False)
        table.to_csv(OUTPUT_DIR / fname)
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
    table.to_csv(OUTPUT_DIR / "claims_single_feature_r2.csv", index=False)
    print("Validation R2 of one-feature models (higher means a stronger single driver)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ----------------------------------------------------------------------------
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", action="store_true", help="run the variant tests first")
    parser.add_argument("--tests", type=int, nargs="+", choices=sorted(VARIANT_TESTS),
                        help="which tests to run (default 1 to 5)")
    parser.add_argument("--with-optional", action="store_true", help="also run Test 6")
    parser.add_argument("--tol", type=float, default=NOISE_TOL,
                        help="relative RMSE margin a later option must beat (default 0.005)")
    parser.add_argument("--claims", action="store_true", help="check the driver claims and exit")
    args = parser.parse_args(argv)

    train, val = load_split("train"), load_split("val")

    if args.claims:
        check_claims(train, val)
        return

    if args.variants:
        tests = sorted(args.tests) if args.tests else [1, 2, 3, 4, 5]
        if args.with_optional and 6 not in tests:
            tests.append(6)
        cfg = run_variant_tests(train, val, tests, args.tol)
    else:
        cfg = load_selected_config()

    train_final(cfg, train, val)


if __name__ == "__main__":
    main()
