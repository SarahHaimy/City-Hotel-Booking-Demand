"""Registered regression model adapter wrapping the implemented Linear Regression ADR pipeline.

When teammates implement additional Milestone 2 regressors, register them here following
`docs/MODEL_CONTRACT.md` and they will appear across the GUI automatically.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import SEED
from src.regression.linear_regression.features import (
    FORBIDDEN_COLUMNS,
    RAW_INPUT_COLUMNS,
    RegressionConfig,
    engineer_features,
)
from src.regression.linear_regression.linear_regression import (
    _clip_nonnegative,
    _expm1_clip,
    _identity,
)
from src.regression.linear_regression.preprocessing import build_preprocessor
from src.registry.registry import register_model
from src.registry.specs import FeatureSpace, ModelOptionSpec, ModelSpec, ParamSpec


def run_linear_regression_variants(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    with_optional: bool = False,
    tol: float = 0.005,
    progress_cb: Any = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Run Linear Regression's sequential variant ablation (Tests 1-5, optional Test 6).

    Pure function: does not touch global output directories; returns (winning_params, extra_tables).
    """
    from dataclasses import replace

    from src.regression.linear_regression.train import VARIANT_TESTS, choose_winner, fit_and_score

    tests = [1, 2, 3, 4, 5, 6] if with_optional else [1, 2, 3, 4, 5]
    cfg = RegressionConfig()
    extra_tables: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []

    for idx, number in enumerate(tests):
        name, options = VARIANT_TESTS[number]
        if progress_cb:
            progress_cb(idx / len(tests), f"Running variant test {number}/{len(tests)} ({name})…")
        rows: list[dict[str, Any]] = []
        for label, overrides in options:
            row, *_ = fit_and_score(replace(cfg, **overrides), train_df, val_df)
            rows.append({"variant": label, **row})

        winner = choose_winner(rows, tol)
        for i, row in enumerate(rows):
            row["selected"] = i == winner
        extra_tables[f"test{number}_{name}"] = pd.DataFrame(rows)
        cfg = replace(cfg, **options[winner][1])
        summary_rows.append(
            {
                "test": number,
                "name": name,
                "winner": options[winner][0],
                "val_RMSE": rows[winner]["val_RMSE"],
                "val_MAE": rows[winner]["val_MAE"],
                "val_R2": rows[winner]["val_R2"],
            }
        )

    extra_tables["variant_summary"] = pd.DataFrame(summary_rows)
    return cfg.to_dict(), extra_tables


def _cfg_from_params(params: dict[str, Any]) -> RegressionConfig:
    return RegressionConfig(
        log_target=bool(params.get("log_target", True)),
        iqr_factor=params.get("iqr_factor", None),
        seasonality=str(params.get("seasonality", "month")),
        nights=str(params.get("nights", "log_share")),
        agent_min_frequency=int(params.get("agent_min_frequency", 10)),
        other_min_frequency=int(params.get("other_min_frequency", 20)),
        use_repeated_guest=bool(params.get("use_repeated_guest", True)),
        use_prior_cancel=bool(params.get("use_prior_cancel", False)),
    )


def build_linear_regression_pipeline(
    params: dict[str, Any],
    feature_space: FeatureSpace = "original",
    n_components: float | int | None = None,
) -> TransformedTargetRegressor:
    cfg = _cfg_from_params(params)
    steps: list[tuple[str, Any]] = [
        ("engineer", FunctionTransformer(engineer_features, validate=False)),
        ("preprocess", build_preprocessor(cfg)),
    ]
    if feature_space == "pca":
        nc = 0.90 if n_components is None else n_components
        steps.append(("pca", PCA(n_components=nc, random_state=SEED)))
    steps.append(("regressor", LinearRegression()))

    pipe = Pipeline(steps)
    func, inv = (np.log1p, _expm1_clip) if cfg.log_target else (_identity, _clip_nonnegative)
    return TransformedTargetRegressor(
        regressor=pipe, func=func, inverse_func=inv, check_inverse=False
    )


register_model(
    ModelSpec(
        model_id="linear_regression",
        task="regression",
        display_name="Linear Regression (OLS)",
        family="linear",
        description=(
            "Implemented ADR regression pipeline (`src/regression/linear_regression/`) wrapping stateless "
            "seasonality & stay-duration engineering, median imputation, scaling, frequency-grouped OneHotEncoder, "
            "and Ordinary Least Squares inside TransformedTargetRegressor."
        ),
        assumptions_and_pros=(
            "Fast, deterministic closed-form fit with interpretable feature coefficients; "
            "6-stage variant ablation selected log1p(ADR) target transform + Agent min_frequency=10."
        ),
        required_inputs=list(RAW_INPUT_COLUMNS),
        forbidden_inputs=list(FORBIDDEN_COLUMNS),
        params=[
            ParamSpec("log_target", "Log1p Target Transform", "bool", True, help="Predict log(1+ADR) and invert via expm1 clipped at 0."),
            ParamSpec("seasonality", "Seasonality Encoding", "choice", "month", choices=("month", "harmonics"), help="One-hot month vs Fourier day-of-year harmonics."),
            ParamSpec("nights", "Stay Duration Representation", "choice", "log_share", choices=("log_share", "raw")),
            ParamSpec("agent_min_frequency", "Agent Min Frequency", "int", 10, low=5, high=250, step=5),
            ParamSpec("other_min_frequency", "Categorical Min Frequency", "int", 20, low=5, high=100, step=5),
            ParamSpec("use_repeated_guest", "Include Repeated Guest Flag", "bool", True),
            ParamSpec("use_prior_cancel", "Include Prior Cancel Flag", "bool", False),
        ],
        baseline_params={
            "log_target": False,
            "seasonality": "month",
            "nights": "log_share",
            "agent_min_frequency": 50,
            "other_min_frequency": 20,
            "use_repeated_guest": True,
            "use_prior_cancel": True,
        },
        tuned_params={
            "log_target": True,
            "seasonality": "month",
            "nights": "log_share",
            "agent_min_frequency": 10,
            "other_min_frequency": 20,
            "use_repeated_guest": True,
            "use_prior_cancel": False,
        },
        search_space={
            "log_target": [False, True],
            "seasonality": ["month", "harmonics"],
            "nights": ["log_share", "raw"],
            "agent_min_frequency": [10, 25, 50, 100],
            "use_prior_cancel": [False, True],
        },
        default_scoring="neg_root_mean_squared_error",
        build_pipeline=build_linear_regression_pipeline,
        cost_hint="fast",
        track_origin="milestone_1",
        supported_actions=("train", "tune", "evaluate"),
        cli_options=[
            ModelOptionSpec(
                flag="--variants",
                dest="variants",
                kind="bool",
                default=False,
                help="Run sequential 5-stage ablation variant tests before fitting the winning config.",
            ),
            ModelOptionSpec(
                flag="--with-optional",
                dest="with_optional",
                kind="bool",
                default=False,
                help="Include optional Test 6 (night features) when --variants is enabled.",
                requires="variants",
            ),
            ModelOptionSpec(
                flag="--tol",
                dest="tol",
                kind="float",
                default=0.005,
                help="Relative validation RMSE improvement tolerance for --variants winner selection.",
                requires="variants",
            ),
        ],
    )
)
