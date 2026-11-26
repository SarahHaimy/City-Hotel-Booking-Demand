"""Registered classification model adapter wrapping the implemented Gaussian Naive Bayes pipeline.

When teammates implement additional Milestone 2 classifiers, register them here following
`docs/MODEL_CONTRACT.md` and they will appear across the GUI automatically.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.classification.bayes_preprocessing import (
    categorical_features as CLF_CAT_FEATURES,
    features as CLF_FEATURES,
    numerical_features as CLF_NUM_FEATURES,
)
from src.config import SEED
from src.data.schema import TASK_FORBIDDEN_COLUMNS
from src.registry.registry import register_model
from src.registry.specs import FeatureSpace, ModelOptionSpec, ModelSpec, ParamSpec

CLF_FORBIDDEN = list(TASK_FORBIDDEN_COLUMNS["classification"])


def sanitize_classification_inputs(X: pd.DataFrame) -> pd.DataFrame:
    """Stateless input normalizer inside the classification Pipeline.

    Selects the exact 13 reservation-time features from `src/classification/bayes_preprocessing.py`,
    coerces numerics, and normalizes categorical text while ignoring any extra columns.
    """
    missing = [c for c in CLF_FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required classification columns: {missing}")

    out: dict[str, Any] = {}
    for col in CLF_NUM_FEATURES:
        out[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CLF_CAT_FEATURES:
        s = X[col].astype("string").str.strip()
        is_missing = s.isna() | s.str.upper().isin(["NULL", "NAN", ""])
        out[col] = s.mask(is_missing, "Unknown").astype(object)
    return pd.DataFrame(out, index=X.index)


def build_clf_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), list(CLF_NUM_FEATURES)),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(CLF_CAT_FEATURES)),
        ],
        remainder="drop",
    )


def build_gaussian_nb_pipeline(
    params: dict[str, Any],
    feature_space: FeatureSpace = "original",
    n_components: float | int | None = None,
) -> Pipeline:
    vs = float(params.get("var_smoothing", 1e-9))
    base = GaussianNB(var_smoothing=vs)
    if bool(params.get("calibrate", False)):
        est: Any = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=5)
    else:
        est = base

    use_scale = feature_space in ("pca", "lda")
    steps: list[tuple[str, Any]] = [
        ("sanitize", FunctionTransformer(sanitize_classification_inputs, validate=False)),
        ("preprocess", build_clf_preprocessor(scale_numeric=use_scale)),
    ]
    if feature_space == "pca":
        nc = 0.90 if n_components is None else n_components
        steps.append(("pca", PCA(n_components=nc, random_state=SEED)))
    elif feature_space == "lda":
        steps.append(("lda", LinearDiscriminantAnalysis(n_components=1)))
    steps.append(("classifier", est))
    return Pipeline(steps)


register_model(
    ModelSpec(
        model_id="gaussian_nb",
        task="classification",
        display_name="Gaussian Naive Bayes",
        family="bayes",
        description=(
            "Implemented cancellation classifier (`src/classification/gaussian_nb_*.py`) pairing one-hot "
            "reservation categories and numerical booking attributes with GaussianNB, variance smoothing "
            "optimization (`1e-5`), optional 5-fold Sigmoid calibration, and decision-threshold tuning."
        ),
        assumptions_and_pros=(
            "Ultra-fast generative classifier; conditional feature independence produces sharp posterior "
            "probabilities that benefit from `var_smoothing`, Sigmoid calibration, and threshold selection."
        ),
        required_inputs=list(CLF_FEATURES),
        forbidden_inputs=CLF_FORBIDDEN,
        params=[
            ParamSpec("var_smoothing", "Variance Smoothing", "log_float", 1e-5, low=1e-10, high=1e-2, help="Portion of largest feature variance added for numerical stability."),
            ParamSpec("calibrate", "Sigmoid Probability Calibration (5-fold)", "bool", False, help="Wrap GaussianNB in CalibratedClassifierCV(method='sigmoid', cv=5)."),
            ParamSpec("threshold", "Decision Threshold", "float", 0.60, low=0.20, high=0.90, step=0.05),
        ],
        baseline_params={"var_smoothing": 1e-9, "calibrate": False, "threshold": 0.50},
        tuned_params={"var_smoothing": 1e-5, "calibrate": False, "threshold": 0.60},
        search_space={"classifier__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 3e-5, 1e-4, 1e-3]},
        default_scoring="f1",
        build_pipeline=build_gaussian_nb_pipeline,
        supports_proba=True,
        supports_pca=True,
        supports_lda=True,
        cost_hint="fast",
        track_origin="milestone_1",
        supported_actions=("train", "tune", "evaluate"),
        cli_options=[
            ModelOptionSpec(
                flag="--calibrate",
                dest="calibrate",
                kind="bool",
                default=False,
                help="Wrap GaussianNB in 5-fold Sigmoid CalibratedClassifierCV.",
            ),
            ModelOptionSpec(
                flag="--var-smoothing",
                dest="var_smoothing",
                kind="float",
                default=None,
                help="Override GaussianNB var_smoothing (e.g. 1e-5).",
            ),
            ModelOptionSpec(
                flag="--threshold",
                dest="threshold",
                kind="float",
                default=None,
                help="Override classification decision threshold (0.0 to 1.0).",
            ),
        ],
    )
)
