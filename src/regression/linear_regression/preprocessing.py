"""
preprocessing.py

Builds the unfitted preprocessing for the linear regression baseline, plus the
row filters used by the regression training code.

Nothing here touches data on import. build_preprocessor() returns an unfitted
ColumnTransformer; it is fit inside the pipeline on the training split only.

Row filters live here (and are called from train.py) rather than in
clean_data, because the IQR limits are learned from data.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regression.linear_regression.features import (  # noqa: E402
    TARGET,
    RegressionConfig,
    feature_groups,
)


def _one_hot(min_frequency: int) -> OneHotEncoder:
    # Categories seen fewer than min_frequency times in training are grouped
    # into one infrequent group. Unknown categories at inference go to that
    # group when it exists, otherwise they encode as all zeros.
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_frequency,
        sparse_output=False,
    )


def build_preprocessor(cfg: RegressionConfig) -> ColumnTransformer:
    """Unfitted ColumnTransformer for the engineered frame."""
    groups = feature_groups(cfg)

    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    transformers = [
        ("num", numeric, groups["numeric"]),
        ("cat", _one_hot(cfg.other_min_frequency), groups["categorical"]),
        # Agent has its own encoder so Test 4 changes only Agent grouping.
        ("agent", _one_hot(cfg.agent_min_frequency), groups["agent"]),
    ]
    if groups["flags"]:
        transformers.append(("flags", "passthrough", groups["flags"]))

    return ColumnTransformer(transformers, remainder="drop")


# ----------------------------------------------------------------------------
# Row filters (regression training code only)
# ----------------------------------------------------------------------------
def paying_only(df: pd.DataFrame) -> pd.DataFrame:
    """Row filter 1: keep ADR > 0. Applied to every split.

    A no-op when clean_data already dropped ADR == 0. If the team switches to
    clean_data(drop_zero_adr=False) for the shared split, this is what keeps
    the regression scope to paying bookings.
    """
    return df[df[TARGET] > 0]


def fit_iqr_limits(y_train, factor: float) -> Tuple[float, float]:
    """Row filter 2 limits: Q1 - factor*IQR and Q3 + factor*IQR, from training only."""
    q1, q3 = np.percentile(np.asarray(y_train, dtype=float), [25, 75])
    iqr = q3 - q1
    return float(q1 - factor * iqr), float(q3 + factor * iqr)


def within_limits(y, limits: Optional[Tuple[float, float]]) -> np.ndarray:
    """Boolean mask of values inside the limits. No limits means keep all."""
    y = np.asarray(y, dtype=float)
    if limits is None:
        return np.ones(len(y), dtype=bool)
    low, high = limits
    return (y >= low) & (y <= high)
