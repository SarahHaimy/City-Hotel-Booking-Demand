"""
features.py

Schema, configuration and stateless feature engineering for the linear
regression ADR baseline.

This file holds no fitted state. It defines:
  * which raw booking fields the model needs (RAW_INPUT_COLUMNS),
  * which fields must never reach the model (FORBIDDEN_COLUMNS),
  * RegressionConfig, the single object that records every choice made by the
    variant tests (target transform, IQR factor, seasonality, Agent grouping,
    flags, night features),
  * engineer_features(), a pure function that turns raw booking fields into
    the derived columns. It runs as the first step of the pipeline, so
    inference accepts raw booking fields.

Everything learned from data (scalers, encoders, imputers, IQR limits) lives
elsewhere and is fit on the training split only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Optional

import numpy as np
import pandas as pd

TARGET = "ADR"

# Trend feature origin. Arrival dates in the data run from July 2015 to August
# 2017, so DaysSinceStart does not extrapolate beyond that range.
START_DATE = pd.Timestamp("2015-07-01")
DAYS_PER_YEAR = 365.25

MONTH_TO_NUM = {
    name: i
    for i, name in enumerate(
        [
            "January", "February", "March", "April", "May", "June", "July",
            "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}

# Categorical fields kept as they are (they share one encoder).
RAW_CATEGORICAL = [
    "Meal",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",  # room booked, not assigned
    "DepositType",
    "CustomerType",
]

# Raw fields engineer_features() reads. Any other column in the input frame
# (IsCanceled, ReservationStatus, ...) is ignored, so leakage columns cannot
# reach the model even if they are present in the dataframe.
RAW_INPUT_COLUMNS = [
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateDayOfMonth",
    "LeadTime",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    *RAW_CATEGORICAL,
    "Agent",
    "IsRepeatedGuest",
    "PreviousCancellations",
]

HARMONIC_COLUMNS = ["DoySin1", "DoyCos1", "DoySin2", "DoyCos2"]

# Dropped because they are the target, the outcome, or not known at booking.
FORBIDDEN_COLUMNS = [
    "ADR",
    "IsCanceled",
    "ReservationStatus",
    "ReservationStatusDate",
    "AssignedRoomType",
    "BookingChanges",
    "DaysInWaitingList",
    "TotalOfSpecialRequests",
    "HasBabies",
    "RequiredCarParkingSpaces",
]

SEASONALITY_OPTIONS = ("month", "harmonics")
NIGHTS_OPTIONS = ("log_share", "raw")


@dataclass
class RegressionConfig:
    """Every choice the variant tests can change. Defaults are the starting point."""

    # Test 1: log1p / expm1 target transform (predictions clipped at 0).
    log_target: bool = False
    # Test 2: IQR factor for training-row outlier removal, None means no filter.
    iqr_factor: Optional[float] = None
    # Test 3: "month" = one hot ArrivalMonth, "harmonics" = sin/cos of day of year.
    seasonality: str = "month"
    # Test 6: "log_share" = TotalNightsLog + WeekendShare, "raw" = weekend and week nights.
    nights: str = "log_share"
    # Test 4: min_frequency of the Agent encoder (other categoricals use other_min_frequency).
    agent_min_frequency: int = 50
    other_min_frequency: int = 20
    # Test 5: flags.
    use_repeated_guest: bool = True
    use_prior_cancel: bool = True

    def __post_init__(self) -> None:
        if self.seasonality not in SEASONALITY_OPTIONS:
            raise ValueError(f"seasonality must be one of {SEASONALITY_OPTIONS}")
        if self.nights not in NIGHTS_OPTIONS:
            raise ValueError(f"nights must be one of {NIGHTS_OPTIONS}")
        if self.iqr_factor is not None and self.iqr_factor <= 0:
            raise ValueError("iqr_factor must be positive or None")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RegressionConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def feature_groups(cfg: RegressionConfig) -> dict:
    """Column groups (of the engineered frame) that the model uses for a config."""
    numeric = ["Adults", "Children", "LeadTimeLog"]
    if cfg.nights == "log_share":
        numeric += ["TotalNightsLog", "WeekendShare"]
    else:
        numeric += ["StaysInWeekendNights", "StaysInWeekNights"]
    numeric.append("DaysSinceStart")

    categorical = [*RAW_CATEGORICAL, "ArrivalDayOfWeek"]
    if cfg.seasonality == "month":
        categorical.append("ArrivalMonth")
    else:
        numeric += HARMONIC_COLUMNS

    flags = []
    if cfg.use_repeated_guest:
        flags.append("IsRepeatedGuest")
    if cfg.use_prior_cancel:
        flags.append("HadPriorCancel")

    return {
        "numeric": numeric,
        "categorical": categorical,
        "agent": ["Agent"],
        "flags": flags,
    }


def assert_no_leakage(feature_names) -> None:
    """Fail loudly if a forbidden column ended up among the model features."""
    for name in feature_names:
        base = name.split("__", 1)[-1]
        for bad in FORBIDDEN_COLUMNS:
            if base == bad or base.startswith(bad + "_"):
                raise AssertionError(f"Forbidden column reached the model: {name}")


# ----------------------------------------------------------------------------
# Stateless cleaning helpers
# ----------------------------------------------------------------------------
def _clean_text(series: pd.Series) -> pd.Series:
    """Strip padding, turn missing values and 'NULL' text into the constant 'null'."""
    text = series.astype("string").str.strip()
    is_null = text.isna() | (text.str.upper() == "NULL").fillna(False)
    return text.mask(is_null, "null").astype(object)


def _clean_agent(series: pd.Series) -> pd.Series:
    """Agent ids as text. 9.0 and ' 9' both become '9'. Missing becomes 'null'."""
    text = _clean_text(series)
    num = pd.to_numeric(series, errors="coerce")
    is_int = num.notna() & (num == num.round())
    as_int = num.round().astype("Int64").astype(str)
    return text.where(~is_int, as_int).astype(object)


def _month_numbers(series: pd.Series) -> pd.Series:
    """Month names ('July', ' july ') or numbers (7) to integers 1 to 12."""
    names = series.astype("string").str.strip().str.capitalize()
    num = names.map(MONTH_TO_NUM).astype("float64")
    num = num.fillna(pd.to_numeric(series, errors="coerce"))
    if num.isna().any() or ((num < 1) | (num > 12)).any():
        raise ValueError("ArrivalDateMonth contains values that are not a month")
    return num.astype(int)


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Raw booking fields in, derived modelling columns out. Pure and stateless.

    Builds every candidate column. The ColumnTransformer then picks the ones the
    current RegressionConfig asks for, so the variant tests only change config.
    ArrivalDate is an intermediate and is not returned.
    """
    missing = [c for c in RAW_INPUT_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"Missing raw booking fields: {missing}")

    arrival = pd.to_datetime(
        pd.DataFrame(
            {
                "year": pd.to_numeric(X["ArrivalDateYear"]).astype(int),
                "month": _month_numbers(X["ArrivalDateMonth"]),
                "day": pd.to_numeric(X["ArrivalDateDayOfMonth"]).astype(int),
            },
            index=X.index,
        )
    )

    lead = pd.to_numeric(X["LeadTime"], errors="coerce")
    weekend = pd.to_numeric(X["StaysInWeekendNights"], errors="coerce")
    week = pd.to_numeric(X["StaysInWeekNights"], errors="coerce")
    total = weekend + week

    # Weekend share is 0 when the stay has no nights.
    share = weekend / total.where(total != 0)
    share = share.where(total != 0, 0.0)

    angle = 2 * np.pi * arrival.dt.dayofyear / DAYS_PER_YEAR

    prior_cancel = pd.to_numeric(X["PreviousCancellations"], errors="coerce").fillna(0)
    repeated = pd.to_numeric(X["IsRepeatedGuest"], errors="coerce").fillna(0)

    out = {
        "Adults": pd.to_numeric(X["Adults"], errors="coerce"),
        "Children": pd.to_numeric(X["Children"], errors="coerce"),
        "LeadTimeLog": np.log1p(lead.clip(lower=0)),
        "TotalNightsLog": np.log1p(total.clip(lower=0)),
        "WeekendShare": share,
        "StaysInWeekendNights": weekend,
        "StaysInWeekNights": week,
        "DaysSinceStart": (arrival - START_DATE).dt.days.astype("float64"),
        "DoySin1": np.sin(angle),
        "DoyCos1": np.cos(angle),
        "DoySin2": np.sin(2 * angle),
        "DoyCos2": np.cos(2 * angle),
        "ArrivalDayOfWeek": arrival.dt.day_name().astype(object),
        "ArrivalMonth": arrival.dt.month_name().astype(object),
        "HadPriorCancel": (prior_cancel > 0).astype(int),
        "IsRepeatedGuest": (repeated > 0).astype(int),
        "Agent": _clean_agent(X["Agent"]),
    }
    for col in RAW_CATEGORICAL:
        out[col] = _clean_text(X[col])

    return pd.DataFrame(out, index=X.index)
