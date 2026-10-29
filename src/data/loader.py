"""Data loading utilities for the City Hotel (H2) dataset.

Loading only fixes formatting problems (padded strings, the literal 'NULL'
marker). It does NOT fit anything on the data, so it is safe to use before
the train/test split without causing leakage.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "H2.csv"

# Exact source labels from Section 4 of the project spec (no renaming).
TARGET_REG = "ADR"
TARGET_CLF = "IsCanceled"


def load_h2(path=DEFAULT_PATH) -> pd.DataFrame:
    """Load H2.csv and normalise string formatting without altering values."""
    df = pd.read_csv(path)

    # Strip padding whitespace (e.g. 'BB       ', '   9') in text columns.
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()

    # The dataset uses the literal string 'NULL' for missing Agent/Company.
    df = df.replace({"NULL": pd.NA})

    df["ReservationStatusDate"] = pd.to_datetime(df["ReservationStatusDate"])
    return df
