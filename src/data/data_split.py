"""
Train / validation / test split (70 / 15 / 15)
Stratified on IsCanceled so all three sets keep the same cancellation rate.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
STRATIFY_COL = "IsCanceled"
ID_DTYPES = {"Agent": "string", "Company": "string"}
DATE_COLS = ["ReservationStatusDate"]


def split_data(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
    stratify_col: str = STRATIFY_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1")

    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[stratify_col]
    )
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, random_state=random_state,
        stratify=train_val[stratify_col],
    )

    # sanity checks: disjoint, complete
    assert not (set(train.index) & set(val.index))
    assert not (set(train.index) & set(test.index))
    assert not (set(val.index) & set(test.index))
    assert len(train) + len(val) + len(test) == len(df)
    return train, val, test


def train_val_combined(train: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([train, val])


def save_splits(train, val, test, out_dir: str | Path = "data/processed") -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(out_dir / f"{name}.csv", index_label="row_id")


def load_splits(in_dir: str | Path = "data/processed"):
    in_dir = Path(in_dir)
    return tuple(
        pd.read_csv(in_dir / f"{n}.csv", index_col="row_id", dtype=ID_DTYPES,
                    parse_dates=DATE_COLS)
        for n in ("train", "val", "test")
    )


def summarize_splits(train, val, test, target: str = STRATIFY_COL) -> pd.DataFrame:
    rows = [(n, len(p), p[target].mean()) for n, p in
            (("train", train), ("val", val), ("test", test))]
    return pd.DataFrame(rows, columns=["split", "rows", f"{target}_rate"])


if __name__ == "__main__":
    from src.data.data_cleaning import clean_data
    from src.data.loader import load_h2

    raw = load_h2()
    clean, report = clean_data(raw)
    print(report.to_string(index=False))

    tr, va, te = split_data(clean)
    print(summarize_splits(tr, va, te).to_string(index=False))
    save_splits(tr, va, te)