"""
Row-level / structural cleaning of H2.csv (City Hotel).

Rules here use NO statistics learned from the data (no means, medians, encoders),
so running them before the train/val/test split cannot leak information.
Anything data-dependent (imputation values, encoders, scalers) belongs in
src/preprocessing and must be fit on the training split only.

Column renames (spec section 4 requires a documented 1-to-1 mapping):
    Babies -> HasBabies   (0/1 flag)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMN_MAPPING = {"Babies": "HasBabies"}   # source label -> name used in code
MAX_ADR = 1000.0                           # rows above this are treated as data errors
                                           # (raw max is 5400; the next largest is ~510)
ID_COLUMNS = ["Agent", "Company"]          # identifiers, not quantities -> kept as strings


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """H2.csv pads many text fields with spaces (e.g. 'HB       ', 'A      ')."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip().replace({"": np.nan, "NULL": np.nan})
    return df


def _ids_to_str(series: pd.Series) -> pd.Series:
    """Agent/Company load as floats (6.0) because of NaNs; turn them into clean string IDs."""
    num = pd.to_numeric(series, errors="coerce").astype("Int64")
    return num.astype("string")


def clean_data(
    df: pd.DataFrame,
    drop_zero_adr: bool = True,
    drop_company: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the cleaning rules in a fixed order.

    Parameters
    ----------
    drop_zero_adr : remove rows with ADR == 0 (see review notes: this is arguably
                    a regression-only filter; set False to keep them for
                    classification / clustering).
    drop_company  : drop the Company column (~95% missing).

    Returns
    -------
    cleaned_df, report   where report lists rows removed at each step.
    The original row index is preserved so every row can be traced to H2.csv.
    """
    steps: list[tuple[str, int]] = []
    n_start = len(df)

    def log(name: str, before: int, after: int) -> None:
        steps.append((name, before - after))

    out = _strip_strings(df)

    # 1. exact duplicates (after whitespace normalisation, on all source columns)
    n = len(out)
    out = out.drop_duplicates()
    log("duplicate rows", n, len(out))

    # 2. missing values
    out["Children"] = out["Children"].fillna(0).astype(int)          # only 4 NaNs
    # Country / Agent / Company: remain NaN here; imputed (e.g. 'Unknown') in the pipeline
    for col in ID_COLUMNS:
        out[col] = _ids_to_str(out[col])
    if drop_company:
        out = out.drop(columns=["Company"])

    # 3. bookings with no guests
    n = len(out)
    guests = out["Adults"] + out["Children"] + out["Babies"]
    out = out[guests > 0]
    log("zero guests", n, len(out))

    # 4. zero-night stays that were NOT cancelled (a cancelled 0-night booking is plausible)
    n = len(out)
    nights = out["StaysInWeekendNights"] + out["StaysInWeekNights"]
    out = out[~((nights == 0) & (out["IsCanceled"] == 0))]
    log("zero nights & not canceled", n, len(out))

    # 5. ADR anomalies
    n = len(out)
    out = out[(out["ADR"] >= 0) & (out["ADR"] <= MAX_ADR)]
    log(f"ADR < 0 or > {MAX_ADR:g}", n, len(out))
    if drop_zero_adr:
        n = len(out)
        out = out[out["ADR"] > 0]
        log("ADR == 0", n, len(out))

    # 6. Babies -> binary flag
    out["HasBabies"] = (out["Babies"] > 0).astype(int)
    out = out.drop(columns=["Babies"])

    # 7. columns dropped/merged above (Company, Babies -> HasBabies) can make rows that
    #    differed only in those columns identical, so de-duplicate again on what remains.
    #    Otherwise identical feature rows could land in both train and test.
    n = len(out)
    out = out.drop_duplicates()
    log("duplicates after column changes", n, len(out))

    report = pd.DataFrame(steps, columns=["step", "rows_removed"])
    report["rows_remaining"] = n_start - report["rows_removed"].cumsum()
    return out, report