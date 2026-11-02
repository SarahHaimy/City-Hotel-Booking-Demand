from __future__ import annotations

import numpy as np
import pandas as pd

COLUMN_MAPPING = {"Babies": "HasBabies"}   
MAX_ADR = 1000.0                           
                                           
ID_COLUMNS = ["Agent", "Company"]          


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip().replace({"": np.nan, "NULL": np.nan})
    return df


def _ids_to_str(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce").astype("Int64")
    return num.astype("string")


def clean_data(
    df: pd.DataFrame,
    drop_zero_adr: bool = False,
    drop_company: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    report = pd.DataFrame(steps, columns=["step", "rows_removed"])
    report["rows_remaining"] = n_start - report["rows_removed"].cumsum()
    return out, report