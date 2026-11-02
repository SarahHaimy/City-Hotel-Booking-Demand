"""Exploratory data analysis for H2.csv.

Usage (works from any directory):
    python src/eda/run_eda1.py

Writes tables to outputs/eda/tables and figures to outputs/eda/figures.
This module only describes the data. Nothing here is fitted or used for
modelling, so it cannot leak information into later stages.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Make `src` importable when this file is run directly as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loader import load_h2, TARGET_CLF, TARGET_REG  # noqa: E402

OUT = ROOT / "outputs" / "eda"
FIG = OUT / "figures"
TAB = OUT / "tables"

NUMERIC = [
    "LeadTime", "StaysInWeekendNights", "StaysInWeekNights", "Adults",
    "Children", "Babies", "PreviousCancellations",
    "PreviousBookingsNotCanceled", "BookingChanges", "DaysInWaitingList",
    "ADR", "RequiredCarParkingSpaces", "TotalOfSpecialRequests",
]
CATEGORICAL = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment",
    "DistributionChannel", "ReservedRoomType", "AssignedRoomType",
    "DepositType", "Agent", "Company", "CustomerType", "ReservationStatus",
]

# Columns that are only known after the booking outcome is decided.
POST_OUTCOME = ["ReservationStatus", "ReservationStatusDate"]


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=130)
    plt.close(fig)


def structure(df):
    section("1. STRUCTURE")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Arrival range: {df['ArrivalDateYear'].min()} to {df['ArrivalDateYear'].max()}")
    dtypes = df.dtypes.astype(str).rename("dtype").to_frame()
    dtypes["n_unique"] = df.nunique()
    dtypes.to_csv(TAB / "dtypes_cardinality.csv")
    print(dtypes.to_string())


def quality(df):
    section("2. DATA QUALITY")
    dup = df.duplicated().sum()
    print(f"Fully duplicated rows: {dup:,} ({dup / len(df):.1%})")

    miss = df.isna().sum()
    miss = miss[miss > 0].rename("missing").to_frame()
    miss["pct"] = (miss["missing"] / len(df) * 100).round(2)
    miss.to_csv(TAB / "missing_values.csv")
    print("\nMissing values:")
    print(miss.to_string())

    guests = df["Adults"] + df["Children"].fillna(0) + df["Babies"]
    nights = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    checks = {
        "Zero guests (Adults+Children+Babies == 0)": (guests == 0).sum(),
        "Zero adults": (df["Adults"] == 0).sum(),
        "Zero nights stayed": (nights == 0).sum(),
        "ADR == 0": (df["ADR"] == 0).sum(),
        "ADR < 0": (df["ADR"] < 0).sum(),
        "Reserved room != assigned room": (df["ReservedRoomType"] != df["AssignedRoomType"]).sum(),
        "IsCanceled=1 but status is Check-Out (inconsistent)": ((df["IsCanceled"] == 1) & (df["ReservationStatus"] == "Check-Out")).sum(),
    }
    out = pd.Series(checks, name="rows").to_frame()
    out["pct"] = (out["rows"] / len(df) * 100).round(2)
    out.to_csv(TAB / "unusual_values.csv")
    print("\nUnusual values:")
    print(out.to_string())


def numeric_summary(df):
    section("3. NUMERIC SUMMARY AND SKEWNESS")
    desc = df[NUMERIC].describe().T
    desc["skew"] = df[NUMERIC].skew()
    q1, q3 = df[NUMERIC].quantile(0.25), df[NUMERIC].quantile(0.75)
    iqr = q3 - q1
    outliers = ((df[NUMERIC] < q1 - 1.5 * iqr) | (df[NUMERIC] > q3 + 1.5 * iqr)).sum()
    desc["iqr_outliers"] = outliers
    desc["iqr_outlier_pct"] = (outliers / len(df) * 100).round(2)
    desc.round(3).to_csv(TAB / "numeric_summary.csv")
    print(desc.round(2).to_string())
    print("\nHighly skewed (|skew| > 1):", ", ".join(desc.index[desc["skew"].abs() > 1]))


def categorical_summary(df):
    section("4. CATEGORICAL CARDINALITY")
    rows = []
    for c in CATEGORICAL:
        vc = df[c].value_counts(normalize=True, dropna=False)
        rows.append({
            "column": c,
            "n_unique": df[c].nunique(),
            "top_value": vc.index[0],
            "top_share": round(vc.iloc[0], 3),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "categorical_cardinality.csv", index=False)
    print(out.to_string(index=False))


def target_analysis(df):
    section("5. TARGETS")
    rate = df[TARGET_CLF].mean()
    print(f"IsCanceled rate: {rate:.1%}  (class balance is {1 - rate:.1%} / {rate:.1%})")
    adr = df[df["ADR"] > 0]["ADR"]
    print(f"ADR (excluding zeros): mean={adr.mean():.1f}, median={adr.median():.1f}, "
          f"p99={adr.quantile(.99):.1f}, max={adr.max():.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.countplot(x=TARGET_CLF, data=df, ax=ax[0], color="#4C72B0")
    ax[0].set_title("IsCanceled class balance")
    sns.histplot(df.loc[df["ADR"].between(0, 500), "ADR"], bins=60, ax=ax[1], color="#4C72B0")
    ax[1].set_title("ADR distribution (0 to 500)")
    save(fig, "targets")


def plots(df):
    section("6. FIGURES")
    # Distributions of skewed numerics, raw and log1p.
    skewed = ["LeadTime", "ADR", "StaysInWeekNights", "BookingChanges", "DaysInWaitingList"]
    fig, axes = plt.subplots(2, len(skewed), figsize=(4 * len(skewed), 6))
    for i, c in enumerate(skewed):
        x = df[c].clip(lower=0)
        sns.histplot(x, bins=50, ax=axes[0, i], color="#4C72B0")
        axes[0, i].set_title(c)
        sns.histplot(np.log1p(x), bins=50, ax=axes[1, i], color="#55A868")
        axes[1, i].set_title(f"log1p({c})")
    save(fig, "skewed_distributions")

    # Boxplots for outliers.
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, c in zip(axes, ["LeadTime", "ADR", "StaysInWeekNights", "BookingChanges"]):
        sns.boxplot(y=df[c], ax=ax, color="#8172B2")
        ax.set_title(c)
    save(fig, "outlier_boxplots")

    # Correlation heatmap (numeric only).
    corr = df[NUMERIC + [TARGET_CLF]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f",
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation (numeric features and IsCanceled)")
    save(fig, "correlation_heatmap")
    corr[[TARGET_CLF]].sort_values(TARGET_CLF).to_csv(TAB / "corr_with_iscanceled.csv")

    # Cancellation rate by key categoricals.
    cols = ["MarketSegment", "DepositType", "CustomerType", "ArrivalDateMonth"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, c in zip(axes.ravel(), cols):
        rate = df.groupby(c)[TARGET_CLF].mean().sort_values()
        rate.plot.barh(ax=ax, color="#C44E52")
        ax.set_title(f"Cancellation rate by {c}")
        ax.set_xlabel("rate")
    save(fig, "cancel_rate_by_category")

    # ADR seasonality and room type.
    order = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]
    d = df[df["ADR"].between(1, 500)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    sns.boxplot(x="ArrivalDateMonth", y="ADR", data=d, order=order, ax=ax[0], color="#4C72B0")
    ax[0].tick_params(axis="x", rotation=60)
    ax[0].set_title("ADR by arrival month")
    sns.boxplot(x="ReservedRoomType", y="ADR", data=d, ax=ax[1], color="#55A868")
    ax[1].set_title("ADR by reserved room type")
    save(fig, "adr_patterns")

    # Missingness.
    miss = (df.isna().mean() * 100)
    miss = miss[miss > 0].sort_values()
    if len(miss):
        fig, ax = plt.subplots(figsize=(6, 3))
        miss.plot.barh(ax=ax, color="#DD8452")
        ax.set_xlabel("% missing")
        ax.set_title("Missing values")
        save(fig, "missing_values")
    print(f"Saved figures to {FIG}")


def leakage_notes(df):
    section("7. EARLY LEAKAGE FLAGS (for the leakage audit)")
    ct = pd.crosstab(df["ReservationStatus"], df[TARGET_CLF])
    print("ReservationStatus vs IsCanceled:")
    print(ct.to_string())
    print(
        "\nReservationStatus and ReservationStatusDate are recorded after the outcome,\n"
        "so they must be excluded from the classification and clustering features.\n"
        "Also review: AssignedRoomType, BookingChanges, DaysInWaitingList, and\n"
        "ADR for the classifier (values can change after booking); and for the\n"
        "ADR regressor, anything only known after the stay (e.g. ReservationStatus)."
    )
    print("\nCancellation rate by DepositType (watch for outcome proxies):")
    print(df.groupby("DepositType")[TARGET_CLF].agg(["mean", "count"]).round(3).to_string())
    ct.to_csv(TAB / "status_vs_iscanceled.csv")


def run():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = load_h2()
    structure(df)
    quality(df)
    numeric_summary(df)
    categorical_summary(df)
    target_analysis(df)
    plots(df)
    leakage_notes(df)
    print(f"\nDone. Tables in {TAB}")


if __name__ == "__main__":
    run()
