"""Tests for loading, cleaning and splitting.  Run from the project root:

    python -m pytest tests/test_data_pipeline.py -v

Cleaning/splitting tests use tiny synthetic data (fast, no CSV needed).
The loader tests use the real data/H2.csv and are skipped if it is missing.
"""
import numpy as np
import pandas as pd
import pytest

from src.data.data_cleaning import MAX_ADR, clean_data
from src.data.loader import DEFAULT_PATH, load_h2
from src.data.data_split import (load_splits, save_splits, split_data,
                                 train_val_combined)


def make_df(**overrides):
    """One valid booking; any field can be overridden."""
    row = dict(IsCanceled=0, LeadTime=10, Adults=2, Children=0.0, Babies=0,
               StaysInWeekendNights=1, StaysInWeekNights=2, ADR=100.0,
               Agent="9", Company=pd.NA, MarketSegment="Online TA")
    row.update(overrides)
    return pd.DataFrame([row])


def stack(*frames):
    return pd.concat(frames, ignore_index=True)


def test_duplicates_removed():
    df = stack(make_df(), make_df(), make_df(LeadTime=11))
    out, rep = clean_data(df)
    assert len(out) == 2
    assert rep.set_index("step").loc["duplicate rows", "rows_removed"] == 1


def test_children_nan_becomes_zero_int():
    out, _ = clean_data(make_df(Children=np.nan))
    assert out["Children"].iloc[0] == 0
    assert pd.api.types.is_integer_dtype(out["Children"])


def test_zero_guests_removed():
    df = stack(make_df(Adults=0, Children=0.0, Babies=0), make_df(LeadTime=11))
    out, _ = clean_data(df)
    assert len(out) == 1 and out["LeadTime"].iloc[0] == 11


def test_adults_zero_with_children_is_kept():
    out, _ = clean_data(make_df(Adults=0, Children=2.0))
    assert len(out) == 1


def test_zero_nights_dropped_only_if_not_canceled():
    kept = make_df(StaysInWeekendNights=0, StaysInWeekNights=0, IsCanceled=1)
    dropped = make_df(StaysInWeekendNights=0, StaysInWeekNights=0,
                      IsCanceled=0, LeadTime=11)
    out, _ = clean_data(stack(kept, dropped))
    assert len(out) == 1 and out["IsCanceled"].iloc[0] == 1


def test_adr_outlier_removed():
    out, _ = clean_data(stack(make_df(ADR=5400.0), make_df(LeadTime=11)))
    assert len(out) == 1 and (out["ADR"] <= MAX_ADR).all()


def test_zero_adr_flag():
    df = stack(make_df(ADR=0.0), make_df(LeadTime=11))
    assert len(clean_data(df)[0]) == 1                          # default: dropped
    assert len(clean_data(df, drop_zero_adr=False)[0]) == 2     # kept


def test_babies_to_binary_flag():
    df = stack(make_df(Babies=0), make_df(Babies=2, LeadTime=11),
               make_df(Babies=10, LeadTime=12))
    out, _ = clean_data(df)
    assert "Babies" not in out.columns
    assert out["HasBabies"].tolist() == [0, 1, 1]


def test_ids_become_clean_strings_and_missing_stays_missing():
    df = stack(make_df(Agent="9"), make_df(Agent=pd.NA, LeadTime=11))
    out, _ = clean_data(df)
    assert out["Agent"].iloc[0] == "9"
    assert pd.isna(out["Agent"].iloc[1])


def test_padded_strings_and_null_marker_normalised():
    out, _ = clean_data(make_df(MarketSegment="  Online TA   ", Agent="  NULL "))
    assert out["MarketSegment"].iloc[0] == "Online TA"
    assert pd.isna(out["Agent"].iloc[0])


def test_original_index_preserved_and_report_consistent():
    df = stack(make_df(), make_df(Adults=0, LeadTime=11), make_df(LeadTime=12))
    out, rep = clean_data(df)
    assert out.index.tolist() == [0, 2]
    assert rep["rows_remaining"].iloc[-1] == len(out)


def test_duplicates_created_by_dropping_company_are_removed():
    df = stack(make_df(Company="12"), make_df(Company="34"))   # differ only in Company
    assert len(clean_data(df, drop_company=False)[0]) == 2
    out, rep = clean_data(df, drop_company=True)
    assert "Company" not in out.columns and len(out) == 1


def test_duplicates_created_by_babies_flag_are_removed():
    df = stack(make_df(Babies=1), make_df(Babies=2))           # both -> HasBabies = 1
    assert len(clean_data(df)[0]) == 1


def test_input_not_mutated():
    df = make_df(Children=np.nan)
    before = df.copy()
    clean_data(df)
    pd.testing.assert_frame_equal(df, before)


@pytest.fixture
def toy():
    rng = np.random.default_rng(0)
    n = 2000
    return pd.DataFrame({"IsCanceled": (rng.random(n) < 0.3).astype(int),
                         "x": rng.normal(size=n)})


def test_split_sizes_disjoint_complete(toy):
    tr, va, te = split_data(toy)
    assert len(tr) + len(va) + len(te) == len(toy)
    assert abs(len(va) / len(toy) - 0.15) < 0.01
    assert abs(len(te) / len(toy) - 0.15) < 0.01
    assert not set(tr.index) & set(va.index)
    assert not set(tr.index) & set(te.index)
    assert not set(va.index) & set(te.index)


def test_split_is_stratified(toy):
    rates = [p["IsCanceled"].mean() for p in split_data(toy)]
    assert max(rates) - min(rates) < 0.005


def test_split_reproducible_and_seed_sensitive(toy):
    a = split_data(toy, random_state=42)[2].index
    b = split_data(toy, random_state=42)[2].index
    c = split_data(toy, random_state=7)[2].index
    assert a.equals(b) and not a.equals(c)


def test_train_val_combined_excludes_test(toy):
    tr, va, te = split_data(toy)
    tv = train_val_combined(tr, va)
    assert len(tv) == len(tr) + len(va) and not set(tv.index) & set(te.index)


def test_save_load_roundtrip(tmp_path):
    df = stack(*[make_df(LeadTime=i, IsCanceled=i % 2) for i in range(60)])
    df["ReservationStatusDate"] = pd.Timestamp("2016-01-01")
    df["Agent"] = df["Agent"].astype("string")
    df["Company"] = df["Company"].astype("string")
    parts = split_data(df)
    save_splits(*parts, out_dir=tmp_path)
    for orig, back in zip(parts, load_splits(tmp_path)):
        assert orig.index.equals(back.index)
        assert back["Agent"].dtype == "string"
        assert pd.api.types.is_datetime64_any_dtype(back["ReservationStatusDate"])


needs_csv = pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="data/H2.csv not found")


@needs_csv
def test_loader_shape_and_formatting():
    df = load_h2()
    assert df.shape == (79330, 31)                       # spec section 2
    assert not (df["Meal"].str.len() != df["Meal"].str.strip().str.len()).any()
    assert not (df.astype(str) == "NULL").any().any()
    assert pd.api.types.is_datetime64_any_dtype(df["ReservationStatusDate"])


@needs_csv
def test_full_pipeline_on_real_data():
    clean, _ = clean_data(load_h2())
    assert not clean.duplicated().any()
    assert (clean["Adults"] + clean["Children"] + clean["HasBabies"] > 0).all()
    assert (clean["ADR"] > 0).all() and (clean["ADR"] <= MAX_ADR).all()
    assert clean["Children"].notna().all()
    tr, va, te = split_data(clean)
    assert len(tr) + len(va) + len(te) == len(clean)
    rates = [p["IsCanceled"].mean() for p in (tr, va, te)]
    assert max(rates) - min(rates) < 0.005