"""Upload ingestion, schema validation, and task split management."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
import pandas as pd

from src.config import cleaning_flags, path as cfg_path, split_dir
from src.data.data_cleaning import MAX_ADR, clean_data
from src.data.data_split import load_splits, save_splits, split_data
from src.data.loader import load_h2
from src.data.schema import (
    COLUMN_ALIASES,
    H2_COLUMNS,
    TASK_FORBIDDEN_COLUMNS,
)

IssueLevel = Literal["error", "warning", "info"]


@dataclass
class ValidationIssue:
    level: IssueLevel
    code: str
    message: str
    column: str | None = None


@dataclass
class ValidationReport:
    ok: bool
    n_rows: int
    n_cols: int
    mapped_columns: dict[str, str] = field(default_factory=dict)
    unmatched_columns: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    present_targets: list[str] = field(default_factory=list)
    dropped_leakage: list[str] = field(default_factory=list)
    missing_counts: dict[str, int] = field(default_factory=dict)
    unseen_categories: dict[str, list[str]] = field(default_factory=dict)
    out_of_range_counts: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    test_overlap_rows: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def infos(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.level == "info"]


def load_uploaded_table(source: str | Path | BinaryIO | bytes, filename: str | None = None) -> pd.DataFrame:
    """Read a CSV or Excel upload and sanitize whitespace/'NULL' strings without fitting state."""
    name = filename or (str(source) if isinstance(source, (str, Path)) else getattr(source, "name", "upload.csv"))
    lower = name.lower()

    if isinstance(source, bytes):
        buf: Any = io.BytesIO(source)
    elif hasattr(source, "read"):
        raw_bytes = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        buf = io.BytesIO(raw_bytes) if isinstance(raw_bytes, bytes) else io.StringIO(raw_bytes)
    else:
        buf = Path(source)

    if lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(buf)
    else:
        try:
            df = pd.read_csv(buf)
        except UnicodeDecodeError:
            if isinstance(buf, io.BytesIO):
                buf.seek(0)
            df = pd.read_csv(buf, encoding="latin-1")

    # Drop accidental unnamed index columns if present alongside real columns
    drop_idx = [c for c in df.columns if str(c).startswith("Unnamed:") or str(c) == "row_id"]
    if drop_idx and len(drop_idx) < len(df.columns):
        df = df.drop(columns=drop_idx)

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            s = df[col].astype("string").str.strip()
            df[col] = s.replace({"": pd.NA, "NULL": pd.NA, "null": pd.NA, "nan": pd.NA, "<NA>": pd.NA})

    return df


def normalize_columns(
    df: pd.DataFrame,
    custom_mapping: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Map snake_case / case variants (and optional user overrides) to exact H2 column labels."""
    out = df.copy()
    custom_mapping = custom_mapping or {}
    rename_map: dict[str, str] = {}
    mapped_report: dict[str, str] = {}
    unmatched: list[str] = []

    used_targets = set(out.columns)
    for col in out.columns:
        col_str = str(col).strip()
        if col_str in custom_mapping and custom_mapping[col_str]:
            target = custom_mapping[col_str]
            rename_map[col] = target
            if col_str != target:
                mapped_report[col_str] = target
            used_targets.add(target)
            continue

        if col_str in H2_COLUMNS or col_str == "HasBabies":
            if col != col_str:
                rename_map[col] = col_str
            continue

        key = col_str.lower().strip()
        canonical = COLUMN_ALIASES.get(key) or COLUMN_ALIASES.get(key.replace(" ", "_"))
        if canonical and canonical not in used_targets:
            rename_map[col] = canonical
            mapped_report[col_str] = canonical
            used_targets.add(canonical)
        else:
            unmatched.append(col_str)

    if rename_map:
        out = out.rename(columns=rename_map)

    # Bidirectional compatibility between raw `Babies` and cleaned `HasBabies`
    if "Babies" in out.columns and "HasBabies" not in out.columns:
        babies_num = pd.to_numeric(out["Babies"], errors="coerce").fillna(0)
        out["HasBabies"] = (babies_num > 0).astype(int)
    elif "HasBabies" in out.columns and "Babies" not in out.columns:
        hb = pd.to_numeric(out["HasBabies"], errors="coerce").fillna(0)
        out["Babies"] = (hb > 0).astype(int)

    return out, mapped_report, unmatched


def _compute_overlap_with_test(df: pd.DataFrame, task: str) -> int:
    """Estimate exact row overlap against the held-out project test split on shared columns."""
    tdir = split_dir("regression" if task == "clustering" else task)
    test_path = tdir / "test.csv"
    if not test_path.exists():
        return 0
    try:
        test_df = pd.read_csv(test_path)
    except Exception:
        return 0
    key_cols = [
        c for c in (
            "LeadTime", "ArrivalDateYear", "ArrivalDateMonth", "ArrivalDateDayOfMonth",
            "StaysInWeekendNights", "StaysInWeekNights", "Adults", "MarketSegment",
            "ReservedRoomType", "ADR",
        )
        if c in df.columns and c in test_df.columns
    ]
    if len(key_cols) < 6:
        return 0
    a = pd.util.hash_pandas_object(df[key_cols].astype(str), index=False)
    b = set(pd.util.hash_pandas_object(test_df[key_cols].astype(str), index=False).tolist())
    return int(a.isin(b).sum())


def validate_dataframe(
    df: pd.DataFrame,
    task: Literal["regression", "classification", "clustering"],
    required_inputs: list[str],
    forbidden_inputs: list[str] | None = None,
    custom_mapping: dict[str, str] | None = None,
    require_target: bool = False,
    min_rows: int = 1,
    check_test_overlap: bool = False,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Validate an uploaded or loaded DataFrame against a task/model schema."""
    norm_df, mapped, unmatched = normalize_columns(df, custom_mapping=custom_mapping)
    issues: list[ValidationIssue] = []

    if mapped:
        pairs = ", ".join(f"{k}->{v}" for k, v in list(mapped.items())[:8])
        issues.append(ValidationIssue("info", "COLUMN_MAPPED", f"Auto-mapped columns to H2 labels: {pairs}"))

    n_rows, n_cols = norm_df.shape
    if n_rows < min_rows:
        issues.append(
            ValidationIssue(
                "error", "TOO_FEW_ROWS",
                f"Dataset has {n_rows:,} rows, but at least {min_rows:,} rows are required.",
            )
        )

    present_targets = [t for t in ("ADR", "IsCanceled") if t in norm_df.columns]
    target_col = "ADR" if task == "regression" else ("IsCanceled" if task == "classification" else None)

    if require_target and target_col and target_col not in norm_df.columns:
        issues.append(
            ValidationIssue(
                "error", "MISSING_TARGET",
                f"Target column '{target_col}' is required for {task} training/evaluation.",
                column=target_col,
            )
        )

    if require_target and task == "classification" and "IsCanceled" in norm_df.columns:
        uniq = pd.to_numeric(norm_df["IsCanceled"], errors="coerce").dropna().unique()
        if len(uniq) < 2:
            issues.append(
                ValidationIssue(
                    "error", "SINGLE_CLASS",
                    "IsCanceled must contain both classes (0 and 1) to train a classifier.",
                    column="IsCanceled",
                )
            )

    missing_req = [c for c in required_inputs if c not in norm_df.columns]
    if missing_req:
        issues.append(
            ValidationIssue(
                "error", "MISSING_REQUIRED",
                f"Missing required input columns: {', '.join(missing_req)}",
            )
        )

    forb_set = set(forbidden_inputs or TASK_FORBIDDEN_COLUMNS.get(task, []))
    if target_col:
        forb_set = forb_set - {target_col}
    dropped_leakage = sorted(c for c in forb_set if c in norm_df.columns and c not in required_inputs)
    if dropped_leakage:
        issues.append(
            ValidationIssue(
                "info", "LEAKAGE_IGNORED",
                f"Outcome/post-booking columns present in file will be excluded from model features: {', '.join(dropped_leakage)}",
            )
        )

    missing_counts: dict[str, int] = {}
    out_of_range: dict[str, int] = {}
    unseen_cats: dict[str, list[str]] = {}

    check_cols = [c for c in set(required_inputs + ([target_col] if target_col else [])) if c and c in norm_df.columns]
    for col in sorted(check_cols):
        na_cnt = int(norm_df[col].isna().sum())
        if na_cnt > 0:
            missing_counts[col] = na_cnt
            spec = H2_COLUMNS.get(col)
            if col == target_col:
                issues.append(
                    ValidationIssue(
                        "error", "TARGET_NAN",
                        f"Target column '{col}' has {na_cnt:,} missing values.", column=col,
                    )
                )
            elif spec and not spec.nullable:
                issues.append(
                    ValidationIssue(
                        "warning", "MISSING_VALUES",
                        f"Column '{col}' has {na_cnt:,} missing values (pipeline imputer/encoder will handle them).",
                        column=col,
                    )
                )

        spec = H2_COLUMNS.get(col)
        if not spec:
            continue
        if spec.kind in ("numeric_int", "numeric_float", "binary"):
            coerced = pd.to_numeric(norm_df[col], errors="coerce")
            bad_parse = int(coerced.isna().sum() - norm_df[col].isna().sum())
            if bad_parse > 0:
                issues.append(
                    ValidationIssue(
                        "error", "NON_NUMERIC",
                        f"Column '{col}' contains {bad_parse:,} non-numeric values.", column=col,
                    )
                )
            norm_df[col] = coerced
            low_bad = int((coerced < spec.min_val).sum()) if spec.min_val is not None else 0
            high_cap = MAX_ADR if col == "ADR" else spec.max_val
            high_bad = int((coerced > high_cap).sum()) if high_cap is not None else 0
            if low_bad + high_bad > 0:
                out_of_range[col] = low_bad + high_bad
                issues.append(
                    ValidationIssue(
                        "warning", "OUT_OF_RANGE",
                        f"Column '{col}' has {low_bad + high_bad:,} values outside [{spec.min_val}, {high_cap}].",
                        column=col,
                    )
                )
        elif spec.kind == "categorical" and spec.categories:
            vals = set(norm_df[col].dropna().astype(str).unique())
            extra = sorted(vals - set(spec.categories))
            if extra and col != "Country":
                unseen_cats[col] = extra[:10]
                issues.append(
                    ValidationIssue(
                        "warning", "UNSEEN_CATEGORIES",
                        f"Column '{col}' has unseen categories ({', '.join(extra[:5])}); OneHotEncoder will safely ignore/group them.",
                        column=col,
                    )
                )

    dup_rows = int(norm_df.duplicated().sum())
    if dup_rows > 0 and n_rows > 1:
        issues.append(
            ValidationIssue(
                "info", "DUPLICATE_ROWS",
                f"Dataset contains {dup_rows:,} exact duplicate rows (enable structural cleaning to deduplicate).",
            )
        )

    overlap = _compute_overlap_with_test(norm_df, task) if check_test_overlap and n_rows >= 20 else 0
    if overlap > 0:
        issues.append(
            ValidationIssue(
                "warning", "TEST_OVERLAP",
                f"{overlap:,} rows match bookings in the held-out project test split; avoid training on test rows.",
            )
        )

    ok = not any(i.level == "error" for i in issues)
    report = ValidationReport(
        ok=ok,
        n_rows=n_rows,
        n_cols=n_cols,
        mapped_columns=mapped,
        unmatched_columns=unmatched,
        missing_required=missing_req,
        present_targets=present_targets,
        dropped_leakage=dropped_leakage,
        missing_counts=missing_counts,
        unseen_categories=unseen_cats,
        out_of_range_counts=out_of_range,
        duplicate_rows=dup_rows,
        test_overlap_rows=overlap,
        issues=issues,
    )
    return norm_df, report


def dataframe_sha256(df: pd.DataFrame) -> str:
    """Fast deterministic hash of a DataFrame's shape, columns, and values."""
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy()
    h = hashlib.sha256()
    h.update(f"{df.shape}:{list(df.columns)}".encode("utf-8"))
    h.update(row_hashes.tobytes())
    return h.hexdigest()[:16]


def ensure_task_splits(task: Literal["regression", "classification"]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load (train, val, test) for a supervised task, generating them reproducibly if missing."""
    sdir = split_dir(task)
    train_p, val_p, test_p = sdir / "train.csv", sdir / "val.csv", sdir / "test.csv"
    if not (train_p.exists() and val_p.exists() and test_p.exists()):
        raw_path = cfg_path("raw_data")
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw dataset {raw_path} not found and splits in {sdir} do not exist.")
        raw = load_h2(raw_path)
        flags = cleaning_flags(task)
        cleaned, _ = clean_data(raw, **flags)
        tr, va, te = split_data(cleaned)
        save_splits(tr, va, te, out_dir=sdir)
    return load_splits(sdir)


def load_clustering_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean H2 for unsupervised clustering, returning (cleaned_df, cleaning_report)."""
    raw = load_h2(cfg_path("raw_data"))
    flags = cleaning_flags("clustering")
    return clean_data(raw, **flags)
