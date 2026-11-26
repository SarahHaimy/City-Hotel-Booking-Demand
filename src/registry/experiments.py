"""Unified flat experiment log across regression, classification, and clustering."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import path as cfg_path
from src.registry.artifacts import ArtifactStore, get_store


def _flatten_card(card) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    splits = card.metrics or {}
    for split_name, mdict in splits.items():
        if not isinstance(mdict, dict) or not mdict:
            continue
        row: dict[str, Any] = {
            "bundle_id": card.bundle_id,
            "task": card.task,
            "model_id": card.model_id,
            "display_name": card.display_name,
            "family": card.family,
            "variant": card.variant,
            "feature_space": card.feature_space,
            "source": card.source,
            "run_name": card.run_name,
            "track_origin": card.track_origin,
            "is_deployed": card.is_deployed,
            "split": split_name,
            "train_time_s": card.train_time_s,
            "n_train_rows": card.n_train_rows,
            "n_features_out": card.n_features_out,
            "threshold": card.threshold,
            "pca_components": card.pca_info.get("components_kept"),
            "pca_variance_kept": card.pca_info.get("variance_kept"),
            "created_at": card.created_at,
        }
        for k, v in mdict.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
        rows.append(row)
    return rows


def sync_experiments_log(store: ArtifactStore | None = None, out_path: Path | None = None) -> pd.DataFrame:
    """Rebuild outputs/experiments/results.csv from all available artifact bundles."""
    store = store or get_store()
    out_path = Path(out_path) if out_path is not None else cfg_path("experiments_log")
    all_rows: list[dict[str, Any]] = []
    for bundle in store.list_bundles():
        all_rows.extend(_flatten_card(bundle.card))
    df = pd.DataFrame(all_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_experiments_log(
    task: str | None = None,
    split: str | None = None,
    source: str | None = None,
    store: ArtifactStore | None = None,
) -> pd.DataFrame:
    """Return the flat experiment table, syncing from bundles so it is never stale."""
    df = sync_experiments_log(store=store)
    if df.empty:
        return df
    if task:
        df = df[df["task"] == task]
    if split:
        df = df[df["split"] == split]
    if source and source != "all":
        df = df[df["source"] == source]
    return df.reset_index(drop=True)
