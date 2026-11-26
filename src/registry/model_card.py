"""Model card metadata schema stored alongside every serialized pipeline bundle."""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn

from src.config import ROOT, SEED

SCHEMA_VERSION = 1


def current_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=2, check=False,
        )
        return res.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


@dataclass
class ModelCard:
    bundle_id: str
    model_id: str
    task: str
    variant: str
    display_name: str
    family: str
    source: str = "project"  # "project" | "user"
    run_name: str = ""
    track_origin: str = "milestone_2_suite"
    is_deployed: bool = False
    selection_criterion: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    feature_space: str = "original"  # "original" | "pca" | "lda"
    pca_info: dict[str, Any] = field(default_factory=dict)
    inputs: list[str] = field(default_factory=list)
    forbidden_checked: list[str] = field(default_factory=list)
    preprocessing_steps: list[str] = field(default_factory=list)
    target: str | None = None
    threshold: float | None = None
    supports_proba: bool = False
    supports_predict_new: bool = True
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    tuning_summary: dict[str, Any] | None = None
    train_time_s: float = 0.0
    n_train_rows: int = 0
    n_features_out: int | None = None
    sampling_info: dict[str, Any] | None = None
    data_version: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    seed: int = SEED
    sklearn_version: str = field(default_factory=lambda: sklearn.__version__)
    python_version: str = field(default_factory=platform.python_version)
    git_commit: str = field(default_factory=current_git_commit)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelCard":
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def load(cls, card_path: Path) -> "ModelCard":
        return cls.from_dict(json.loads(card_path.read_text(encoding="utf-8")))

    def save(self, card_path: Path) -> None:
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(self.to_json(), encoding="utf-8")

    def primary_metric(self) -> tuple[str, float | None]:
        """Return (metric_name, value) on the task's primary evaluation split."""
        if self.task == "regression":
            val = self.metrics.get("val", {}).get("RMSE")
            if val is None:
                val = self.metrics.get("test", {}).get("RMSE")
            return "RMSE", (float(val) if val is not None else None)
        if self.task == "classification":
            val = self.metrics.get("val", {}).get("F1")
            if val is None:
                val = self.metrics.get("test", {}).get("F1")
            return "F1", (float(val) if val is not None else None)
        val = self.metrics.get("train", {}).get("silhouette")
        if val is None:
            val = self.metrics.get("val", {}).get("silhouette")
        return "silhouette", (float(val) if val is not None else None)
