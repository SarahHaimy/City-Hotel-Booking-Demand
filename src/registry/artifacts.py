"""Artifact bundle persistence, health verification, deployment pointers, and ZIP import/export."""
from __future__ import annotations

import io
import json
import shutil
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from src.config import path as cfg_path
from src.registry.model_card import ModelCard


@dataclass
class BundleHealth:
    bundle_id: str
    status: str  # "ok" | "warning" | "error"
    message: str
    size_bytes: int = 0
    version_mismatch: bool = False


@dataclass
class ArtifactBundle:
    card: ModelCard
    bundle_dir: Path
    _pipeline_cache: Any = None

    @property
    def pipeline_path(self) -> Path:
        return self.bundle_dir / "pipeline.joblib"

    @property
    def card_path(self) -> Path:
        return self.bundle_dir / "model_card.json"

    def load_pipeline(self) -> Any:
        if self._pipeline_cache is None:
            if not self.pipeline_path.exists():
                raise FileNotFoundError(f"Pipeline file missing at {self.pipeline_path}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._pipeline_cache = joblib.load(self.pipeline_path)
        return self._pipeline_cache

    def read_csv(self, filename: str) -> pd.DataFrame | None:
        for candidate in (self.bundle_dir / filename, self.bundle_dir / "extras" / filename):
            if candidate.exists():
                return pd.read_csv(candidate)
        return None

    def read_json(self, filename: str) -> Any | None:
        for candidate in (self.bundle_dir / filename, self.bundle_dir / "extras" / filename):
            if candidate.exists():
                return json.loads(candidate.read_text(encoding="utf-8"))
        return None

    def size_bytes(self) -> int:
        if not self.bundle_dir.exists():
            return 0
        return sum(p.stat().st_size for p in self.bundle_dir.rglob("*") if p.is_file())


class ArtifactStore:
    """Manages official (`artifacts/<task>/<model_id>/<variant>`) and user (`artifacts/user/<run_id>`) bundles."""

    def __init__(self, root_dir: Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir is not None else cfg_path("artifacts")
        self.user_dir = self.root_dir / "user"
        self.deployed_file = self.root_dir / "deployed.json"

    def bundle_dir_for(self, card: ModelCard) -> Path:
        if card.source == "user" or card.bundle_id.startswith("user/"):
            slug = card.bundle_id.split("/", 1)[-1]
            return self.user_dir / slug
        return self.root_dir / card.task / card.model_id / card.variant

    def _read_deployed_map(self) -> dict[str, str]:
        if self.deployed_file.exists():
            try:
                return json.loads(self.deployed_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _write_deployed_map(self, mapping: dict[str, str]) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.deployed_file.write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    def save_bundle(
        self,
        card: ModelCard,
        pipeline: Any,
        predictions_val: pd.DataFrame | None = None,
        predictions_test: pd.DataFrame | None = None,
        cv_scores: dict[str, Any] | None = None,
        tuning_results: pd.DataFrame | None = None,
        extra_tables: dict[str, pd.DataFrame] | None = None,
        extra_json: dict[str, Any] | None = None,
        set_as_deployed: bool = False,
        replace_existing: bool = True,
    ) -> ArtifactBundle:
        bdir = self.bundle_dir_for(card)
        if replace_existing and bdir.exists():
            shutil.rmtree(bdir)
        bdir.mkdir(parents=True, exist_ok=True)
        extras_dir = bdir / "extras"
        extras_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, bdir / "pipeline.joblib")

        if predictions_val is not None:
            predictions_val.to_csv(bdir / "predictions_val.csv", index=False)
        if predictions_test is not None:
            predictions_test.to_csv(bdir / "predictions_test.csv", index=False)
        if cv_scores is not None:
            (bdir / "cv_scores.json").write_text(json.dumps(cv_scores, indent=2), encoding="utf-8")
        if tuning_results is not None:
            tuning_results.to_csv(bdir / "tuning_results.csv", index=False)
        for name, table in (extra_tables or {}).items():
            fname = name if name.endswith(".csv") else f"{name}.csv"
            table.to_csv(extras_dir / fname, index=False)
        for name, payload in (extra_json or {}).items():
            fname = name if name.endswith(".json") else f"{name}.json"
            (extras_dir / fname).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

        deployed_map = self._read_deployed_map()
        if set_as_deployed or card.is_deployed:
            deployed_map[card.task] = card.bundle_id
            self._write_deployed_map(deployed_map)
            card.is_deployed = True
        elif card.task not in deployed_map and card.source == "project":
            deployed_map[card.task] = card.bundle_id
            self._write_deployed_map(deployed_map)
            card.is_deployed = True
        else:
            card.is_deployed = deployed_map.get(card.task) == card.bundle_id

        card.save(bdir / "model_card.json")
        return ArtifactBundle(card=card, bundle_dir=bdir, _pipeline_cache=pipeline)

    def list_bundles(
        self,
        task: str | None = None,
        source: str | None = None,
        model_id: str | None = None,
        variant: str | None = None,
        feature_space: str | None = None,
    ) -> list[ArtifactBundle]:
        if not self.root_dir.exists():
            return []
        deployed_map = self._read_deployed_map()
        bundles: list[ArtifactBundle] = []
        for card_path in sorted(self.root_dir.rglob("model_card.json")):
            try:
                card = ModelCard.load(card_path)
            except Exception:
                continue
            card.is_deployed = deployed_map.get(card.task) == card.bundle_id
            if task and card.task != task:
                continue
            if source and card.source != source:
                continue
            if model_id and card.model_id != model_id:
                continue
            if variant and card.variant != variant:
                continue
            if feature_space and card.feature_space != feature_space:
                continue
            bundles.append(ArtifactBundle(card=card, bundle_dir=card_path.parent))
        return bundles

    def get_bundle(self, bundle_id: str) -> ArtifactBundle:
        for b in self.list_bundles():
            if b.card.bundle_id == bundle_id:
                return b
        raise KeyError(f"Artifact bundle '{bundle_id}' not found in {self.root_dir}")

    def get_deployed(self, task: str) -> ArtifactBundle | None:
        deployed_map = self._read_deployed_map()
        target_id = deployed_map.get(task)
        bundles = self.list_bundles(task=task)
        if not bundles:
            return None
        if target_id:
            for b in bundles:
                if b.card.bundle_id == target_id:
                    return b
        # Fallback: prefer project tuned -> project baseline -> first available
        for pref_source, pref_var in (("project", "tuned"), ("project", "baseline")):
            for b in bundles:
                if b.card.source == pref_source and b.card.variant == pref_var:
                    return b
        return bundles[0]

    def set_deployed(self, task: str, bundle_id: str) -> None:
        bundle = self.get_bundle(bundle_id)
        if bundle.card.task != task:
            raise ValueError(f"Bundle '{bundle_id}' belongs to task '{bundle.card.task}', not '{task}'.")
        deployed_map = self._read_deployed_map()
        deployed_map[task] = bundle_id
        self._write_deployed_map(deployed_map)
        for b in self.list_bundles(task=task):
            b.card.is_deployed = b.card.bundle_id == bundle_id
            b.card.save(b.card_path)

    def verify_bundle(self, bundle: ArtifactBundle | str) -> BundleHealth:
        try:
            b = self.get_bundle(bundle) if isinstance(bundle, str) else bundle
        except KeyError as exc:
            return BundleHealth(str(bundle), "error", str(exc))

        if not b.pipeline_path.exists():
            return BundleHealth(b.card.bundle_id, "error", "pipeline.joblib is missing", b.size_bytes())

        version_mismatch = b.card.sklearn_version != sklearn.__version__
        try:
            _ = b.load_pipeline()
        except Exception as exc:
            return BundleHealth(
                b.card.bundle_id, "error", f"Failed to load pipeline: {exc}",
                b.size_bytes(), version_mismatch=version_mismatch,
            )

        if version_mismatch:
            return BundleHealth(
                b.card.bundle_id, "warning",
                f"Saved with scikit-learn {b.card.sklearn_version} (installed: {sklearn.__version__})",
                b.size_bytes(), version_mismatch=True,
            )
        return BundleHealth(b.card.bundle_id, "ok", "Healthy — loads cleanly", b.size_bytes())

    def delete_bundle(self, bundle_id: str, allow_project: bool = False) -> None:
        b = self.get_bundle(bundle_id)
        if b.card.source == "project" and not allow_project:
            raise PermissionError("Cannot delete official project bundles without allow_project=True.")
        shutil.rmtree(b.bundle_dir, ignore_errors=True)
        deployed_map = self._read_deployed_map()
        if deployed_map.get(b.card.task) == bundle_id:
            deployed_map.pop(b.card.task, None)
            self._write_deployed_map(deployed_map)

    def export_bundle_zip(self, bundle_id: str) -> bytes:
        b = self.get_bundle(bundle_id)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in sorted(b.bundle_dir.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, arcname=str(file_path.relative_to(b.bundle_dir)))
        return buf.getvalue()

    def import_bundle_zip(self, zip_bytes: bytes, force_user_namespace: bool = True) -> ArtifactBundle:
        buf = io.BytesIO(zip_bytes)
        with zipfile.ZipFile(buf, "r") as zf:
            names = set(zf.namelist())
            if "model_card.json" not in names or "pipeline.joblib" not in names:
                raise ValueError("Invalid bundle archive: must contain model_card.json and pipeline.joblib")
            card_data = json.loads(zf.read("model_card.json").decode("utf-8"))
            card = ModelCard.from_dict(card_data)
            if force_user_namespace:
                slug = card.bundle_id.replace("/", "_")
                if not slug.startswith("imported_"):
                    slug = f"imported_{slug}"
                card.source = "user"
                card.bundle_id = f"user/{slug}"
                card.is_deployed = False
            target_dir = self.bundle_dir_for(card)
            target_dir.mkdir(parents=True, exist_ok=True)
            for member in zf.namelist():
                member_path = Path(member)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise ValueError(f"Unsafe path in archive: {member}")
                zf.extract(member, path=target_dir)
            card.save(target_dir / "model_card.json")
        bundle = ArtifactBundle(card=card, bundle_dir=target_dir)
        _ = bundle.load_pipeline()
        return bundle


def get_store() -> ArtifactStore:
    return ArtifactStore()
