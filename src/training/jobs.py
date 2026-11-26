"""Background job execution, live progress reporting, log persistence, and cooperative cancellation."""
from __future__ import annotations

import json
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.config import path as cfg_path


class JobCancelledError(RuntimeError):
    """Raised inside a running training/tuning job when cancellation is requested."""


@dataclass
class JobRecord:
    job_id: str
    title: str
    kind: str  # "train" | "tune" | "cluster" | "rebuild"
    status: str = "queued"  # "queued" | "running" | "succeeded" | "failed" | "cancelled"
    progress: float = 0.0
    message: str = "Queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    result_bundle_id: str | None = None
    result_summary: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class JobRunner:
    """Single-worker background executor writing status & logs to `runs/<job_id>/`."""

    def __init__(self, runs_dir: Path | None = None, max_workers: int = 1) -> None:
        self.runs_dir = Path(runs_dir) if runs_dir is not None else cfg_path("runs")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ml_job")
        self._cancel_flags: set[str] = set()

    def _job_dir(self, job_id: str) -> Path:
        d = self.runs_dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_record(self, rec: JobRecord) -> None:
        jdir = self._job_dir(rec.job_id)
        tmp = jdir / "status.json.tmp"
        target = jdir / "status.json"
        tmp.write_text(json.dumps(rec.to_dict(), indent=2, default=str), encoding="utf-8")
        tmp.replace(target)

    def _append_log(self, job_id: str, line: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        log_path = self._job_dir(job_id) / "job.log"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[{ts}] {line}\n")

    def submit(
        self,
        title: str,
        kind: str,
        fn: Callable[..., Any],
        kwargs: dict[str, Any] | None = None,
    ) -> JobRecord:
        job_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        rec = JobRecord(job_id=job_id, title=title, kind=kind)
        self._write_record(rec)
        self._append_log(job_id, f"Submitted job '{title}' ({kind})")

        call_kwargs = dict(kwargs or {})

        def _worker() -> None:
            if job_id in self._cancel_flags:
                rec.status = "cancelled"
                rec.message = "Cancelled before start"
                rec.finished_at = datetime.now(timezone.utc).isoformat()
                self._write_record(rec)
                return

            rec.status = "running"
            rec.started_at = datetime.now(timezone.utc).isoformat()
            rec.message = "Starting…"
            self._write_record(rec)
            self._append_log(job_id, "Worker started")

            def _progress(frac: float, msg: str) -> None:
                if job_id in self._cancel_flags or self._is_cancel_marked(job_id):
                    raise JobCancelledError("Cancelled by user.")
                rec.progress = float(max(0.0, min(1.0, frac)))
                rec.message = msg
                self._write_record(rec)
                self._append_log(job_id, f"({rec.progress * 100:.0f}%) {msg}")

            call_kwargs["progress_cb"] = _progress
            try:
                result = fn(**call_kwargs)
                rec.status = "succeeded"
                rec.progress = 1.0
                rec.message = "Completed successfully"
                rec.finished_at = datetime.now(timezone.utc).isoformat()
                if hasattr(result, "card"):
                    rec.result_bundle_id = result.card.bundle_id
                    m_name, m_val = result.card.primary_metric()
                    rec.result_summary = {
                        "bundle_id": result.card.bundle_id,
                        "primary_metric": m_name,
                        "primary_value": m_val,
                        "train_time_s": result.card.train_time_s,
                    }
                elif isinstance(result, list):
                    rec.result_summary = {"built_count": len(result), "bundles": result}
                self._write_record(rec)
                self._append_log(job_id, f"Finished OK -> {rec.result_bundle_id or rec.result_summary}")
            except JobCancelledError as exc:
                rec.status = "cancelled"
                rec.message = str(exc)
                rec.finished_at = datetime.now(timezone.utc).isoformat()
                self._write_record(rec)
                self._append_log(job_id, "Job cancelled by user")
            except Exception as exc:
                rec.status = "failed"
                rec.error = f"{type(exc).__name__}: {exc}"
                rec.message = rec.error
                rec.finished_at = datetime.now(timezone.utc).isoformat()
                self._write_record(rec)
                self._append_log(job_id, f"FAILED: {rec.error}\n{traceback.format_exc()}")

        self._executor.submit(_worker)
        return rec

    def _is_cancel_marked(self, job_id: str) -> bool:
        rec = self.get_job(job_id)
        return bool(rec and rec.cancel_requested)

    def cancel(self, job_id: str) -> None:
        self._cancel_flags.add(job_id)
        rec = self.get_job(job_id)
        if rec:
            rec.cancel_requested = True
            if rec.status == "queued":
                rec.status = "cancelled"
                rec.message = "Cancelled"
                rec.finished_at = datetime.now(timezone.utc).isoformat()
            self._write_record(rec)
            self._append_log(job_id, "Cancellation requested")

    def get_job(self, job_id: str) -> JobRecord | None:
        status_path = self.runs_dir / job_id / "status.json"
        if not status_path.exists():
            return None
        try:
            return JobRecord.from_dict(json.loads(status_path.read_text(encoding="utf-8")))
        except Exception:
            return None

    def read_logs(self, job_id: str, tail_lines: int = 80) -> str:
        log_path = self.runs_dir / job_id / "job.log"
        if not log_path.exists():
            return ""
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail_lines:])

    def list_jobs(self, limit: int = 30) -> list[JobRecord]:
        if not self.runs_dir.exists():
            return []
        records: list[JobRecord] = []
        for p in sorted(self.runs_dir.glob("*/status.json"), reverse=True):
            try:
                records.append(JobRecord.from_dict(json.loads(p.read_text(encoding="utf-8"))))
            except Exception:
                continue
            if len(records) >= limit:
                break
        return records


_GLOBAL_RUNNER: JobRunner | None = None


def get_job_runner() -> JobRunner:
    global _GLOBAL_RUNNER
    if _GLOBAL_RUNNER is None:
        _GLOBAL_RUNNER = JobRunner()
    return _GLOBAL_RUNNER
