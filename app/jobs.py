"""Sequential run queue for TriCast Studio's live mode.

One worker thread runs the jobs in submission order: the server owns one GPU, and two models loaded
at once would compete for its memory. A job moves queued → running → done | error. While it works the
runner calls ``progress(stage, fraction, partial=None)``; a ``partial`` carrying ``baseline`` is served
at once, so the native output shows before the emulated pass ends. A request equal to one that is
queued, running or done on the same server identity (``fields["server"]``: device and code hashes) gets
that job back instead of a new run; a failed request runs again.

Every finished job is saved atomically as ``runs_dir/<id>.json``. The newest ``keep`` jobs stay in
memory, and at startup the newest ``keep`` saved jobs are loaded back, so the run list and the result
cache survive a restart; :meth:`JobQueue.get` reads older saved jobs from disk. A runner failure is
logged here with its traceback and reaches the client only as ``{"code", "message"}``.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

_OUTPUT_KEYS = ("env", "timing", "baseline", "emulated", "metrics", "evidence", "cached_baseline")


class Progress(Protocol):
    def __call__(self, stage: str, fraction: float, partial: dict | None = None) -> None: ...


Work = Callable[[Progress], dict]


@dataclass(frozen=True)
class Job:
    """One run. Instances are replaced, never mutated, so a record built from one stays consistent.

    ``fields`` holds what is known at submission (``request``, ``mma_label``, ``preset``, ``recipe``);
    ``result`` collects the outputs, and at the end also ``summary`` and ``preview`` for listings.
    A saved job file is this dataclass as JSON."""

    id: str
    fields: dict
    created_utc: str
    status: str = "queued"
    stage: str | None = None
    progress: float = 0.0
    result: dict = field(default_factory=dict)
    error: dict | None = None

    def record(self) -> dict:
        """The ``GET /api/runs/{id}`` shape; outputs stay null until they exist."""
        record = {"id": self.id, "status": self.status, "stage": self.stage, "progress": self.progress,
                  **self.fields}
        record.update({key: self.result.get(key) for key in _OUTPUT_KEYS})
        if self.error is not None:
            record["error"] = self.error
        return record

    def summary(self) -> dict:
        """The ``GET /api/runs`` entry of a live run."""
        return {"id": self.id, "status": self.status, **self.fields["request"],
                "mma_label": self.fields["mma_label"], "created_utc": self.created_utc,
                "summary": self.result.get("summary"), "preview": self.result.get("preview")}


def _load(path: Path) -> Job:
    return Job(**json.loads(path.read_text(encoding="utf-8")))


class JobQueue:
    """Thread-safe store of jobs plus the single worker thread that runs them."""

    def __init__(self, runs_dir: Path, keep: int = 50) -> None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        self._runs_dir = runs_dir
        self._keep = keep
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        for path in sorted(runs_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)[-keep:]:
            try:
                job = _load(path)
            except (ValueError, TypeError):  # not JSON, or not a saved job
                log.warning("skipping %s: not a saved run", path)
                continue
            self._jobs[job.id] = job
        self._pending: queue.Queue[tuple[str, Work]] = queue.Queue()
        threading.Thread(target=self._serve, name="studio-jobs", daemon=True).start()

    def submit(self, fields: dict, work: Work) -> tuple[str, str]:
        """Queue ``work(progress) -> result`` unless a job with the same ``fields["request"]`` and
        ``fields["server"]`` is queued, running or done; return the job's id (uuid4 hex when new) and its
        status."""
        key = (fields["request"], fields.get("server"))
        with self._lock:
            same = next((job for job in reversed(self._jobs.values()) if job.status != "error"
                         and (job.fields["request"], job.fields.get("server")) == key), None)
            if same is not None:
                return same.id, same.status
            job = Job(uuid.uuid4().hex, fields, datetime.now(timezone.utc).isoformat(timespec="seconds"))
            self._jobs[job.id] = job
        self._pending.put((job.id, work))
        return job.id, job.status

    def get(self, job_id: str) -> dict | None:
        """The job's record from memory, else from ``runs_dir``; ``job_id`` must be a safe file name."""
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job.record()
        path = self._runs_dir / f"{job_id}.json"
        return _load(path).record() if path.is_file() else None

    def summaries(self) -> list[dict]:
        """Summaries of the jobs in memory, newest first."""
        with self._lock:
            jobs = list(self._jobs.values())
        return [job.summary() for job in reversed(jobs)]

    def counts(self) -> dict:
        """How many jobs wait and how many run; unfinished jobs are never dropped from memory."""
        with self._lock:
            statuses = [job.status for job in self._jobs.values()]
        return {"queued": statuses.count("queued"), "running": statuses.count("running")}

    def _serve(self) -> None:
        while True:
            self._execute(*self._pending.get())

    def _execute(self, job_id: str, work: Work) -> None:
        def progress(stage: str, fraction: float, partial: dict | None = None) -> None:
            self._report(job_id, stage, fraction, partial)

        with self._lock:
            self._jobs[job_id] = replace(self._jobs[job_id], status="running")
        try:
            job, text = self._final(job_id, status="done", progress=1.0, result=work(progress))
        except Exception as exc:  # any failure becomes the job's result; the worker keeps serving
            log.exception("run %s failed", job_id)
            job, text = self._final(job_id, status="error",
                                    error={"code": "run_failed", "message": f"{type(exc).__name__}: {exc}"})
        self._save(job.id, text)
        self._publish(job)

    def _report(self, job_id: str, stage: str, fraction: float, partial: dict | None) -> None:
        baseline = (partial or {}).get("baseline")
        if baseline is not None:
            # Checked on arrival: a non-JSON output fails the run here instead of breaking its error record.
            json.dumps(baseline, allow_nan=False)
        with self._lock:
            job = self._jobs[job_id]
            result = job.result if baseline is None else {**job.result, "baseline": baseline}
            self._jobs[job_id] = replace(job, stage=stage, progress=fraction, result=result)

    def _final(self, job_id: str, **changes: object) -> tuple[Job, str]:
        """The finished job and its file text. Serializing first means a result that is not strict JSON
        (NaN, tensors) raises here, before anything is published, and the caller records an error."""
        with self._lock:
            job = replace(self._jobs[job_id], **changes)
        return job, json.dumps(asdict(job), ensure_ascii=False, allow_nan=False)

    def _save(self, job_id: str, text: str) -> None:
        """Write the file before the job is published as finished. A failed write is logged and the job is
        still published; it then lives in memory only and is gone after a restart."""
        path = self._runs_dir / f"{job_id}.json"
        partial_file = path.with_name(f"{path.name}.tmp")
        try:
            partial_file.write_text(text, encoding="utf-8")
            os.replace(partial_file, path)  # readers see no file or the whole file, never a part
        except OSError:
            log.exception("could not save run %s", job_id)

    def _publish(self, job: Job) -> None:
        """Make the finished job visible and drop the oldest other finished jobs beyond ``keep``."""
        with self._lock:
            self._jobs[job.id] = job
            finished = [key for key, item in self._jobs.items()
                        if item.status in ("done", "error") and key != job.id]
            for key in finished[:max(0, len(self._jobs) - self._keep)]:
                del self._jobs[key]
