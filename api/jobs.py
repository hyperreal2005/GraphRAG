"""Background jobs for the API (ROADMAP D3): ingest and removal, one at a time.

Both write to SQLite, Neo4j and Qdrant, and removal refuses to run while an ingest is unfinished,
so jobs run strictly in turn on one worker thread. The registry is in memory: a restart forgets
finished jobs, and the stores - not the registry - are the record of what was done. D4's
scheduled jobs run through the same queue.
"""

from __future__ import annotations

import logging
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    job_id: str
    kind: str
    params: dict[str, Any]
    status: str = "queued"  # queued -> running -> succeeded | failed
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }


class JobRunner:
    """A single worker thread running jobs in the order they were submitted."""

    def __init__(self, max_kept: int = 200) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: queue.Queue[tuple[Job, Callable[[], Any]] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._max_kept = max(10, int(max_kept))
        self._worker = threading.Thread(target=self._work, name="graphrag-jobs", daemon=True)
        self._worker.start()

    def submit(self, kind: str, params: dict[str, Any], fn: Callable[[], Any]) -> Job:
        job = Job(job_id=uuid.uuid4().hex, kind=kind, params=params)
        with self._lock:
            self._jobs[job.job_id] = job
            self._order.append(job.job_id)
            self._forget_oldest_finished()
        self._queue.put((job, fn))
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            return [self._jobs[job_id] for job_id in reversed(self._order[-max(1, limit):])]

    def wait(self, timeout: float | None = None) -> bool:
        """Block until every submitted job has finished; for tests and shutdown."""
        done = threading.Event()
        self._queue.put((Job(job_id="_barrier", kind="_barrier", params={}), done.set))
        return done.wait(timeout)

    def stop(self) -> None:
        self._queue.put(None)

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            job, fn = item
            if job.kind == "_barrier":
                fn()
                continue
            job.status, job.started_at = "running", _now()
            try:
                job.result = fn()
                job.status = "succeeded"
            except Exception as exc:
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                logger.warning("job_failed id=%s kind=%s\n%s", job.job_id, job.kind, traceback.format_exc())
            finally:
                job.finished_at = _now()

    def _forget_oldest_finished(self) -> None:
        while len(self._order) > self._max_kept:
            oldest = next(
                (j for j in self._order if self._jobs[j].status in {"succeeded", "failed"}), None
            )
            if oldest is None:
                return
            self._order.remove(oldest)
            self._jobs.pop(oldest, None)
