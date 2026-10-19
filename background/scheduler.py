"""The scheduler (ROADMAP D4): decides what is due and submits it to the job queue.

It runs inside the API process by default (open question 4), as a thread that wakes every
`scheduler_tick_seconds`. It never runs work itself: it submits jobs to the same queue as ingest
and removal, so a community rebuild never reads the graph while a write is half done. Two jobs:

* **community rebuild** when `community_rebuild_every_n_documents` documents have changed since
  the last build (architecture 6.3, N=50), counting removals;
* **healer report**, read-only, once a day from `healer_report_hour` local time.

`due_jobs` is the whole decision, as a pure function of the clock and the stored records, so it
is tested without threads. `main_admin.py due-jobs --run` runs the same decision from an OS task.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any, Callable

from background.community_job import adopt_current_build, build_record, staleness
from background.healer_job import last_report

logger = logging.getLogger(__name__)


def due_jobs(settings: Any, store: Any, now: datetime) -> dict[str, Any]:
    """What is due now, and why. `now` is local time."""
    community = staleness(store, int(settings.community_rebuild_every_n_documents))
    last = last_report(store)
    ran_today = False
    if last and last.get("at"):
        try:
            ran_today = datetime.fromisoformat(last["at"]).astimezone().date() == now.date()
        except ValueError:
            ran_today = False
    heal_due = bool(settings.healer_report_enabled) and now.hour >= int(settings.healer_report_hour) and not ran_today
    return {
        "communities": community,
        "healer": {
            "enabled": bool(settings.healer_report_enabled),
            "hour": int(settings.healer_report_hour),
            "last": last,
            "due": heal_due,
        },
    }


class Scheduler:
    def __init__(
        self,
        settings: Any,
        store: Any,
        runner: Any,
        jobs: dict[str, Callable[[], Any]],
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.settings = settings
        self.store = store
        self.runner = runner
        self.jobs = jobs
        self.clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick(self) -> list[str]:
        """Submit whatever is due and not already queued or running. Returns the kinds submitted."""
        if build_record(self.store) is None:
            adopt_current_build(self.store)
            logger.info("community_build_adopted: no build record existed; the current corpus is the baseline")
        due = due_jobs(self.settings, self.store, self.clock())
        submitted = []
        if due["communities"]["due"]:
            submitted += self._submit("communities", {"trigger": "changed_documents", **due["communities"]})
        if due["healer"]["due"]:
            submitted += self._submit("heal-report", {"trigger": "nightly"})
        return submitted

    def _submit(self, kind: str, params: dict[str, Any]) -> list[str]:
        if any(job.kind == kind and job.status in {"queued", "running"} for job in self.runner.list(200)):
            return []
        self.runner.submit(kind, {k: v for k, v in params.items() if k != "versions"}, self.jobs[kind])
        logger.info("scheduled_job_submitted kind=%s", kind)
        return [kind]

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="graphrag-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                logger.warning("scheduler_tick_failed error_type=%s", type(exc).__name__)
            self._stop.wait(max(5.0, float(self.settings.scheduler_tick_seconds)))
