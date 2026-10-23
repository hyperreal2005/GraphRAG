"""Cohere calls, counted as they happen (ROADMAP F2).

The trial key allows 1,000 calls a month, and twice a run lost its reranking to the cap mid-way
(Round 5, C2). Spend was reconstructed afterwards from logs. Each rerank attempt now appends one
line to `data/usage/cohere-YYYY-MM.jsonl`, keyed by a short hash of the API key - never the key
itself - so any process can say how many calls this key has made this month.

Per-run counts come from an in-process counter instead, so an evaluation reports exactly what it
spent even while the API serves queries alongside it. Gemini is counted there too: the SDK's
`generate_content`, `generate_content_stream` and `embed_content` are wrapped once per process,
the way tracing already instruments it, so every call site is covered without touching any of
them. A count is of calls; one `embed_content` call may embed a batch, and SDK-internal retries
are not visible.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import threading
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_CALLS: Counter[tuple[str, str, str]] = Counter()  # (provider, kind, model) -> calls, this process
_GEMINI_COUNTED = False
_GEMINI_METHODS = ("generate_content", "generate_content_stream", "embed_content")


def _count(provider: str, kind: str, model: str | None) -> None:
    with _LOCK:
        _CALLS[(provider, kind, model or "unknown")] += 1


def count_gemini_calls() -> bool:
    """Count every Gemini call this process makes from now on. Safe to call more than once."""
    global _GEMINI_COUNTED
    with _LOCK:
        if _GEMINI_COUNTED:
            return True
        try:
            from google.genai import models
        except ImportError:
            return False
        for cls in (models.Models, models.AsyncModels):
            for name in _GEMINI_METHODS:
                setattr(cls, name, _counted(getattr(cls, name), name))
        _GEMINI_COUNTED = True
    return True


def _counted(original: Any, kind: str) -> Any:
    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def counted_async(self: Any, *args: Any, **kwargs: Any) -> Any:
            _count("gemini", kind, kwargs.get("model"))
            return await original(self, *args, **kwargs)

        return counted_async

    @functools.wraps(original)
    def counted(self: Any, *args: Any, **kwargs: Any) -> Any:
        _count("gemini", kind, kwargs.get("model"))
        return original(self, *args, **kwargs)

    return counted


def process_calls() -> Counter[tuple[str, str, str]]:
    """A snapshot to diff against with `calls_since`."""
    with _LOCK:
        return Counter(_CALLS)


def calls_since(before: Counter[tuple[str, str, str]]) -> dict[str, dict[str, dict[str, int]]]:
    """Calls made since `before`, as provider -> kind -> model -> count."""
    nested: dict[str, dict[str, dict[str, int]]] = {}
    for (provider, kind, model), count in sorted((process_calls() - before).items()):
        nested.setdefault(provider, {}).setdefault(kind, {})[model] = count
    return nested


def key_fingerprint(api_key: str | None) -> str:
    """A short, non-reversible label for a key: enough to tell keys apart, useless to anyone else."""
    if not api_key:
        return "none"
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:10]


def _ledger(settings: Any, month: str | None = None) -> Path:
    month = month or datetime.now(timezone.utc).strftime("%Y-%m")
    return Path(settings.usage_dir) / f"cohere-{month}.jsonl"


def _started(settings: Any) -> date | None:
    """The day the ledger began counting, or None if it has never counted anything.

    Calls before it are invisible, so a month it began part-way through has an unknown start:
    on 2026-10-22 a trial key ran out at what the ledger read as 498 calls left, because the
    month's first 400-odd calls predated it.
    """
    usage_dir = Path(settings.usage_dir)
    try:
        return date.fromisoformat((usage_dir / "started.txt").read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pass
    earliest: date | None = None
    for path in usage_dir.glob("cohere-*.jsonl"):
        try:
            with path.open(encoding="utf-8") as handle:
                at = datetime.fromisoformat(json.loads(handle.readline())["at"]).date()
        except (OSError, ValueError, KeyError, TypeError):
            continue
        earliest = at if earliest is None or at < earliest else earliest
    return earliest


def record_cohere_call(settings: Any, kind: str = "rerank") -> None:
    """Count one call. A ledger that cannot be written must never fail the query."""
    _count("cohere", kind, getattr(settings, "query_rerank_model", None))
    try:
        marker = Path(settings.usage_dir) / "started.txt"
        if not marker.exists():
            started = _started(settings) or datetime.now(timezone.utc).date()
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(started.isoformat(), encoding="utf-8")
        path = _ledger(settings)
        line = json.dumps({
            "at": datetime.now(timezone.utc).isoformat(),
            "key": key_fingerprint(settings.cohere_api_key),
            "kind": kind,
        })
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:  # noqa: BLE001 - accounting is never worth an answer
        pass


def month_calls(settings: Any, month: str | None = None) -> int:
    """Calls the configured key has made this month, by this ledger's count."""
    path = _ledger(settings, month)
    if not path.exists():
        return 0
    key = key_fingerprint(settings.cohere_api_key)
    count = 0
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            try:
                count += json.loads(raw).get("key") == key
            except ValueError:
                continue
    return count


def summary(settings: Any) -> dict[str, Any]:
    """This month's calls on the configured key. `remaining` is exact only when the ledger has
    counted the whole month; otherwise it is reported as `remaining_at_most`."""
    calls = month_calls(settings)
    limit = int(settings.cohere_monthly_call_limit)
    started = _started(settings)
    whole_month = started is not None and started <= datetime.now(timezone.utc).date().replace(day=1)
    out: dict[str, Any] = {
        "key": key_fingerprint(settings.cohere_api_key),
        "month": datetime.now(timezone.utc).strftime("%Y-%m"),
        "calls_this_month": calls,
        "monthly_limit": limit or None,
        "counted_from": started.isoformat() if started else None,
    }
    if limit:
        if whole_month:
            out["remaining"] = max(0, limit - calls)
        else:
            out["remaining"] = None
            out["remaining_at_most"] = max(0, limit - calls)
    return out
