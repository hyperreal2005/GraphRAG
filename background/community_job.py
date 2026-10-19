"""Community rebuilds and their staleness (ROADMAP D4; architecture 6.3).

Communities are derived: a build summarises the graph as it was, and nothing updates it until the
next build. Until D4 nothing recorded *what* a build saw, so nothing could say how far the corpus
had moved since - and a removed document's facts lived on in the summaries indefinitely.

Each build now records the active version of every document it saw. Staleness is the number of
documents added, removed or re-versioned since; the architecture's trigger is 50.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

RECORD_KEY = "community_build"


def active_versions(store: Any) -> dict[str, str]:
    """Each document's active version, from SQLite, the store every read path fences on."""
    with store._connection() as conn:  # noqa: SLF001
        rows = conn.execute("SELECT doc_id, version_id FROM document_versions WHERE status = 'active'").fetchall()
    return {str(row["doc_id"]): str(row["version_id"]) for row in rows}


def build_record(store: Any) -> dict[str, Any] | None:
    raw = store.get_system_metadata(RECORD_KEY)
    if not raw:
        return None
    try:
        record = json.loads(raw)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


def record_build(store: Any, *, build_id: str, communities: int, source: str = "build") -> dict[str, Any]:
    record = {
        "build_id": build_id,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "communities": communities,
        "source": source,
        "versions": active_versions(store),
    }
    store.set_system_metadata(RECORD_KEY, json.dumps(record))
    return record


def adopt_current_build(store: Any) -> dict[str, Any]:
    """Record the corpus as it stands as the last build's, without rebuilding.

    For a store whose communities were built before builds were recorded: rebuilding only to
    learn the baseline would cost a provider call per community. Marked "adopted" so the record
    says it was assumed, not observed.
    """
    return record_build(store, build_id="adopted", communities=-1, source="adopted: no build record existed")


def staleness(store: Any, threshold: int) -> dict[str, Any]:
    """How many documents changed since the last build, and whether a rebuild is due."""
    record = build_record(store)
    if record is None:
        return {"recorded": False, "changed_documents": None, "threshold": threshold, "due": False}
    then = record.get("versions") or {}
    now = active_versions(store)
    changed = sum(1 for doc_id in set(then) | set(now) if then.get(doc_id) != now.get(doc_id))
    return {
        "recorded": True,
        "build_id": record.get("build_id"),
        "built_at": record.get("built_at"),
        "source": record.get("source"),
        "changed_documents": changed,
        "threshold": threshold,
        "due": threshold > 0 and changed >= threshold,
    }


def run_community_build(settings: Any, store: Any, builder_factory: Any | None = None) -> dict[str, Any]:
    """Rebuild every community of this store and record what the build saw."""
    if builder_factory is None:
        from ingestion.community_builder import CommunityBuilder

        builder_factory = CommunityBuilder
    builder = builder_factory(settings)
    try:
        report = builder.build()
    finally:
        builder.close()
    result = {
        "build_id": report.build_id,
        "detected": report.detected,
        "kept": report.kept,
        "summarised": report.summarised,
        "indexed": report.indexed,
        "failed": report.failed,
        "provider_calls": report.provider_calls,
    }
    if report.summarised:
        record_build(store, build_id=report.build_id, communities=report.indexed)
    else:
        # The previous build is still the one being served; its record stands.
        logger.warning("community_build_empty build_id=%s; previous build kept", report.build_id)
    return result
