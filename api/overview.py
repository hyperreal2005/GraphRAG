"""The graph overview behind the console's Graph tab (ROADMAP D5; architecture 9.4).

The architecture's schema emergence dashboard: the top entity types and relation types, graph
density over time, and the communities with their sizes. Everything is fenced to this store's
active versions, because Neo4j holds other corpora beside it (D0). Each source fails on its own:
an unreachable Qdrant leaves the community list empty and says why, not the whole overview.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from background.community_job import staleness
from background.healer_job import last_report

logger = logging.getLogger(__name__)

ENTITY_FENCE = "any(v IN coalesce(e.active_source_versions, []) WHERE v IN $versions)"


def _graph(settings: Any, versions: list[str], driver: Any) -> dict[str, Any]:
    with driver.session() as session:
        # One scan for both the total and the breakdown: the fence makes each scan cost seconds.
        all_types = session.run(
            f"MATCH (e:Entity) WHERE {ENTITY_FENCE} "
            "RETURN coalesce(e.type, 'Entity') AS name, count(*) AS count ORDER BY count DESC",
            versions=versions,
        ).data()
        entities = sum(row["count"] for row in all_types)
        types = all_types[:20]
        predicates = session.run(
            "MATCH (a:Assertion) WHERE a.document_version_id IN $versions "
            "RETURN coalesce(a.predicate_canonical, a.predicate_normalized, a.predicate_raw) AS name, "
            "count(*) AS count ORDER BY count DESC LIMIT 50",
            versions=versions,
        ).data()
        per_version = session.run(
            "MATCH (a:Assertion) WHERE a.document_version_id IN $versions "
            "RETURN a.document_version_id AS version, count(*) AS assertions, "
            "sum(CASE WHEN a.review_status = 'accepted' THEN 1 ELSE 0 END) AS accepted",
            versions=versions,
        ).data()
    assertions = sum(row["assertions"] for row in per_version)
    return {
        "entities": entities,
        "assertions": assertions,
        "accepted_assertions": sum(row["accepted"] or 0 for row in per_version),
        "entity_types": types,
        "relation_types": predicates,
        "_per_version": {row["version"]: row["assertions"] for row in per_version},
    }


def _density(store: Any, per_version: dict[str, int]) -> list[dict[str, Any]]:
    """Documents and assertions by the day their active version was activated, cumulatively."""
    with store._connection() as conn:  # noqa: SLF001
        rows = conn.execute(
            "SELECT version_id, coalesce(activated_at, created_at) AS at FROM document_versions WHERE status = 'active'"
        ).fetchall()
    by_day: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        day = str(row["at"] or "")[:10] or "unknown"
        by_day[day][0] += 1
        by_day[day][1] += per_version.get(str(row["version_id"]), 0)
    series, documents, assertions = [], 0, 0
    for day in sorted(by_day):
        documents += by_day[day][0]
        assertions += by_day[day][1]
        series.append({
            "day": day,
            "documents_added": by_day[day][0],
            "documents": documents,
            "assertions": assertions,
            "assertions_per_document": round(assertions / documents, 2) if documents else 0.0,
        })
    return series


def _communities(settings: Any, client: Any) -> list[dict[str, Any]]:
    collection = settings.qdrant_communities_collection
    if not client.collection_exists(collection):
        return []
    points, offset = [], None
    while True:
        batch, offset = client.scroll(collection, limit=256, offset=offset, with_payload=True, with_vectors=False)
        points += batch
        if offset is None:
            break
    rows = [
        {
            "title": str((p.payload or {}).get("title") or ""),
            "size": int((p.payload or {}).get("size") or 0),
            "documents": len((p.payload or {}).get("doc_ids") or []),
        }
        for p in points
    ]
    return sorted(rows, key=lambda r: (-r["size"], r["title"]))


def _healer(store: Any) -> dict[str, Any] | None:
    last = last_report(store)
    if not last:
        return None
    try:
        report = json.loads(Path(last["report"]).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return {"at": last.get("at"), "available": False}
    kinds = Counter(str(c.get("kind")) for c in report.get("contradictions") or [])
    return {
        "at": last.get("at"),
        "available": True,
        **{k: v for k, v in report.items() if isinstance(v, (int, bool))},
        "contradiction_kinds": dict(kinds),
    }


class GraphOverview:
    """Builds the overview, keeping it for `ttl_seconds` while the corpus stays the same."""

    def __init__(self, settings: Any, store: Any, driver_factory: Any = None, qdrant_factory: Any = None,
                 ttl_seconds: int = 600) -> None:
        self.settings = settings
        self.store = store
        self.driver_factory = driver_factory or self._driver
        self.qdrant_factory = qdrant_factory or self._qdrant
        self.ttl_seconds = ttl_seconds
        self._cached: tuple[float, tuple[str, ...], dict[str, Any]] | None = None

    def get(self) -> dict[str, Any]:
        versions = tuple(sorted(self.store.get_queryable_document_version_ids() or []))
        if self._cached and self._cached[1] == versions and time.monotonic() - self._cached[0] < self.ttl_seconds:
            return self._cached[2]
        overview = self._build(list(versions))
        # Only a complete overview is kept: one built while a store was down would otherwise
        # hide its recovery for the whole TTL.
        self._cached = None if overview["errors"] else (time.monotonic(), versions, overview)
        return overview

    def _build(self, versions: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {"documents": len(versions), "errors": {}}
        per_version: dict[str, int] = {}
        try:
            driver = self.driver_factory(self.settings)
            try:
                graph = _graph(self.settings, versions, driver)
            finally:
                driver.close()
            per_version = graph.pop("_per_version")
            result["graph"] = graph
        except Exception as exc:
            logger.warning("overview_graph_failed error_type=%s", type(exc).__name__)
            result["graph"] = None
            result["errors"]["graph"] = type(exc).__name__
        result["density"] = _density(self.store, per_version)
        try:
            result["communities"] = _communities(self.settings, self.qdrant_factory(self.settings))
        except Exception as exc:
            logger.warning("overview_communities_failed error_type=%s", type(exc).__name__)
            result["communities"] = []
            result["errors"]["communities"] = type(exc).__name__
        result["community_build"] = {
            k: v for k, v in staleness(self.store, int(self.settings.community_rebuild_every_n_documents)).items()
        }
        result["healer"] = _healer(self.store)
        return result

    @staticmethod
    def _driver(settings: Any) -> Any:
        from neo4j import GraphDatabase

        return GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_query_user or settings.neo4j_user, settings.neo4j_query_password or settings.neo4j_password),
        )

    @staticmethod
    def _qdrant(settings: Any) -> Any:
        from qdrant_client import QdrantClient

        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=10)
