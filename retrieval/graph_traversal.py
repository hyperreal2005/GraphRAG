"""Level-wise graph traversal (REFACTOR2 WP4-B, REVIEW1 finding B5).

The previous templates used undirected variable-length Cypher —
``(start)-[:SUBJECT_OF|OBJECT*1..6]-(endpoint)`` — with the active, version-fence and
predicate filters applied afterwards as an ``all(assertion IN nodes(p) ...)`` predicate.
Neo4j therefore had to enumerate every path before it could discard any, and the WP1
baseline measured graph retrieval at a 18.3s p95 against a 15s budget on a 155-chunk
corpus. WP2 raises graph density by design, which makes that strictly worse.

This module replaces both templates with breadth-first expansion: one bounded query per
logical hop, every filter inlined in that level's own ``MATCH`` so it prunes instead of
post-filtering, and a fan-out cap per level. Cost becomes one round trip per hop with a
predictable worst case, rather than unbounded path enumeration.

The Cypher built here is application-owned and fully parameterised; the model never
contributes to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

# One logical hop is two relationships: Entity -> Assertion -> Entity.
LEVEL_MATCH = (
    "UNWIND $frontier AS anchor_id "
    "MATCH (source:Entity {id: anchor_id})-[:SUBJECT_OF|OBJECT]-(assertion:Assertion) "
    "WHERE assertion.active = true "
    "AND (NOT $enforce_version_fence OR assertion.document_version_id IN $active_version_ids) "
    "AND (size($relationship_types) = 0 OR assertion.predicate_normalized IN $relationship_types) "
    "AND ($temporal_start IS NULL OR assertion.valid_until IS NULL "
    "OR assertion.valid_until >= $temporal_start) "
    "AND ($temporal_end IS NULL OR assertion.valid_from IS NULL "
    "OR assertion.valid_from <= $temporal_end) "
    "MATCH (subject:Entity)-[:SUBJECT_OF]->(assertion) "
    "OPTIONAL MATCH (assertion)-[:OBJECT]->(object:Entity) "
    "RETURN DISTINCT anchor_id AS source_entity_id, assertion.id AS assertion_id, "
    "assertion.claim_key AS claim_key, subject.id AS subject_entity_id, "
    "subject.name AS subject_name, assertion.predicate_raw AS predicate_raw, "
    "assertion.predicate_canonical AS predicate_canonical, "
    "assertion.predicate_normalized AS predicate_normalized, "
    "assertion.object_kind AS object_kind, object.id AS object_entity_id, "
    "object.name AS object_name, assertion.literal_value_json AS literal_value_json, "
    "assertion.properties_json AS properties_json, assertion.confidence AS confidence, "
    "assertion.doc_id AS doc_id, assertion.document_version_id AS document_version_id, "
    "assertion.chunk_id AS chunk_id, assertion.source_ref AS source_ref, "
    "assertion.valid_from AS valid_from, assertion.valid_until AS valid_until "
    "ORDER BY assertion.confidence DESC, assertion.id ASC "
    "LIMIT $level_limit"
)


def build_level_query() -> str:
    """The single bounded query used for every BFS level."""
    return LEVEL_MATCH


def _fallback_identity(row: dict[str, Any]) -> str:
    """Dedupe key for a row that carries no assertion_id (legacy path format)."""
    parts = [
        str(row.get("source_entity_id") or ""),
        "|".join(str(item) for item in (row.get("node_names") or [])),
        "|".join(str(item) for item in (row.get("rel_types") or [])),
        str(row.get("subject_entity_id") or ""),
        str(row.get("object_entity_id") or ""),
    ]
    return "legacy:" + "::".join(parts)


def neighbours(row: dict[str, Any], source_entity_id: str) -> list[str]:
    """Entity IDs reachable from ``source_entity_id`` through this assertion."""
    found: list[str] = []
    for key in ("subject_entity_id", "object_entity_id"):
        value = row.get(key)
        if value and str(value) != source_entity_id:
            found.append(str(value))
    return found


@dataclass
class TraversalResult:
    """Assertion rows discovered, annotated with the hop at which each was reached."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    levels_executed: int = 0
    reached: set[str] = field(default_factory=set)
    target_found: bool = False

    def limited(self, path_limit: int) -> list[dict[str, Any]]:
        """Closest first, then most confident — the ordering the old LIMIT applied."""
        ordered = sorted(
            self.rows,
            key=lambda row: (
                int(row.get("hop_count") or 1),
                -float(row.get("confidence") or 0.0),
                str(row.get("assertion_id") or ""),
            ),
        )
        return ordered[: max(1, int(path_limit))]


def expand(
    execute: Callable[[str, dict[str, Any]], list[dict[str, Any]]],
    *,
    anchor_ids: Iterable[str],
    max_hops: int,
    level_limit: int,
    params: dict[str, Any],
    targets: set[str] | None = None,
) -> TraversalResult:
    """Breadth-first expansion from ``anchor_ids``, at most ``max_hops`` logical hops.

    ``execute`` runs one level's Cypher and returns its rows, so this stays testable
    without a database and works for both the sync and async drivers.

    When ``targets`` is supplied the walk stops as soon as one is reached and only the
    assertions on the connecting path are returned; that replaces ``shortestPath``, whose
    post-filter silently yielded "no path" whenever the shortest route happened to carry
    one inactive assertion.
    """
    result = TraversalResult()
    frontier = [str(item) for item in anchor_ids if str(item)]
    if not frontier:
        return result

    visited: set[str] = set(frontier)
    result.reached.update(frontier)
    # entity_id -> (assertion row, entity it was reached from); used to rebuild a path.
    parents: dict[str, tuple[dict[str, Any], str]] = {}
    seen_assertions: set[str] = set()

    for level in range(1, max(1, int(max_hops)) + 1):
        if not frontier:
            break
        level_params = {
            **params,
            "frontier": frontier,
            "level_limit": max(1, int(level_limit)),
        }
        rows = execute(LEVEL_MATCH, level_params)
        result.levels_executed = level

        next_frontier: list[str] = []
        for row in rows:
            source_entity_id = str(row.get("source_entity_id") or "")
            annotated = {**row, "hop_count": level}
            # Rows from the pre-assertion graph carry no assertion_id; dedupe those on
            # their path shape instead of discarding them, so legacy reads keep working
            # until WP5-A removes that format for good.
            identity = str(row.get("assertion_id") or "") or _fallback_identity(row)
            if identity not in seen_assertions:
                seen_assertions.add(identity)
                result.rows.append(annotated)

            for neighbour in neighbours(row, source_entity_id):
                result.reached.add(neighbour)
                if neighbour not in parents and neighbour not in visited:
                    parents[neighbour] = (annotated, source_entity_id)
                if neighbour not in visited:
                    visited.add(neighbour)
                    next_frontier.append(neighbour)

        if targets:
            hit = next((item for item in targets if item in result.reached), None)
            if hit is not None:
                result.target_found = True
                result.rows = _path_rows(parents, hit)
                return result

        frontier = next_frontier

    if targets and not result.target_found:
        # A partial neighbourhood is not a connecting path; say so rather than
        # returning unrelated assertions.
        result.rows = []
    return result


def _path_rows(
    parents: dict[str, tuple[dict[str, Any], str]],
    target: str,
) -> list[dict[str, Any]]:
    """Walk the parent chain back from ``target`` to the anchor, nearest hop first."""
    path: list[dict[str, Any]] = []
    current = target
    guard = 0
    while current in parents and guard < 64:
        row, previous = parents[current]
        path.append(row)
        current = previous
        guard += 1
    path.reverse()
    return path
