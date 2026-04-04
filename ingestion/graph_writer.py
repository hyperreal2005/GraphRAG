"""Neo4j graph writer for Phase 2 ingestion."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable

from config import Settings
from ingestion.relation_normalizer import sanitize_rel_type


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_properties(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        except json.JSONDecodeError:
            pass
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return json.dumps(str(value))


def normalize_aliases(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item).strip()]
    return [str(value)]


def build_source_ref(doc_id: str, source_chunk: str) -> str:
    return f"{doc_id}:{source_chunk}"


def build_edge_key(head_id: str, relation: str, tail_id: str, properties_json: str) -> str:
    payload = f"{head_id}|{relation}|{tail_id}|{properties_json}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GraphWriter:
    """Write entities and relations to Neo4j with MERGE semantics."""

    def __init__(self, settings: Settings) -> None:
        try:
            from neo4j import GraphDatabase
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("neo4j driver is required for graph writes.") from exc
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self._driver.close()

    def write_nodes(self, doc_id: str, nodes: list[dict[str, Any]]) -> None:
        query = (
            "MERGE (n:Entity {id: $id}) "
            "ON CREATE SET "
            "n.name = $name, "
            "n.type = $type, "
            "n.properties = $properties, "
            "n.aliases = $aliases, "
            "n.sources = [$doc_id], "
            "n.confidence = $confidence, "
            "n.created_at = $created_at "
            "ON MATCH SET "
            "n.sources = CASE WHEN $doc_id IN n.sources THEN n.sources ELSE n.sources + [$doc_id] END, "
            "n.updated_at = $updated_at, "
            "n.aliases = $aliases "
        )
        with self._driver.session() as session:
            for node in nodes:
                payload = {
                    "id": node["id"],
                    "name": node["name"],
                    "type": node["type"],
                    "properties": normalize_properties(node.get("properties", {})),
                    "aliases": normalize_aliases(node.get("aliases", [])),
                    "confidence": float(node.get("confidence", 0.0)),
                    "doc_id": doc_id,
                    "created_at": node.get("created_at", utc_now_iso()),
                    "updated_at": utc_now_iso(),
                }
                session.run(query, payload)

    def write_edges(self, doc_id: str, edges: list[dict[str, Any]]) -> None:
        with self._driver.session() as session:
            for edge in edges:
                rel_type = sanitize_rel_type(edge["relation"])
                source_chunk = str(edge.get("source_chunk") or "unknown")
                properties_json = normalize_properties(edge.get("properties", {}))
                source_ref = build_source_ref(doc_id, source_chunk)
                edge_key = build_edge_key(
                    str(edge["head_id"]),
                    str(edge["relation"]),
                    str(edge["tail_id"]),
                    properties_json,
                )
                query = (
                    "MATCH (h:Entity {id: $head_id}) "
                    "MATCH (t:Entity {id: $tail_id}) "
                    f"MERGE (h)-[r:{rel_type} {{edge_key: $edge_key}}]->(t) "
                    "ON CREATE SET "
                    "r.type = $relation, "
                    "r.raw_relations = [$raw_relation], "
                    "r.properties = $properties, "
                    "r.weight = $weight, "
                    "r.sources = [$doc_id], "
                    "r.source_chunks = [$source_chunk], "
                    "r.source_refs = [$source_ref], "
                    "r.evidence_count = 1, "
                    "r.created_at = $created_at "
                    "ON MATCH SET "
                    "r.raw_relations = CASE "
                    "WHEN $raw_relation IN coalesce(r.raw_relations, []) THEN coalesce(r.raw_relations, []) "
                    "ELSE coalesce(r.raw_relations, []) + [$raw_relation] "
                    "END, "
                    "r.sources = CASE WHEN $doc_id IN coalesce(r.sources, []) THEN coalesce(r.sources, []) "
                    "ELSE coalesce(r.sources, []) + [$doc_id] END, "
                    "r.source_chunks = CASE WHEN $source_chunk IN coalesce(r.source_chunks, []) "
                    "THEN coalesce(r.source_chunks, []) "
                    "ELSE coalesce(r.source_chunks, []) + [$source_chunk] END, "
                    "r.source_refs = CASE WHEN $source_ref IN coalesce(r.source_refs, []) "
                    "THEN coalesce(r.source_refs, []) "
                    "ELSE coalesce(r.source_refs, []) + [$source_ref] END, "
                    "r.weight = CASE WHEN $weight > coalesce(r.weight, 0.0) THEN $weight ELSE coalesce(r.weight, 0.0) END, "
                    "r.updated_at = $updated_at "
                    "SET r.evidence_count = size(coalesce(r.source_refs, [])) "
                )
                payload = {
                    "head_id": edge["head_id"],
                    "tail_id": edge["tail_id"],
                    "edge_key": edge_key,
                    "relation": edge["relation"],
                    "raw_relation": str(edge.get("raw_relation") or edge["relation"]),
                    "properties": properties_json,
                    "weight": float(edge.get("weight", 0.0)),
                    "source_chunk": source_chunk,
                    "source_ref": source_ref,
                    "doc_id": doc_id,
                    "created_at": edge.get("created_at", utc_now_iso()),
                    "updated_at": utc_now_iso(),
                }
                session.run(query, payload)

    def retract_chunks(self, doc_id: str, chunk_ids: Iterable[str]) -> None:
        chunk_list = list(chunk_ids)
        if not chunk_list:
            return
        source_refs = [build_source_ref(doc_id, str(chunk_id)) for chunk_id in chunk_list]
        delete_legacy_query = (
            "MATCH ()-[r]->() "
            "WHERE r.source_chunk IN $chunk_ids "
            "DELETE r"
        )
        query = (
            "MATCH ()-[r]->() "
            "WHERE any(chunk IN coalesce(r.source_chunks, []) WHERE chunk IN $chunk_ids) "
            "WITH r, "
            "[chunk IN coalesce(r.source_chunks, []) WHERE NOT chunk IN $chunk_ids] AS kept_chunks, "
            "[ref IN coalesce(r.source_refs, []) WHERE NOT ref IN $source_refs] AS kept_refs "
            "SET r.source_chunks = kept_chunks, "
            "r.source_refs = kept_refs, "
            "r.evidence_count = size(kept_refs), "
            "r.sources = [src IN coalesce(r.sources, []) WHERE any(ref IN kept_refs WHERE ref STARTS WITH src + ':')] "
            "WITH r "
            "WHERE coalesce(r.evidence_count, 0) = 0 "
            "DELETE r"
        )
        with self._driver.session() as session:
            session.run(delete_legacy_query, {"chunk_ids": chunk_list})
            session.run(query, {"chunk_ids": chunk_list, "source_refs": source_refs})

    def retract_entities(self, doc_id: str, entity_ids: Iterable[str]) -> None:
        entity_list = list(entity_ids)
        if not entity_list:
            return
        query = (
            "MATCH (n:Entity) "
            "WHERE n.id IN $entity_ids "
            "SET n.sources = [s IN n.sources WHERE s <> $doc_id] "
            "WITH n "
            "WHERE size(n.sources) = 0 "
            "DETACH DELETE n"
        )
        with self._driver.session() as session:
            session.run(query, {"entity_ids": entity_list, "doc_id": doc_id})

    def get_entities_for_doc(self, doc_id: str) -> list[dict[str, Any]]:
        """Fetch entity nodes sourced from a specific document."""
        query = (
            "MATCH (n:Entity) "
            "WHERE $doc_id IN coalesce(n.sources, []) "
            "RETURN n.id AS id, n.name AS name, n.type AS type, n.properties AS properties"
        )
        with self._driver.session() as session:
            records = session.run(query, {"doc_id": doc_id})
            return [dict(record) for record in records]
