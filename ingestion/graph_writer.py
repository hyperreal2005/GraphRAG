"""Neo4j graph writer for Phase 2 ingestion."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from config import Settings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_rel_type(relation: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in relation.upper())
    cleaned = cleaned.strip("_") or "RELATED_TO"
    return cleaned


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
                    "properties": node.get("properties", {}),
                    "aliases": node.get("aliases", []),
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
                query = (
                    "MATCH (h:Entity {id: $head_id}) "
                    "MATCH (t:Entity {id: $tail_id}) "
                    f"MERGE (h)-[r:{rel_type} {{source_chunk: $source_chunk}}]->(t) "
                    "ON CREATE SET "
                    "r.type = $relation, "
                    "r.properties = $properties, "
                    "r.weight = $weight, "
                    "r.sources = [$doc_id], "
                    "r.created_at = $created_at "
                    "ON MATCH SET "
                    "r.sources = CASE WHEN $doc_id IN r.sources THEN r.sources ELSE r.sources + [$doc_id] END "
                )
                payload = {
                    "head_id": edge["head_id"],
                    "tail_id": edge["tail_id"],
                    "relation": edge["relation"],
                    "properties": edge.get("properties", {}),
                    "weight": float(edge.get("weight", 0.0)),
                    "source_chunk": edge.get("source_chunk"),
                    "doc_id": doc_id,
                    "created_at": edge.get("created_at", utc_now_iso()),
                }
                session.run(query, payload)

    def retract_chunks(self, chunk_ids: Iterable[str]) -> None:
        chunk_list = list(chunk_ids)
        if not chunk_list:
            return
        query = (
            "MATCH ()-[r]->() "
            "WHERE r.source_chunk IN $chunk_ids "
            "DELETE r"
        )
        with self._driver.session() as session:
            session.run(query, {"chunk_ids": chunk_list})

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
