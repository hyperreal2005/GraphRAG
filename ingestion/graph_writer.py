"""Neo4j writer for canonical entities and evidence-bearing assertions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable

from config import Settings
from ingestion.relation_normalizer import sanitize_rel_type
from models.refactor import AssertionRecord


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_properties(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, (dict, list)):
                return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except json.JSONDecodeError:
            pass
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def normalize_aliases(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [str(item) for item in value if item is not None]
    else:
        values = [str(value)]
    return list(dict.fromkeys(item.strip() for item in values if item.strip()))


def build_source_ref(doc_id: str, source_chunk: str) -> str:
    """Legacy two-part source reference retained for compatibility."""
    return f"{doc_id}:{source_chunk}"


def build_edge_key(head_id: str, relation: str, tail_id: str, properties_json: str) -> str:
    """Legacy direct-edge key retained for migration tests/read compatibility."""
    payload = f"{head_id}|{relation}|{tail_id}|{properties_json}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


ENTITY_BATCH_QUERY = """
UNWIND $rows AS row
MERGE (n:Entity {id: row.id})
ON CREATE SET
    n.name = row.name,
    n.type = row.type,
    n.properties = row.properties_json,
    n.aliases = row.aliases,
    n.sources = [row.doc_id],
    n.source_versions = [row.document_version_id],
    n.confidence = row.confidence,
    n.created_at = row.created_at
ON MATCH SET
    n.name = CASE
        WHEN trim(coalesce(n.name, '')) = '' OR toLower(n.name) = 'entity' THEN row.name
        ELSE n.name
    END,
    n.type = CASE
        WHEN trim(coalesce(n.type, '')) = '' OR toLower(n.type) = 'entity' THEN row.type
        ELSE n.type
    END,
    n.properties = CASE
        WHEN coalesce(n.properties, '{}') = '{}' THEN row.properties_json
        ELSE n.properties
    END,
    n.aliases = reduce(acc = coalesce(n.aliases, []), alias IN row.aliases |
        CASE WHEN alias IN acc THEN acc ELSE acc + alias END),
    n.sources = CASE
        WHEN row.doc_id IN coalesce(n.sources, []) THEN coalesce(n.sources, [])
        ELSE coalesce(n.sources, []) + row.doc_id
    END,
    n.source_versions = CASE
        WHEN row.document_version_id IN coalesce(n.source_versions, []) THEN coalesce(n.source_versions, [])
        ELSE coalesce(n.source_versions, []) + row.document_version_id
    END,
    n.confidence = CASE
        WHEN row.confidence > coalesce(n.confidence, 0.0) THEN row.confidence
        ELSE coalesce(n.confidence, 0.0)
    END,
    n.updated_at = row.updated_at
"""


ENTITY_ASSERTION_BATCH_QUERY = """
UNWIND $rows AS row
MATCH (subject:Entity {id: row.subject_entity_id})
MATCH (object:Entity {id: row.object_entity_id})
MERGE (assertion:Assertion {id: row.id})
SET assertion.claim_key = row.claim_key,
    assertion.subject_local_entity_id = row.subject_local_entity_id,
    assertion.predicate_raw = row.predicate_raw,
    assertion.predicate_normalized = row.predicate_normalized,
    assertion.object_kind = row.object_kind,
    assertion.object_local_entity_id = row.object_local_entity_id,
    assertion.literal_value_json = NULL,
    assertion.properties_json = row.properties_json,
    assertion.confidence = row.confidence,
    assertion.doc_id = row.doc_id,
    assertion.document_version_id = row.document_version_id,
    assertion.chunk_id = row.chunk_id,
    assertion.source_ref = row.source_ref,
    assertion.extractor_model = row.extractor_model,
    assertion.extraction_prompt_version = row.extraction_prompt_version,
    assertion.valid_from = row.valid_from,
    assertion.valid_until = row.valid_until,
    assertion.active = row.active,
    assertion.review_status = row.review_status,
    assertion.created_at = coalesce(assertion.created_at, row.created_at),
    assertion.updated_at = row.updated_at
MERGE (subject)-[:SUBJECT_OF]->(assertion)
MERGE (assertion)-[:OBJECT]->(object)
"""


LITERAL_ASSERTION_BATCH_QUERY = """
UNWIND $rows AS row
MATCH (subject:Entity {id: row.subject_entity_id})
MERGE (assertion:Assertion {id: row.id})
SET assertion.claim_key = row.claim_key,
    assertion.subject_local_entity_id = row.subject_local_entity_id,
    assertion.predicate_raw = row.predicate_raw,
    assertion.predicate_normalized = row.predicate_normalized,
    assertion.object_kind = row.object_kind,
    assertion.object_local_entity_id = NULL,
    assertion.literal_value_json = row.literal_value_json,
    assertion.properties_json = row.properties_json,
    assertion.confidence = row.confidence,
    assertion.doc_id = row.doc_id,
    assertion.document_version_id = row.document_version_id,
    assertion.chunk_id = row.chunk_id,
    assertion.source_ref = row.source_ref,
    assertion.extractor_model = row.extractor_model,
    assertion.extraction_prompt_version = row.extraction_prompt_version,
    assertion.valid_from = row.valid_from,
    assertion.valid_until = row.valid_until,
    assertion.active = row.active,
    assertion.review_status = row.review_status,
    assertion.created_at = coalesce(assertion.created_at, row.created_at),
    assertion.updated_at = row.updated_at
MERGE (subject)-[:SUBJECT_OF]->(assertion)
WITH assertion
OPTIONAL MATCH (assertion)-[legacyObject:OBJECT]->()
DELETE legacyObject
"""


class GraphWriter:
    """Write versioned entities/assertions with bounded batched transactions."""

    def __init__(self, settings: Settings, driver: Any | None = None) -> None:
        self.settings = settings
        if driver is not None:
            self._driver = driver
        else:
            try:
                from neo4j import GraphDatabase
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("neo4j driver is required for graph writes.") from exc
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT assertion_id_unique IF NOT EXISTS FOR (a:Assertion) REQUIRE a.id IS UNIQUE",
            "CREATE INDEX assertion_claim_key IF NOT EXISTS FOR (a:Assertion) ON (a.claim_key)",
            "CREATE INDEX assertion_version IF NOT EXISTS FOR (a:Assertion) ON (a.document_version_id)",
            "CREATE INDEX assertion_chunk IF NOT EXISTS FOR (a:Assertion) ON (a.chunk_id)",
            "CREATE INDEX assertion_active IF NOT EXISTS FOR (a:Assertion) ON (a.active)",
        ]
        with self._driver.session() as session:
            for statement in statements:
                result = session.run(statement)
                consume = getattr(result, "consume", None)
                if callable(consume):
                    consume()

    def close(self) -> None:
        self._driver.close()

    def write_graph_version(
        self,
        *,
        doc_id: str,
        document_version_id: str,
        nodes: list[dict[str, Any]],
        assertions: list[AssertionRecord],
        batch_size: int = 500,
    ) -> None:
        """Stage one version's complete graph contribution in one write transaction."""
        now = utc_now_iso()
        node_rows = [
            {
                "id": str(node["id"]),
                "name": str(node.get("name") or ""),
                "type": str(node.get("type") or "Entity"),
                "properties_json": normalize_properties(node.get("properties", {})),
                "aliases": normalize_aliases(node.get("aliases", [])),
                "confidence": float(node.get("confidence", 0.0)),
                "doc_id": doc_id,
                "document_version_id": document_version_id,
                "created_at": str(node.get("created_at") or now),
                "updated_at": now,
            }
            for node in nodes
        ]
        assertion_rows = [self._assertion_row(item) for item in assertions]
        entity_rows = [row for row in assertion_rows if row["object_kind"] == "entity"]
        literal_rows = [row for row in assertion_rows if row["object_kind"] == "literal"]

        def write_transaction(tx: Any) -> None:
            for batch in self._batches(node_rows, batch_size):
                tx.run(ENTITY_BATCH_QUERY, rows=batch).consume()
            for batch in self._batches(entity_rows, batch_size):
                tx.run(ENTITY_ASSERTION_BATCH_QUERY, rows=batch).consume()
            for batch in self._batches(literal_rows, batch_size):
                tx.run(LITERAL_ASSERTION_BATCH_QUERY, rows=batch).consume()

        with self._driver.session() as session:
            execute_write = getattr(session, "execute_write", None)
            if callable(execute_write):
                execute_write(write_transaction)
            else:  # compatible test doubles and older drivers
                write_transaction(session)

    def write_nodes(self, doc_id: str, nodes: list[dict[str, Any]]) -> None:
        """Legacy compatibility wrapper; Refactor 1 pipeline uses write_graph_version."""
        self.write_graph_version(
            doc_id=doc_id,
            document_version_id=f"legacy:{doc_id}",
            nodes=nodes,
            assertions=[],
        )

    def write_edges(self, doc_id: str, edges: list[dict[str, Any]]) -> None:
        """Legacy direct-edge writer retained only for old-store compatibility."""
        with self._driver.session() as session:
            for edge in edges:
                rel_type = sanitize_rel_type(edge["relation"])
                source_chunk = str(edge.get("source_chunk") or "unknown")
                properties_json = normalize_properties(edge.get("properties", {}))
                edge_key = build_edge_key(
                    str(edge["head_id"]),
                    str(edge["relation"]),
                    str(edge["tail_id"]),
                    properties_json,
                )
                query = (
                    "MATCH (h:Entity {id: $head_id}) MATCH (t:Entity {id: $tail_id}) "
                    f"MERGE (h)-[r:{rel_type} {{edge_key: $edge_key}}]->(t) "
                    "SET r.type=$relation, r.raw_relations=[$raw_relation], "
                    "r.properties=$properties, r.weight=$weight, r.sources=[$doc_id], "
                    "r.source_chunks=[$source_chunk], r.source_refs=[$source_ref]"
                )
                session.run(
                    query,
                    {
                        "head_id": edge["head_id"],
                        "tail_id": edge["tail_id"],
                        "edge_key": edge_key,
                        "relation": edge["relation"],
                        "raw_relation": str(edge.get("raw_relation") or edge["relation"]),
                        "properties": properties_json,
                        "weight": float(edge.get("weight", 0.0)),
                        "source_chunk": source_chunk,
                        "source_ref": build_source_ref(doc_id, source_chunk),
                        "doc_id": doc_id,
                    },
                )

    def verify_document_version(self, document_version_id: str) -> dict[str, int]:
        """Return staged count/integrity metrics for an assertion version."""
        query = """
        MATCH (a:Assertion {document_version_id: $version_id})
        OPTIONAL MATCH (:Entity)-[subjectEdge:SUBJECT_OF]->(a)
        OPTIONAL MATCH (a)-[objectEdge:OBJECT]->(:Entity)
        RETURN count(DISTINCT a) AS assertions,
               count(DISTINCT subjectEdge) AS subject_edges,
               count(DISTINCT CASE WHEN a.object_kind = 'entity' THEN a END) AS entity_assertions,
               count(DISTINCT objectEdge) AS object_edges
        """
        with self._driver.session() as session:
            record = session.run(query, {"version_id": document_version_id}).single()
        if record is None:
            return {"assertions": 0, "subject_edges": 0, "entity_assertions": 0, "object_edges": 0}
        return {key: int(record[key] or 0) for key in ("assertions", "subject_edges", "entity_assertions", "object_edges")}

    def activate_document_version(self, document_version_id: str, previous_version_id: str | None) -> None:
        """Activate staged assertions and atomically deactivate the superseded version."""
        query = """
        MATCH (candidate:Assertion {document_version_id: $version_id})
        SET candidate.active = true, candidate.updated_at = $updated_at
        WITH count(candidate) AS activated
        OPTIONAL MATCH (old:Assertion {document_version_id: $previous_version_id})
        WHERE $previous_version_id IS NOT NULL
        SET old.active = false, old.updated_at = $updated_at
        RETURN activated, count(old) AS deactivated
        """
        with self._driver.session() as session:
            session.run(
                query,
                {
                    "version_id": document_version_id,
                    "previous_version_id": previous_version_id,
                    "updated_at": utc_now_iso(),
                },
            ).consume()

    def retract_document_version(self, document_version_id: str) -> None:
        """Delete one version's assertions and clean only assertion-free entity residue."""
        with self._driver.session() as session:
            session.run(
                "MATCH (a:Assertion {document_version_id: $version_id}) DETACH DELETE a",
                {"version_id": document_version_id},
            ).consume()
            session.run(
                """
                MATCH (n:Entity)
                WHERE $version_id IN coalesce(n.source_versions, [])
                SET n.source_versions = [v IN coalesce(n.source_versions, []) WHERE v <> $version_id]
                WITH n
                OPTIONAL MATCH (n)-[:SUBJECT_OF]->(subjectAssertion:Assertion)
                OPTIONAL MATCH (objectAssertion:Assertion)-[:OBJECT]->(n)
                WITH n, count(DISTINCT subjectAssertion) + count(DISTINCT objectAssertion) AS evidence_count
                WHERE evidence_count = 0 AND size(coalesce(n.source_versions, [])) = 0
                DETACH DELETE n
                """,
                {"version_id": document_version_id},
            ).consume()

    def retract_assertions_for_chunks(self, document_version_id: str, chunk_ids: Iterable[str]) -> None:
        chunk_list = sorted(set(str(item) for item in chunk_ids if str(item)))
        if not chunk_list:
            return
        with self._driver.session() as session:
            session.run(
                """
                MATCH (a:Assertion {document_version_id: $version_id})
                WHERE a.chunk_id IN $chunk_ids
                DETACH DELETE a
                """,
                {"version_id": document_version_id, "chunk_ids": chunk_list},
            ).consume()

    def retract_chunks(self, doc_id: str, chunk_ids: Iterable[str]) -> None:
        """Retract legacy direct edges plus assertion evidence for selected chunks."""
        chunk_list = sorted(set(str(item) for item in chunk_ids if str(item)))
        if not chunk_list:
            return
        with self._driver.session() as session:
            session.run(
                "MATCH (a:Assertion {doc_id: $doc_id}) WHERE a.chunk_id IN $chunk_ids DETACH DELETE a",
                {"doc_id": doc_id, "chunk_ids": chunk_list},
            ).consume()
            session.run(
                "MATCH ()-[r]->() WHERE r.source_chunk IN $chunk_ids DELETE r",
                {"chunk_ids": chunk_list},
            ).consume()

    def retract_entities(self, doc_id: str, entity_ids: Iterable[str]) -> set[str]:
        """Remove a document source and delete only assertion-free orphan entities."""
        entity_list = sorted(set(str(item) for item in entity_ids if str(item)))
        if not entity_list:
            return set()
        update_query = """
        MATCH (n:Entity)
        WHERE n.id IN $entity_ids
        SET n.sources = [s IN coalesce(n.sources, []) WHERE s <> $doc_id]
        WITH n
        OPTIONAL MATCH (n)-[:SUBJECT_OF]->(subjectAssertion:Assertion)
        OPTIONAL MATCH (objectAssertion:Assertion)-[:OBJECT]->(n)
        WITH n, count(DISTINCT subjectAssertion) + count(DISTINCT objectAssertion) AS assertion_count
        RETURN n.id AS entity_id,
               size(coalesce(n.sources, [])) = 0 AND assertion_count = 0 AS orphaned
        """
        delete_query = "MATCH (n:Entity) WHERE n.id IN $entity_ids DETACH DELETE n"
        with self._driver.session() as session:
            rows = session.run(update_query, {"entity_ids": entity_list, "doc_id": doc_id})
            orphaned = {str(row["entity_id"]) for row in rows if bool(row["orphaned"])}
            if orphaned:
                session.run(delete_query, {"entity_ids": sorted(orphaned)}).consume()
        return orphaned

    def get_entities_for_doc(self, doc_id: str) -> list[dict[str, Any]]:
        query = """
        MATCH (n:Entity)
        WHERE $doc_id IN coalesce(n.sources, [])
        RETURN n.id AS id, n.name AS name, n.type AS type,
               n.properties AS properties, n.aliases AS aliases
        """
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, {"doc_id": doc_id})]

    def get_active_assertions_for_doc(self, doc_id: str) -> list[dict[str, Any]]:
        query = """
        MATCH (subject:Entity)-[:SUBJECT_OF]->(a:Assertion {doc_id: $doc_id, active: true})
        OPTIONAL MATCH (a)-[:OBJECT]->(object:Entity)
        RETURN a.id AS id, a.claim_key AS claim_key,
               subject.id AS subject_entity_id, subject.name AS subject_name,
               a.subject_local_entity_id AS subject_local_entity_id,
               a.predicate_raw AS predicate_raw,
               a.predicate_normalized AS predicate_normalized,
               a.object_kind AS object_kind,
               object.id AS object_entity_id, object.name AS object_name,
               a.object_local_entity_id AS object_local_entity_id,
               a.literal_value_json AS literal_value_json,
               a.properties_json AS properties_json, a.confidence AS confidence,
               a.document_version_id AS document_version_id,
               a.chunk_id AS chunk_id, a.source_ref AS source_ref,
               a.valid_from AS valid_from, a.valid_until AS valid_until,
               a.review_status AS review_status
        ORDER BY a.chunk_id, a.claim_key
        """
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, {"doc_id": doc_id})]

    def find_contradictions(self, doc_id: str | None = None) -> list[dict[str, Any]]:
        """Surface active claims with one subject/predicate and multiple object values."""
        query = """
        MATCH (subject:Entity)-[:SUBJECT_OF]->(a:Assertion {active: true})
        WHERE $doc_id IS NULL OR a.doc_id = $doc_id
        OPTIONAL MATCH (a)-[:OBJECT]->(object:Entity)
        WITH subject.id AS subject_id,
             a.predicate_normalized AS predicate,
             collect(DISTINCT coalesce(object.id, a.literal_value_json)) AS values,
             collect(DISTINCT a.id) AS assertion_ids
        WHERE size(values) > 1
        RETURN subject_id, predicate, values, assertion_ids
        ORDER BY subject_id, predicate
        """
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, {"doc_id": doc_id})]

    def rollback_merged_entity(
        self,
        *,
        doc_id: str,
        document_version_id: str,
        local_entity_id: str,
        merged_entity_id: str,
        restored_entity_id: str,
        name: str,
        entity_type: str,
    ) -> int:
        """Rewrite only assertion endpoints carrying one local merge provenance key."""
        query = """
        MATCH (subject:Entity)-[:SUBJECT_OF]->(a:Assertion {document_version_id: $version_id})
        OPTIONAL MATCH (a)-[:OBJECT]->(object:Entity)
        WHERE (a.subject_local_entity_id = $local_id AND subject.id = $merged_id)
           OR (a.object_local_entity_id = $local_id AND object.id = $merged_id)
        RETURN a.id AS id, subject.id AS subject_entity_id,
               a.subject_local_entity_id AS subject_local_entity_id,
               a.predicate_raw AS predicate_raw, a.object_kind AS object_kind,
               object.id AS object_entity_id,
               a.object_local_entity_id AS object_local_entity_id,
               a.literal_value_json AS literal_value_json,
               a.properties_json AS properties_json, a.confidence AS confidence,
               a.chunk_id AS chunk_id, a.extractor_model AS extractor_model,
               a.extraction_prompt_version AS extraction_prompt_version,
               a.valid_from AS valid_from, a.valid_until AS valid_until,
               a.active AS active, a.review_status AS review_status
        """
        with self._driver.session() as session:
            rows = [
                dict(record)
                for record in session.run(
                    query,
                    {
                        "version_id": document_version_id,
                        "local_id": local_entity_id,
                        "merged_id": merged_entity_id,
                    },
                )
            ]

        rewritten: list[AssertionRecord] = []
        old_ids: list[str] = []
        for row in rows:
            old_ids.append(str(row["id"]))
            subject_id = str(row["subject_entity_id"])
            object_id = str(row["object_entity_id"]) if row.get("object_entity_id") else None
            if row.get("subject_local_entity_id") == local_entity_id and subject_id == merged_entity_id:
                subject_id = restored_entity_id
            if row.get("object_local_entity_id") == local_entity_id and object_id == merged_entity_id:
                object_id = restored_entity_id
            literal_value = None
            if row.get("literal_value_json") is not None:
                literal_value = json.loads(str(row["literal_value_json"]))
            properties = json.loads(str(row.get("properties_json") or "{}"))
            rewritten.append(
                AssertionRecord.create(
                    subject_entity_id=subject_id,
                    subject_local_entity_id=self._optional_text(row.get("subject_local_entity_id")),
                    predicate_raw=str(row["predicate_raw"]),
                    object_kind=str(row["object_kind"]),
                    object_entity_id=object_id,
                    object_local_entity_id=self._optional_text(row.get("object_local_entity_id")),
                    literal_value=literal_value,
                    properties=properties,
                    confidence=float(row.get("confidence") or 0.0),
                    doc_id=doc_id,
                    document_version_id=document_version_id,
                    chunk_id=str(row["chunk_id"]),
                    extractor_model=str(row.get("extractor_model") or self.settings.gemini_model),
                    extraction_prompt_version=str(
                        row.get("extraction_prompt_version") or self.settings.extraction_prompt_version
                    ),
                    valid_from=self._optional_text(row.get("valid_from")),
                    valid_until=self._optional_text(row.get("valid_until")),
                    active=bool(row.get("active")),
                    review_status=str(row.get("review_status") or "accepted"),
                )
            )

        self.write_graph_version(
            doc_id=doc_id,
            document_version_id=document_version_id,
            nodes=[
                {
                    "id": restored_entity_id,
                    "name": name,
                    "type": entity_type,
                    "aliases": [name],
                    "properties": {},
                    "confidence": 0.0,
                }
            ],
            assertions=rewritten,
        )
        if old_ids:
            with self._driver.session() as session:
                session.run(
                    "MATCH (a:Assertion) WHERE a.id IN $assertion_ids DETACH DELETE a",
                    {"assertion_ids": old_ids},
                ).consume()
        return len(rewritten)

    @staticmethod
    def _assertion_row(assertion: AssertionRecord) -> dict[str, Any]:
        row = assertion.model_dump(mode="json")
        row["created_at"] = assertion.created_at.isoformat()
        row["updated_at"] = assertion.updated_at.isoformat()
        return row

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    @staticmethod
    def _batches(rows: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
        size = max(1, int(batch_size))
        for start in range(0, len(rows), size):
            yield rows[start : start + size]
