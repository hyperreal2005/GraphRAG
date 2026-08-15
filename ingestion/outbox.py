"""Durable Refactor 1 orchestration across SQLite, Neo4j, and Qdrant."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import Any, Iterable

from config import Settings
from ingestion.graph_writer import GraphWriter
from ingestion.state_store import MetadataStore
from ingestion.vector_indexer import (
    ChunkVectorInput,
    EntityVectorInput,
    QdrantVectorIndexer,
    build_entity_summary,
)
from models.refactor import AssertionRecord, OutboxEvent, OutboxEventType, canonical_json


logger = logging.getLogger(__name__)


class IngestionOutboxProcessor:
    """Apply version transitions idempotently and retain complete failure state."""

    def __init__(
        self,
        *,
        settings: Settings,
        store: MetadataStore,
        graph_writer: GraphWriter,
        vector_indexer: QdrantVectorIndexer,
    ) -> None:
        self.settings = settings
        self.store = store
        self.graph_writer = graph_writer
        self.vector_indexer = vector_indexer

    @staticmethod
    def serialize_initial_payload(
        *,
        nodes: list[dict[str, Any]],
        assertions: Iterable[AssertionRecord],
        chunks: Iterable[ChunkVectorInput],
    ) -> dict[str, Any]:
        """Create the durable graph-stage payload from validated records."""
        return {
            "nodes": nodes,
            "assertions": [item.model_dump(mode="json") for item in assertions],
            "chunks": [asdict(item) for item in chunks],
        }

    def make_event(
        self,
        *,
        event_type: OutboxEventType,
        run_id: str,
        doc_id: str,
        document_version_id: str,
        payload: dict[str, Any],
        discriminator: str = "",
    ) -> OutboxEvent:
        """Build a deterministic event whose replay cannot duplicate work."""
        return OutboxEvent(
            event_id=self.store.build_outbox_event_id(
                document_version_id,
                event_type,
                discriminator,
            ),
            event_type=event_type,
            run_id=run_id,
            doc_id=doc_id,
            document_version_id=document_version_id,
            payload_json=canonical_json(payload),
        )

    def reconcile(
        self,
        *,
        document_version_id: str | None = None,
        wait_for_retries: bool = False,
        max_events: int = 100,
    ) -> int:
        """Process available work; optionally wait through bounded retry delays."""
        processed = 0
        while processed < max(1, max_events):
            event = self.store.claim_next_outbox_event(
                max_attempts=self.settings.outbox_max_attempts,
                document_version_id=document_version_id,
                lease_seconds=self.settings.outbox_lease_seconds,
            )
            if event is None:
                break
            started = time.perf_counter()
            try:
                self._dispatch(event)
                self.store.complete_outbox_event(event.event_id)
                self._record_event_metric(
                    event,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    status="completed",
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                status = self.store.fail_outbox_event(
                    event.event_id,
                    error,
                    max_attempts=self.settings.outbox_max_attempts,
                    retry_base_seconds=self.settings.outbox_retry_base_seconds,
                )
                logger.exception(
                    "outbox_event_failed event_id=%s type=%s attempt=%d status=%s",
                    event.event_id,
                    event.event_type,
                    event.attempts,
                    status,
                )
                self._record_event_metric(
                    event,
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    status=status,
                    error=error,
                )
                if status == "dead_letter":
                    version = self.store.get_document_version(event.document_version_id)
                    if version is not None and version.status == "active":
                        self.store.record_document_version_error(event.document_version_id, error)
                    else:
                        self.store.fail_document_version(event.document_version_id, error)
                    self.store.fail_document(event.doc_id, error)
                    break
                if wait_for_retries:
                    delay = max(0.0, float(self.settings.outbox_retry_base_seconds)) * (
                        2 ** max(0, event.attempts - 1)
                    )
                    if delay:
                        time.sleep(min(delay, 10.0))
                else:
                    break
            processed += 1
        return processed

    def _dispatch(self, event: OutboxEvent) -> None:
        payload = self._payload(event)
        handlers = {
            "GRAPH_STAGE_VERSION": self._stage_graph,
            "VECTOR_STAGE_VERSION": self._stage_vectors,
            "ACTIVATE_VERSION": self._activate,
            "RETRACT_VERSION": self._retract,
            "REFRESH_ENTITY_VECTOR": self._refresh_entities,
        }
        handlers[event.event_type](event, payload)

    def _stage_graph(self, event: OutboxEvent, payload: dict[str, Any]) -> None:
        nodes = self._nodes(payload)
        assertions = [AssertionRecord.model_validate(item) for item in payload.get("assertions", [])]
        if any(item.document_version_id != event.document_version_id for item in assertions):
            raise ValueError("Graph-stage assertion belongs to a different document version.")
        self.graph_writer.write_graph_version(
            doc_id=event.doc_id,
            document_version_id=event.document_version_id,
            nodes=nodes,
            assertions=assertions,
        )
        counts = self.graph_writer.verify_document_version(event.document_version_id)
        entity_assertions = sum(1 for item in assertions if item.object_kind == "entity")
        expected = {
            "entities": len(nodes),
            "assertions": len(assertions),
            "subject_edges": len(assertions),
            "entity_assertions": entity_assertions,
            "object_edges": entity_assertions,
        }
        if counts != expected:
            raise RuntimeError(f"Neo4j staged-version integrity mismatch: expected {expected}, got {counts}.")

        self.store.stage_document_version(event.document_version_id, "GRAPH_STAGED")
        vector_event = self.make_event(
            event_type="VECTOR_STAGE_VERSION",
            run_id=event.run_id,
            doc_id=event.doc_id,
            document_version_id=event.document_version_id,
            payload={"nodes": nodes, "chunks": payload.get("chunks", [])},
        )
        self.store.enqueue_outbox_event(vector_event)

    def _stage_vectors(self, event: OutboxEvent, payload: dict[str, Any]) -> None:
        chunks = [ChunkVectorInput(**item) for item in payload.get("chunks", [])]
        if any(item.document_version_id != event.document_version_id for item in chunks):
            raise ValueError("Chunk vector input belongs to a different document version.")
        nodes = self._nodes(payload)
        entity_inputs = [
            EntityVectorInput(
                neo4j_node_id=str(node["id"]),
                name=str(node.get("name") or ""),
                entity_type=str(node.get("type") or "Entity"),
                summary=build_entity_summary(
                    name=str(node.get("name") or ""),
                    entity_type=str(node.get("type") or "Entity"),
                    properties=node.get("properties"),
                ),
                document_version_id=event.document_version_id,
                doc_id=event.doc_id,
                active=False,
            )
            for node in nodes
        ]
        self.vector_indexer.upsert_chunk_vectors(chunks)
        self.vector_indexer.upsert_entity_vectors(entity_inputs)
        counts = self.vector_indexer.count_document_version(event.document_version_id)
        expected = {"chunks": len(chunks), "entities": len(entity_inputs)}
        if counts != expected:
            raise RuntimeError(f"Qdrant staged-version integrity mismatch: expected {expected}, got {counts}.")

        self.store.stage_document_version(event.document_version_id, "VECTOR_STAGED")
        previous = self.store.get_active_document_version(event.doc_id)
        activate_event = self.make_event(
            event_type="ACTIVATE_VERSION",
            run_id=event.run_id,
            doc_id=event.doc_id,
            document_version_id=event.document_version_id,
            payload={"previous_version_id": previous.version_id if previous else None},
        )
        self.store.enqueue_outbox_event(activate_event)

    def _activate(self, event: OutboxEvent, payload: dict[str, Any]) -> None:
        previous_version_id = self._optional_text(payload.get("previous_version_id"))
        self.vector_indexer.set_document_version_active(event.document_version_id, True)
        if previous_version_id:
            self.vector_indexer.set_document_version_active(previous_version_id, False)
        self.graph_writer.activate_document_version(event.document_version_id, previous_version_id)

        active = self.store.get_active_document_version(event.doc_id)
        if active is None or active.version_id != event.document_version_id:
            sqlite_previous = self.store.activate_document_version(event.document_version_id)
            if sqlite_previous != previous_version_id:
                raise RuntimeError(
                    "SQLite activation observed a different superseded version "
                    f"({sqlite_previous!r} != {previous_version_id!r})."
                )
        if previous_version_id:
            self.store.enqueue_outbox_event(
                self.make_event(
                    event_type="RETRACT_VERSION",
                    run_id=event.run_id,
                    doc_id=event.doc_id,
                    document_version_id=event.document_version_id,
                    payload={
                        "activated_version_id": event.document_version_id,
                        "retract_version_id": previous_version_id,
                    },
                    discriminator=previous_version_id,
                )
            )
            return

        self._finish_active_version(event.doc_id, event.document_version_id)

    def _retract(self, event: OutboxEvent, payload: dict[str, Any]) -> None:
        retract_version_id = self._optional_text(payload.get("retract_version_id"))
        if retract_version_id is None:
            # Compatibility with events written by the first Refactor 1 trial build.
            retract_version_id = event.document_version_id
        activated_version_id = self._optional_text(payload.get("activated_version_id"))
        if activated_version_id is None:
            raise ValueError("Retraction event is missing activated_version_id.")
        active = self.store.get_active_document_version(event.doc_id)
        if active is None or active.version_id != activated_version_id:
            raise RuntimeError("Retraction owner is not the SQLite-active document version.")
        if active.version_id == retract_version_id:
            raise RuntimeError("Refusing to retract the active document version.")
        self.graph_writer.retract_document_version(retract_version_id)
        self.vector_indexer.delete_document_version(retract_version_id)
        self._finish_active_version(event.doc_id, activated_version_id)

    def _finish_active_version(self, doc_id: str, document_version_id: str) -> None:
        self.store.complete_document_version(document_version_id)
        doc_row = self.store.get_document_by_id(doc_id)
        self.store.finish_document(
            doc_id,
            int(doc_row["word_count"] or 0) if doc_row else 0,
            "DONE",
        )

    def _record_event_metric(
        self,
        event: OutboxEvent,
        *,
        duration_ms: float,
        status: str,
        error: str | None = None,
    ) -> None:
        self.store.record_ingestion_stage_metric(
            run_id=event.run_id,
            stage=event.event_type,
            attempt=event.attempts,
            doc_id=event.doc_id,
            document_version_id=event.document_version_id,
            duration_ms=duration_ms,
            status=status,
            details={"event_id": event.event_id, "error": error},
            metric_id=f"{event.event_id}:{event.attempts}",
        )

    def _refresh_entities(self, event: OutboxEvent, payload: dict[str, Any]) -> None:
        """Refresh selected active entity vectors from authoritative Neo4j state."""
        entity_ids = {str(item) for item in payload.get("entity_ids", []) if str(item)}
        version_entity_ids = {
            str(item["entity_id"])
            for item in self.store.get_doc_entities(
                event.doc_id,
                event.document_version_id,
                active_only=True,
            )
        }
        nodes = self.graph_writer.get_entities_for_doc(event.doc_id)
        allowed_ids = version_entity_ids & entity_ids if entity_ids else version_entity_ids
        nodes = [item for item in nodes if str(item.get("id")) in allowed_ids]
        inputs = [
            EntityVectorInput(
                neo4j_node_id=str(node["id"]),
                name=str(node.get("name") or ""),
                entity_type=str(node.get("type") or "Entity"),
                summary=build_entity_summary(
                    str(node.get("name") or ""),
                    str(node.get("type") or "Entity"),
                    node.get("properties"),
                ),
                document_version_id=event.document_version_id,
                doc_id=event.doc_id,
                active=True,
            )
            for node in nodes
        ]
        self.vector_indexer.upsert_entity_vectors(inputs)

    @staticmethod
    def _payload(event: OutboxEvent) -> dict[str, Any]:
        payload = json.loads(event.payload_json)
        if not isinstance(payload, dict):
            raise ValueError("Outbox payload must be a JSON object.")
        return payload

    @staticmethod
    def _nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw = payload.get("nodes", [])
        if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
            raise ValueError("Outbox nodes must be a list of objects.")
        unique: dict[str, dict[str, Any]] = {}
        for item in raw:
            node_id = str(item.get("id") or "").strip()
            if not node_id:
                raise ValueError("Every graph node requires an id.")
            unique[node_id] = item
        return [unique[key] for key in sorted(unique)]

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        clean = str(value or "").strip()
        return clean or None
