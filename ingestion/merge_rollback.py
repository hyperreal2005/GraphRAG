"""Audited and surgical rollback for an incorrect cross-document merge."""

from __future__ import annotations

import hashlib

from ingestion.graph_writer import GraphWriter
from ingestion.outbox import IngestionOutboxProcessor
from ingestion.state_store import MetadataStore
from ingestion.vector_indexer import QdrantVectorIndexer
from models.refactor import MergeDecision


class MergeRollbackService:
    """Separate one local canonical entity without erasing the original audit record."""

    def __init__(
        self,
        *,
        store: MetadataStore,
        graph_writer: GraphWriter,
        vector_indexer: QdrantVectorIndexer,
        outbox_processor: IngestionOutboxProcessor,
    ) -> None:
        self.store = store
        self.graph_writer = graph_writer
        self.vector_indexer = vector_indexer
        self.outbox_processor = outbox_processor

    def rollback(self, decision_id: str, *, run_id: str) -> MergeDecision:
        """Reverse one merge and refresh only its affected active-version vectors."""
        original = self.store.get_merge_decision(decision_id)
        if original is None:
            raise KeyError(decision_id)
        if original.decision not in {"exact_merge", "auto_merge", "llm_merge"}:
            raise ValueError("Only a successful merge decision can be rolled back.")
        if original.reversed_at is not None:
            raise ValueError("Merge decision has already been reversed.")
        if not original.candidate_entity_id:
            raise ValueError("Merge decision does not identify the merged entity.")
        version = self.store.get_document_version(original.document_version_id)
        if version is None or version.status != "active":
            raise ValueError("Refactor 1 merge rollback is limited to an active document version.")

        restored_id = original.incoming_local_entity_id
        merged_id = original.candidate_entity_id
        if restored_id == merged_id:
            raise ValueError("Merge decision is already separated.")
        reversal_id = hashlib.sha256(
            f"rollback|{decision_id}|{restored_id}".encode("utf-8")
        ).hexdigest()
        reversal = MergeDecision(
            decision_id=reversal_id,
            run_id=run_id,
            doc_id=original.doc_id,
            document_version_id=original.document_version_id,
            incoming_local_entity_id=original.incoming_local_entity_id,
            incoming_name=original.incoming_name,
            incoming_type=original.incoming_type,
            candidate_entity_id=merged_id,
            candidate_name=original.candidate_name,
            candidate_type=original.candidate_type,
            top1_similarity=original.top1_similarity,
            top2_similarity=original.top2_similarity,
            score_margin=original.score_margin,
            type_compatible=original.type_compatible,
            decision="rollback",
            rationale=f"Reverses audited merge decision {decision_id}.",
            reverses_decision_id=decision_id,
        )
        self.store.record_merge_decision(reversal)

        self.graph_writer.rollback_merged_entity(
            doc_id=original.doc_id,
            document_version_id=original.document_version_id,
            local_entity_id=original.incoming_local_entity_id,
            merged_entity_id=merged_id,
            restored_entity_id=restored_id,
            name=original.incoming_name,
            entity_type=original.incoming_type,
        )
        self.store.rollback_version_entity_mapping(
            document_version_id=original.document_version_id,
            local_entity_id=original.incoming_local_entity_id,
            merged_entity_id=merged_id,
            restored_entity_id=restored_id,
            canonical_name=original.incoming_name,
            aliases=[original.incoming_name],
        )
        self.vector_indexer.delete_version_entity_vectors(
            original.document_version_id,
            [merged_id, restored_id],
        )
        refresh = self.outbox_processor.make_event(
            event_type="REFRESH_ENTITY_VECTOR",
            run_id=run_id,
            doc_id=original.doc_id,
            document_version_id=original.document_version_id,
            payload={"entity_ids": [merged_id, restored_id]},
            discriminator=reversal_id,
        )
        self.store.enqueue_outbox_event(refresh)
        self.outbox_processor.reconcile(
            document_version_id=original.document_version_id,
            wait_for_retries=True,
        )
        refreshed = next(
            event
            for event in self.store.list_outbox_events(
                document_version_id=original.document_version_id
            )
            if event.event_id == refresh.event_id
        )
        if refreshed.status != "completed":
            raise RuntimeError(
                f"Merge rollback vector refresh did not complete: {refreshed.last_error or refreshed.status}"
            )
        self.store.mark_merge_decision_reversed(decision_id)
        return reversal

