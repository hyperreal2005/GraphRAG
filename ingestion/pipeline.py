"""Phase 1 ingestion pipeline orchestration."""

from __future__ import annotations

import json
import hashlib
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from config import Settings
from ingestion.chunker import CHUNK_FINGERPRINT_VERSION, SemanticChunker
from ingestion.coref import CoreferenceResolver
from ingestion.content_classifier import classify_content_sequence, is_graph_extractable
from ingestion.parser import DocumentParser, compute_sha256, detect_format
from ingestion.assertion_builder import AssertionBuilder
from ingestion.outbox import IngestionOutboxProcessor
from ingestion.state_store import MetadataStore
from ingestion.vector_indexer import ChunkVectorInput, EntityVectorInput, QdrantVectorIndexer, build_entity_summary
from models.ingestion import (
    DeltaSummary,
    FileIngestionResult,
    IngestionAction,
    IngestionRunSummary,
    ParsedDocument,
)
from models.refactor import DocumentVersion


def utc_now_iso() -> str:
    """Current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


INGESTION_PIPELINE_VERSION = "refactor1-versioned-assertions-v6"
STAGE_ORDER = {"PARSING": 1, "EXTRACTED": 2, "GRAPH_WRITTEN": 3, "VECTORS_WRITTEN": 4, "DONE": 5}


logger = logging.getLogger(__name__)


class PhaseOnePipeline:
    """Implements Stage 1-7 for ingestion plane."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.store = MetadataStore(self.settings.metadata_db_path)
        self.parser = DocumentParser()
        self.coref = CoreferenceResolver(
            profile=self.settings.coref_profile,
            strict=self.settings.strict_coref,
        )
        self.chunker = SemanticChunker(
            target_tokens=self.settings.chunk_target_tokens,
            max_tokens=self.settings.chunk_max_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        self.extractor = None
        self.disambiguator = None
        self.cross_doc_merger = None
        self.graph_writer = None
        self.vector_indexer = None
        self.assertion_builder = None
        self.outbox_processor = None
        if self.settings.enable_graph:
            from ingestion.cross_doc_merger import CrossDocumentMerger
            from ingestion.disambiguator import EntityDisambiguator
            from ingestion.extractor import GeminiExtractor
            from ingestion.graph_writer import GraphWriter

            self._ensure_embedding_metadata()
            self.extractor = GeminiExtractor(self.settings)
            self.disambiguator = EntityDisambiguator(self.settings)
            self.cross_doc_merger = CrossDocumentMerger(self.settings)
            self.graph_writer = GraphWriter(self.settings)
            self.vector_indexer = QdrantVectorIndexer(self.settings)
            self.assertion_builder = AssertionBuilder(self.settings)
            self.outbox_processor = IngestionOutboxProcessor(
                settings=self.settings,
                store=self.store,
                graph_writer=self.graph_writer,
                vector_indexer=self.vector_indexer,
            )

    def ingest_path(self, input_path: Path, recursive: bool = True) -> IngestionRunSummary:
        """Ingest one file or directory."""
        files = self._discover_files(input_path, recursive=recursive)
        if self.settings.enable_graph and self.settings.auto_backfill_vectors:
            self._backfill_graph_written_docs()
        summary = IngestionRunSummary()
        for file_path in files:
            summary.add(self.ingest_file(file_path))
        return summary

    def close(self) -> None:
        """Close external clients owned by this pipeline."""
        resources = [
            self.graph_writer,
            getattr(self.vector_indexer, "_client", None),
            getattr(self.cross_doc_merger, "_qdrant_client", None),
        ]
        seen: set[int] = set()
        for resource in resources:
            if resource is None or id(resource) in seen:
                continue
            seen.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    def ingest_file(self, source_path: Path) -> FileIngestionResult:
        """Ingest one file, preserving the last active version until activation succeeds."""
        started_at = utc_now_iso()
        run_id = str(uuid.uuid4())
        absolute_path = source_path.resolve()
        source_key = str(absolute_path)
        doc_format = detect_format(absolute_path)
        checksum = compute_sha256(absolute_path)
        existing_row = self.store.get_document_by_source(source_key)
        target_stage = "DONE" if self.settings.enable_graph else "PARSING"
        pipeline_fingerprint = self._pipeline_fingerprint(target_stage)

        if existing_row is not None and self._is_active_generation_current(
            existing_row,
            checksum=checksum,
            target_stage=target_stage,
            pipeline_fingerprint=pipeline_fingerprint,
        ):
            doc_id = str(existing_row["doc_id"])
            active_version = self.store.get_active_document_version(doc_id)
            self.store.log_ingestion_run(
                run_id=run_id,
                doc_id=doc_id,
                started_at=started_at,
                finished_at=utc_now_iso(),
                status="skipped",
                document_version_id=active_version.version_id if active_version else None,
            )
            return FileIngestionResult(
                source_path=source_key,
                action=IngestionAction.UNCHANGED,
                doc_id=doc_id,
                skipped=True,
            )

        action = IngestionAction.NEW if existing_row is None else IngestionAction.CHANGED
        doc_id = str(uuid.uuid4()) if existing_row is None else str(existing_row["doc_id"])
        document_version_id: str | None = None

        try:
            self.store.start_document(
                doc_id,
                source_key,
                doc_format,
                checksum,
                target_stage=target_stage,
                pipeline_fingerprint=pipeline_fingerprint,
                run_id=run_id,
            )
            if self.settings.enable_graph:
                document_version_id = str(uuid.uuid4())
                self.store.create_document_version(
                    DocumentVersion(
                        version_id=document_version_id,
                        doc_id=doc_id,
                        checksum=checksum,
                        pipeline_fingerprint=pipeline_fingerprint,
                        generative_model=self.settings.gemini_model,
                        embedding_model=self.settings.embedding_model_id,
                        embedding_dim=self.settings.embedding_dim,
                        embedding_prompt_version=self.settings.embedding_prompt_version,
                        embedding_schema_version=self.settings.embedding_version,
                        embedding_normalization_policy=self.settings.embedding_normalization_policy,
                        document_embedding_instruction=self.settings.embedding_document_prefix,
                        query_embedding_instruction=self.settings.embedding_query_prefix,
                        extraction_prompt_version=self.settings.extraction_prompt_version,
                        graph_schema_version=self.settings.graph_schema_version,
                    )
                )

            parse_started = time.perf_counter()
            parsed = self.parser.parse_document(
                source_path=absolute_path,
                doc_id=doc_id,
                checksum=checksum,
            )
            resolved = self.coref.resolve_document(parsed)
            chunks = self.chunker.chunk_document(resolved)
            content_kinds = (
                classify_content_sequence([chunk.section_heading for chunk in chunks])
                if self.settings.graph_exclude_reference_sections
                else ["body" for _ in chunks]
            )
            chunks = [
                replace(
                    chunk,
                    content_kind=content_kind,
                )
                for chunk, content_kind in zip(chunks, content_kinds)
            ]
            active_version = self.store.get_active_document_version(doc_id)
            old_hash_to_chunk_id = self.store.get_chunk_ids_by_hash(
                doc_id,
                active_version.version_id if active_version else None,
                active_only=active_version is not None,
            )

            if document_version_id:
                chunks = [
                    replace(
                        chunk,
                        chunk_id=str(
                            uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                f"graphrag:{document_version_id}:{chunk.position}:{chunk.chunk_hash}",
                            )
                        ),
                    )
                    for chunk in chunks
                ]
            else:
                chunks = [
                    replace(chunk, chunk_id=old_hash_to_chunk_id.get(chunk.chunk_hash, chunk.chunk_id))
                    for chunk in chunks
                ]

            new_hashes = {chunk.chunk_hash for chunk in chunks}
            old_hashes = set(old_hash_to_chunk_id)
            delta = DeltaSummary(
                added_chunk_hashes=sorted(new_hashes - old_hashes),
                removed_chunk_hashes=sorted(old_hashes - new_hashes),
                unchanged_chunk_hashes=sorted(new_hashes & old_hashes),
            )
            self.store.update_pipeline_stage(
                doc_id,
                "PARSING",
                word_count=resolved.metadata.word_count,
            )
            self._write_artifacts(
                run_id=run_id,
                parsed_document=resolved,
                chunks=chunks,
                action=action,
                delta=delta,
            )
            self.store.record_ingestion_stage_metric(
                run_id=run_id,
                stage="PARSE_COREF_CHUNK",
                attempt=1,
                doc_id=doc_id,
                document_version_id=document_version_id,
                duration_ms=(time.perf_counter() - parse_started) * 1000.0,
                status="completed",
                details={
                    "chunks": len(chunks),
                    "reference_chunks": sum(
                        1 for chunk in chunks if chunk.content_kind == "reference"
                    ),
                },
            )

            entities_created = 0
            entities_merged = 0
            relations_created = 0
            if document_version_id:
                entities_created, entities_merged, relations_created = self._stage_graph_generation(
                    run_id=run_id,
                    doc_id=doc_id,
                    document_version_id=document_version_id,
                    doc_format=doc_format,
                    chunks=chunks,
                    active_version_id=active_version.version_id if active_version else None,
                )
                current = self.store.get_active_document_version(doc_id)
                if current is None or current.version_id != document_version_id:
                    failed = self.store.get_document_version(document_version_id)
                    detail = failed.last_error if failed else "candidate version disappeared"
                    raise RuntimeError(f"Document version activation did not complete: {detail}")
                self.store.set_chunk_embedding_version(
                    [chunk.chunk_id for chunk in chunks],
                    self.settings.embedding_version,
                )
            else:
                self.store.delete_chunks_by_hashes(doc_id, delta.removed_chunk_hashes)
                self.store.upsert_chunks(chunks)
                self.store.finish_document(doc_id, resolved.metadata.word_count, target_stage)

            self.store.log_ingestion_run(
                run_id=run_id,
                doc_id=doc_id,
                started_at=started_at,
                finished_at=utc_now_iso(),
                status="complete",
                entities_created=entities_created,
                relations_created=relations_created,
                entities_merged=entities_merged,
                document_version_id=document_version_id,
            )
            return FileIngestionResult(
                source_path=source_key,
                action=action,
                doc_id=doc_id,
                chunk_count=len(chunks),
                skipped=False,
                delta=delta,
            )
        except Exception as exc:
            if document_version_id:
                self.store.fail_document_version(document_version_id, str(exc))
            self.store.fail_document(doc_id=doc_id, error_message=str(exc))
            self.store.log_ingestion_run(
                run_id=run_id,
                doc_id=doc_id,
                started_at=started_at,
                finished_at=utc_now_iso(),
                status="failed",
                document_version_id=document_version_id,
            )
            return FileIngestionResult(
                source_path=source_key,
                action=action,
                doc_id=doc_id,
                skipped=False,
                error=str(exc),
            )

    def _stage_graph_generation(
        self,
        *,
        run_id: str,
        doc_id: str,
        document_version_id: str,
        doc_format: str,
        chunks: list,
        active_version_id: str | None,
    ) -> tuple[int, int, int]:
        """Extract and durably stage a complete immutable graph/vector version."""
        if any(
            component is None
            for component in (
                self.extractor,
                self.disambiguator,
                self.cross_doc_merger,
                self.assertion_builder,
                self.outbox_processor,
            )
        ):
            raise RuntimeError("Refactor 1 ingestion components are not initialized.")

        def run_extract(chunk):  # noqa: ANN001, ANN202
            raw_result = self.extractor.extract(chunk.text, doc_format)
            return chunk, self._scope_extraction_result(chunk.chunk_id, raw_result)

        extractable_chunks = [
            chunk for chunk in chunks if is_graph_extractable(chunk.content_kind)
        ]
        skipped_reference_chunks = len(chunks) - len(extractable_chunks)
        stats_before = self._extractor_stats()
        extraction_started = time.perf_counter()
        max_workers = max(1, int(self.settings.gemini_max_workers))
        extraction_results = []
        completed_chunks = 0
        progress_interval = max(1, int(self.settings.ingestion_progress_interval_chunks))
        try:
            if max_workers == 1 or len(extractable_chunks) <= 1:
                for chunk in extractable_chunks:
                    extraction_results.append(run_extract(chunk))
                    completed_chunks += 1
                    self._log_extraction_progress(
                        completed_chunks,
                        len(extractable_chunks),
                        progress_interval,
                    )
            else:
                with ThreadPoolExecutor(
                    max_workers=min(max_workers, len(extractable_chunks))
                ) as executor:
                    futures = [executor.submit(run_extract, chunk) for chunk in extractable_chunks]
                    try:
                        for future in as_completed(futures):
                            extraction_results.append(future.result())
                            completed_chunks += 1
                            self._log_extraction_progress(
                                completed_chunks,
                                len(extractable_chunks),
                                progress_interval,
                            )
                    except Exception:
                        for pending in futures:
                            pending.cancel()
                        raise
        except Exception as exc:
            self._record_extraction_metric(
                run_id=run_id,
                doc_id=doc_id,
                document_version_id=document_version_id,
                started=extraction_started,
                status="failed",
                stats_before=stats_before,
                chunks_completed=completed_chunks,
                chunks_total=len(extractable_chunks),
                reference_chunks_skipped=skipped_reference_chunks,
                error=str(exc),
            )
            raise

        self._record_extraction_metric(
            run_id=run_id,
            doc_id=doc_id,
            document_version_id=document_version_id,
            started=extraction_started,
            status="completed",
            stats_before=stats_before,
            chunks_completed=completed_chunks,
            chunks_total=len(extractable_chunks),
            reference_chunks_skipped=skipped_reference_chunks,
        )
        extraction_results.sort(key=lambda item: (item[0].position, item[0].chunk_id))

        extraction_entities = []
        extraction_relations: list[dict[str, object]] = []
        raw_entity_chunk_ids: dict[str, str] = {}
        raw_mentions: list[tuple[str, str, str, float]] = []
        for chunk, result in extraction_results:
            extraction_entities.extend(result.entities)
            for entity in result.entities:
                raw_entity_chunk_ids[entity.entity_id] = chunk.chunk_id
                raw_mentions.append((entity.entity_id, chunk.chunk_id, doc_id, entity.confidence))
            for relation in result.relations:
                extraction_relations.append(
                    {
                        "head_id": relation.head_id,
                        "tail_id": relation.tail_id,
                        "relation": relation.relation,
                        "raw_relation": relation.raw_relation,
                        "properties": relation.properties,
                        "weight": relation.confidence,
                        "source_chunk": chunk.chunk_id,
                        "valid_from": relation.valid_from,
                        "valid_until": relation.valid_until,
                    }
                )

        resolution_started = time.perf_counter()
        merge_stats_before = self._merger_stats()
        existing_entities = (
            self.store.get_local_entities_for_version(doc_id, active_version_id)
            if active_version_id is not None
            else []
        )
        disamb = self.disambiguator.disambiguate(extraction_entities, existing_entities)
        canonical_nodes: dict[str, dict[str, object]] = {}
        type_counts: dict[str, dict[str, int]] = {}
        confidence_values: dict[str, list[float]] = {}
        for cluster in disamb.canonical_entities:
            aliases = list(dict.fromkeys([cluster.canonical_name, *cluster.aliases]))
            canonical_nodes[cluster.entity_id] = {
                "id": cluster.entity_id,
                "name": cluster.canonical_name,
                "type": "Entity",
                "aliases": aliases,
                "properties": {},
                "confidence": 0.0,
            }
            type_counts[cluster.entity_id] = {}
            confidence_values[cluster.entity_id] = []

        for entity in extraction_entities:
            local_id = disamb.entity_id_map.get(entity.entity_id)
            if not local_id or local_id not in canonical_nodes:
                continue
            node = canonical_nodes[local_id]
            display_properties = node.get("properties")
            if isinstance(display_properties, dict):
                for key, value in entity.properties.items():
                    display_properties.setdefault(key, value)
            aliases = node.get("aliases")
            if isinstance(aliases, list) and entity.name not in aliases:
                aliases.append(entity.name)
            counts = type_counts[local_id]
            counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1
            confidence_values[local_id].append(entity.confidence)
        for local_id, counts in type_counts.items():
            if counts:
                canonical_nodes[local_id]["type"] = max(counts, key=counts.get)
            values = confidence_values[local_id]
            if values:
                canonical_nodes[local_id]["confidence"] = sum(values) / len(values)

        merge_result = self.cross_doc_merger.resolve(
            list(canonical_nodes.values()),
            run_id=run_id,
            doc_id=doc_id,
            document_version_id=document_version_id,
            store=self.store,
        )
        local_to_final = {node_id: node_id for node_id in canonical_nodes}
        local_to_final.update(merge_result.local_to_final_id)
        canonical_nodes = self._remap_canonical_nodes(canonical_nodes, local_to_final)
        raw_to_final = {
            raw_id: local_to_final.get(local_id, local_id)
            for raw_id, local_id in disamb.entity_id_map.items()
        }

        mapped_mentions = sorted(
            {
                (
                    raw_to_final.get(entity_id, entity_id),
                    chunk_id,
                    mention_doc_id,
                    confidence,
                    disamb.entity_id_map.get(entity_id, entity_id),
                )
                for entity_id, chunk_id, mention_doc_id, confidence in raw_mentions
            }
        )
        mentions_by_chunk: dict[str, list[str]] = {}
        for entity_id, chunk_id, _, _, _ in mapped_mentions:
            mentions_by_chunk.setdefault(chunk_id, []).append(entity_id)

        mapped_relations: list[dict[str, object]] = []
        for relation in extraction_relations:
            head_id = raw_to_final.get(str(relation["head_id"]), str(relation["head_id"]))
            tail_id = raw_to_final.get(str(relation["tail_id"]), str(relation["tail_id"]))
            if head_id not in canonical_nodes or tail_id not in canonical_nodes:
                raise ValueError("Canonical relation endpoint is missing after cross-document resolution.")
            raw_relation = str(relation.get("raw_relation") or relation.get("relation") or "").strip()
            if not raw_relation:
                raise ValueError("Extracted relation has an empty predicate.")
            mapped_relations.append(
                {
                    **relation,
                    "head_id": head_id,
                    "tail_id": tail_id,
                    "head_local_id": disamb.entity_id_map.get(str(relation["head_id"])),
                    "tail_local_id": disamb.entity_id_map.get(str(relation["tail_id"])),
                    "relation": str(relation.get("relation") or raw_relation),
                    "raw_relation": raw_relation,
                }
            )

        assertions = self.assertion_builder.build(
            doc_id=doc_id,
            document_version_id=document_version_id,
            mapped_relations=mapped_relations,
            extracted_entities=extraction_entities,
            raw_to_final_entity_id=raw_to_final,
            raw_to_local_entity_id=disamb.entity_id_map,
            raw_entity_chunk_ids=raw_entity_chunk_ids,
        )
        merge_stats_after = self._merger_stats()

        def merge_delta(key: str) -> int:
            return max(0, merge_stats_after.get(key, 0) - merge_stats_before.get(key, 0))

        merge_call_count = merge_delta("call_count")
        self.store.record_ingestion_stage_metric(
            run_id=run_id,
            stage="DISAMBIGUATE_MERGE_ASSERT",
            attempt=1,
            doc_id=doc_id,
            document_version_id=document_version_id,
            duration_ms=(time.perf_counter() - resolution_started) * 1000.0,
            status="completed",
            provider="google" if merge_call_count else None,
            model_id=self.settings.gemini_model if merge_call_count else None,
            input_tokens=merge_delta("input_tokens"),
            output_tokens=merge_delta("output_tokens"),
            total_tokens=merge_delta("total_tokens"),
            call_count=merge_call_count,
            details={
                "entities_extracted": len(extraction_entities),
                "canonical_entities": len(canonical_nodes),
                "assertions": len(assertions),
                "auto_or_llm_merges": merge_result.merged_total,
                "llm_adjudication_calls": merge_call_count,
            },
        )
        node_rows = [canonical_nodes[key] for key in sorted(canonical_nodes)]
        doc_entities = [
            {
                "doc_id": doc_id,
                "entity_id": node["id"],
                "canonical_name": node["name"],
                "aliases": node.get("aliases", []),
            }
            for node in node_rows
        ]
        chunk_inputs = [
            ChunkVectorInput(
                chunk_id=chunk.chunk_id,
                doc_id=doc_id,
                text=chunk.text,
                section_heading=chunk.section_heading,
                page=chunk.page_number,
                entity_ids_mentioned=mentions_by_chunk.get(chunk.chunk_id, []),
                document_version_id=document_version_id,
                active=False,
                content_kind=chunk.content_kind,
            )
            for chunk in chunks
        ]
        payload = self.outbox_processor.serialize_initial_payload(
            nodes=node_rows,
            assertions=assertions,
            chunks=chunk_inputs,
        )
        initial_event = self.outbox_processor.make_event(
            event_type="GRAPH_STAGE_VERSION",
            run_id=run_id,
            doc_id=doc_id,
            document_version_id=document_version_id,
            payload=payload,
        )
        sqlite_stage_started = time.perf_counter()
        self.store.stage_version_records(
            document_version_id=document_version_id,
            chunks=chunks,
            mentions=mapped_mentions,
            doc_entities=doc_entities,
            initial_event=initial_event,
        )
        self.store.record_ingestion_stage_metric(
            run_id=run_id,
            stage="SQLITE_STAGE_VERSION",
            attempt=1,
            doc_id=doc_id,
            document_version_id=document_version_id,
            duration_ms=(time.perf_counter() - sqlite_stage_started) * 1000.0,
            status="completed",
            details={
                "chunks": len(chunks),
                "mentions": len(mapped_mentions),
                "entities": len(doc_entities),
                "assertions": len(assertions),
            },
        )
        self.store.update_pipeline_stage(doc_id, "EXTRACTED")
        self.outbox_processor.reconcile(
            document_version_id=document_version_id,
            wait_for_retries=True,
            max_events=100,
        )
        incomplete_events = [
            event
            for event in self.store.list_outbox_events(
                document_version_id=document_version_id
            )
            if event.status != "completed"
        ]
        if incomplete_events:
            states = ", ".join(
                f"{event.event_type}:{event.status}" for event in incomplete_events
            )
            raise RuntimeError(f"Document version reconciliation is incomplete: {states}.")
        completed_version = self.store.get_document_version(document_version_id)
        if (
            completed_version is None
            or completed_version.status != "active"
            or completed_version.last_completed_stage != "DONE"
        ):
            raise RuntimeError("Document version did not reach the DONE activation checkpoint.")
        return merge_result.new_nodes, merge_result.merged_total, len(assertions)

    def _extractor_stats(self) -> dict[str, int]:
        snapshot = getattr(self.extractor, "snapshot_stats", None)
        if not callable(snapshot):
            return {}
        stats = snapshot()
        return {
            key: int(getattr(stats, key, 0) or 0)
            for key in (
                "entities_count",
                "relations_count",
                "call_count",
                "input_tokens",
                "output_tokens",
                "total_tokens",
            )
        }

    def _merger_stats(self) -> dict[str, int]:
        snapshot = getattr(self.cross_doc_merger, "snapshot_stats", None)
        if not callable(snapshot):
            return {}
        stats = snapshot()
        return {
            key: int(getattr(stats, key, 0) or 0)
            for key in (
                "call_count",
                "input_tokens",
                "output_tokens",
                "total_tokens",
            )
        }

    def _record_extraction_metric(
        self,
        *,
        run_id: str,
        doc_id: str,
        document_version_id: str,
        started: float,
        status: str,
        stats_before: dict[str, int],
        chunks_completed: int,
        chunks_total: int,
        reference_chunks_skipped: int,
        error: str | None = None,
    ) -> None:
        stats_after = self._extractor_stats()

        def delta(key: str) -> int:
            return max(0, stats_after.get(key, 0) - stats_before.get(key, 0))

        self.store.record_ingestion_stage_metric(
            run_id=run_id,
            stage="GEMINI_EXTRACTION",
            attempt=1,
            doc_id=doc_id,
            document_version_id=document_version_id,
            duration_ms=(time.perf_counter() - started) * 1000.0,
            status=status,
            provider="google",
            model_id=self.settings.gemini_model,
            input_tokens=delta("input_tokens"),
            output_tokens=delta("output_tokens"),
            total_tokens=delta("total_tokens"),
            call_count=delta("call_count"),
            details={
                "chunks_completed": chunks_completed,
                "chunks_total": chunks_total,
                "reference_chunks_skipped": reference_chunks_skipped,
                "entities": delta("entities_count"),
                "relations": delta("relations_count"),
                "error": error,
            },
        )

    @staticmethod
    def _log_extraction_progress(completed: int, total: int, interval: int) -> None:
        if completed == total or completed % max(1, interval) == 0:
            logger.info(
                "ingestion_extraction_progress completed=%d total=%d",
                completed,
                total,
            )

    def _pipeline_fingerprint(self, target_stage: str) -> str:
        payload = {
            "pipeline_version": INGESTION_PIPELINE_VERSION,
            "chunk_fingerprint_version": CHUNK_FINGERPRINT_VERSION,
            "target_stage": target_stage,
            "coref_profile": self.settings.coref_profile,
            "strict_coref": self.settings.strict_coref,
            "chunk_target_tokens": self.settings.chunk_target_tokens,
            "chunk_max_tokens": self.settings.chunk_max_tokens,
            "chunk_overlap_tokens": self.settings.chunk_overlap_tokens,
            "gemini_model": self.settings.gemini_model if self.settings.enable_graph else None,
            "extraction_prompt_version": self.settings.extraction_prompt_version if self.settings.enable_graph else None,
            "graph_exclude_reference_sections": (
                self.settings.graph_exclude_reference_sections
                if self.settings.enable_graph
                else None
            ),
            "disamb_model": self.settings.disamb_model if self.settings.enable_graph else None,
            "disamb_threshold": self.settings.disamb_threshold if self.settings.enable_graph else None,
            "embedding_model_id": self.settings.embedding_model_id if self.settings.enable_graph else None,
            "embedding_version": self.settings.embedding_version if self.settings.enable_graph else None,
            "embedding_prompt_version": self.settings.embedding_prompt_version if self.settings.enable_graph else None,
            "embedding_normalization_policy": self.settings.embedding_normalization_policy if self.settings.enable_graph else None,
            "embedding_document_prefix": self.settings.embedding_document_prefix if self.settings.enable_graph else None,
            "embedding_query_prefix": self.settings.embedding_query_prefix if self.settings.enable_graph else None,
            "graph_schema_version": self.settings.graph_schema_version if self.settings.enable_graph else None,
            "cross_doc_auto_merge_threshold": self.settings.cross_doc_auto_merge_threshold if self.settings.enable_graph else None,
            "cross_doc_auto_merge_margin": self.settings.cross_doc_auto_merge_margin if self.settings.enable_graph else None,
            "cross_doc_llm_confidence_threshold": self.settings.cross_doc_llm_confidence_threshold if self.settings.enable_graph else None,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _is_active_generation_current(
        row,  # noqa: ANN001
        *,
        checksum: str,
        target_stage: str,
        pipeline_fingerprint: str,
    ) -> bool:
        active_checksum = str(row["active_checksum"] or row["checksum"] or "")
        active_fingerprint = str(row["active_pipeline_fingerprint"] or "")
        completed_stage = str(row["last_completed_stage"] or row["pipeline_stage"] or "")
        return (
            str(row["status"] or "") == "complete"
            and active_checksum == checksum
            and active_fingerprint == pipeline_fingerprint
            and STAGE_ORDER.get(completed_stage, 0) >= STAGE_ORDER[target_stage]
        )

    @staticmethod
    def _needs_full_reprocess(row, checksum: str, pipeline_fingerprint: str) -> bool:  # noqa: ANN001
        if row is None:
            return False
        active_checksum = str(row["active_checksum"] or row["checksum"] or "")
        desired_checksum = str(row["desired_checksum"] or "")
        active_fingerprint = str(row["active_pipeline_fingerprint"] or "")
        same_attempted_content = checksum in {active_checksum, desired_checksum}
        return same_attempted_content and (
            str(row["status"] or "") != "complete" or active_fingerprint != pipeline_fingerprint
        )

    @staticmethod
    def _scope_extraction_result(chunk_id: str, result):  # noqa: ANN001, ANN205
        """Namespace model-local entity IDs and validate relation endpoints."""
        raw_to_scoped: dict[str, str] = {}
        scoped_entities = []
        for entity in result.entities:
            raw_id = str(entity.entity_id or "").strip()
            if not raw_id:
                raise ValueError(f"Extraction returned an entity without an ID for chunk {chunk_id}.")
            if raw_id in raw_to_scoped:
                raise ValueError(f"Extraction returned duplicate entity ID {raw_id!r} for chunk {chunk_id}.")
            scoped_id = f"{chunk_id}::{raw_id}"
            raw_to_scoped[raw_id] = scoped_id
            scoped_entities.append(entity.model_copy(update={"entity_id": scoped_id}))

        scoped_relations = []
        for relation in result.relations:
            raw_head = str(relation.head_id or "").strip()
            raw_tail = str(relation.tail_id or "").strip()
            if raw_head not in raw_to_scoped or raw_tail not in raw_to_scoped:
                raise ValueError(
                    "Extraction relation references an entity outside its chunk: "
                    f"{raw_head!r} -> {raw_tail!r} in chunk {chunk_id}."
                )
            scoped_relations.append(
                relation.model_copy(
                    update={
                        "head_id": raw_to_scoped[raw_head],
                        "tail_id": raw_to_scoped[raw_tail],
                    }
                )
            )

        return result.model_copy(update={"entities": scoped_entities, "relations": scoped_relations})

    def _discover_files(self, input_path: Path, recursive: bool) -> list[Path]:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        if input_path.is_file():
            detect_format(input_path)
            return [input_path]

        pattern = "**/*" if recursive else "*"
        candidates = [
            path
            for path in input_path.glob(pattern)
            if path.is_file() and path.suffix.lower().lstrip(".") in self.settings.supported_formats
        ]
        return sorted(candidates)

    def _write_artifacts(
        self,
        run_id: str,
        parsed_document: ParsedDocument,
        chunks: list,
        action: IngestionAction,
        delta: DeltaSummary,
    ) -> None:
        run_dir = self.settings.artifact_dir / parsed_document.doc_id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        (run_dir / "parsed_document.json").write_text(
            json.dumps(parsed_document.to_dict(), indent=2),
            encoding="utf-8",
        )
        (run_dir / "chunks.json").write_text(
            json.dumps([chunk.to_dict() for chunk in chunks], indent=2),
            encoding="utf-8",
        )
        (run_dir / "delta.json").write_text(
            json.dumps(delta.to_dict(), indent=2),
            encoding="utf-8",
        )
        (run_dir / "run_summary.json").write_text(
            json.dumps(
                {
                    "action": action.value,
                    "generated_at": utc_now_iso(),
                    "chunk_count": len(chunks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _backfill_graph_written_docs(self) -> None:
        """Backfill vectors for complete docs that reached GRAPH_WRITTEN before Stage 7 existed."""
        if self.graph_writer is None or self.vector_indexer is None:
            return
        rows = self.store.list_documents(status="complete", pipeline_stage="GRAPH_WRITTEN")
        for row in rows:
            doc_id = str(row["doc_id"])
            try:
                chunks = self.store.get_chunks_for_doc(doc_id)
                mentions_by_chunk = self.store.get_entity_mentions_map(doc_id)
                chunk_inputs = [
                    ChunkVectorInput(
                        chunk_id=str(chunk["chunk_id"]),
                        doc_id=str(chunk["doc_id"]),
                        text=str(chunk["text"] or ""),
                        section_heading=str(chunk["section"]) if chunk["section"] is not None else None,
                        page=int(chunk["page"]) if chunk["page"] is not None else None,
                        entity_ids_mentioned=mentions_by_chunk.get(str(chunk["chunk_id"]), []),
                    )
                    for chunk in chunks
                ]
                entity_rows = self.graph_writer.get_entities_for_doc(doc_id)
                entity_inputs = [
                    EntityVectorInput(
                        neo4j_node_id=str(entity_row.get("id") or ""),
                        name=str(entity_row.get("name") or ""),
                        entity_type=str(entity_row.get("type") or "Entity"),
                        summary=build_entity_summary(
                            name=str(entity_row.get("name") or ""),
                            entity_type=str(entity_row.get("type") or "Entity"),
                            properties=entity_row.get("properties"),
                        ),
                    )
                    for entity_row in entity_rows
                    if str(entity_row.get("id") or "").strip()
                ]
                self.vector_indexer.upsert_chunk_vectors(chunk_inputs)
                self.vector_indexer.upsert_entity_vectors(entity_inputs)
                self.store.set_chunk_embedding_version(
                    [item.chunk_id for item in chunk_inputs],
                    self.settings.embedding_version,
                )
                self.store.update_pipeline_stage(doc_id, "VECTORS_WRITTEN")
                self.store.finish_document(
                    doc_id=doc_id,
                    word_count=int(row["word_count"] or 0),
                    pipeline_stage="DONE",
                )
            except Exception as exc:
                self.store.fail_document(doc_id=doc_id, error_message=f"Vector backfill failed: {exc}")
                raise RuntimeError(f"Vector backfill failed for doc_id={doc_id}: {exc}") from exc

    def _ensure_embedding_metadata(self) -> None:
        expected = {
            "embedding_model_id": self.settings.embedding_model_id,
            "embedding_dim": str(self.settings.embedding_dim),
            "embedding_version": self.settings.embedding_version,
            "embedding_prompt_version": self.settings.embedding_prompt_version,
            "embedding_normalization_policy": self.settings.embedding_normalization_policy,
            "embedding_document_prefix": self.settings.embedding_document_prefix,
            "embedding_query_prefix": self.settings.embedding_query_prefix,
        }
        stored_model = self.store.get_system_metadata("embedding_model_id")
        if stored_model is None:
            for key, value in expected.items():
                self.store.set_system_metadata(key, value)
            return
        mismatches = {
            key: (self.store.get_system_metadata(key), value)
            for key, value in expected.items()
            if self.store.get_system_metadata(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Embedding corpus metadata mismatch. Rebuild vectors before ingestion: "
                f"{mismatches}"
            )

    def _remap_canonical_nodes(
        self,
        canonical_nodes: dict[str, dict[str, object]],
        local_to_final_map: dict[str, str],
    ) -> dict[str, dict[str, object]]:
        merged: dict[str, dict[str, object]] = {}
        for local_id, node in canonical_nodes.items():
            final_id = str(local_to_final_map.get(local_id, local_id))
            existing = merged.get(final_id)
            if existing is None:
                node_copy = dict(node)
                node_copy["id"] = final_id
                node_copy["aliases"] = list(node.get("aliases", []))
                node_copy["properties"] = dict(node.get("properties", {}))
                merged[final_id] = node_copy
                continue

            existing_aliases = set(str(item) for item in existing.get("aliases", []))
            for alias in node.get("aliases", []):
                alias_text = str(alias)
                if alias_text not in existing_aliases:
                    existing.setdefault("aliases", []).append(alias_text)
                    existing_aliases.add(alias_text)

            existing_props = existing.get("properties", {})
            if not isinstance(existing_props, dict):
                existing_props = {}
            incoming_props = node.get("properties", {})
            if isinstance(incoming_props, dict):
                for key, value in incoming_props.items():
                    if key not in existing_props:
                        existing_props[key] = value
            existing["properties"] = existing_props

            existing_type = str(existing.get("type", "Entity"))
            incoming_type = str(node.get("type", "Entity"))
            if existing_type.strip().lower() == "entity" and incoming_type.strip():
                existing["type"] = incoming_type

            existing_name = str(existing.get("name", "")).strip()
            incoming_name = str(node.get("name", "")).strip()
            if (not existing_name or existing_name.lower() == "entity") and incoming_name:
                existing["name"] = incoming_name

            existing_conf = float(existing.get("confidence", 0.0))
            incoming_conf = float(node.get("confidence", 0.0))
            existing["confidence"] = max(existing_conf, incoming_conf)
        return merged
