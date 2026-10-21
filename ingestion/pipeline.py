"""Phase 1 ingestion pipeline orchestration."""

from __future__ import annotations

import json
import hashlib
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from config import Settings
from ingestion.chunker import CHUNK_FINGERPRINT_VERSION, SemanticChunker
from ingestion.coref import ANNOTATION_VERSION as COREF_ANNOTATION_VERSION
from ingestion.coref import CoreferenceResolver
from ingestion.document_cards import CorpusCatalog, fallback_card
from ingestion.extractor import ROW_BATCH_VERSION
from ingestion.content_classifier import classify_content_sequence, is_graph_extractable
from ingestion.entity_types import display_type, dominant_coarse_class
from ingestion.coverage import coverage_report
from ingestion.parser import DocumentParser, compute_sha256, detect_format
from ingestion.section_grouper import group_sections
from observability import tracing
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
from models.refactor import AssertionRecord, DocumentVersion


def utc_now_iso() -> str:
    """Current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


INGESTION_PIPELINE_VERSION = "refactor2-grouped-sections-v1"

# Fingerprint keys added after corpora were already stamped without them. A stored
# fingerprint may predate any of these, so reproducing one must be tried without them too.
FINGERPRINT_KEYS_ADDED_LATER = ("chunk_table_block_target_tokens",)


def pipeline_fingerprint_payload(settings: Settings, target_stage: str) -> dict[str, object]:
    """Every setting that decides what an ingest writes. A document stamped with a different
    fingerprint is re-ingested, so a setting missing here can change the output silently.

    `chunk_table_block_target_tokens` was missing until 2026-09-27: one chunk per table row
    and merged table blocks produce different chunks from the same file, yet corpora built
    each way carried the same pipeline identity (REFACTOR3 WP20).
    """
    graph = settings.enable_graph
    payload = _base_fingerprint_payload(settings, target_stage, graph)
    # Present only when coreference annotation is on, so every corpus built while it was a
    # no-op keeps its identity, and a corpus built with it gets a different one.
    if settings.coref_enabled:
        payload["coref_annotation"] = COREF_ANNOTATION_VERSION
    # Likewise present only when on: batched row extraction writes different assertions.
    if settings.extraction_table_batch_tokens > 0:
        payload["extraction_table_batch"] = (
            f"{ROW_BATCH_VERSION}:{settings.extraction_table_batch_tokens}"
            f"/{settings.extraction_table_batch_max_rows}"
        )
    # ROADMAP E1: the partial-name rule and the permutation guard change which mentions become
    # one entity, so a corpus built with them is a different corpus. Present only when on.
    if graph and (settings.disamb_partial_name_classes or settings.disamb_permutation_guard):
        payload["disamb_rules"] = {
            "partial_name_classes": sorted(settings.disamb_partial_name_classes),
            "permutation_guard": bool(settings.disamb_permutation_guard),
        }
    # Likewise: the relaxed lexical check changes which cross-document merges are made.
    if graph and settings.cross_doc_relaxed_lexical_support:
        payload["cross_doc_relaxed_lexical_support"] = True
    return payload


def _base_fingerprint_payload(settings: Settings, target_stage: str, graph: bool) -> dict[str, object]:
    return {
        "pipeline_version": INGESTION_PIPELINE_VERSION,
        "chunk_fingerprint_version": CHUNK_FINGERPRINT_VERSION,
        "target_stage": target_stage,
        "coref_profile": settings.coref_profile,
        "strict_coref": settings.strict_coref,
        "chunk_target_tokens": settings.chunk_target_tokens,
        "chunk_max_tokens": settings.chunk_max_tokens,
        "chunk_overlap_tokens": settings.chunk_overlap_tokens,
        "chunk_table_block_target_tokens": settings.chunk_table_block_target_tokens,
        "section_group_enabled": settings.section_group_enabled,
        "section_group_target_tokens": settings.section_group_target_tokens,
        "section_group_max_tokens": settings.section_group_max_tokens,
        "section_group_page_gap_tolerance": settings.section_group_page_gap_tolerance,
        # Named for when one model did everything; it has always meant the model ingestion
        # extracts with, which is now `ingestion_model` (same value, so no stamp changed).
        "gemini_model": settings.ingestion_model if graph else None,
        "extraction_prompt_version": settings.extraction_prompt_version if graph else None,
        "graph_exclude_reference_sections": settings.graph_exclude_reference_sections if graph else None,
        "disamb_model": settings.disamb_model if graph else None,
        "disamb_threshold": settings.disamb_threshold if graph else None,
        "embedding_model_id": settings.embedding_model_id if graph else None,
        "embedding_version": settings.embedding_version if graph else None,
        "embedding_prompt_version": settings.embedding_prompt_version if graph else None,
        "embedding_normalization_policy": settings.embedding_normalization_policy if graph else None,
        "embedding_document_prefix": settings.embedding_document_prefix if graph else None,
        "embedding_query_prefix": settings.embedding_query_prefix if graph else None,
        "graph_schema_version": settings.graph_schema_version if graph else None,
        "cross_doc_auto_merge_threshold": settings.cross_doc_auto_merge_threshold if graph else None,
        "cross_doc_auto_merge_margin": settings.cross_doc_auto_merge_margin if graph else None,
        "cross_doc_llm_confidence_threshold": settings.cross_doc_llm_confidence_threshold if graph else None,
    }


def embedding_contract(settings: Settings) -> dict[str, str]:
    """The store-level embedding record. A store holds vectors from one embedding only, so
    ingestion refuses to start when this differs from the settings it runs with."""
    return {
        "embedding_model_id": settings.embedding_model_id,
        "embedding_dim": str(settings.embedding_dim),
        "embedding_version": settings.embedding_version,
        "embedding_prompt_version": settings.embedding_prompt_version,
        "embedding_normalization_policy": settings.embedding_normalization_policy,
        "embedding_document_prefix": settings.embedding_document_prefix,
        "embedding_query_prefix": settings.embedding_query_prefix,
    }


def fingerprint_of(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def pipeline_fingerprint(settings: Settings, target_stage: str) -> str:
    return fingerprint_of(pipeline_fingerprint_payload(settings, target_stage))


@dataclass
class ExtractionReusePlan:
    """Chunks whose assertions carry forward unchanged, and the entities they reference."""

    assertions_by_chunk: dict[str, list[dict]] = field(default_factory=dict)
    mentions_by_chunk: dict[str, list[str]] = field(default_factory=dict)
    entities: dict[str, dict] = field(default_factory=dict)
    # Unchanged chunks re-extracted because the extractor changed since - counted so a
    # prompt roll-out shows how much of the corpus it actually reached.
    declined_extractor_changed: int = 0

    @property
    def assertion_count(self) -> int:
        return sum(len(rows) for rows in self.assertions_by_chunk.values())

STAGE_ORDER = {"PARSING": 1, "EXTRACTED": 2, "GRAPH_WRITTEN": 3, "VECTORS_WRITTEN": 4, "DONE": 5}


logger = logging.getLogger(__name__)


class PhaseOnePipeline:
    """Implements Stage 1-7 for ingestion plane."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.settings.ensure_directories()
        self.store = MetadataStore(self.settings.metadata_db_path)
        self.parser = DocumentParser()
        self.catalog = CorpusCatalog.load(
            self.settings.corpus_manifest_path, self.settings.corpus_news_index_path
        )
        self.coref = CoreferenceResolver(
            profile=self.settings.coref_profile,
            strict=self.settings.strict_coref,
            enabled=self.settings.coref_enabled,
        )
        tracing.configure_tracing(settings)
        self.chunker = SemanticChunker(
            target_tokens=self.settings.chunk_target_tokens,
            max_tokens=self.settings.chunk_max_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
            table_block_target_tokens=self.settings.chunk_table_block_target_tokens,
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
            # WP3-B: pick up work abandoned by an earlier crashed run before starting a
            # new one. Without this a partially staged version was never revisited.
            try:
                recovered = self.outbox_processor.reconcile(max_events=200)
                if recovered:
                    logger.info("startup_reconcile_recovered events=%d", recovered)
            except Exception as exc:
                logger.warning("startup_reconcile_failed error=%s", type(exc).__name__)

    def ingest_path(self, input_path: Path, recursive: bool = True) -> IngestionRunSummary:
        """Ingest one file or directory."""
        files = self._discover_files(input_path, recursive=recursive)
        summary = IngestionRunSummary()
        for file_path in files:
            summary.add(self.ingest_file(file_path))
        if input_path.is_dir():
            summary.missing_sources = [
                {"doc_id": str(row["doc_id"]), "source_path": str(row["source_path"]), "title": str(row["title"] or "")}
                for row in self.store.list_documents_missing_source(input_path)
            ]
            for item in summary.missing_sources:
                logger.warning(
                    "source_missing doc_id=%s path=%s (still served; remove with main_admin.py remove-document)",
                    item["doc_id"],
                    item["source_path"],
                )
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
        # One span per document. The extraction and embedding calls it makes are
        # auto-instrumented and nest underneath, which is what architecture 10.1 meant by
        # "extraction outputs during ingestion": what the model returned for *this* file.
        with tracing.span(
            "graphrag.ingest_file",
            **{"document.source": source_path.name, "document.format": detect_format(source_path.resolve())},
        ) as current:
            result = self._ingest_file(source_path)
            tracing.annotate(
                current,
                **{
                    "document.action": getattr(getattr(result, "action", None), "value", None),
                    "document.chunks": int(getattr(result, "chunk_count", 0) or 0),
                    "document.degraded_chunks": int(getattr(result, "degraded_chunks", 0) or 0),
                    "document.skipped": bool(getattr(result, "skipped", False)),
                    "document.error": getattr(result, "error", None),
                },
            )
            return result

    def _ingest_file(self, source_path: Path) -> FileIngestionResult:
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
            # A store built before document cards existed gains them on its next ingest,
            # with no reprocessing: the card is metadata, not part of the fingerprint.
            if not self.store.get_document_cards([doc_id]):
                card = self.catalog.card_for(absolute_path)
                if card:
                    self.store.set_document_card(doc_id, card)
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
                        generative_model=self.settings.ingestion_model,
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
            self.store.set_document_card(
                doc_id,
                self.catalog.card_for(absolute_path)
                or fallback_card(absolute_path, (section.heading for section in parsed.sections)),
            )
            # WP2-A: group parser items into logical sections before coreference, so
            # coref sees a whole section (reference 3.2) and the chunker has enough text
            # to window at 700/800 with overlap instead of degenerating to a passthrough.
            if self.settings.section_group_enabled:
                parsed = replace(
                    parsed,
                    sections=group_sections(
                        parsed.sections,
                        target_tokens=self.settings.section_group_target_tokens,
                        max_tokens=self.settings.section_group_max_tokens,
                        page_gap_tolerance=self.settings.section_group_page_gap_tolerance,
                    ),
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

            # WP2-D: the delta is about content, not position. Comparing occurrence
            # hashes reported every chunk after an insertion as changed, which made the
            # summary useless and blocked assertion reuse.
            old_content_hashes = set(
                self.store.get_content_hashes(
                    doc_id,
                    active_version.version_id if active_version else None,
                    active_only=active_version is not None,
                )
            )
            new_hashes = {chunk.content_hash for chunk in chunks}
            old_hashes = old_content_hashes
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
                    # Counted so a coreference stage that annotates nothing reads as zero.
                    # For months it returned every section unchanged and reported success.
                    "coref_sections": self.coref.last_stats.sections,
                    "coref_annotations": self.coref.last_stats.annotations,
                    "coref_failures": self.coref.last_stats.failures,
                    "annotated_chunks": sum(1 for chunk in chunks if chunk.extraction_text),
                },
            )

            entities_created = 0
            entities_merged = 0
            relations_created = 0
            degraded_chunks = 0
            if document_version_id:
                (
                    entities_created,
                    entities_merged,
                    relations_created,
                    degraded_chunks,
                ) = self._stage_graph_generation(
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

            if doc_format == "pdf":
                self._record_coverage(absolute_path, doc_id, document_version_id, chunks)
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
                degraded_chunks=degraded_chunks,
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

    def _plan_extraction_reuse(
        self,
        *,
        doc_id: str,
        chunks: list,
        active_version_id: str | None,
        doc_format: str = "",
    ) -> ExtractionReusePlan:
        """Decide which chunks can carry their assertions forward instead of re-extracting.

        Architecture finding `A4`, second half. A chunk qualifies when the text extraction
        would read is identical to what an active chunk of the previous version was
        extracted from, so a second call would buy a re-derivation of facts already held.
        That text is the coreference-annotated copy where there is one (WP20): matching on
        the stored text alone would carry unannotated extractions forward when annotation is
        switched on, and it would never take effect. The extractor must be the same too:
        assertions from another extraction prompt version or model are re-extracted unless
        `extraction_reuse_across_prompts` opts in (WP22). Reuse is also refused, and the
        chunk extracted normally, whenever an entity the prior assertions reference is no
        longer readable in the graph: a reused assertion must never point at an endpoint the
        new version does not contain.
        """
        empty = ExtractionReusePlan()
        if not self.settings.extraction_reuse_enabled or not active_version_id:
            return empty
        if self.graph_writer is None:
            return empty

        prior_by_input = {
            text: chunk_id
            for chunk_id, text in self.store.get_extraction_inputs(doc_id, active_version_id).items()
            if text
        }
        if not prior_by_input:
            return empty
        candidates = {
            chunk.chunk_id: prior_by_input[chunk.extraction_text or chunk.text]
            for chunk in chunks
            if (chunk.extraction_text or chunk.text) in prior_by_input
        }
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if not candidates:
            return empty

        try:
            rows = self.graph_writer.get_assertions_for_version_chunks(
                active_version_id, set(candidates.values())
            )
        except Exception as exc:
            logger.warning("extraction_reuse_lookup_failed error=%s", type(exc).__name__)
            return empty
        if not rows:
            return empty

        by_prior_chunk: dict[str, list[dict]] = {}
        for row in rows:
            by_prior_chunk.setdefault(str(row.get("chunk_id") or ""), []).append(row)

        referenced = {
            str(row[key])
            for row in rows
            for key in ("subject_entity_id", "object_entity_id")
            if row.get(key)
        }
        try:
            available = {
                str(node["id"]): node for node in self.graph_writer.get_entities_by_ids(referenced)
            }
        except Exception as exc:
            logger.warning("extraction_reuse_entity_lookup_failed error=%s", type(exc).__name__)
            return empty

        plan = ExtractionReusePlan()
        for new_chunk_id, prior_chunk_id in candidates.items():
            prior_rows = by_prior_chunk.get(prior_chunk_id)
            if not prior_rows:
                continue
            expected = self._expected_prompt_version(chunk_by_id.get(new_chunk_id), doc_format)
            if not self._extracted_by_current_extractor(prior_rows, expected):
                plan.declined_extractor_changed += 1
                continue
            endpoints = {
                str(row[key])
                for row in prior_rows
                for key in ("subject_entity_id", "object_entity_id")
                if row.get(key)
            }
            missing = endpoints - available.keys()
            if missing:
                logger.info(
                    "extraction_reuse_declined chunk=%s missing_entities=%d",
                    new_chunk_id,
                    len(missing),
                )
                continue
            plan.assertions_by_chunk[new_chunk_id] = prior_rows
            # The chunk's mentions are reused with it. Without this the entities carried
            # forward would have no mention rows, and `stage_version_records` would
            # reject the version for exactly that inconsistency.
            plan.mentions_by_chunk[new_chunk_id] = sorted(endpoints)
            for entity_id in endpoints:
                plan.entities[entity_id] = available[entity_id]
        return plan

    def _extracted_by_current_extractor(self, rows: list[dict], expected_prompt_version: str) -> bool:
        """Would the current extractor have produced these assertions? Same input is not
        enough: a new extraction prompt or model exists to extract differently, and reusing
        across it meant a prompt change never reached text that had not changed. Judged per
        assertion, from the provenance each carries, so assertions already carried across a
        change by the opt-in are recognised for what produced them."""
        if self.settings.extraction_reuse_across_prompts:
            return True
        return all(
            str(row.get("extraction_prompt_version") or "") == expected_prompt_version
            and str(row.get("extractor_model") or "") == self.settings.ingestion_model
            for row in rows
        )

    def _row_batching(self, doc_format: str) -> bool:
        # CSV and JSON rows take the GLiNER-seeded relation prompt, which does not batch.
        return self.settings.extraction_table_batch_tokens > 0 and doc_format not in {"csv", "json"}

    def _batched_prompt_version(self) -> str:
        return f"{self.settings.extraction_prompt_version}+{ROW_BATCH_VERSION}"

    def _expected_prompt_version(self, chunk, doc_format: str) -> str:  # noqa: ANN001
        """The prompt version this chunk's assertions would carry if extracted now."""
        if self._row_batching(doc_format) and getattr(chunk, "table_row", False):
            return self._batched_prompt_version()
        return self.settings.extraction_prompt_version

    def _extraction_units(self, chunks: list, doc_format: str) -> list[list]:
        """One chunk per unit, except runs of consecutive rows of one table - same heading,
        consecutive positions - which are grouped up to the token and row limits."""
        ordered = sorted(chunks, key=lambda chunk: (chunk.position, chunk.chunk_id))
        if not self._row_batching(doc_format):
            return [[chunk] for chunk in ordered]
        budget = int(self.settings.extraction_table_batch_tokens)
        max_rows = max(1, int(self.settings.extraction_table_batch_max_rows or 1))
        units: list[list] = []
        current: list = []
        tokens = 0
        for chunk in ordered:
            if not chunk.table_row:
                if current:
                    units.append(current)
                    current, tokens = [], 0
                units.append([chunk])
                continue
            same_table = bool(current) and (
                chunk.section_heading == current[-1].section_heading
                and chunk.position == current[-1].position + 1
            )
            if current and (
                not same_table or tokens + chunk.token_count > budget or len(current) >= max_rows
            ):
                units.append(current)
                current, tokens = [], 0
            current.append(chunk)
            tokens += chunk.token_count
        if current:
            units.append(current)
        return units

    def _rebuild_reused_assertions(
        self,
        *,
        doc_id: str,
        document_version_id: str,
        plan: ExtractionReusePlan,
    ) -> list:
        """Re-key prior assertions onto the new version.

        Everything that identifies the *claim* is preserved — subject, object, both
        predicate forms, properties, confidence, temporal bounds, review status — so
        `claim_key` is unchanged by construction. Only `assertion_id`, which is per source
        occurrence, is new.
        """
        rebuilt = []
        for chunk_id, rows in plan.assertions_by_chunk.items():
            for row in rows:
                literal_value = None
                if row.get("literal_value_json") is not None:
                    literal_value = json.loads(str(row["literal_value_json"]))
                predicate_raw = str(row["predicate_raw"])
                rebuilt.append(
                    AssertionRecord.create(
                        subject_entity_id=str(row["subject_entity_id"]),
                        subject_local_entity_id=self._optional_text(row.get("subject_local_entity_id")),
                        predicate_raw=predicate_raw,
                        predicate_canonical=self._optional_text(row.get("predicate_canonical"))
                        or predicate_raw,
                        object_kind=str(row["object_kind"]),
                        object_entity_id=self._optional_text(row.get("object_entity_id")),
                        object_local_entity_id=self._optional_text(row.get("object_local_entity_id")),
                        literal_value=literal_value,
                        properties=json.loads(str(row.get("properties_json") or "{}")),
                        confidence=float(row.get("confidence") or 0.0),
                        doc_id=doc_id,
                        document_version_id=document_version_id,
                        chunk_id=chunk_id,
                        extractor_model=str(row.get("extractor_model") or self.settings.ingestion_model),
                        extraction_prompt_version=str(
                            row.get("extraction_prompt_version")
                            or self.settings.extraction_prompt_version
                        ),
                        valid_from=self._optional_text(row.get("valid_from")),
                        valid_until=self._optional_text(row.get("valid_until")),
                        active=False,
                        review_status=str(row.get("review_status") or "accepted"),
                    )
                )
        return rebuilt

    @staticmethod
    def _optional_text(value) -> str | None:  # noqa: ANN001
        normalized = str(value or "").strip()
        return normalized or None

    def _stage_graph_generation(
        self,
        *,
        run_id: str,
        doc_id: str,
        document_version_id: str,
        doc_format: str,
        chunks: list,
        active_version_id: str | None,
    ) -> tuple[int, int, int, int]:
        """Extract and durably stage a complete immutable graph/vector version.

        Returns entities created, entities merged, relations created, and the number of
        chunks quarantined by WP3-C.
        """
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

        quarantined: list[tuple[str, str]] = []

        def run_extract(chunk):  # noqa: ANN001, ANN202
            """Extract one chunk, quarantining it rather than failing the document.

            REVIEW1 finding B11: any chunk whose relations referenced an undeclared
            entity aborted the whole document. That was sized for ~15 chunks per
            document; at the measured 194 it made a total loss near-certain,
            discarding 193 good extractions to punish one bad one.
            """
            try:
                # The annotated copy when coreference found something to name; the chunk's
                # stored text otherwise. Stored text is what gets cited, so it stays verbatim.
                raw_result = self.extractor.extract(chunk.extraction_text or chunk.text, doc_format)
                return chunk, self._scope_extraction_result(chunk.chunk_id, raw_result)
            except Exception as exc:
                quarantined.append((chunk.chunk_id, f"{type(exc).__name__}: {exc}"))
                logger.warning(
                    "chunk_extraction_quarantined chunk=%s error=%s",
                    chunk.chunk_id,
                    type(exc).__name__,
                )
                return chunk, None

        # Gate 18 (WP24): consecutive table rows are extracted in one call and split back
        # into per-row results. Rows extracted that way record the batched prompt version;
        # any row the batch could not serve is extracted alone, as before.
        batched_chunk_ids: list[str] = []

        def run_extract_unit(unit):  # noqa: ANN001, ANN202
            if not (self._row_batching(doc_format) and unit[0].table_row):
                return [run_extract(chunk) for chunk in unit]
            try:
                results = self.extractor.extract_rows(
                    [chunk.extraction_text or chunk.text for chunk in unit], doc_format
                )
            except Exception as exc:
                logger.warning("row_batch_failed rows=%d error=%s", len(unit), type(exc).__name__)
                results = [None] * len(unit)
            outcomes = []
            for chunk, result in zip(unit, results):
                if result is not None:
                    try:
                        outcomes.append((chunk, self._scope_extraction_result(chunk.chunk_id, result)))
                        batched_chunk_ids.append(chunk.chunk_id)
                        continue
                    except Exception:
                        pass
                outcomes.append(run_extract(chunk))
            return outcomes

        graph_chunks = [chunk for chunk in chunks if is_graph_extractable(chunk.content_kind)]
        skipped_reference_chunks = len(chunks) - len(graph_chunks)

        # Provider spend for one document is bounded here, before the first call is made.
        # Nothing else in the ingest path limits how many chunks a document may produce,
        # and a file that is not really a document — a data bundle, a log, an export —
        # looks exactly like a very long one until the bill arrives. A 3.2 MB annotation
        # bundle chunked into 987 pieces and spent 999 extraction calls before anyone
        # noticed (docs/ERRORS.md, 2026-08-27).
        chunk_cap = int(self.settings.ingestion_max_chunks_per_document)
        if chunk_cap > 0 and len(graph_chunks) > chunk_cap:
            raise RuntimeError(
                f"Document produces {len(graph_chunks)} extractable chunks, above the "
                f"{chunk_cap}-chunk limit, and would cost about that many provider calls. "
                "Raise INGESTION_MAX_CHUNKS_PER_DOCUMENT if this really is one document."
            )
        reuse = self._plan_extraction_reuse(
            doc_id=doc_id,
            chunks=graph_chunks,
            active_version_id=active_version_id,
            doc_format=doc_format,
        )
        extractable_chunks = [
            chunk for chunk in graph_chunks if chunk.chunk_id not in reuse.assertions_by_chunk
        ]
        units = self._extraction_units(extractable_chunks, doc_format)
        stats_before = self._extractor_stats()
        extraction_started = time.perf_counter()
        max_workers = max(1, int(self.settings.gemini_max_workers))
        extraction_results = []
        completed_chunks = 0
        progress_interval = max(1, int(self.settings.ingestion_progress_interval_chunks))
        try:
            if max_workers == 1 or len(units) <= 1:
                for unit in units:
                    for outcome in run_extract_unit(unit):
                        if outcome[1] is not None:
                            extraction_results.append(outcome)
                    completed_chunks += len(unit)
                    self._log_extraction_progress(
                        completed_chunks,
                        len(extractable_chunks),
                        progress_interval,
                    )
            else:
                with ThreadPoolExecutor(
                    max_workers=min(max_workers, len(units))
                ) as executor:
                    futures = [executor.submit(run_extract_unit, unit) for unit in units]
                    try:
                        for future in as_completed(futures):
                            outcomes = future.result()
                            for outcome in outcomes:
                                if outcome[1] is not None:
                                    extraction_results.append(outcome)
                            completed_chunks += len(outcomes)
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

        # A document is lost only when the failure is systemic, not incidental.
        failure_ratio = (
            len(quarantined) / len(extractable_chunks) if extractable_chunks else 0.0
        )
        if failure_ratio > float(self.settings.extraction_chunk_failure_ratio):
            first_error = quarantined[0][1] if quarantined else "n/a"
            message = (
                f"Extraction failed for {len(quarantined)} of "
                f"{len(extractable_chunks)} chunks ({failure_ratio:.1%}), above the "
                f"{self.settings.extraction_chunk_failure_ratio:.0%} quarantine limit. "
                f"First error: {first_error}"
            )
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
                error=message,
                degraded_chunks=len(quarantined),
            )
            raise RuntimeError(message)
        for chunk_id, error in quarantined:
            self.store.record_chunk_extraction_failure(
                document_version_id=document_version_id,
                chunk_id=chunk_id,
                attempts=int(self.settings.extraction_max_attempts),
                error=error,
            )

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
            degraded_chunks=len(quarantined),
            reused_chunks=len(reuse.assertions_by_chunk),
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
                # WP2-B: the display label and the coarse class are both decided once
                # per canonical entity from every observed type, never per mention.
                observed = [
                    raw for raw, count in counts.items() for _ in range(count)
                ]
                canonical_nodes[local_id]["type"] = display_type(observed)
                canonical_nodes[local_id]["coarse_class"] = dominant_coarse_class(observed)
                canonical_nodes[local_id]["entity_types"] = dict(counts)
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

        # WP7: entities referenced only by reused assertions are already canonical — they
        # came from the previous version of this document and went through cross-document
        # resolution then. They are added under their existing ids so the new version
        # contains every endpoint its assertions point at.
        for entity_id, node in reuse.entities.items():
            if entity_id in canonical_nodes:
                continue
            canonical_nodes[entity_id] = {
                "id": entity_id,
                "name": str(node.get("name") or entity_id),
                "type": str(node.get("type") or "Entity"),
                "aliases": list(node.get("aliases") or [str(node.get("name") or entity_id)]),
                "properties": node.get("properties") or {},
                "confidence": float(node.get("confidence") or 0.0),
                "coarse_class": str(node.get("coarse_class") or "UNKNOWN"),
            }
        raw_to_final = {
            raw_id: local_to_final.get(local_id, local_id)
            for raw_id, local_id in disamb.entity_id_map.items()
        }

        mention_rows = {
            (
                raw_to_final.get(entity_id, entity_id),
                chunk_id,
                mention_doc_id,
                confidence,
                disamb.entity_id_map.get(entity_id, entity_id),
            )
            for entity_id, chunk_id, mention_doc_id, confidence in raw_mentions
        }
        # WP7: a reused chunk brings its mentions with it. These entities are already
        # canonical, so the final and local ids are the same.
        for chunk_id, entity_ids in reuse.mentions_by_chunk.items():
            for entity_id in entity_ids:
                mention_rows.add((entity_id, chunk_id, doc_id, 1.0, entity_id))
        mapped_mentions = sorted(mention_rows)
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
            prompt_versions={
                chunk_id: self._batched_prompt_version() for chunk_id in batched_chunk_ids
            },
        )
        assertions.extend(
            self._rebuild_reused_assertions(
                doc_id=doc_id,
                document_version_id=document_version_id,
                plan=reuse,
            )
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
            model_id=self.settings.ingestion_model if merge_call_count else None,
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
                "reused_chunks": len(reuse.assertions_by_chunk),
                "reused_assertions": reuse.assertion_count,
                "reuse_declined_extractor_changed": reuse.declined_extractor_changed,
                "rejected_assertions": len(getattr(self.assertion_builder, "rejected", [])),
            },
        )
        node_rows = [canonical_nodes[key] for key in sorted(canonical_nodes)]
        doc_entities = [
            {
                "doc_id": doc_id,
                "entity_id": node["id"],
                "canonical_name": node["name"],
                "aliases": node.get("aliases", []),
                "entity_types": node.get("entity_types", {}),
                "coarse_class": node.get("coarse_class", "UNKNOWN"),
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
        return (
            merge_result.new_nodes,
            merge_result.merged_total,
            len(assertions),
            len(quarantined),
        )

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
                "row_batches",
                "rows_batched",
                "row_entities_ungrounded",
                "row_properties_ungrounded",
                "row_relations_cross_row",
                "row_results_invalid",
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
        degraded_chunks: int = 0,
        reused_chunks: int = 0,
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
            model_id=self.settings.ingestion_model,
            input_tokens=delta("input_tokens"),
            output_tokens=delta("output_tokens"),
            total_tokens=delta("total_tokens"),
            call_count=delta("call_count"),
            details={
                "chunks_completed": chunks_completed,
                "chunks_total": chunks_total,
                "reference_chunks_skipped": reference_chunks_skipped,
                "degraded_chunks": degraded_chunks,
                "reused_chunks": reused_chunks,
                "entities": delta("entities_count"),
                "relations": delta("relations_count"),
                # Gate 18 batching: calls saved, and what grounding refused.
                "row_batches": delta("row_batches"),
                "rows_batched": delta("rows_batched"),
                "row_entities_ungrounded": delta("row_entities_ungrounded"),
                "row_properties_ungrounded": delta("row_properties_ungrounded"),
                "row_relations_cross_row": delta("row_relations_cross_row"),
                "row_results_invalid": delta("row_results_invalid"),
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

    def _record_coverage(self, path: Path, doc_id: str, document_version_id: str | None, chunks: list) -> None:
        """ROADMAP E3: measure and keep how much of a PDF reached its chunks. Never fails an ingest."""
        if not document_version_id:
            return
        try:
            report = coverage_report(path, [str(getattr(chunk, "text", "") or "") for chunk in chunks],
                                     self.parser.last_page_recovery)
            self.store.set_document_coverage(doc_id, document_version_id, report)
            if report.get("missing") or report.get("no_text"):
                logger.warning("pdf_coverage_gaps doc_id=%s missing=%s no_text=%s", doc_id, report.get("missing"), report.get("no_text"))
        except Exception as exc:
            logger.warning("pdf_coverage_unmeasured doc_id=%s error=%s", doc_id, type(exc).__name__)

    def _pipeline_fingerprint(self, target_stage: str) -> str:
        return pipeline_fingerprint(self.settings, target_stage)

    @staticmethod
    def _is_active_generation_current(
        row,  # noqa: ANN001
        *,
        checksum: str,
        target_stage: str,
        pipeline_fingerprint: str,
    ) -> bool:
        active_checksum = str(row["active_checksum"] or "")
        active_fingerprint = str(row["active_pipeline_fingerprint"] or "")
        completed_stage = str(row["last_completed_stage"] or row["pipeline_stage"] or "")
        return (
            str(row["status"] or "") == "complete"
            and active_checksum == checksum
            and active_fingerprint == pipeline_fingerprint
            and STAGE_ORDER.get(completed_stage, 0) >= STAGE_ORDER[target_stage]
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

    def _ensure_embedding_metadata(self) -> None:
        expected = embedding_contract(self.settings)
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
                "Embedding corpus metadata mismatch. Rebuild vectors before ingestion "
                "(scripts/reembed_corpus.py); if they were already rebuilt, record it with "
                f"scripts/restamp_after_reembed.py: {mismatches}"
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

            merged_types = dict(existing.get("entity_types", {}))
            for raw_type, count in (node.get("entity_types", {}) or {}).items():
                merged_types[raw_type] = merged_types.get(raw_type, 0) + int(count)
            if merged_types:
                observed = [raw for raw, count in merged_types.items() for _ in range(count)]
                existing["entity_types"] = merged_types
                existing["type"] = display_type(observed)
                existing["coarse_class"] = dominant_coarse_class(observed)

            existing_name = str(existing.get("name", "")).strip()
            incoming_name = str(node.get("name", "")).strip()
            if (not existing_name or existing_name.lower() == "entity") and incoming_name:
                existing["name"] = incoming_name

            existing_conf = float(existing.get("confidence", 0.0))
            incoming_conf = float(node.get("confidence", 0.0))
            existing["confidence"] = max(existing_conf, incoming_conf)
        return merged
