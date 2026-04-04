"""Phase 1 ingestion pipeline orchestration."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from config import Settings
from ingestion.chunker import SemanticChunker
from ingestion.coref import CoreferenceResolver
from ingestion.parser import DocumentParser, compute_sha256, detect_format
from ingestion.relation_normalizer import canonicalize_relation
from ingestion.state_store import MetadataStore
from ingestion.vector_indexer import ChunkVectorInput, EntityVectorInput, QdrantVectorIndexer, build_entity_summary
from models.ingestion import (
    DeltaSummary,
    FileIngestionResult,
    IngestionAction,
    IngestionRunSummary,
    ParsedDocument,
)


def utc_now_iso() -> str:
    """Current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


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
        self.graph_writer = None
        self.vector_indexer = None
        if self.settings.enable_graph:
            from ingestion.disambiguator import EntityDisambiguator
            from ingestion.extractor import GeminiExtractor
            from ingestion.graph_writer import GraphWriter

            self._ensure_embedding_metadata()
            self.extractor = GeminiExtractor(self.settings)
            self.disambiguator = EntityDisambiguator(self.settings)
            self.graph_writer = GraphWriter(self.settings)
            self.vector_indexer = QdrantVectorIndexer(self.settings)

    def ingest_path(self, input_path: Path, recursive: bool = True) -> IngestionRunSummary:
        """Ingest one file or directory."""
        files = self._discover_files(input_path, recursive=recursive)
        if self.settings.enable_graph and self.settings.auto_backfill_vectors:
            self._backfill_graph_written_docs()
        summary = IngestionRunSummary()
        for file_path in files:
            summary.add(self.ingest_file(file_path))
        return summary

    def ingest_file(self, source_path: Path) -> FileIngestionResult:
        """Ingest a single file from Stage 1 through Stage 7."""
        started_at = utc_now_iso()
        run_id = str(uuid.uuid4())
        absolute_path = source_path.resolve()
        source_key = str(absolute_path)

        doc_format = detect_format(absolute_path)
        checksum = compute_sha256(absolute_path)
        existing_row = self.store.get_document_by_source(source_key)

        if existing_row is not None and str(existing_row["checksum"]) == checksum:
            doc_id = str(existing_row["doc_id"])
            result = FileIngestionResult(
                source_path=source_key,
                action=IngestionAction.UNCHANGED,
                doc_id=doc_id,
                skipped=True,
            )
            self.store.log_ingestion_run(
                run_id=run_id,
                doc_id=doc_id,
                started_at=started_at,
                finished_at=utc_now_iso(),
                status="skipped",
            )
            return result

        action = IngestionAction.NEW if existing_row is None else IngestionAction.CHANGED
        doc_id = str(uuid.uuid4()) if existing_row is None else str(existing_row["doc_id"])
        self.store.start_document(doc_id, source_key, doc_format, checksum)

        try:
            parsed = self.parser.parse_document(
                source_path=absolute_path,
                doc_id=doc_id,
                checksum=checksum,
            )
            resolved = self.coref.resolve_document(parsed)
            chunks = self.chunker.chunk_document(resolved)

            old_hash_to_chunk_id = self.store.get_chunk_ids_by_hash(doc_id)
            for index, chunk in enumerate(chunks):
                old_chunk_id = old_hash_to_chunk_id.get(chunk.chunk_hash)
                if old_chunk_id:
                    chunks[index] = replace(chunk, chunk_id=old_chunk_id)

            new_hashes = {chunk.chunk_hash for chunk in chunks}
            old_hashes = set(old_hash_to_chunk_id.keys())
            added = sorted(new_hashes - old_hashes)
            removed = sorted(old_hashes - new_hashes)
            unchanged = sorted(new_hashes & old_hashes)

            self.store.delete_chunks_by_hashes(doc_id, removed)
            self.store.upsert_chunks(chunks)
            self.store.finish_document(doc_id, resolved.metadata.word_count, "PARSING")

            delta = DeltaSummary(
                added_chunk_hashes=added,
                removed_chunk_hashes=removed,
                unchanged_chunk_hashes=unchanged,
            )
            entities_created = 0
            relations_created = 0

            if self.settings.enable_graph:
                if (
                    self.extractor is None
                    or self.disambiguator is None
                    or self.graph_writer is None
                    or self.vector_indexer is None
                ):
                    raise RuntimeError("Extraction components are not initialized.")

                removed_chunk_ids = [old_hash_to_chunk_id[h] for h in removed if h in old_hash_to_chunk_id]
                entity_ids_to_retract: set[str] = set()
                if removed_chunk_ids:
                    entity_ids_removed = self.store.get_entity_ids_for_chunks(doc_id, removed_chunk_ids)
                    self.store.delete_entity_mentions_for_chunks(doc_id, removed_chunk_ids)
                    remaining_entities = self.store.get_entity_ids_with_mentions(doc_id)
                    entity_ids_to_retract = entity_ids_removed - remaining_entities
                    self.graph_writer.retract_chunks(doc_id, removed_chunk_ids)
                    self.graph_writer.retract_entities(doc_id, entity_ids_to_retract)
                    self.store.delete_doc_entities(doc_id, list(entity_ids_to_retract))
                    self.vector_indexer.delete_chunk_vectors(removed_chunk_ids)
                    self.vector_indexer.delete_entity_vectors(list(entity_ids_to_retract))

                chunks_to_process = (
                    [chunk for chunk in chunks if chunk.chunk_hash in added]
                    if action is IngestionAction.CHANGED
                    else chunks
                )

                extraction_entities = []
                extraction_relations = []
                entity_mentions: list[tuple[str, str, str, float]] = []

                def run_extract(chunk):
                    return chunk, self.extractor.extract(chunk.text, doc_format)

                max_workers = max(1, int(self.settings.gemini_max_workers))
                if max_workers == 1 or len(chunks_to_process) <= 1:
                    results_iter = (run_extract(chunk) for chunk in chunks_to_process)
                else:
                    workers = min(max_workers, len(chunks_to_process))
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = [executor.submit(run_extract, chunk) for chunk in chunks_to_process]
                        results_iter = (future.result() for future in as_completed(futures))

                for chunk, result in results_iter:
                    extraction_entities.extend(result.entities)
                    extraction_relations.extend(
                        [
                            {
                                "head_id": relation.head_id,
                                "head_type": relation.head_type,
                                "relation": relation.relation,
                                "raw_relation": relation.raw_relation,
                                "tail_id": relation.tail_id,
                                "tail_type": relation.tail_type,
                                "properties": relation.properties,
                                "weight": relation.confidence,
                                "source_chunk": chunk.chunk_id,
                            }
                            for relation in result.relations
                        ]
                    )
                    for entity in result.entities:
                        entity_mentions.append((entity.entity_id, chunk.chunk_id, doc_id, 1.0))

                existing_doc_entities = self.store.get_doc_entities(doc_id)
                disamb = self.disambiguator.disambiguate(extraction_entities, existing_doc_entities)

                canonical_nodes = {}
                entity_type_counts: dict[str, dict[str, int]] = {}
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
                    entity_type_counts[cluster.entity_id] = {}

                for entity in extraction_entities:
                    canonical_id = disamb.entity_id_map.get(entity.entity_id)
                    if not canonical_id or canonical_id not in canonical_nodes:
                        continue
                    entry = canonical_nodes[canonical_id]
                    entry["properties"].update(entity.properties or {})
                    if entity.name and entity.name not in entry["aliases"]:
                        entry["aliases"].append(entity.name)
                    if entity.entity_type:
                        counts = entity_type_counts[canonical_id]
                        counts[entity.entity_type] = counts.get(entity.entity_type, 0) + 1

                for canonical_id, counts in entity_type_counts.items():
                    if counts:
                        canonical_nodes[canonical_id]["type"] = max(counts, key=counts.get)

                canonical_entities = [
                    {
                        "entity_id": node["id"],
                        "canonical_name": node["name"],
                        "aliases": node["aliases"],
                    }
                    for node in canonical_nodes.values()
                ]
                self.store.upsert_doc_entities(doc_id, canonical_entities)

                self.store.delete_entity_mentions_for_chunks(doc_id, [chunk.chunk_id for chunk in chunks_to_process])
                mapped_mentions = [
                    (disamb.entity_id_map.get(entity_id, entity_id), chunk_id, doc_id, confidence)
                    for entity_id, chunk_id, doc_id, confidence in entity_mentions
                ]
                self.store.upsert_entity_mentions(mapped_mentions)
                mentions_by_chunk: dict[str, list[str]] = {}
                for entity_id, chunk_id, _, _ in mapped_mentions:
                    mentions_by_chunk.setdefault(chunk_id, []).append(entity_id)

                mapped_edges = []
                for edge in extraction_relations:
                    head_id = disamb.entity_id_map.get(edge["head_id"], edge["head_id"])
                    tail_id = disamb.entity_id_map.get(edge["tail_id"], edge["tail_id"])
                    if head_id not in canonical_nodes or tail_id not in canonical_nodes:
                        continue
                    raw_relation = str(edge.get("raw_relation") or edge.get("relation") or "").strip()
                    if not raw_relation:
                        continue
                    canonical_relation = canonicalize_relation(str(edge.get("relation") or raw_relation))
                    mapped_edges.append(
                        {
                            "head_id": head_id,
                            "tail_id": tail_id,
                            "relation": canonical_relation,
                            "raw_relation": raw_relation,
                            "properties": edge["properties"],
                            "weight": edge["weight"],
                            "source_chunk": edge["source_chunk"],
                        }
                    )

                node_confidence: dict[str, list[float]] = {node_id: [] for node_id in canonical_nodes}
                for edge in mapped_edges:
                    weight = float(edge.get("weight", 0.0))
                    node_confidence[edge["head_id"]].append(weight)
                    node_confidence[edge["tail_id"]].append(weight)

                for node_id, weights in node_confidence.items():
                    if weights:
                        canonical_nodes[node_id]["confidence"] = sum(weights) / len(weights)

                self.store.update_pipeline_stage(doc_id, "EXTRACTED")
                self.graph_writer.write_nodes(doc_id, list(canonical_nodes.values()))
                self.graph_writer.write_edges(doc_id, mapped_edges)
                self.store.update_pipeline_stage(doc_id, "GRAPH_WRITTEN")

                chunk_vector_inputs = [
                    ChunkVectorInput(
                        chunk_id=chunk.chunk_id,
                        doc_id=doc_id,
                        text=chunk.text,
                        section_heading=chunk.section_heading,
                        page=chunk.page_number,
                        entity_ids_mentioned=mentions_by_chunk.get(chunk.chunk_id, []),
                    )
                    for chunk in chunks_to_process
                ]
                entity_vector_inputs = [
                    EntityVectorInput(
                        neo4j_node_id=node["id"],
                        name=node["name"],
                        entity_type=node["type"],
                        summary=build_entity_summary(
                            name=str(node["name"]),
                            entity_type=str(node["type"]),
                            properties=node.get("properties", {}),
                        ),
                    )
                    for node in canonical_nodes.values()
                ]
                self.vector_indexer.upsert_chunk_vectors(chunk_vector_inputs)
                self.vector_indexer.upsert_entity_vectors(entity_vector_inputs)
                self.store.set_chunk_embedding_version(
                    [item.chunk_id for item in chunk_vector_inputs],
                    self.settings.embedding_version,
                )
                self.store.update_pipeline_stage(doc_id, "VECTORS_WRITTEN")

                entities_created = len(canonical_nodes)
                relations_created = len(mapped_edges)
                self.store.finish_document(doc_id, resolved.metadata.word_count, "DONE")

            self._write_artifacts(
                run_id=run_id,
                parsed_document=resolved,
                chunks=chunks,
                action=action,
                delta=delta,
            )
            self.store.log_ingestion_run(
                run_id=run_id,
                doc_id=doc_id,
                started_at=started_at,
                finished_at=utc_now_iso(),
                status="complete",
                entities_created=entities_created,
                relations_created=relations_created,
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
            self.store.fail_document(doc_id=doc_id, error_message=str(exc))
            self.store.log_ingestion_run(
                run_id=run_id,
                doc_id=doc_id,
                started_at=started_at,
                finished_at=utc_now_iso(),
                status="failed",
            )
            return FileIngestionResult(
                source_path=source_key,
                action=action,
                doc_id=doc_id,
                skipped=False,
                error=str(exc),
            )

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
        stored_model = self.store.get_system_metadata("embedding_model_id")
        stored_dim = self.store.get_system_metadata("embedding_dim")
        stored_version = self.store.get_system_metadata("embedding_version")
        if stored_model is None:
            self.store.set_system_metadata("embedding_model_id", self.settings.embedding_model_id)
            self.store.set_system_metadata("embedding_dim", str(self.settings.embedding_dim))
            self.store.set_system_metadata("embedding_version", self.settings.embedding_version)
            return
        if stored_model != self.settings.embedding_model_id or stored_dim != str(self.settings.embedding_dim):
            raise RuntimeError(
                "Embedding model mismatch. Re-embed vectors or fix config before ingestion."
            )
        if stored_version is None:
            self.store.set_system_metadata("embedding_version", self.settings.embedding_version)
