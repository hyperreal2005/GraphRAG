"""Typed contracts for Phase 1 ingestion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


class SectionType(str, Enum):
    """Supported section types after parsing."""

    TEXT = "text"
    TABLE_ROW = "table_row"


class IngestionAction(str, Enum):
    """High-level action taken for a file."""

    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


@dataclass
class DocumentMetadata:
    """Metadata captured per parsed document."""

    ingested_at: str = field(default_factory=utc_now_iso)
    word_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"ingested_at": self.ingested_at, "word_count": self.word_count}


@dataclass
class SectionProvenance:
    """Where one parser item came from, preserved when items are grouped.

    Docling emits one item per layout element and WP2-A merges consecutive items into a
    logical section. Without this the merge would erase the page and bounding box that
    exact citation depends on. ``char_start``/``char_end`` are item-local, exactly as
    Docling reports them; they are not offsets into the grouped section.
    """

    section_id: str
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    heading: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"section_id": self.section_id}
        if self.heading is not None:
            payload["heading"] = self.heading
        if self.page is not None:
            payload["page"] = self.page
        if self.char_start is not None:
            payload["char_start"] = self.char_start
        if self.char_end is not None:
            payload["char_end"] = self.char_end
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox)
        return payload


@dataclass
class Section:
    """Structured section emitted by parser stage."""

    section_id: str
    section_type: SectionType
    heading: str | None = None
    text: str | None = None
    page: int | None = None
    table_headers: list[str] | None = None
    table_row: list[str] | None = None
    char_start: int | None = None
    char_end: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    provenance: list[SectionProvenance] = field(default_factory=list)
    # Coreference annotations: (offset, text) insertions into `text.strip()`, applied only
    # to the text extraction reads. The section's own text is never rewritten, so stored,
    # embedded and cited chunks stay verbatim.
    coref_annotations: list[tuple[int, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "section_id": self.section_id,
            "type": self.section_type.value,
        }
        if self.heading is not None:
            payload["heading"] = self.heading
        if self.text is not None:
            payload["text"] = self.text
        if self.page is not None:
            payload["page"] = self.page
        if self.table_headers is not None:
            payload["table_headers"] = self.table_headers
        if self.table_row is not None:
            payload["table_row"] = self.table_row
        if self.char_start is not None:
            payload["char_start"] = self.char_start
        if self.char_end is not None:
            payload["char_end"] = self.char_end
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox)
        if self.provenance:
            payload["provenance"] = [item.to_dict() for item in self.provenance]
        if self.coref_annotations:
            payload["coref_annotations"] = [list(item) for item in self.coref_annotations]
        return payload


@dataclass
class ParsedDocument:
    """Parsed document schema from Stage 1/2."""

    doc_id: str
    source: str
    format: str
    checksum: str
    sections: list[Section]
    metadata: DocumentMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "format": self.format,
            "checksum": self.checksum,
            "sections": [section.to_dict() for section in self.sections],
            "metadata": self.metadata.to_dict(),
        }


@dataclass
class Chunk:
    """Semantic chunk emitted by Stage 3."""

    doc_id: str
    chunk_id: str
    position: int
    section_heading: str | None
    page_number: int | None
    text: str
    token_count: int
    char_start: int | None
    char_end: int | None
    chunk_hash: str
    # Position-free identity of the text itself (WP2-D). chunk_hash stays the
    # occurrence identity; this one survives an edit elsewhere in the document.
    content_hash: str = ""
    content_kind: str = "body"
    provenance: list[SectionProvenance] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    # What extraction reads when it differs from `text`: the same words with coreference
    # annotations inserted. None when there is nothing to annotate.
    extraction_text: str | None = None
    # One rendered table row. Ingestion-time only: row chunks may be extracted in batches.
    table_row: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "position": self.position,
            "section_heading": self.section_heading,
            "page_number": self.page_number,
            "text": self.text,
            "token_count": self.token_count,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "chunk_hash": self.chunk_hash,
            "content_hash": self.content_hash,
            "content_kind": self.content_kind,
            "pages": list(self.pages),
            "provenance": [item.to_dict() for item in self.provenance],
        }
        if self.extraction_text is not None:
            payload["extraction_text"] = self.extraction_text
        return payload


@dataclass
class DeltaSummary:
    """Chunk-level diff summary for changed documents."""

    added_chunk_hashes: list[str] = field(default_factory=list)
    removed_chunk_hashes: list[str] = field(default_factory=list)
    unchanged_chunk_hashes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_chunk_hashes": self.added_chunk_hashes,
            "removed_chunk_hashes": self.removed_chunk_hashes,
            "unchanged_chunk_hashes": self.unchanged_chunk_hashes,
        }


@dataclass
class FileIngestionResult:
    """Per-file pipeline result."""

    source_path: str
    action: IngestionAction
    doc_id: str
    chunk_count: int = 0
    skipped: bool = False
    error: str | None = None
    delta: DeltaSummary = field(default_factory=DeltaSummary)
    # WP3-C: chunks withheld from the graph without failing the document.
    degraded_chunks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "action": self.action.value,
            "doc_id": self.doc_id,
            "chunk_count": self.chunk_count,
            "skipped": self.skipped,
            "error": self.error,
            "delta": self.delta.to_dict(),
            "degraded_chunks": self.degraded_chunks,
        }


@dataclass
class IngestionRunSummary:
    """Aggregate run summary for CLI output."""

    processed: int = 0
    changed: int = 0
    skipped: int = 0
    failed: int = 0
    results: list[FileIngestionResult] = field(default_factory=list)

    def add(self, result: FileIngestionResult) -> None:
        self.results.append(result)
        if result.error:
            self.failed += 1
            return
        if result.skipped:
            self.skipped += 1
            return
        if result.action is IngestionAction.CHANGED:
            self.changed += 1
        self.processed += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "changed": self.changed,
            "skipped": self.skipped,
            "failed": self.failed,
            "results": [result.to_dict() for result in self.results],
        }
