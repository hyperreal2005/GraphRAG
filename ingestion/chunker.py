"""Semantic chunking stage for Phase 1 ingestion."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass

from models.ingestion import Chunk, ParsedDocument, Section, SectionType


SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)


def normalize_for_hash(text: str) -> str:
    """Canonicalize chunk text before hashing."""
    return re.sub(r"\s+", " ", text).strip().lower()


def chunk_hash(text: str) -> str:
    """Compute stable SHA-256 for chunk text."""
    canonical_text = normalize_for_hash(text)
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def token_count(text: str) -> int:
    """Approximate token count for chunk sizing."""
    return len(text.split())


@dataclass
class SentenceSpan:
    text: str
    start: int
    end: int
    tokens: int


class SemanticChunker:
    """Chunk text with sentence boundaries and token overlap."""

    def __init__(self, target_tokens: int = 700, max_tokens: int = 800, overlap_tokens: int = 100) -> None:
        if target_tokens > max_tokens:
            raise ValueError("target_tokens cannot exceed max_tokens")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk all sections of a parsed document."""
        chunks: list[Chunk] = []
        position = 0

        for section in document.sections:
            if section.section_type is SectionType.TABLE_ROW:
                chunk = self._chunk_table_row(document.doc_id, section, position)
                if chunk is not None:
                    chunks.append(chunk)
                    position += 1
                continue

            if section.section_type is SectionType.TEXT:
                text_chunks = self._chunk_text_section(document.doc_id, section, position)
                chunks.extend(text_chunks)
                position += len(text_chunks)

        return chunks

    def _chunk_table_row(self, doc_id: str, section: Section, position: int) -> Chunk | None:
        headers = section.table_headers or []
        row_values = section.table_row or []
        if not headers and not row_values:
            return None

        rendered = " | ".join(f"{header}: {value}" for header, value in zip(headers, row_values))
        digest = chunk_hash(rendered)
        return Chunk(
            doc_id=doc_id,
            chunk_id=str(uuid.uuid4()),
            position=position,
            section_heading=section.heading,
            page_number=section.page,
            text=rendered,
            token_count=token_count(rendered),
            char_start=None,
            char_end=None,
            chunk_hash=digest,
        )

    def _chunk_text_section(self, doc_id: str, section: Section, base_position: int) -> list[Chunk]:
        text = (section.text or "").strip()
        if not text:
            return []

        spans = self._sentence_spans(text)
        if not spans:
            digest = chunk_hash(text)
            return [
                Chunk(
                    doc_id=doc_id,
                    chunk_id=str(uuid.uuid4()),
                    position=base_position,
                    section_heading=section.heading,
                    page_number=section.page,
                    text=text,
                    token_count=token_count(text),
                    char_start=0,
                    char_end=len(text),
                    chunk_hash=digest,
                )
            ]

        chunks: list[Chunk] = []
        start_index = 0
        local_position = 0

        while start_index < len(spans):
            current: list[SentenceSpan] = []
            running_tokens = 0
            cursor = start_index

            while cursor < len(spans):
                sentence = spans[cursor]
                sentence_tokens = sentence.tokens

                if current and running_tokens + sentence_tokens > self.max_tokens:
                    break

                if not current and sentence_tokens > self.max_tokens:
                    current.append(sentence)
                    running_tokens += sentence_tokens
                    cursor += 1
                    break

                current.append(sentence)
                running_tokens += sentence_tokens
                cursor += 1

                if running_tokens >= self.target_tokens:
                    break

            if not current:
                break

            chunk_text = " ".join(span.text for span in current).strip()
            digest = chunk_hash(chunk_text)
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=str(uuid.uuid4()),
                    position=base_position + local_position,
                    section_heading=section.heading,
                    page_number=section.page,
                    text=chunk_text,
                    token_count=running_tokens,
                    char_start=current[0].start,
                    char_end=current[-1].end,
                    chunk_hash=digest,
                )
            )
            local_position += 1

            if cursor >= len(spans):
                break

            overlap_start = cursor
            overlap_running = 0
            while overlap_start > start_index:
                previous = spans[overlap_start - 1]
                if overlap_running and overlap_running + previous.tokens > self.overlap_tokens:
                    break
                overlap_running += previous.tokens
                overlap_start -= 1
                if overlap_running >= self.overlap_tokens:
                    break

            if overlap_start == start_index and cursor == start_index + 1:
                start_index = cursor
            else:
                start_index = overlap_start

        return chunks

    def _sentence_spans(self, text: str) -> list[SentenceSpan]:
        spans: list[SentenceSpan] = []
        for match in SENTENCE_PATTERN.finditer(text):
            sentence_text = re.sub(r"\s+", " ", match.group(0)).strip()
            if not sentence_text:
                continue
            spans.append(
                SentenceSpan(
                    text=sentence_text,
                    start=match.start(),
                    end=match.end(),
                    tokens=token_count(sentence_text),
                )
            )
        return spans
