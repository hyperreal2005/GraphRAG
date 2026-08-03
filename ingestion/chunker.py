"""Semantic chunking stage for Phase 1 ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass

from models.ingestion import Chunk, ParsedDocument, Section, SectionType


SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.MULTILINE)
CHUNK_FINGERPRINT_VERSION = "v2"


def normalize_for_hash(text: str) -> str:
    """Normalize insignificant whitespace without changing semantic case."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_hash(
    text: str,
    *,
    section_heading: str | None = None,
    page_number: int | None = None,
    position: int | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
) -> str:
    """Compute an occurrence-aware fingerprint for extraction/index deltas."""
    payload = {
        "version": CHUNK_FINGERPRINT_VERSION,
        "text": normalize_for_hash(text),
        "section_heading": normalize_for_hash(section_heading or ""),
        "page_number": page_number,
        "position": position,
        "char_start": char_start,
        "char_end": char_end,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
        if target_tokens <= 0 or max_tokens <= 0:
            raise ValueError("target_tokens and max_tokens must be positive")
        if target_tokens > max_tokens:
            raise ValueError("target_tokens cannot exceed max_tokens")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk_document(self, document: ParsedDocument) -> list[Chunk]:
        """Chunk all sections of a parsed document."""
        chunks: list[Chunk] = []
        position = 0

        for section in document.sections:
            if section.section_type is SectionType.TABLE_ROW:
                table_chunks = self._chunk_table_row(document.doc_id, section, position)
                chunks.extend(table_chunks)
                position += len(table_chunks)
                continue

            if section.section_type is SectionType.TEXT:
                text_chunks = self._chunk_text_section(document.doc_id, section, position)
                chunks.extend(text_chunks)
                position += len(text_chunks)

        return chunks

    def _chunk_table_row(self, doc_id: str, section: Section, position: int) -> list[Chunk]:
        headers = section.table_headers or []
        row_values = section.table_row or []
        if not headers and not row_values:
            return []

        rendered = " | ".join(f"{header}: {value}" for header, value in zip(headers, row_values))
        segments = self._split_oversized_text(rendered)
        chunks: list[Chunk] = []
        for offset, segment in enumerate(segments):
            chunk_position = position + offset
            digest = chunk_hash(
                segment,
                section_heading=section.heading,
                page_number=section.page,
                position=chunk_position,
            )
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=str(uuid.uuid4()),
                    position=chunk_position,
                    section_heading=section.heading,
                    page_number=section.page,
                    text=segment,
                    token_count=token_count(segment),
                    char_start=None,
                    char_end=None,
                    chunk_hash=digest,
                )
            )
        return chunks

    def _chunk_text_section(self, doc_id: str, section: Section, base_position: int) -> list[Chunk]:
        text = (section.text or "").strip()
        if not text:
            return []

        spans = self._sentence_spans(text)
        if not spans:
            absolute_start = section.char_start if section.char_start is not None else 0
            absolute_end = absolute_start + len(text)
            digest = chunk_hash(
                text,
                section_heading=section.heading,
                page_number=section.page,
                position=base_position,
                char_start=absolute_start,
                char_end=absolute_end,
            )
            return [
                Chunk(
                    doc_id=doc_id,
                    chunk_id=str(uuid.uuid4()),
                    position=base_position,
                    section_heading=section.heading,
                    page_number=section.page,
                    text=text,
                    token_count=token_count(text),
                    char_start=absolute_start,
                    char_end=absolute_end,
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
                    raise RuntimeError("Oversized sentence was not split before chunk assembly.")

                current.append(sentence)
                running_tokens += sentence_tokens
                cursor += 1

                if running_tokens >= self.target_tokens:
                    break

            if not current:
                break

            chunk_text = " ".join(span.text for span in current).strip()
            chunk_position = base_position + local_position
            section_offset = section.char_start if section.char_start is not None else 0
            absolute_start = section_offset + current[0].start
            absolute_end = section_offset + current[-1].end
            digest = chunk_hash(
                chunk_text,
                section_heading=section.heading,
                page_number=section.page,
                position=chunk_position,
                char_start=absolute_start,
                char_end=absolute_end,
            )
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=str(uuid.uuid4()),
                    position=chunk_position,
                    section_heading=section.heading,
                    page_number=section.page,
                    text=chunk_text,
                    token_count=running_tokens,
                    char_start=absolute_start,
                    char_end=absolute_end,
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

            # Overlap must never prevent forward progress. This invariant also
            # protects against a short overlapped prefix followed by a sentence
            # that exactly fills or exceeds the remaining token budget.
            start_index = max(overlap_start, start_index + 1)

        return chunks

    def _sentence_spans(self, text: str) -> list[SentenceSpan]:
        spans: list[SentenceSpan] = []
        for match in SENTENCE_PATTERN.finditer(text):
            sentence_text = re.sub(r"\s+", " ", match.group(0)).strip()
            if not sentence_text:
                continue
            sentence_tokens = token_count(sentence_text)
            if sentence_tokens <= self.max_tokens:
                spans.append(
                    SentenceSpan(
                        text=sentence_text,
                        start=match.start(),
                        end=match.end(),
                        tokens=sentence_tokens,
                    )
                )
                continue

            token_matches = list(re.finditer(r"\S+", match.group(0)))
            for offset in range(0, len(token_matches), self.max_tokens):
                token_group = token_matches[offset : offset + self.max_tokens]
                segment = " ".join(item.group(0) for item in token_group)
                spans.append(
                    SentenceSpan(
                        text=segment,
                        start=match.start() + token_group[0].start(),
                        end=match.start() + token_group[-1].end(),
                        tokens=len(token_group),
                    )
                )
        return spans

    def _split_oversized_text(self, text: str) -> list[str]:
        tokens = re.findall(r"\S+", text)
        if not tokens:
            return []
        return [" ".join(tokens[index : index + self.max_tokens]) for index in range(0, len(tokens), self.max_tokens)]
