"""Semantic chunking stage for Phase 1 ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass

from models.ingestion import Chunk, ParsedDocument, Section, SectionType


# Abbreviations whose trailing period must not end a sentence. Kept explicit rather
# than inferred: a wrong guess here silently changes chunk boundaries (REVIEW1 B12).
SENTENCE_ABBREVIATIONS = (
    "Dr", "Mr", "Mrs", "Ms", "Prof", "Sr", "Jr", "St",
    "Fig", "Eq", "Sec", "Ch", "No", "Vol", "pp", "p", "al", "Ref", "Refs",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
    "Inc", "Ltd", "Co", "Corp", "Univ", "Dept", "Est", "approx", "cf", "vs",
    "e.g", "i.e", "etc", "U.S", "U.K", "E.U",
)
_ABBREVIATION_SET = {item.casefold() for item in SENTENCE_ABBREVIATIONS}
# Python's re requires a fixed-width lookbehind, so abbreviations cannot be expressed in
# the pattern. Find candidate boundaries cheaply, then reject them in code.
SENTENCE_CANDIDATE_PATTERN = re.compile(
    r"[.!?][\"'”’)\]]*\s+(?=[\"'“‘(\[]*[A-Z0-9])"
)
TRAILING_TOKEN_PATTERN = re.compile(r"([A-Za-z][A-Za-z.]*|\d+)\.$")
PARAGRAPH_BOUNDARY_PATTERN = re.compile(r"\n\s*\n")
CHUNK_FINGERPRINT_VERSION = "v3"


def normalize_for_hash(text: str) -> str:
    """Normalize insignificant whitespace without changing semantic case."""
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str, *, section_heading: str | None = None) -> str:
    """Identity of what a chunk *says*, independent of where it sits.

    REVIEW1 finding A4: `chunk_hash` folds in position and character offsets, so
    inserting one paragraph changes the hash of every chunk after it. Change detection
    and the delta summary were therefore meaningless on any real edit, and Principle 4
    ("surgically updated") could not hold.

    Two identities are needed, not one:
      content_hash  what the chunk says      -> change detection, assertion reuse (WP7)
      chunk_hash    which occurrence it is   -> chunk_id derivation, stable ordering
    """
    payload = {
        "version": CHUNK_FINGERPRINT_VERSION,
        "text": normalize_for_hash(text),
        "section_heading": normalize_for_hash(section_heading or ""),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


def _pages_of(section: Section) -> list[int]:
    """Ordered, de-duplicated pages a section spans, for page-range citation."""
    pages = [item.page for item in section.provenance if item.page is not None]
    if not pages and section.page is not None:
        pages = [int(section.page)]
    ordered: list[int] = []
    for page in pages:
        value = int(page)
        if value not in ordered:
            ordered.append(value)
    return ordered


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

    def __init__(
        self,
        target_tokens: int = 700,
        max_tokens: int = 800,
        overlap_tokens: int = 100,
        table_block_target_tokens: int | None = None,
    ) -> None:
        if target_tokens <= 0 or max_tokens <= 0:
            raise ValueError("target_tokens and max_tokens must be positive")
        if target_tokens > max_tokens:
            raise ValueError("target_tokens cannot exceed max_tokens")
        # Tables get their own target. At the prose target (700) a table becomes one diffuse
        # block: measured on wp2g, that took MRR 0.749 to 0.678 and graph utility 0.610 to
        # 0.171 against row-per-chunk, while cutting extraction calls 3.3x. A smaller block
        # keeps most of the saving and leaves a table granular enough to rank.
        # None -> the prose target; <= 0 -> one chunk per row (no grouping at all).
        self.table_block_target_tokens = (
            target_tokens if table_block_target_tokens is None else int(table_block_target_tokens)
        )
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

        # Consecutive table rows under one heading are accumulated and emitted as a block
        # rather than one chunk per row. Measured on the working corpus: 72.2% of PDF body
        # chunks were single rows under 100 tokens, which is why the mean active chunk size
        # reads 153.8 against a 600-800 gate, and why extraction costs one provider call
        # per row - 2,000 calls for PDF bodies against 676 for blocks, a 3.0x difference.
        #
        # A row is not the right unit for either job. For extraction, batching rows into
        # one call without merging them would need per-assertion provenance the schema does
        # not carry, so an assertion could silently land on the wrong row and break the
        # exactness gate 6 protects. For retrieval, a 44-token row carries no context: the
        # column meanings live in the header and the comparison lives in the sibling rows.
        # Merging keeps one chunk to one extraction and one citation, by construction.
        pending_rows: list[Section] = []

        def flush_rows() -> None:
            nonlocal position
            if not pending_rows:
                return
            if self.table_block_target_tokens <= 0:
                block_chunks = []
                for row in pending_rows:
                    block_chunks.extend(
                        self._chunk_table_row(document.doc_id, row, position + len(block_chunks))
                    )
            else:
                block_chunks = self._chunk_table_block(
                    document.doc_id, list(pending_rows), position
                )
            chunks.extend(block_chunks)
            position += len(block_chunks)
            pending_rows.clear()

        for section in document.sections:
            if section.section_type is SectionType.TABLE_ROW:
                if pending_rows and section.heading != pending_rows[-1].heading:
                    flush_rows()
                pending_rows.append(section)
                continue

            flush_rows()
            if section.section_type is SectionType.TEXT:
                text_chunks = self._chunk_text_section(document.doc_id, section, position)
                chunks.extend(text_chunks)
                position += len(text_chunks)

        flush_rows()
        return chunks

    def _chunk_table_block(
        self, doc_id: str, rows: list[Section], position: int
    ) -> list[Chunk]:
        """One chunk per run of table rows, split at the token target.

        Falls back to the per-row path for a single row, so a lone row in a section is
        unchanged and the existing behaviour is preserved where there is nothing to group.
        """
        if len(rows) == 1:
            return self._chunk_table_row(doc_id, rows[0], position)

        rendered_rows: list[str] = []
        for section in rows:
            headers = section.table_headers or []
            values = section.table_row or []
            if not headers and not values:
                continue
            rendered_rows.append(
                " | ".join(f"{header}: {value}" for header, value in zip(headers, values))
            )
        if not rendered_rows:
            return []

        blocks: list[list[str]] = []
        current: list[str] = []
        running = 0
        for rendered in rendered_rows:
            tokens = token_count(rendered)
            if current and running + tokens > self.table_block_target_tokens:
                blocks.append(current)
                current, running = [], 0
            current.append(rendered)
            running += tokens
        if current:
            blocks.append(current)

        anchor = rows[0]
        provenance: list[Any] = []
        for section in rows:
            provenance.extend(section.provenance)

        chunks: list[Chunk] = []
        for offset, block in enumerate(blocks):
            text = "\n".join(block)
            chunk_position = position + offset
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=str(uuid.uuid4()),
                    position=chunk_position,
                    section_heading=anchor.heading,
                    page_number=anchor.page,
                    text=text,
                    token_count=token_count(text),
                    char_start=None,
                    char_end=None,
                    chunk_hash=chunk_hash(
                        text,
                        section_heading=anchor.heading,
                        page_number=anchor.page,
                        position=chunk_position,
                    ),
                    content_hash=content_hash(text, section_heading=anchor.heading),
                    provenance=provenance,
                )
            )
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
            content_digest = content_hash(segment, section_heading=section.heading)
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
                    content_hash=content_digest,
                    provenance=list(section.provenance),
                    pages=_pages_of(section),
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
            content_digest = content_hash(text, section_heading=section.heading)
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
                    content_hash=content_digest,
                    provenance=list(section.provenance),
                    pages=_pages_of(section),
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

            chunk_text = self.slice_text(text, current[0].start, current[-1].end)
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
            content_digest = content_hash(chunk_text, section_heading=section.heading)
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
                    content_hash=content_digest,
                    provenance=list(section.provenance),
                    pages=_pages_of(section),
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

    @staticmethod
    def slice_text(source: str, start: int, end: int) -> str:
        """Take chunk text from the source, collapsing only whitespace.

        REVIEW1 finding B13: the previous implementation rebuilt chunk text by joining
        split sentence fragments with a space, so every split point gained one. Because
        the splitter broke on every period, `$26.3 billion` was stored as `$26. 3
        billion` in 11.6% of chunks - and stored text is what gets embedded, retrieved,
        cited and validated against. Slicing the original removes the mutation
        regardless of where the boundaries fall.
        """
        return " ".join(source[max(0, start):max(0, end)].split())

    @staticmethod
    def _is_real_boundary(text: str, period_index: int) -> bool:
        """Reject a period that belongs to a number, an initial, or an abbreviation."""
        if text[period_index] != ".":
            return True  # ! and ? are never ambiguous here
        # A decimal point: digit before and digit after.
        if period_index + 1 < len(text) and text[period_index + 1].isdigit():
            if period_index > 0 and text[period_index - 1].isdigit():
                return False
        match = TRAILING_TOKEN_PATTERN.search(text[max(0, period_index - 24) : period_index + 1])
        if match is None:
            return True
        token = match.group(1)
        if token.casefold() in _ABBREVIATION_SET:
            return False
        if token.casefold().rstrip(".") in _ABBREVIATION_SET:
            return False
        # A single capital letter is an initial (J. Smith), not a sentence end.
        return not (len(token) == 1 and token.isupper())

    def _sentence_boundaries(self, text: str) -> list[int]:
        """Offsets where a sentence may end. Paragraph breaks always qualify."""
        offsets = {match.start() for match in PARAGRAPH_BOUNDARY_PATTERN.finditer(text)}
        for match in SENTENCE_CANDIDATE_PATTERN.finditer(text):
            if self._is_real_boundary(text, match.start()):
                offsets.add(match.end())
        return sorted(offset for offset in offsets if 0 < offset < len(text))

    def _sentence_spans(self, text: str) -> list[SentenceSpan]:
        spans: list[SentenceSpan] = []
        cuts = [0, *self._sentence_boundaries(text), len(text)]
        for index in range(len(cuts) - 1):
            start, end = cuts[index], cuts[index + 1]
            sentence_text = self.slice_text(text, start, end)
            if not sentence_text:
                continue
            sentence_tokens = token_count(sentence_text)
            if sentence_tokens <= self.max_tokens:
                spans.append(
                    SentenceSpan(text=sentence_text, start=start, end=end, tokens=sentence_tokens)
                )
                continue

            # One sentence longer than the hard ceiling is split on word boundaries,
            # still by offset so the text is never rebuilt from pieces.
            token_matches = list(re.finditer(r"\S+", text[start:end]))
            for offset in range(0, len(token_matches), self.max_tokens):
                token_group = token_matches[offset : offset + self.max_tokens]
                segment_start = start + token_group[0].start()
                segment_end = start + token_group[-1].end()
                spans.append(
                    SentenceSpan(
                        text=self.slice_text(text, segment_start, segment_end),
                        start=segment_start,
                        end=segment_end,
                        tokens=len(token_group),
                    )
                )
        return spans

    def _split_oversized_text(self, text: str) -> list[str]:
        tokens = re.findall(r"\S+", text)
        if not tokens:
            return []
        return [" ".join(tokens[index : index + self.max_tokens]) for index in range(0, len(tokens), self.max_tokens)]
