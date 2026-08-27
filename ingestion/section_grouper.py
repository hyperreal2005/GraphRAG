"""Merge parser items into logical sections before coreference and chunking.

REVIEW1 finding `B1`: `DocumentParser._sections_from_docling` emits one `Section` per
Docling layout item — a paragraph, a list item, a figure caption. `SemanticChunker` then
chunks each section independently and never packs across boundaries, so no section ever
approaches the 700-token target and the chunker degenerates into a passthrough. Measured
on the PDF layer: sections equal chunks exactly, mean 42-56 words, zero chunks reaching
600, and therefore no overlap emitted anywhere in the corpus.

This stage restores the architecture's Stage 2/3 contract. It groups consecutive text
items that belong to the same logical section so that:

  * coreference sees a whole section, which is what reference section 3.2 requires and
    warns will fail on short spans;
  * the chunker receives enough text to actually window at 700/800 with 100 overlap.

The group target (`section_group_max_tokens`, default 2000) is deliberately well above
the chunk target. Grouping to ~700 would yield exactly one chunk per group and overlap
would still never fire; grouping to ~2000 gives the chunker roughly three windows to
work with.

Two rules were added after measuring the first implementation on real papers, where it
only lifted the mean chunk from 56 to 115 words:

1. **A table row does not break a text run.** Docling interleaves tables through body
   text, and flushing on every one shattered the runs: 83 flushes against 30 genuine
   heading changes in a single paper. Table rows still become their own chunks, as the
   architecture requires; they simply no longer split the prose around them.

2. **A content-kind change is a hard boundary.** Body text must never merge with the
   bibliography: doing so would feed reference entries back to the extractor, undoing
   the contamination fix recorded in ERRORS.md on 2026-08-11 and defeating
   `graph_exclude_reference_sections`. The sequential classifier already used after
   chunking decides this, so grouping and extraction agree on where references begin.

3. **A heading change is a soft boundary.** Docling's `section_header` granularity is
   subsection-level - 31 headings across 181 items, roughly 320 words each - so grouping
   strictly by heading cannot reach the 600-800 band however well it works. A heading
   change therefore only ends a group once that group already carries
   `section_group_target_tokens`; below that the group absorbs the next subsection. Large
   sections keep their own identity, tiny ones merge with their neighbours.
"""

from __future__ import annotations

from dataclasses import replace

from ingestion.content_classifier import classify_content_sequence
from models.ingestion import Section, SectionProvenance, SectionType


def _token_count(text: str | None) -> int:
    return len((text or "").split())


def _normalized_heading(section: Section) -> str:
    return " ".join(str(section.heading or "").split()).casefold()


def _provenance_for(section: Section) -> list[SectionProvenance]:
    """Existing provenance if the parser supplied it, else derive it from the item."""
    if section.provenance:
        return [
            item if item.heading is not None else replace(item, heading=section.heading)
            for item in section.provenance
        ]
    return [
        SectionProvenance(
            section_id=section.section_id,
            page=section.page,
            char_start=section.char_start,
            char_end=section.char_end,
            bbox=section.bbox,
            heading=section.heading,
        )
    ]


def _pages_adjacent(left: Section, right: Section, tolerance: int) -> bool:
    """A section may run across a page break, but not across an arbitrary gap."""
    if left.page is None or right.page is None:
        return True
    return abs(int(right.page) - int(left.page)) <= max(0, int(tolerance))


def _can_extend(
    members: list[Section],
    candidate: Section,
    *,
    accumulated_tokens: int,
    target_tokens: int,
    max_tokens: int,
    page_gap_tolerance: int,
    same_content_kind: bool,
) -> bool:
    if not members:
        return True
    # Hard boundary: prose must never merge into the bibliography.
    if not same_content_kind:
        return False
    if not _pages_adjacent(members[-1], candidate, page_gap_tolerance):
        return False
    # Soft heading boundary: split on a heading change only once the group already
    # carries enough text to stand on its own.
    if _normalized_heading(members[-1]) != _normalized_heading(candidate):
        if accumulated_tokens >= max(1, int(target_tokens)):
            return False
    # Hard cap so coreference and chunking both stay bounded. A single item that already
    # exceeds the cap still forms its own group rather than being dropped.
    return accumulated_tokens + _token_count(candidate.text) <= max(1, int(max_tokens))


def _merge(members: list[Section], index: int) -> Section:
    """Fuse a run of text items into one logical section."""
    if len(members) == 1:
        single = members[0]
        return replace(single, provenance=_provenance_for(single))

    texts = [str(item.text or "").strip() for item in members]
    provenance: list[SectionProvenance] = []
    for item in members:
        provenance.extend(_provenance_for(item))

    pages = [item.page for item in members if item.page is not None]
    return Section(
        section_id=f"group-{index}",
        section_type=SectionType.TEXT,
        heading=members[0].heading,
        # Blank line between items keeps paragraph structure legible to the model and
        # gives the sentence splitter an unambiguous boundary.
        text="\n\n".join(text for text in texts if text),
        page=pages[0] if pages else None,
        char_start=members[0].char_start,
        char_end=members[-1].char_end,
        bbox=members[0].bbox,
        provenance=provenance,
    )


def group_sections(
    sections: list[Section],
    *,
    target_tokens: int = 700,
    max_tokens: int = 2000,
    page_gap_tolerance: int = 1,
) -> list[Section]:
    """Merge consecutive text sections into logical sections; keep table rows separate.

    Table rows are never merged into a text group - the architecture requires one chunk
    per row with its headers prepended - but they no longer interrupt the prose run
    around them either.
    """
    # Content kind is decided over the whole ordered sequence, because "References"
    # continues until an appendix heading rather than ending at the next heading.
    content_kinds = classify_content_sequence([item.heading for item in sections])

    grouped: list[Section] = []
    deferred_rows: list[Section] = []
    pending: list[Section] = []
    pending_kind: str | None = None
    pending_tokens = 0
    group_index = 1

    def flush() -> None:
        nonlocal pending, pending_tokens, group_index, deferred_rows, pending_kind
        if pending:
            grouped.append(_merge(pending, group_index))
            group_index += 1
            pending = []
            pending_tokens = 0
            pending_kind = None
        if deferred_rows:
            grouped.extend(deferred_rows)
            deferred_rows = []

    for section, content_kind in zip(sections, content_kinds):
        if section.section_type is not SectionType.TEXT:
            # Emitted after the surrounding text group closes, so an interleaved table
            # cannot shatter the prose run (see module docstring, rule 1).
            deferred_rows.append(section)
            continue
        if not str(section.text or "").strip():
            continue

        if _can_extend(
            pending,
            section,
            accumulated_tokens=pending_tokens,
            target_tokens=target_tokens,
            max_tokens=max_tokens,
            page_gap_tolerance=page_gap_tolerance,
            same_content_kind=(pending_kind is None or pending_kind == content_kind),
        ):
            pending.append(section)
            pending_tokens += _token_count(section.text)
            pending_kind = content_kind
            continue

        flush()
        pending = [section]
        pending_tokens = _token_count(section.text)
        pending_kind = content_kind

    flush()
    return grouped
