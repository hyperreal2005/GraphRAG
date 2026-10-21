"""How much of a PDF reached the corpus, page by page (ROADMAP E3).

Docling can report success while dropping a page, and a page with no text layer was refused
outright, so "ingested" never meant "covered". The first attempt to measure this compared
`chunks.page` values, but that column holds a chunk's first page only, and it called pages
missing that were inside multi-page chunks (docs/ERRORS.md, 2026-10-14). This measures content.

Each page's text layer is compared with the document's stored chunk text two ways:

* **sequence overlap** - the share of the page's 5-word sequences found in the chunks. Order
  matters, so prose is matched exactly;
* **word overlap** - the share of the page's distinct words of 3 or more letters found anywhere
  in the chunks. Order does not matter, so a table that Docling linearised differently from the
  text layer still counts.

A page is **covered** at sequence overlap >= 0.5, **partial** at word overlap >= 0.7, otherwise
**missing**; a page whose text layer is empty is **no_text** and needs OCR, not comparison. The
thresholds were fixed before any document was measured.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

_WORD = re.compile(r"[a-z0-9]+")
SHINGLE = 5
COVERED_SEQUENCE = 0.5
PARTIAL_WORDS = 0.7


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def _shingles(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


@dataclass
class PageCoverage:
    page: int
    status: str  # covered | partial | missing | no_text
    sequence_overlap: float | None = None
    word_overlap: float | None = None


@dataclass
class DocumentCoverage:
    pages: int
    pages_with_text: int
    covered: int
    partial: int
    missing: list[int] = field(default_factory=list)
    no_text: list[int] = field(default_factory=list)
    per_page: list[PageCoverage] = field(default_factory=list)

    @property
    def coverage(self) -> float | None:
        """Covered or partially covered pages, over the pages that have a text layer."""
        if not self.pages_with_text:
            return None
        return round((self.covered + self.partial) / self.pages_with_text, 4)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["coverage"] = self.coverage
        return data


def classify_page(page_text: str, chunk_shingles: set[tuple[str, ...]], chunk_words: set[str], page: int) -> PageCoverage:
    words = _words(page_text)
    if not words:
        return PageCoverage(page=page, status="no_text")
    shingles = _shingles(words)
    sequence = (len(shingles & chunk_shingles) / len(shingles)) if shingles else None
    distinct = {w for w in words if len(w) >= 3}
    word = (len(distinct & chunk_words) / len(distinct)) if distinct else None
    if sequence is not None and sequence >= COVERED_SEQUENCE:
        status = "covered"
    elif word is not None and word >= PARTIAL_WORDS:
        status = "partial"
    elif sequence is None and word is not None and word >= COVERED_SEQUENCE:
        status = "covered"  # a page too short for a 5-word sequence, judged on its words
    else:
        status = "missing"
    return PageCoverage(
        page=page,
        status=status,
        sequence_overlap=None if sequence is None else round(sequence, 4),
        word_overlap=None if word is None else round(word, 4),
    )


def coverage_report(pdf_path: Path, chunk_texts: Iterable[str], recovery: dict | None = None) -> dict:
    """The stored record: the measure, with the parser's account of pages without a text layer.

    A page the parser found blank is not a gap; a page it read by OCR is content whose text
    layer cannot be compared, so it is reported as such rather than as missing.
    """
    recovery = recovery or {}
    blank, ocr = set(recovery.get("blank") or []), set(recovery.get("ocr") or [])
    report = measure_pdf_coverage(pdf_path, chunk_texts).to_dict()
    report["blank"] = sorted(blank)
    report["ocr"] = sorted(ocr)
    report["no_text"] = [p for p in report["no_text"] if p not in blank and p not in ocr]
    return report


def measure_pdf_coverage(pdf_path: Path, chunk_texts: Iterable[str]) -> DocumentCoverage:
    """Compare every page's text layer with the document's chunk text."""
    from pypdf import PdfReader

    words = _words("\n".join(chunk_texts))
    chunk_shingles, chunk_words = _shingles(words), set(words)
    reader = PdfReader(str(pdf_path))
    per_page = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        per_page.append(classify_page(text, chunk_shingles, chunk_words, number))
    return DocumentCoverage(
        pages=len(per_page),
        pages_with_text=sum(p.status != "no_text" for p in per_page),
        covered=sum(p.status == "covered" for p in per_page),
        partial=sum(p.status == "partial" for p in per_page),
        missing=[p.page for p in per_page if p.status == "missing"],
        no_text=[p.page for p in per_page if p.status == "no_text"],
        per_page=per_page,
    )
