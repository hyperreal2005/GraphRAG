"""What a passage's document is: its title, and for news its publisher and date.

MultiHop-RAG questions name publishers and dates ("reports by The Age on October 23, 2023"),
and arXiv questions name papers. The passages the system showed carried none of that, only
chunk text and a doc_id UUID. So the generator could not attribute a fact it had read, and
both judges ruled such passages unrelated or such claims unsupported (REFACTOR3 WP19, WP23).

The corpus already records all three. `data/corpus/MANIFEST.json` has each file's title, and
`structured/news_index.csv` has each article's publisher and publication time, joined to the
manifest by URL. A card is stored on the document row at ingest and shown as one line with
every passage from that document.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DocumentCard:
    title: str = ""
    publisher: str = ""
    published: str = ""  # YYYY-MM-DD

    def render(self) -> str:
        """`"Title" - Publisher, 2023-10-31`, or whichever parts are known."""
        parts = []
        if self.title:
            parts.append(f'"{self.title}"')
        meta = ", ".join(part for part in (self.publisher, self.published) if part)
        if meta:
            parts.append(meta)
        return " - ".join(parts)

    def __bool__(self) -> bool:
        return bool(self.title or self.publisher or self.published)


class CorpusCatalog:
    """Cards for the files the corpus builder recorded, looked up by file name."""

    def __init__(self, cards: dict[str, DocumentCard]) -> None:
        self._cards = cards

    def __len__(self) -> int:
        return len(self._cards)

    @classmethod
    def load(cls, manifest: Path, news_index: Path | None = None) -> "CorpusCatalog":
        if not manifest.exists():
            return cls({})
        data = json.loads(manifest.read_text(encoding="utf-8"))
        by_url: dict[str, dict[str, str]] = {}
        by_title: dict[str, dict[str, str]] = {}
        if news_index is not None and news_index.exists():
            with news_index.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    by_url[(row.get("url") or "").strip()] = row
                    by_title[(row.get("title") or "").strip()] = row
        cards: dict[str, DocumentCard] = {}
        for layer in (data.get("layers") or {}).values():
            for entry in layer.get("files") or []:
                title = str(entry.get("title") or "").strip()
                row = by_url.get(str(entry.get("origin_url") or "").strip()) or by_title.get(title)
                publisher = str((row or {}).get("source") or "").strip()
                published = str((row or {}).get("published_at") or "").strip()[:10]
                if title or publisher:
                    cards[Path(str(entry["path"])).name] = DocumentCard(title, publisher, published)
        return cls(cards)

    def card_for(self, source_path: Path | str) -> DocumentCard | None:
        return self._cards.get(Path(str(source_path)).name)

    def publishers(self) -> set[str]:
        """Every outlet the corpus knows of, including ones whose articles were never
        ingested - a question can name "The Age" whether or not the corpus holds it."""
        return {card.publisher for card in self._cards.values() if card.publisher}


def fallback_card(source_path: Path | str, headings: Iterable[str | None] = ()) -> DocumentCard:
    """For a file the manifest does not know: its first heading, else its file name."""
    for heading in headings:
        if heading and heading.strip():
            return DocumentCard(title=" ".join(heading.split()))
    return DocumentCard(title=Path(str(source_path)).stem.replace("-", " ").replace("_", " "))
