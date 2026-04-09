"""Deterministic fusion for graph/vector/community retrieval outputs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from config import Settings
from ingestion.state_store import MetadataStore
from models.query import Citation, FusionOutput, RetrieverResult


SOURCE_WEIGHTS = {
    "graph": 1.0,
    "vector": 0.9,
    "community": 0.7,
}
SOURCE_PRIORITY = {
    "graph": 0,
    "vector": 1,
    "community": 2,
}
BOILERPLATE_HINTS = (
    "e-issn",
    "issn",
    "all rights reserved",
    "copyright",
    "<! -- image -->",
    "www.",
    "vol.",
    "no.",
    "doi:",
    "journal",
)


class FusionEngine:
    """Merge retrieval results with deterministic dedupe/ranking."""

    def __init__(self, settings: Settings, store: MetadataStore | None = None) -> None:
        self.settings = settings
        self.store = store or MetadataStore(settings.metadata_db_path)

    def fuse(
        self,
        graph_results: Iterable[RetrieverResult],
        vector_results: Iterable[RetrieverResult],
        community_results: Iterable[RetrieverResult],
        community_fallback_used: bool = False,
    ) -> FusionOutput:
        merged: list[RetrieverResult] = []
        merged.extend(graph_results)
        merged.extend(vector_results)
        merged.extend(community_results)

        deduped = self._dedupe_and_score(merged)
        ranked = sorted(deduped.values(), key=lambda item: item.score_fused, reverse=True)
        selected = ranked[: max(1, int(self.settings.query_context_max_items))]
        top_scores = [float(item.score_fused) for item in selected[:3]]

        prioritized = sorted(
            selected,
            key=lambda item: (SOURCE_PRIORITY.get(item.source_type, 99), -item.score_fused),
        )
        citations = self._build_citations(prioritized)
        graph_paths = [item.text_or_path for item in prioritized if item.source_type == "graph" and item.text_or_path]
        retrieval_sources = list(dict.fromkeys(item.source_type for item in prioritized))
        context_text = self._build_context_text(prioritized)

        return FusionOutput(
            context_items=prioritized,
            citations=citations,
            graph_paths=graph_paths,
            retrieval_sources=retrieval_sources,
            top_scores=top_scores,
            context_text=context_text,
            community_fallback_used=community_fallback_used,
        )

    def _dedupe_and_score(self, results: list[RetrieverResult]) -> dict[str, RetrieverResult]:
        deduped: dict[str, RetrieverResult] = {}
        for item in results:
            if (
                item.source_type == "vector"
                and self.settings.query_filter_boilerplate_chunks
                and self._is_boilerplate_chunk(item.text_or_path)
            ):
                continue
            source_weight = SOURCE_WEIGHTS.get(item.source_type, 0.6)
            base_score = max(0.0, min(1.0, float(item.score_raw)))
            item.score_fused = base_score * source_weight
            key = item.dedupe_key()
            existing = deduped.get(key)
            if existing is None or item.score_fused > existing.score_fused:
                deduped[key] = item
        return deduped

    def _build_citations(self, results: list[RetrieverResult]) -> list[Citation]:
        vector_items = [item for item in results if item.source_type == "vector"]
        doc_name_map = self._build_doc_name_map(vector_items)
        citations: list[Citation] = []
        seen_chunks: set[str] = set()
        max_citations = max(1, int(self.settings.query_citation_max))

        for item in vector_items:
            if len(citations) >= max_citations:
                break
            chunk_id = str(item.provenance.get("chunk_id") or item.source_id or "")
            if not chunk_id or chunk_id in seen_chunks:
                continue
            seen_chunks.add(chunk_id)
            doc_id = item.provenance.get("doc_id")
            page = item.provenance.get("page")
            citations.append(
                Citation(
                    chunk_id=chunk_id,
                    doc_id=str(doc_id) if doc_id else None,
                    doc_name=doc_name_map.get(str(doc_id), str(doc_id) if doc_id else None),
                    page=int(page) if isinstance(page, int) or (isinstance(page, str) and page.isdigit()) else None,
                    excerpt=self._trim_excerpt(item.text_or_path),
                )
            )
        return citations

    def _is_boilerplate_chunk(self, text: str) -> bool:
        normalized = " ".join((text or "").lower().split())
        if not normalized:
            return True
        hint_hits = sum(1 for hint in BOILERPLATE_HINTS if hint in normalized)
        # Frequent PDF table/header fragments often combine separators and metadata-heavy tokens.
        has_table_rule = "|" in normalized and normalized.count("-") >= 8
        has_meta_pattern = bool(re.search(r"\bvol\.\s*\d+\b", normalized)) and bool(
            re.search(r"\b(?:no\.?|number)\s*\d+\b", normalized)
        )
        alpha_chars = sum(1 for char in normalized if char.isalpha())
        non_alpha_chars = len(normalized) - alpha_chars
        alpha_ratio = alpha_chars / max(1, len(normalized))
        noisy_surface = non_alpha_chars > alpha_chars and alpha_ratio < 0.45

        if hint_hits >= 3:
            return True
        if hint_hits >= 2 and (has_table_rule or has_meta_pattern or noisy_surface):
            return True
        if has_table_rule and noisy_surface:
            return True
        return False

    def _build_doc_name_map(self, vector_items: list[RetrieverResult]) -> dict[str, str]:
        doc_name_map: dict[str, str] = {}
        for item in vector_items:
            doc_id = item.provenance.get("doc_id")
            if not isinstance(doc_id, str) or not doc_id or doc_id in doc_name_map:
                continue
            row = self.store.get_document_by_id(doc_id)
            if row is None:
                doc_name_map[doc_id] = doc_id
                continue
            source_path = str(row["source_path"] or doc_id)
            doc_name_map[doc_id] = Path(source_path).name
        return doc_name_map

    def _trim_excerpt(self, text: str, limit: int = 280) -> str:
        normalized = " ".join((text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    def _build_context_text(self, results: list[RetrieverResult]) -> str:
        if not results:
            return ""

        max_chars = max(500, int(self.settings.query_context_max_chars))
        lines: list[str] = []
        total = 0
        for item in results:
            block = self._context_block(item)
            if not block:
                continue
            projected = total + len(block) + 1
            if projected > max_chars:
                break
            lines.append(block)
            total = projected
        return "\n".join(lines)

    def _context_block(self, item: RetrieverResult) -> str:
        if item.source_type == "graph":
            return f"[GRAPH PATH]\n{item.text_or_path}"
        if item.source_type == "community":
            return f"[COMMUNITY SUMMARY]\n{item.text_or_path}"
        chunk_id = str(item.provenance.get("chunk_id") or item.source_id)
        doc_id = item.provenance.get("doc_id")
        section = item.provenance.get("section_heading")
        page = item.provenance.get("page")
        source_line = f"Source: doc_id={doc_id}, chunk_id={chunk_id}, section={section}, page={page}"
        return f"[DOCUMENT CHUNK]\n{source_line}\nText: {item.text_or_path}"
