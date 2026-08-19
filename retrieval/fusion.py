"""Deterministic fusion for graph/vector/community retrieval outputs."""

from __future__ import annotations

import logging
import hashlib
import re
from pathlib import Path
from typing import Iterable

from config import Settings
from ingestion.state_store import MetadataStore
from models.query import Citation, FusionOutput, RetrieverResult
from retrieval.reranker import QueryReranker

logger = logging.getLogger(__name__)

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
RELEVANCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "through",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}
RELEVANCE_ACRONYMS = {
    "llm": {"large", "language", "model"},
    "rag": {"retrieval", "augmented", "generation"},
    "nlp": {"natural", "language", "processing"},
}


class FusionEngine:
    """Merge retrieval results with deterministic dedupe/ranking."""

    def __init__(
        self,
        settings: Settings,
        store: MetadataStore | None = None,
        reranker: QueryReranker | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or MetadataStore(settings.metadata_db_path)
        self.reranker = reranker or QueryReranker(settings)

    def fuse(
        self,
        graph_results: Iterable[RetrieverResult],
        vector_results: Iterable[RetrieverResult],
        community_results: Iterable[RetrieverResult],
        community_fallback_used: bool = False,
        user_query: str = "",
    ) -> FusionOutput:
        merged: list[RetrieverResult] = []
        merged.extend(graph_results)
        merged.extend(vector_results)
        merged.extend(community_results)
        self._ensure_evidence_ids(merged)

        deduped = self._dedupe_and_score(merged, user_query=user_query)
        ranked = sorted(deduped.values(), key=self._ranking_score, reverse=True)
        reranked, rerank_mode = self._apply_rerank(user_query=user_query, ranked=ranked)
        selected = self._select_context_items(reranked)

        # _select_context_items places the best graph and vector candidates first.
        # Preserve that order so a graph-heavy result set cannot exhaust the
        # character budget before the selected vector evidence is packed.
        packed_items, context_text = self._pack_context(selected)
        top_scores = [self._confidence_score(item) for item in packed_items[:3]]
        citations = self._build_citations(packed_items)
        graph_paths = [
            str(item.provenance.get("graph_path") or item.text_or_path)
            for item in packed_items
            if item.source_type == "graph" and item.text_or_path
        ]
        retrieval_sources = list(dict.fromkeys(item.source_type for item in packed_items))
        logger.info(
            "fusion_rerank mode=%s merged=%d deduped=%d selected=%d",
            rerank_mode,
            len(merged),
            len(deduped),
            len(packed_items),
        )

        return FusionOutput(
            context_items=packed_items,
            citations=citations,
            graph_paths=graph_paths,
            retrieval_sources=retrieval_sources,
            top_scores=top_scores,
            context_text=context_text,
            community_fallback_used=community_fallback_used,
        )

    def _apply_rerank(self, user_query: str, ranked: list[RetrieverResult]) -> tuple[list[RetrieverResult], str]:
        if not ranked:
            return [], "empty"
        try:
            return self.reranker.rerank(user_query, ranked)
        except Exception:
            return ranked, "deterministic_fallback"

    def _select_context_items(self, ranked: list[RetrieverResult]) -> list[RetrieverResult]:
        max_items = max(1, int(self.settings.query_context_max_items))
        has_graph = any(item.source_type == "graph" for item in ranked)
        has_vector = any(item.source_type == "vector" for item in ranked)
        if not (has_graph and has_vector):
            return ranked[:max_items]

        selected: list[RetrieverResult] = []
        selected_indices: set[int] = set()
        graph_index = next((index for index, item in enumerate(ranked) if item.source_type == "graph"), None)
        vector_index = next((index for index, item in enumerate(ranked) if item.source_type == "vector"), None)
        for index in (graph_index, vector_index):
            if index is None or index in selected_indices:
                continue
            selected.append(ranked[index])
            selected_indices.add(index)

        for index, item in enumerate(ranked):
            if len(selected) >= max_items:
                break
            if index in selected_indices:
                continue
            selected.append(item)
            selected_indices.add(index)

        return selected[:max_items]

    def _dedupe_and_score(
        self,
        results: list[RetrieverResult],
        *,
        user_query: str = "",
    ) -> dict[str, RetrieverResult]:
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
            relevance = self._lexical_relevance(user_query, item)
            item.score_fused = base_score * source_weight * (0.75 + (0.25 * relevance))
            item.score_reranked = None
            key = item.dedupe_key()
            existing = deduped.get(key)
            if existing is None or item.score_fused > existing.score_fused:
                deduped[key] = item
        return deduped

    def _lexical_relevance(self, user_query: str, item: RetrieverResult) -> float:
        query_tokens = self._relevance_tokens(user_query)
        if not query_tokens:
            return 1.0
        if item.source_type == "graph":
            evidence_text = str(
                item.provenance.get("graph_path")
                or item.text_or_path.split("\nExact source chunk:", 1)[0]
            )
        else:
            evidence_text = item.text_or_path
        evidence_tokens = self._relevance_tokens(evidence_text)
        if not evidence_tokens:
            return 0.0
        overlap = len(query_tokens.intersection(evidence_tokens))
        return max(0.0, min(1.0, overlap / max(1, min(8, len(query_tokens)))))

    @staticmethod
    def _relevance_tokens(value: str) -> set[str]:
        raw_tokens = re.findall(r"[a-z0-9]+", str(value or "").casefold())
        tokens: set[str] = set()
        for token in raw_tokens:
            normalized = token[:-1] if len(token) > 3 and token.endswith("s") else token
            if normalized and normalized not in RELEVANCE_STOPWORDS:
                tokens.add(normalized)
                tokens.update(RELEVANCE_ACRONYMS.get(normalized, set()))
        if "how" in raw_tokens and "many" in raw_tokens:
            tokens.add("count")
        return tokens

    def _ranking_score(self, item: RetrieverResult) -> float:
        if item.score_reranked is not None:
            return float(item.score_reranked)
        return float(item.score_fused)

    def _confidence_score(self, item: RetrieverResult) -> float:
        score = self._ranking_score(item)
        return max(0.0, min(1.0, score))

    def _build_citations(self, results: list[RetrieverResult]) -> list[Citation]:
        citable_items = [item for item in results if item.provenance.get("chunk_id")]
        doc_name_map = self._build_doc_name_map(citable_items)
        citations: list[Citation] = []
        by_chunk: dict[str, Citation] = {}
        max_citations = max(1, int(self.settings.query_citation_max))

        for item in citable_items:
            chunk_id = str(item.provenance.get("chunk_id") or item.source_id or "")
            if not chunk_id:
                continue
            evidence_id = str(item.provenance.get("evidence_id") or "")
            assertion_id = str(item.provenance.get("assertion_id") or "")
            existing = by_chunk.get(chunk_id)
            if existing is not None:
                if evidence_id and evidence_id not in existing.evidence_ids:
                    existing.evidence_ids.append(evidence_id)
                if assertion_id and assertion_id not in existing.assertion_ids:
                    existing.assertion_ids.append(assertion_id)
                continue
            if len(citations) >= max_citations:
                continue
            doc_id = item.provenance.get("doc_id")
            page = item.provenance.get("page")
            excerpt = str(item.provenance.get("evidence_text") or item.text_or_path)
            citation = Citation(
                citation_id=f"citation:{chunk_id}",
                chunk_id=chunk_id,
                doc_id=str(doc_id) if doc_id else None,
                doc_name=doc_name_map.get(str(doc_id), str(doc_id) if doc_id else None),
                page=int(page) if isinstance(page, int) or (isinstance(page, str) and page.isdigit()) else None,
                excerpt=self._trim_excerpt(excerpt),
                evidence_ids=[evidence_id] if evidence_id else [],
                assertion_ids=[assertion_id] if assertion_id else [],
            )
            citations.append(citation)
            by_chunk[chunk_id] = citation
        return citations

    def _ensure_evidence_ids(self, results: list[RetrieverResult]) -> None:
        for item in results:
            existing = item.provenance.get("evidence_id")
            if isinstance(existing, str) and existing:
                continue
            if item.source_type == "vector":
                chunk_id = str(item.provenance.get("chunk_id") or item.source_id)
                item.provenance["evidence_id"] = f"chunk:{chunk_id}"
            elif item.source_type == "graph" and item.provenance.get("assertion_id"):
                item.provenance["evidence_id"] = f"assertion:{item.provenance['assertion_id']}"
            else:
                digest = hashlib.sha256(
                    f"{item.source_type}|{item.source_id}|{item.text_or_path}".encode("utf-8")
                ).hexdigest()[:20]
                item.provenance["evidence_id"] = f"{item.source_type}:{digest}"

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

    def _pack_context(self, results: list[RetrieverResult]) -> tuple[list[RetrieverResult], str]:
        if not results:
            return [], ""

        max_chars = max(500, int(self.settings.query_context_max_chars))
        lines: list[str] = []
        packed: list[RetrieverResult] = []
        hydrated_graph_chunks: set[str] = set()
        total = 0
        for item in results:
            graph_chunk_id = ""
            include_graph_source = True
            if item.source_type == "graph":
                graph_chunk_id = str(item.provenance.get("chunk_id") or "")
                include_graph_source = not graph_chunk_id or graph_chunk_id not in hydrated_graph_chunks
            block = self._context_block(item, include_graph_source=include_graph_source)
            if not block:
                continue
            projected = total + len(block) + 1
            if projected > max_chars:
                if not lines:
                    lines.append(block[:max_chars].rstrip())
                    packed.append(item)
                    if graph_chunk_id and include_graph_source:
                        hydrated_graph_chunks.add(graph_chunk_id)
                break
            lines.append(block)
            packed.append(item)
            if graph_chunk_id and include_graph_source:
                hydrated_graph_chunks.add(graph_chunk_id)
            total = projected
        return packed, "\n".join(lines)

    def _context_block(
        self,
        item: RetrieverResult,
        *,
        include_graph_source: bool = True,
    ) -> str:
        evidence_id = str(item.provenance.get("evidence_id") or item.source_id)
        if item.source_type == "graph":
            graph_path = str(item.provenance.get("graph_path") or item.text_or_path)
            block = f"[EVIDENCE id={evidence_id} type=graph_assertion]\nAssertion: {graph_path}"
            evidence_text = str(item.provenance.get("evidence_text") or "").strip()
            if include_graph_source and evidence_text:
                chunk_id = item.provenance.get("chunk_id")
                doc_id = item.provenance.get("doc_id")
                section = item.provenance.get("section_heading")
                page = item.provenance.get("page")
                block += (
                    f"\nSource: doc_id={doc_id}, chunk_id={chunk_id}, section={section}, page={page}"
                    f"\nExact source chunk: {evidence_text}"
                )
            return block
        if item.source_type == "community":
            return f"[EVIDENCE id={evidence_id} type=community]\n{item.text_or_path}"
        chunk_id = str(item.provenance.get("chunk_id") or item.source_id)
        doc_id = item.provenance.get("doc_id")
        section = item.provenance.get("section_heading")
        page = item.provenance.get("page")
        source_line = f"Source: doc_id={doc_id}, chunk_id={chunk_id}, section={section}, page={page}"
        return f"[EVIDENCE id={evidence_id} type=document_chunk]\n{source_line}\nText: {item.text_or_path}"
