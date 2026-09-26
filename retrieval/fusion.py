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
# Reciprocal-rank fusion constant. Sources are combined by *position* within their own
# ranking rather than by raw score, because the raw scores are not the same quantity: a
# graph item's is assertion confidence (median 0.950, p90 1.000) and a vector item's is
# cosine similarity (median 0.691, max 0.766). Multiplying those by source weights left
# the two ranges non-overlapping, so every graph item outranked every vector item whatever
# the question was. 60 is the constant from the RRF literature; measured MRR was flat
# across k from 5 to 60 once the scale mismatch was gone, so nothing here is tuned to it.
RRF_K = 60
RELEVANCE_WEIGHT = 0.25
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
            rerank_mode=rerank_mode,
        )

    def _apply_rerank(self, user_query: str, ranked: list[RetrieverResult]) -> tuple[list[RetrieverResult], str]:
        if not ranked:
            return [], "empty"
        try:
            return self.reranker.rerank(user_query, ranked)
        except Exception:
            return ranked, "deterministic_fallback"

    def _select_context_items(self, ranked: list[RetrieverResult]) -> list[RetrieverResult]:
        """Take the best items in rank order.

        This used to force the top graph item and the top vector item to the front, so
        that a graph-heavy ranking could not starve the context of chunk text. That was a
        workaround for scores that were not comparable across sources; with rank fusion
        the ranking already interleaves the two, and promoting a fixed item past better
        evidence costs the top position for no reason. Measured on 60 held-out questions,
        removing it moved MRR from 0.690 to 0.798 with recall unchanged.
        """
        return ranked[: max(1, int(self.settings.query_context_max_items))]

    def _dedupe_and_score(
        self,
        results: list[RetrieverResult],
        *,
        user_query: str = "",
    ) -> dict[str, RetrieverResult]:
        ranks = self._source_ranks(results)
        deduped: dict[str, RetrieverResult] = {}
        for item in results:
            if (
                item.source_type == "vector"
                and self.settings.query_filter_boilerplate_chunks
                and self._is_boilerplate_chunk(item.text_or_path)
            ):
                continue
            source_weight = SOURCE_WEIGHTS.get(item.source_type, 0.6)
            reciprocal_rank = 1.0 / (RRF_K + ranks[id(item)])
            relevance = self._lexical_relevance(user_query, item)
            item.score_fused = (
                reciprocal_rank
                * source_weight
                * ((1.0 - RELEVANCE_WEIGHT) + (RELEVANCE_WEIGHT * relevance))
            )
            item.score_reranked = None
            key = item.dedupe_key()
            existing = deduped.get(key)
            if existing is None or item.score_fused > existing.score_fused:
                deduped[key] = item
        return deduped

    def _source_ranks(self, results: list[RetrieverResult]) -> dict[int, int]:
        """Position of each item within its own source's ranking, best first.

        Graph assertions are ordered by the cosine of the chunk they cite rather than by
        their own confidence. Confidence says how sure the extractor was that a fact is
        true, which is a property of the fact and not of the question -- ranking by it
        puts graph evidence in an order unrelated to what was asked. Where the vector
        source retrieved the same chunk, its cosine is a real relevance signal already
        on hand; assertions citing a chunk the vector source did not surface sort last,
        which is the correct default for evidence with no measured relevance.
        """
        chunk_cosine: dict[str, float] = {}
        for item in results:
            if item.source_type != "vector":
                continue
            chunk_id = str(item.provenance.get("chunk_id") or "")
            if chunk_id:
                chunk_cosine[chunk_id] = max(
                    chunk_cosine.get(chunk_id, 0.0), float(item.score_raw)
                )

        def relevance_key(item: RetrieverResult) -> float:
            if item.source_type != "graph":
                return float(item.score_raw)
            return chunk_cosine.get(str(item.provenance.get("chunk_id") or ""), 0.0)

        grouped: dict[str, list[RetrieverResult]] = {}
        for item in results:
            grouped.setdefault(item.source_type, []).append(item)

        ranks: dict[int, int] = {}
        for group in grouped.values():
            ordered = sorted(group, key=relevance_key, reverse=True)
            for position, item in enumerate(ordered, start=1):
                ranks[id(item)] = position
        return ranks

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
        # A graph block that displayed its source passage makes that passage citable as
        # `chunk:<id>`. Without this alias a claim citing the passage it was shown maps to
        # no citation, and the claim is discarded as unsupported.
        for item in citable_items:
            if item.source_type != "graph" or not item.provenance.get("source_chunk_shown"):
                continue
            chunk_id = str(item.provenance.get("chunk_id") or "")
            citation = by_chunk.get(chunk_id)
            alias = f"chunk:{chunk_id}"
            if citation is not None and alias not in citation.evidence_ids:
                citation.evidence_ids.append(alias)
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
                if str(item.provenance.get("evidence_text") or "").strip():
                    # The block displayed its source passage, not just the assertion. The
                    # generator treats that passage as citable in its own right; see
                    # AnswerGenerator._alias_shown_source_chunks.
                    item.provenance["source_chunk_shown"] = True
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
