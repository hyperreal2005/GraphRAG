"""Typed contracts for query pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    """Supported query categories for retrieval routing."""

    LOCAL = "local"
    MULTIHOP = "multihop"
    COMPARATIVE = "comparative"
    GLOBAL = "global"
    HYBRID = "hybrid"

    @classmethod
    def from_value(cls, value: str | None) -> "QueryType":
        normalized = str(value or "").strip().lower()
        for item in cls:
            if item.value == normalized:
                return item
        return cls.LOCAL


@dataclass
class QueryAnalysis:
    """Structured output from analyzer stage."""

    query_type: QueryType
    mentioned_entities: list[str] = field(default_factory=list)
    retrieval_flags: dict[str, bool] = field(default_factory=dict)
    sub_queries: list[str] = field(default_factory=list)
    temporal_filter: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "mentioned_entities": list(self.mentioned_entities),
            "retrieval_flags": dict(self.retrieval_flags),
            "sub_queries": list(self.sub_queries),
            "temporal_filter": self.temporal_filter,
        }


@dataclass
class LinkedEntity:
    """Entity-linking decision for one query mention."""

    mention: str
    status: str  # resolved | ambiguous | unresolved
    similarity: float
    entity_id: str | None = None
    entity_name: str | None = None
    entity_type: str | None = None
    candidate_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mention": self.mention,
            "status": self.status,
            "similarity": float(self.similarity),
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "candidate_ids": list(self.candidate_ids),
        }


@dataclass
class RetrieverResult:
    """Unified retrieval item contract for graph/vector/community sources."""

    source_type: str  # graph | vector | community
    source_id: str
    text_or_path: str
    score_raw: float
    provenance: dict[str, Any] = field(default_factory=dict)
    score_fused: float = 0.0
    score_reranked: float | None = None

    def dedupe_key(self) -> str:
        chunk_id = self.provenance.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            return f"{self.source_type}:{chunk_id}"
        return f"{self.source_type}:{self.source_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "text_or_path": self.text_or_path,
            "score_raw": float(self.score_raw),
            "score_fused": float(self.score_fused),
            "score_reranked": float(self.score_reranked) if self.score_reranked is not None else None,
            "provenance": dict(self.provenance),
        }


@dataclass
class Citation:
    """Citation payload attached to final answer."""

    chunk_id: str
    doc_id: str | None = None
    doc_name: str | None = None
    page: int | None = None
    excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "page": self.page,
            "excerpt": self.excerpt,
        }


@dataclass
class FusionOutput:
    """Deterministic fusion output fed into generation."""

    context_items: list[RetrieverResult] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    graph_paths: list[str] = field(default_factory=list)
    retrieval_sources: list[str] = field(default_factory=list)
    top_scores: list[float] = field(default_factory=list)
    context_text: str = ""
    community_fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_items": [item.to_dict() for item in self.context_items],
            "citations": [item.to_dict() for item in self.citations],
            "graph_paths": list(self.graph_paths),
            "retrieval_sources": list(self.retrieval_sources),
            "top_scores": [float(score) for score in self.top_scores],
            "context_text": self.context_text,
            "community_fallback_used": self.community_fallback_used,
        }


@dataclass
class GraphRAGResponse:
    """Structured response contract."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    graph_paths_used: list[str] = field(default_factory=list)
    confidence: float = 0.0
    query_type: str = QueryType.LOCAL.value
    retrieval_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "graph_paths_used": list(self.graph_paths_used),
            "confidence": float(self.confidence),
            "query_type": self.query_type,
            "retrieval_sources": list(self.retrieval_sources),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphRAGResponse":
        citations = [
            Citation(
                chunk_id=str(item.get("chunk_id") or ""),
                doc_id=item.get("doc_id"),
                doc_name=item.get("doc_name"),
                page=item.get("page"),
                excerpt=str(item.get("excerpt") or ""),
            )
            for item in payload.get("citations", [])
            if isinstance(item, dict)
        ]
        return cls(
            answer=str(payload.get("answer") or ""),
            citations=citations,
            graph_paths_used=[str(path) for path in payload.get("graph_paths_used", [])],
            confidence=float(payload.get("confidence") or 0.0),
            query_type=str(payload.get("query_type") or QueryType.LOCAL.value),
            retrieval_sources=[str(item) for item in payload.get("retrieval_sources", [])],
        )
