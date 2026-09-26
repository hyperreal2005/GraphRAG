"""Validated contracts for the query pipeline and public response."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class QueryModel(BaseModel):
    """Strict mutable contract shared by query-plane boundaries."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


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


class QueryAnalysis(QueryModel):
    """Structured output from analyzer stage."""

    query_type: QueryType
    mentioned_entities: list[str] = Field(default_factory=list)
    retrieval_flags: dict[str, bool] = Field(default_factory=dict)
    sub_queries: list[str] = Field(default_factory=list)
    temporal_filter: str | None = None


class LinkedEntity(QueryModel):
    """Entity-linking decision for one query mention."""

    mention: str
    status: str
    similarity: float = Field(ge=0.0, le=1.0)
    entity_id: str | None = None
    entity_name: str | None = None
    entity_type: str | None = None
    candidate_ids: list[str] = Field(default_factory=list)


class RetrieverResult(QueryModel):
    """Unified retrieval item contract for graph/vector/community sources."""

    source_type: str
    source_id: str
    text_or_path: str
    score_raw: float
    provenance: dict[str, Any] = Field(default_factory=dict)
    score_fused: float = 0.0
    score_reranked: float | None = None

    def dedupe_key(self) -> str:
        evidence_id = self.provenance.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id:
            return f"evidence:{evidence_id}"
        chunk_id = self.provenance.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id:
            return f"{self.source_type}:{chunk_id}"
        return f"{self.source_type}:{self.source_id}"


class Citation(QueryModel):
    """Exact evidence citation attached to one or more answer claims."""

    chunk_id: str
    doc_id: str | None = None
    doc_name: str | None = None
    page: int | None = None
    excerpt: str = ""
    citation_id: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    assertion_ids: list[str] = Field(default_factory=list)


class AnswerClaim(QueryModel):
    """One answer claim explicitly bound to retrieved evidence."""

    claim_id: str
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    supported: bool = False


class FusionOutput(QueryModel):
    """Deterministic fusion output fed into generation."""

    context_items: list[RetrieverResult] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    graph_paths: list[str] = Field(default_factory=list)
    retrieval_sources: list[str] = Field(default_factory=list)
    top_scores: list[float] = Field(default_factory=list)
    context_text: str = ""
    community_fallback_used: bool = False
    # Which reranking tier actually ordered this context: "cohere", "llm",
    # "local_cross_encoder", "unranked_passthrough", "disabled" or "empty". Recorded
    # because every tier degrades silently to the next on error, so a rate-limited or
    # failing reranker produces a plausible result from a component that never ran -
    # which is how Phase 6's reranking went unexecuted for four phases unnoticed.
    rerank_mode: str = ""


class GraphRAGResponse(QueryModel):
    """Backward-compatible, claim-aware public response contract."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    graph_paths_used: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    query_type: str = QueryType.LOCAL.value
    retrieval_sources: list[str] = Field(default_factory=list)
    claims: list[AnswerClaim] = Field(default_factory=list)
    degraded: bool = False
    warnings: list[str] = Field(default_factory=list)
    confidence_kind: str = "retrieval_grounding_score"
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    provider_usage: dict[str, dict[str, int]] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphRAGResponse":
        return cls.model_validate(payload)
