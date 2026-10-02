"""Phase 8: retrieve community summaries for GLOBAL and HYBRID queries.

This was a stub returning `[]` from Phase 4 until 2026-09-01, so the third retrieval source
never contributed anything and `community_fallback_used` was true on every global query.
See `ingestion/community_builder.py` for how the summaries are produced.

No version fence is applied here. Communities are derived data rebuilt wholesale, and the
builder recreates the collection per build, so whatever the collection holds is the current
build by construction - unlike chunks and entities, which are versioned and must be fenced
on `document_version_id`.

The embedding *is* checked, as it is for chunks. The first build was embedded with
`gemini-embedding-2`, the corpus moved to `-001`, and for four weeks every GLOBAL query
ranked summaries by the cosine between two models' vectors (REFACTOR3 WP26).
"""

from __future__ import annotations

import logging
from typing import Any

from config import Settings
from ingestion.community_builder import embedding_stamp
from ingestion.embedder import GeminiEmbedder
from models.query import RetrieverResult

logger = logging.getLogger(__name__)


class CommunityEmbeddingMismatch(RuntimeError):
    """The summaries were indexed by another embedding than the one embedding the query.

    Raised rather than returning nothing, so the orchestrator reports
    `community_retrieval_failed:CommunityEmbeddingMismatch`. A stale collection is a
    retriever that did nothing, and must not read as one that found nothing.
    """


class CommunityRetriever:
    """Vector search over community summaries."""

    def __init__(
        self,
        settings: Settings,
        embedder: GeminiEmbedder | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self._embedder = embedder
        self._client = client

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrieverResult]:
        clean_query = (query or "").strip()
        if not clean_query:
            return []
        client = self._ensure_client()
        if client is None:
            return []

        collection = self.settings.qdrant_communities_collection
        try:
            vector = self._ensure_embedder().embed_texts(
                [clean_query], task_type=self.settings.query_embedding_task_type
            )[0]
            hits = client.query_points(
                collection_name=collection,
                query=vector,
                limit=max(1, int(top_k)),
                with_payload=True,
            ).points
        except Exception as exc:
            # A collection that was never built is the normal case before the first
            # build, not an error worth failing a query over.
            logger.info(
                "community_retrieval_unavailable collection=%s error_type=%s",
                collection,
                type(exc).__name__,
            )
            return []

        self._check_embedding(collection, hits)
        results: list[RetrieverResult] = []
        for hit in hits:
            payload = getattr(hit, "payload", None) or {}
            summary = str(payload.get("summary") or "").strip()
            if not summary:
                continue
            title = str(payload.get("title") or "").strip()
            community_id = str(payload.get("community_id") or getattr(hit, "id", ""))
            results.append(
                RetrieverResult(
                    source_type="community",
                    source_id=community_id,
                    text_or_path=f"{title}\n{summary}" if title else summary,
                    score_raw=float(getattr(hit, "score", 0.0) or 0.0),
                    provenance={
                        "community_id": community_id,
                        "build_id": payload.get("build_id"),
                        "title": title,
                        "size": payload.get("size"),
                        "doc_ids": payload.get("doc_ids") or [],
                        "evidence_id": f"community:{community_id}",
                    },
                )
            )
        return results

    def _check_embedding(self, collection: str, hits: list[Any]) -> None:
        if not self.settings.enforce_embedding_metadata:
            return
        expected = embedding_stamp(self.settings)
        for hit in hits:
            payload = getattr(hit, "payload", None) or {}
            found = {key: payload.get(key) for key in expected}
            if found != expected:
                raise CommunityEmbeddingMismatch(
                    f"Collection {collection!r} was indexed with "
                    f"{found.get('embedding_model_id') or 'an unrecorded embedding'}; queries use "
                    f"{expected['embedding_model_id']} ({expected['embedding_schema_version']}). "
                    "Rebuild it: python main_admin.py communities --db <store>."
                )

    def _ensure_embedder(self) -> GeminiEmbedder:
        if self._embedder is None:
            self._embedder = GeminiEmbedder(self.settings)
        return self._embedder

    def _ensure_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except Exception:
            return None
        self._client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=self.settings.qdrant_timeout_seconds,
        )
        return self._client
