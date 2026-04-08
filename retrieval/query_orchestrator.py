"""End-to-end query orchestration."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

from config import Settings
from generation.generator import AnswerGenerator
from models.query import GraphRAGResponse, RetrieverResult
from retrieval.analyzer import QueryAnalyzer
from retrieval.cache import NoOpCache
from retrieval.community_retriever import CommunityRetriever
from retrieval.entity_linker import EntityLinker
from retrieval.fusion import FusionEngine
from retrieval.graph_retriever import GraphRetriever
from retrieval.vector_retriever import VectorRetriever


class QueryOrchestrator:
    """Run the 7-step query pipeline."""

    def __init__(
        self,
        settings: Settings,
        cache: Any | None = None,
        analyzer: QueryAnalyzer | None = None,
        entity_linker: EntityLinker | None = None,
        graph_retriever: GraphRetriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        community_retriever: CommunityRetriever | None = None,
        fusion_engine: FusionEngine | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache or NoOpCache()
        self.analyzer = analyzer or QueryAnalyzer(settings)
        self.entity_linker = entity_linker or EntityLinker(settings)
        self.graph_retriever = graph_retriever or GraphRetriever(settings)
        self.vector_retriever = vector_retriever or VectorRetriever(settings)
        self.community_retriever = community_retriever or CommunityRetriever(settings)
        self.fusion_engine = fusion_engine or FusionEngine(settings)
        self.generator = generator or AnswerGenerator(settings)

    async def query(self, user_query: str, session_id: str = "default") -> GraphRAGResponse:
        """Execute analyze -> retrieve -> fuse -> generate."""
        cache_key = self._build_cache_key(user_query=user_query, session_id=session_id)
        if self.settings.query_cache_enabled:
            cached_payload = self.cache.get_query(cache_key)
            if isinstance(cached_payload, dict):
                return GraphRAGResponse.from_dict(cached_payload)

        analysis = await asyncio.to_thread(self.analyzer.analyze, user_query)
        linked_entities = await asyncio.to_thread(self.entity_linker.link_entities, analysis.mentioned_entities)

        graph_task = self._to_thread_or_empty(
            enabled=bool(analysis.retrieval_flags.get("graph")),
            fn=self.graph_retriever.retrieve,
            analysis=analysis,
            linked_entities=linked_entities,
        )
        vector_task = self._to_thread_or_empty(
            enabled=bool(analysis.retrieval_flags.get("vector", True)),
            fn=self.vector_retriever.retrieve,
            query=user_query,
            top_k=int(self.settings.query_vector_top_k),
        )
        community_task = self._to_thread_or_empty(
            enabled=bool(analysis.retrieval_flags.get("community")),
            fn=self.community_retriever.retrieve,
            query=user_query,
        )

        graph_results, vector_results, community_results = await asyncio.gather(
            graph_task,
            vector_task,
            community_task,
        )
        community_fallback_used = bool(analysis.retrieval_flags.get("community")) and not community_results

        fusion_output = await asyncio.to_thread(
            self.fusion_engine.fuse,
            graph_results,
            vector_results,
            community_results,
            community_fallback_used,
        )
        response = await asyncio.to_thread(
            self.generator.generate,
            user_query,
            analysis,
            fusion_output,
        )

        if self.settings.query_cache_enabled:
            self.cache.set_query(
                query_hash=cache_key,
                response_payload=response.to_dict(),
                ttl_seconds=int(self.settings.query_cache_ttl_seconds),
            )
        self.cache.append_session_entry(
            session_id=session_id,
            entry={
                "query": user_query,
                "linked_entities": [entity.to_dict() for entity in linked_entities],
            },
        )
        return response

    async def _to_thread_or_empty(self, enabled: bool, fn: Any, **kwargs: Any) -> list[RetrieverResult]:
        if not enabled:
            return []
        try:
            return await asyncio.to_thread(fn, **kwargs)
        except Exception:
            return []

    def _build_cache_key(self, user_query: str, session_id: str) -> str:
        payload = f"{session_id}|{user_query.strip().lower()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def close(self) -> None:
        self.graph_retriever.close()
