"""End-to-end query orchestration."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import time
from typing import Any

from config import Settings
from generation.generator import AnswerGenerator
from ingestion.state_store import MetadataStore
from models.query import FusionOutput, GraphRAGResponse, QueryType, RetrieverResult
from retrieval.analyzer import QueryAnalyzer
from retrieval.cache import NoOpCache
from retrieval.community_grounding import ground_communities
from retrieval.community_retriever import CommunityRetriever
from retrieval.entity_linker import EntityLinker
from retrieval.fusion import FusionEngine
from retrieval.graph_retriever import GraphRetriever
from retrieval.vector_retriever import VectorRetriever
from observability import tracing


logger = logging.getLogger(__name__)


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
        store: MetadataStore | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache or NoOpCache()
        self.store = store
        if self.store is None and (graph_retriever is None or fusion_engine is None):
            self.store = MetadataStore(settings.metadata_db_path)
        self.analyzer = analyzer or QueryAnalyzer(settings)
        self.entity_linker = entity_linker or EntityLinker(settings, store=self.store)
        self.graph_retriever = graph_retriever or GraphRetriever(settings, store=self.store)
        self.vector_retriever = vector_retriever or VectorRetriever(settings, store=self.store)
        self.community_retriever = community_retriever or CommunityRetriever(settings)
        self.fusion_engine = fusion_engine or FusionEngine(settings, store=self.store)
        self.generator = generator or AnswerGenerator(settings)
        tracing.configure_tracing(settings)

    @staticmethod
    def _traced(stage: str, fn: Any, **attributes: Any) -> Any:
        """Wrap a stage callable in a span named after the stage. No-op when tracing is off.

        Applied to the callable rather than around the await so the existing timeout and
        timing logic is untouched; `asyncio.to_thread` copies contextvars, so the span
        nests under the query's root span even though the stage runs on a worker thread.
        """

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with tracing.span(f"graphrag.{stage}", **attributes) as current:
                result = fn(*args, **kwargs)
                if current is not None and isinstance(result, list):
                    tracing.annotate(current, **{"result.count": len(result)})
                return result

        async def awrapped(*args: Any, **kwargs: Any) -> Any:
            with tracing.span(f"graphrag.{stage}", **attributes) as current:
                result = await fn(*args, **kwargs)
                if current is not None and isinstance(result, list):
                    tracing.annotate(current, **{"result.count": len(result)})
                return result

        import inspect

        return awrapped if inspect.iscoroutinefunction(fn) else wrapped

    async def query(self, user_query: str, session_id: str = "default") -> GraphRAGResponse:
        """Execute analyze -> retrieve -> fuse -> generate, as one trace."""
        with tracing.span(
            "graphrag.query",
            **{"session.id": session_id, "query.chars": len(user_query or "")},
        ) as root:
            response = await self._query(user_query, session_id)
            tracing.annotate(
                root,
                **{
                    "query.type": response.query_type,
                    "retrieval.sources": response.retrieval_sources,
                    "response.degraded": response.degraded,
                    "response.warnings": response.warnings,
                    "response.claims": len(response.claims),
                    "response.citations": len(response.citations),
                    "response.confidence": response.confidence,
                },
            )
            return response

    async def _query(self, user_query: str, session_id: str = "default") -> GraphRAGResponse:
        """Execute analyze -> retrieve -> fuse -> generate."""
        cache_key = self._build_cache_key(user_query=user_query, session_id=session_id)
        if self.settings.query_cache_enabled:
            cached_payload = self.cache.get_query(cache_key)
            if isinstance(cached_payload, dict):
                return GraphRAGResponse.from_dict(cached_payload)

        warnings: list[str] = []
        stage_timings_ms: dict[str, float] = {}
        if self.store is not None and self.store.has_pending_corpus_update():
            warnings.append("corpus_update_in_progress")
        stage_started = time.perf_counter()
        try:
            analysis = await asyncio.wait_for(
                asyncio.to_thread(self._traced("analysis", self.analyzer.analyze), user_query),
                timeout=max(0.1, float(self.settings.query_analysis_timeout_seconds)),
            )
        except Exception as exc:
            logger.warning("query_analysis_failed error_type=%s", type(exc).__name__)
            stage_timings_ms["analysis"] = (time.perf_counter() - stage_started) * 1000.0
            return GraphRAGResponse(
                answer="I could not safely analyze this question within the configured limits.",
                confidence=0.0,
                degraded=True,
                warnings=[*warnings, f"query_analysis_failed:{type(exc).__name__}"],
                stage_timings_ms=stage_timings_ms,
                provider_usage=self._provider_usage(),
            )
        stage_timings_ms["analysis"] = (time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        try:
            async_link = getattr(self.entity_linker, "alink_entities", None)
            link_operation = (
                self._traced("entity_linking", async_link)(
                    analysis.mentioned_entities, query_context=user_query
                )
                if callable(async_link)
                else asyncio.to_thread(
                    self._traced("entity_linking", self.entity_linker.link_entities),
                    analysis.mentioned_entities,
                    user_query,
                )
            )
            linked_entities = await asyncio.wait_for(
                link_operation,
                timeout=max(0.1, float(self.settings.query_entity_link_timeout_seconds)),
            )
        except Exception as exc:
            logger.warning("entity_linking_failed error_type=%s", type(exc).__name__)
            warnings.append(f"entity_linking_failed:{type(exc).__name__}")
            linked_entities = []
        stage_timings_ms["entity_linking"] = (time.perf_counter() - stage_started) * 1000.0

        graph_task = self._to_thread_or_empty(
            component="graph_retrieval",
            enabled=bool(analysis.retrieval_flags.get("graph")),
            fn=self._traced("graph_retrieval", getattr(self.graph_retriever, "aretrieve", self.graph_retriever.retrieve)),
            analysis=analysis,
            linked_entities=linked_entities,
            user_query=user_query,
            timeout_seconds=self.settings.query_graph_timeout_seconds,
        )
        vector_task = self._to_thread_or_empty(
            component="vector_retrieval",
            enabled=bool(analysis.retrieval_flags.get("vector", True)),
            fn=self._traced(
                "vector_retrieval",
                self._aretrieve_vector_queries
                if callable(getattr(self.vector_retriever, "aretrieve", None))
                else self._retrieve_vector_queries,
            ),
            query=user_query,
            sub_queries=analysis.sub_queries,
            timeout_seconds=self.settings.query_vector_timeout_seconds,
        )
        community_task = self._to_thread_or_empty(
            component="community_retrieval",
            enabled=bool(analysis.retrieval_flags.get("community")),
            fn=self._traced("community_retrieval", self.community_retriever.retrieve),
            query=user_query,
            timeout_seconds=self.settings.query_vector_timeout_seconds,
            # ROADMAP C2: like grounding, for the question types it was measured on.
            **(
                {"size_prior": float(self.settings.query_community_size_prior)}
                if self.settings.query_community_size_prior
                and analysis.query_type in (QueryType.GLOBAL, QueryType.HYBRID)
                else {}
            ),
        )

        graph_outcome, vector_outcome, community_outcome = await asyncio.gather(
            graph_task,
            vector_task,
            community_task,
        )
        graph_results, graph_warning, graph_duration = graph_outcome
        vector_results, vector_warning, vector_duration = vector_outcome
        community_results, community_warning, community_duration = community_outcome
        stage_timings_ms.update(
            {
                "graph_retrieval": graph_duration,
                "vector_retrieval": vector_duration,
                "community_retrieval": community_duration,
            }
        )
        warnings.extend(
            warning
            for warning in (graph_warning, vector_warning, community_warning)
            if warning is not None
        )
        community_fallback_used = bool(analysis.retrieval_flags.get("community")) and not community_results
        vector_results = list(vector_results) + await self._ground_communities(
            analysis, community_results, warnings, stage_timings_ms
        )

        stage_started = time.perf_counter()
        try:
            fusion_output = await asyncio.wait_for(
                asyncio.to_thread(
                    self._traced("fusion", self.fusion_engine.fuse),
                    graph_results,
                    vector_results,
                    community_results,
                    community_fallback_used,
                    user_query,
                ),
                timeout=max(0.1, float(self.settings.query_rerank_timeout_seconds)),
            )
        except Exception as exc:
            warnings.append(f"fusion_or_rerank_failed:{type(exc).__name__}")
            fusion_output = FusionOutput()
        stage_timings_ms["fusion_rerank"] = (time.perf_counter() - stage_started) * 1000.0
        stage_started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._traced("generation", self.generator.generate),
                    user_query,
                    analysis,
                    fusion_output,
                ),
                timeout=max(0.1, float(self.settings.query_generation_timeout_seconds)),
            )
        except Exception as exc:
            warnings.append(f"generation_failed:{type(exc).__name__}")
            response = GraphRAGResponse(
                answer="I could not produce a validated answer within the configured limits.",
                query_type=analysis.query_type.value,
                confidence=0.0,
                degraded=True,
            )
        stage_timings_ms["generation"] = (time.perf_counter() - stage_started) * 1000.0
        if warnings:
            response.degraded = True
            response.warnings = list(dict.fromkeys([*response.warnings, *warnings]))
        response.stage_timings_ms = stage_timings_ms
        response.provider_usage = self._provider_usage()

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

    async def _ground_communities(
        self,
        analysis: Any,
        community_results: list[RetrieverResult],
        warnings: list[str],
        stage_timings_ms: dict[str, float],
    ) -> list[RetrieverResult]:
        """Central member passages for the retrieved communities (ROADMAP C2).

        GLOBAL and HYBRID only, so questions of every other type keep the evidence they
        were measured with even when the analyzer switches community retrieval on.
        """
        per_community = int(self.settings.query_community_grounding_passages)
        if (
            per_community <= 0
            or not community_results
            or self.store is None
            or analysis.query_type not in (QueryType.GLOBAL, QueryType.HYBRID)
        ):
            return []
        results, warning, duration = await self._to_thread_or_empty(
            component="community_grounding",
            enabled=True,
            fn=self._traced("community_grounding", ground_communities),
            communities=community_results,
            store=self.store,
            per_community=per_community,
            timeout_seconds=self.settings.query_vector_timeout_seconds,
        )
        stage_timings_ms["community_grounding"] = duration
        if warning is not None:
            warnings.append(warning)
        return results

    async def _to_thread_or_empty(
        self,
        component: str,
        enabled: bool,
        fn: Any,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> tuple[list[RetrieverResult], str | None, float]:
        started = time.perf_counter()
        if not enabled:
            return [], None, 0.0
        try:
            if inspect.iscoroutinefunction(fn):
                operation = fn(**kwargs)
            else:
                operation = asyncio.to_thread(fn, **kwargs)
            results = await asyncio.wait_for(
                operation,
                timeout=max(0.1, float(timeout_seconds)),
            )
            return results, None, (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            logger.warning("%s_failed error_type=%s", component, type(exc).__name__)
            return (
                [],
                f"{component}_failed:{type(exc).__name__}",
                (time.perf_counter() - started) * 1000.0,
            )

    def _provider_usage(self) -> dict[str, dict[str, int]]:
        usage: dict[str, dict[str, int]] = {}
        for name, component in (("analysis", self.analyzer), ("generation", self.generator)):
            payload = getattr(component, "last_usage", None)
            if isinstance(payload, dict) and payload:
                usage[name] = {
                    str(key): int(value)
                    for key, value in payload.items()
                    if isinstance(value, (int, float))
                }
        return usage

    def _retrieve_vector_queries(self, query: str, sub_queries: list[str]) -> list[RetrieverResult]:
        queries = list(dict.fromkeys([query, *sub_queries]))
        results: list[RetrieverResult] = []
        for index, value in enumerate(queries):
            for item in self.vector_retriever.retrieve(
                value,
                top_k=int(self.settings.query_vector_top_k),
            ):
                item.provenance.setdefault("subquery", value)
                item.provenance.setdefault("subquery_index", index)
                results.append(item)
        deduped: dict[str, RetrieverResult] = {}
        for item in results:
            key = item.dedupe_key()
            if key not in deduped or item.score_raw > deduped[key].score_raw:
                deduped[key] = item
        return list(deduped.values())

    async def _aretrieve_vector_queries(
        self,
        query: str,
        sub_queries: list[str],
    ) -> list[RetrieverResult]:
        queries = list(dict.fromkeys([query, *sub_queries]))
        groups = await asyncio.gather(
            *(
                self.vector_retriever.aretrieve(
                    value,
                    top_k=int(self.settings.query_vector_top_k),
                )
                for value in queries
            )
        )
        deduped: dict[str, RetrieverResult] = {}
        for index, (value, items) in enumerate(zip(queries, groups)):
            for item in items:
                item.provenance.setdefault("subquery", value)
                item.provenance.setdefault("subquery_index", index)
                key = item.dedupe_key()
                if key not in deduped or item.score_raw > deduped[key].score_raw:
                    deduped[key] = item
        return list(deduped.values())

    def _build_cache_key(self, user_query: str, session_id: str) -> str:
        payload = f"{session_id}|{user_query.strip().lower()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def close(self) -> None:
        resources = [
            self.graph_retriever,
            getattr(self.entity_linker, "_client", None),
            getattr(self.vector_retriever, "_client", None),
        ]
        seen: set[int] = set()
        for resource in resources:
            if resource is None or id(resource) in seen:
                continue
            seen.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    async def aclose(self) -> None:
        for resource in (self.graph_retriever, self.entity_linker, self.vector_retriever):
            async_close = getattr(resource, "aclose", None)
            if callable(async_close):
                await async_close()
        self.close()
