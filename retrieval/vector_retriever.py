"""Vector retrieval over Qdrant chunk embeddings."""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

from config import Settings
from ingestion.embedder import GeminiEmbedder
from models.query import RetrieverResult

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retrieve semantic chunk matches from Qdrant."""

    def __init__(
        self,
        settings: Settings,
        embedder: GeminiEmbedder | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or GeminiEmbedder(settings)
        if client is not None:
            self._client = client
            self._async_client = None
            return
        try:
            from qdrant_client import AsyncQdrantClient, QdrantClient
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("qdrant-client package is required for vector retrieval.") from exc
        self._client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=self.settings.qdrant_timeout_seconds,
        )
        self._async_client = AsyncQdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=self.settings.qdrant_timeout_seconds,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrieverResult]:
        """Retrieve top chunk matches for user query."""
        clean_query = query.strip()
        if not clean_query:
            return []

        limit = max(1, int(top_k or self.settings.query_vector_top_k))
        mmr_enabled = bool(self.settings.query_mmr_enabled)
        candidate_multiplier = max(1, int(self.settings.query_mmr_candidate_multiplier))
        candidate_limit = limit * candidate_multiplier if mmr_enabled else limit

        query_vector = self.embedder.embed_texts(
            [clean_query],
            task_type=self.settings.query_embedding_task_type,
        )[0]
        hits = self._search_chunks(
            query_vector=query_vector,
            limit=candidate_limit,
            with_vectors=mmr_enabled,
        )

        return self._rank_hits(query_vector, hits, limit=limit, mmr_enabled=mmr_enabled)

    def _rank_hits(
        self,
        query_vector: list[float],
        hits: list[Any],
        *,
        limit: int,
        mmr_enabled: bool,
    ) -> list[RetrieverResult]:
        candidates: list[tuple[RetrieverResult, list[float] | None]] = []
        for hit in hits:
            payload = self._extract_payload(hit)
            if not self._embedding_payload_matches(payload):
                continue
            point_id = self._extract_point_id(hit, payload)
            score = self._clamp01(self._extract_score(hit))
            text = str(payload.get("text") or "")
            result = RetrieverResult(
                source_type="vector",
                source_id=point_id,
                text_or_path=text,
                score_raw=score,
                provenance={
                    "chunk_id": str(payload.get("chunk_id") or point_id),
                    "doc_id": payload.get("doc_id"),
                    "section_heading": payload.get("section_heading"),
                    "page": payload.get("page"),
                    "query_similarity": score,
                    "document_version_id": payload.get("document_version_id"),
                    "evidence_id": f"chunk:{str(payload.get('chunk_id') or point_id)}",
                },
            )
            candidates.append((result, self._extract_vector(hit)))

        if not candidates:
            logger.info(
                "vector_retriever_mmr enabled=%s candidate_pool=0 selected=0 fallback=empty",
                mmr_enabled,
            )
            return []

        selected_results: list[RetrieverResult]
        fallback_reason = "none"
        if not mmr_enabled or len(candidates) <= limit:
            selected_results = [item for item, _ in candidates[:limit]]
            fallback_reason = "disabled_or_small_pool"
        else:
            mmr_indices = self._select_mmr_indices(query_vector, candidates, limit)
            if not mmr_indices:
                selected_results = [item for item, _ in candidates[:limit]]
                fallback_reason = "missing_vectors"
            else:
                selected_results = []
                for rank, index in enumerate(mmr_indices):
                    item = candidates[index][0]
                    item.provenance["mmr_rank"] = rank
                    item.provenance["mmr_score"] = self._mmr_score_for_index(
                        index=index,
                        selected_indices=mmr_indices[:rank],
                        query_vector=query_vector,
                        candidates=candidates,
                    )
                    selected_results.append(item)

        logger.info(
            "vector_retriever_mmr enabled=%s candidate_pool=%d selected=%d fallback=%s",
            mmr_enabled,
            len(candidates),
            len(selected_results),
            fallback_reason,
        )
        return selected_results

    async def aretrieve(self, query: str, top_k: int | None = None) -> list[RetrieverResult]:
        """Retrieve through async Gemini/Qdrant clients when available."""
        if self._async_client is None:
            return await asyncio.to_thread(self.retrieve, query, top_k)
        clean_query = query.strip()
        if not clean_query:
            return []
        limit = max(1, int(top_k or self.settings.query_vector_top_k))
        mmr_enabled = bool(self.settings.query_mmr_enabled)
        multiplier = max(1, int(self.settings.query_mmr_candidate_multiplier))
        candidate_limit = limit * multiplier if mmr_enabled else limit
        async_embed = getattr(self.embedder, "aembed_texts", None)
        if callable(async_embed):
            query_vector = (
                await async_embed(
                    [clean_query],
                    task_type=self.settings.query_embedding_task_type,
                )
            )[0]
        else:
            query_vector = (
                await asyncio.to_thread(
                    self.embedder.embed_texts,
                    [clean_query],
                    self.settings.query_embedding_task_type,
                )
            )[0]
        response = await self._async_client.query_points(
            collection_name=self.settings.qdrant_chunks_collection,
            query=query_vector,
            query_filter=self._active_filter(),
            limit=candidate_limit,
            with_payload=True,
            with_vectors=mmr_enabled,
        )
        return self._rank_hits(
            query_vector,
            self._extract_points(response),
            limit=limit,
            mmr_enabled=mmr_enabled,
        )

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()

    def _search_chunks(self, query_vector: list[float], limit: int, with_vectors: bool) -> list[Any]:
        query_filter = self._active_filter()
        if hasattr(self._client, "query_points"):
            try:
                try:
                    response = self._client.query_points(
                        collection_name=self.settings.qdrant_chunks_collection,
                        query=query_vector,
                        query_filter=query_filter,
                        limit=limit,
                        with_payload=True,
                        with_vectors=with_vectors,
                    )
                except TypeError:
                    response = self._client.query_points(
                        collection_name=self.settings.qdrant_chunks_collection,
                        query=query_vector,
                        limit=limit,
                        with_payload=True,
                        with_vectors=with_vectors,
                    )
                return self._extract_points(response)
            except Exception:
                pass

        if hasattr(self._client, "search"):
            try:
                response = self._client.search(
                    collection_name=self.settings.qdrant_chunks_collection,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
            except TypeError:
                response = self._client.search(
                    collection_name=self.settings.qdrant_chunks_collection,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                )
            return self._extract_points(response)

        if hasattr(self._client, "search_points"):
            try:
                response = self._client.search_points(
                    collection_name=self.settings.qdrant_chunks_collection,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=with_vectors,
                )
            except TypeError:
                response = self._client.search_points(
                    collection_name=self.settings.qdrant_chunks_collection,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=True,
                )
            return self._extract_points(response)

        raise RuntimeError("Qdrant client does not expose a compatible search API.")

    def _extract_points(self, response: Any) -> list[Any]:
        if isinstance(response, list):
            return list(response)
        points = getattr(response, "points", None)
        if isinstance(points, list):
            return points
        result = getattr(response, "result", None)
        if isinstance(result, list):
            return result
        if result is not None and hasattr(result, "points"):
            nested = getattr(result, "points", None)
            if isinstance(nested, list):
                return nested
        return []

    def _extract_payload(self, point: Any) -> dict[str, Any]:
        if isinstance(point, dict):
            payload = point.get("payload")
            return payload if isinstance(payload, dict) else {}
        payload = getattr(point, "payload", None)
        return payload if isinstance(payload, dict) else {}

    def _extract_score(self, point: Any) -> float:
        if isinstance(point, dict):
            return float(point.get("score") or 0.0)
        return float(getattr(point, "score", 0.0) or 0.0)

    def _extract_vector(self, point: Any) -> list[float] | None:
        if isinstance(point, dict):
            raw_vector = point.get("vector")
        else:
            raw_vector = getattr(point, "vector", None)
        return self._coerce_vector(raw_vector)

    def _coerce_vector(self, raw_vector: Any) -> list[float] | None:
        if raw_vector is None:
            return None
        if isinstance(raw_vector, list):
            try:
                return [float(value) for value in raw_vector]
            except (TypeError, ValueError):
                return None
        if isinstance(raw_vector, tuple):
            try:
                return [float(value) for value in raw_vector]
            except (TypeError, ValueError):
                return None
        if isinstance(raw_vector, dict):
            if "" in raw_vector:
                return self._coerce_vector(raw_vector.get(""))
            for value in raw_vector.values():
                vector = self._coerce_vector(value)
                if vector:
                    return vector
            return None
        values = getattr(raw_vector, "values", None)
        if isinstance(values, list):
            return self._coerce_vector(values)
        return None

    def _extract_point_id(self, point: Any, payload: dict[str, Any]) -> str:
        if isinstance(point, dict):
            point_id = point.get("id")
        else:
            point_id = getattr(point, "id", None)
        return str(payload.get("chunk_id") or point_id or "")

    def _select_mmr_indices(
        self,
        query_vector: list[float],
        candidates: list[tuple[RetrieverResult, list[float] | None]],
        limit: int,
    ) -> list[int]:
        if not candidates:
            return []
        vectors: list[list[float]] = []
        query_dim = len(query_vector)
        for _, vector in candidates:
            if not isinstance(vector, list) or len(vector) != query_dim:
                return []
            vectors.append(vector)

        lambda_mult = max(0.0, min(1.0, float(self.settings.query_mmr_lambda)))
        selected: list[int] = []
        remaining = set(range(len(candidates)))

        while remaining and len(selected) < limit:
            best_index: int | None = None
            best_value = float("-inf")
            for index in remaining:
                relevance = self._cosine_similarity(query_vector, vectors[index])
                diversity_penalty = 0.0
                if selected:
                    diversity_penalty = max(
                        self._cosine_similarity(vectors[index], vectors[chosen])
                        for chosen in selected
                    )
                mmr_value = (lambda_mult * relevance) - ((1.0 - lambda_mult) * diversity_penalty)
                if best_index is None or mmr_value > best_value:
                    best_index = index
                    best_value = mmr_value

            if best_index is None:
                break
            selected.append(best_index)
            remaining.remove(best_index)

        return selected

    def _mmr_score_for_index(
        self,
        index: int,
        selected_indices: list[int],
        query_vector: list[float],
        candidates: list[tuple[RetrieverResult, list[float] | None]],
    ) -> float:
        vector = candidates[index][1]
        if not isinstance(vector, list):
            return 0.0
        lambda_mult = max(0.0, min(1.0, float(self.settings.query_mmr_lambda)))
        relevance = self._cosine_similarity(query_vector, vector)
        diversity_penalty = 0.0
        if selected_indices:
            diversity_penalty = max(
                self._cosine_similarity(vector, candidates[chosen][1] or [])
                for chosen in selected_indices
            )
        return (lambda_mult * relevance) - ((1.0 - lambda_mult) * diversity_penalty)

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        cosine = dot / (left_norm * right_norm)
        return max(-1.0, min(1.0, cosine))

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _embedding_payload_matches(self, payload: dict[str, Any]) -> bool:
        if not self.settings.enforce_embedding_metadata:
            return True
        return (
            payload.get("embedding_model_id") == self.settings.embedding_model_id
            and int(payload.get("embedding_dimension") or 0) == int(self.settings.embedding_dim)
            and payload.get("embedding_prompt_version") == self.settings.embedding_prompt_version
            and payload.get("embedding_schema_version") == self.settings.embedding_version
            and payload.get("active") is True
        )

    @staticmethod
    def _active_filter() -> Any:
        try:
            from qdrant_client.http import models
        except Exception:
            return {"must": [{"key": "active", "match": {"value": True}}]}
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="active",
                    match=models.MatchValue(value=True),
                )
            ]
        )
