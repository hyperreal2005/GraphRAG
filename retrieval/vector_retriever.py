"""Vector retrieval over Qdrant chunk embeddings."""

from __future__ import annotations

import asyncio
import logging
import math

import numpy as np
from typing import Any

from config import Settings
from ingestion.embedder import GeminiEmbedder
from ingestion.state_store import MetadataStore
from ingestion.sparse_encoder import encode_query
from ingestion.vector_indexer import SPARSE_VECTOR_NAME, build_active_version_filter
from models.query import RetrieverResult

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retrieve semantic chunk matches from Qdrant."""

    def __init__(
        self,
        settings: Settings,
        embedder: GeminiEmbedder | None = None,
        client: Any | None = None,
        store: MetadataStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or GeminiEmbedder(settings)
        self.store = store
        self._sparse_ready: bool | None = None
        self._sparse_models = None
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
        active_version_ids = self._queryable_version_ids()
        if active_version_ids == []:
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
            active_version_ids=active_version_ids,
        )
        hits = self._fuse_with_sparse(
            clean_query,
            hits,
            limit=candidate_limit,
            with_vectors=mmr_enabled,
            active_version_ids=active_version_ids,
        )

        return self._rank_hits(
            query_vector,
            hits,
            limit=limit,
            mmr_enabled=mmr_enabled,
            active_version_ids=active_version_ids,
        )

    def _rank_hits(
        self,
        query_vector: list[float],
        hits: list[Any],
        *,
        limit: int,
        mmr_enabled: bool,
        active_version_ids: list[str] | None = None,
    ) -> list[RetrieverResult]:
        candidates: list[tuple[RetrieverResult, list[float] | None]] = []
        for hit in hits:
            payload = self._extract_payload(hit)
            if not self._embedding_payload_matches(payload, active_version_ids):
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
                mmr_scores = self._mmr_scores(query_vector, candidates, mmr_indices)
                for rank, index in enumerate(mmr_indices):
                    item = candidates[index][0]
                    item.provenance["mmr_rank"] = rank
                    item.provenance["mmr_score"] = mmr_scores[rank]
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
        active_version_ids = self._queryable_version_ids()
        if active_version_ids == []:
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
            query_filter=build_active_version_filter(
                active_version_ids,
                exclude_reference_chunks=not self.settings.query_include_reference_chunks,
            ),
            limit=candidate_limit,
            with_payload=True,
            with_vectors=mmr_enabled,
        )
        return self._rank_hits(
            query_vector,
            self._extract_points(response),
            limit=limit,
            mmr_enabled=mmr_enabled,
            active_version_ids=active_version_ids,
        )

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()

    def _sparse_available(self) -> bool:
        """Whether a sparse query can actually be served, decided once per retriever."""
        if not self.settings.query_sparse_enabled:
            return False
        if self._sparse_ready is None:
            try:
                from qdrant_client import models

                self._sparse_models = models
                info = self._client.get_collection(self.settings.qdrant_chunks_collection)
                params = getattr(getattr(info, "config", None), "params", None)
                sparse = getattr(params, "sparse_vectors", None) or {}
                self._sparse_ready = SPARSE_VECTOR_NAME in sparse
            except Exception:
                # A dense-only collection, or a client that cannot answer: fall back
                # silently. Hybrid retrieval is an improvement, not a precondition.
                self._sparse_ready = False
            if not self._sparse_ready:
                logger.info("sparse_retrieval_unavailable falling_back_to_dense_only")
        return bool(self._sparse_ready)

    def _fuse_with_sparse(
        self,
        query: str,
        dense_hits: list[Any],
        *,
        limit: int,
        with_vectors: bool,
        active_version_ids: list[str] | None,
    ) -> list[Any]:
        """Reciprocal-rank fusion of the dense and lexical channels (WP6, finding `A5`).

        RRF is used rather than score blending because the two channels' scores are not
        comparable: cosine similarity is bounded and dense, BM25 is unbounded and depends
        on collection statistics. Rank is the only quantity both produce on the same
        scale. MMR then runs over the fused pool, so diversification still sees every
        candidate either channel found.
        """
        if not self._sparse_available():
            return dense_hits
        sparse_query = encode_query(query)
        if not sparse_query:
            return dense_hits
        try:
            sparse_hits = self._search_chunks(
                query_vector=self._sparse_models.SparseVector(
                    indices=sparse_query.indices, values=sparse_query.values
                ),
                limit=limit,
                with_vectors=with_vectors,
                active_version_ids=active_version_ids,
                using=SPARSE_VECTOR_NAME,
            )
        except Exception as exc:
            logger.warning("sparse_query_failed error=%s", type(exc).__name__)
            return dense_hits
        if not sparse_hits:
            return dense_hits

        k = max(1.0, float(self.settings.query_rrf_k))
        fused: dict[str, float] = {}
        best_hit: dict[str, Any] = {}
        for channel in (dense_hits, sparse_hits):
            for rank, hit in enumerate(channel):
                payload = self._extract_payload(hit)
                key = self._extract_point_id(hit, payload)
                fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank + 1)
                # Keep whichever copy carries a vector, so MMR is not starved.
                incumbent = best_hit.get(key)
                if incumbent is None or (
                    self._extract_vector(incumbent) is None
                    and self._extract_vector(hit) is not None
                ):
                    best_hit[key] = hit
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        return [best_hit[key] for key, _ in ordered[:limit]]

    def _search_chunks(
        self,
        query_vector: Any,
        limit: int,
        with_vectors: bool,
        active_version_ids: list[str] | None = None,
        using: str | None = None,
    ) -> list[Any]:
        query_filter = build_active_version_filter(
            active_version_ids,
            exclude_reference_chunks=not self.settings.query_include_reference_chunks,
        )
        # WP6: only `query_points` can address a named sparse channel. The older
        # `search` APIs below are dense-only, so a sparse request must fail here rather
        # than silently fall through and be answered as a dense query.
        named = {"using": using} if using else {}
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
                        **named,
                    )
                except TypeError:
                    response = self._client.query_points(
                        collection_name=self.settings.qdrant_chunks_collection,
                        query=query_vector,
                        limit=limit,
                        with_payload=True,
                        with_vectors=with_vectors,
                        **named,
                    )
                return self._extract_points(response)
            except Exception:
                if using:
                    raise
        if using:
            raise RuntimeError("Qdrant client cannot query a named sparse vector.")

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

    def _mmr_matrices(
        self,
        query_vector: list[float],
        candidates: list[tuple[RetrieverResult, list[float] | None]],
    ) -> tuple[Any, Any] | None:
        """Precompute relevance and pairwise similarity once.

        The previous implementation recomputed both inside the greedy loop in pure
        Python: roughly 3e7 float operations per sub-query at 3072 dimensions, measured
        at a 9.8s p50 against a 15s budget (REVIEW1 finding B4). Normalising once and
        letting NumPy do two matrix products makes it a few milliseconds.
        """
        query_dim = len(query_vector)
        if not candidates or query_dim == 0:
            return None
        for _, vector in candidates:
            if not isinstance(vector, list) or len(vector) != query_dim:
                return None

        matrix = np.asarray([vector for _, vector in candidates], dtype=np.float64)
        query = np.asarray(query_vector, dtype=np.float64)

        # Normalise so a dot product is the cosine. Zero vectors stay zero, matching
        # _cosine_similarity's guard rather than producing NaN.
        row_norms = np.linalg.norm(matrix, axis=1)
        row_norms[row_norms == 0.0] = 1.0
        matrix = matrix / row_norms[:, np.newaxis]
        query_norm = float(np.linalg.norm(query))
        query = query / (query_norm or 1.0)

        relevance = np.clip(matrix @ query, -1.0, 1.0)
        similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
        return relevance, similarity

    def _select_mmr_indices(
        self,
        query_vector: list[float],
        candidates: list[tuple[RetrieverResult, list[float] | None]],
        limit: int,
    ) -> list[int]:
        matrices = self._mmr_matrices(query_vector, candidates)
        if matrices is None:
            return []
        relevance, similarity = matrices

        lambda_mult = max(0.0, min(1.0, float(self.settings.query_mmr_lambda)))
        count = len(candidates)
        selected: list[int] = []
        # Running max similarity to anything already selected; -inf until the first pick,
        # so the first selection is driven purely by relevance.
        penalty = np.full(count, -np.inf)
        available = np.ones(count, dtype=bool)

        for _ in range(min(limit, count)):
            diversity = np.where(np.isneginf(penalty), 0.0, penalty)
            scores = (lambda_mult * relevance) - ((1.0 - lambda_mult) * diversity)
            scores = np.where(available, scores, -np.inf)
            best_index = int(np.argmax(scores))  # ties resolve to the lowest index
            if not available[best_index]:
                break
            selected.append(best_index)
            available[best_index] = False
            penalty = np.maximum(penalty, similarity[best_index])

        return selected

    def _mmr_scores(
        self,
        query_vector: list[float],
        candidates: list[tuple[RetrieverResult, list[float] | None]],
        selected: list[int],
    ) -> list[float]:
        """MMR score each selection had at the moment it was chosen."""
        matrices = self._mmr_matrices(query_vector, candidates)
        if matrices is None:
            return [0.0] * len(selected)
        relevance, similarity = matrices

        lambda_mult = max(0.0, min(1.0, float(self.settings.query_mmr_lambda)))
        scores: list[float] = []
        penalty = np.full(len(candidates), -np.inf)
        for index in selected:
            diversity = 0.0 if np.isneginf(penalty[index]) else float(penalty[index])
            scores.append((lambda_mult * float(relevance[index])) - ((1.0 - lambda_mult) * diversity))
            penalty = np.maximum(penalty, similarity[index])
        return scores

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

    def _embedding_payload_matches(
        self,
        payload: dict[str, Any],
        active_version_ids: list[str] | None = None,
    ) -> bool:
        if active_version_ids is not None and str(payload.get("document_version_id") or "") not in set(
            active_version_ids
        ):
            return False
        if (
            not self.settings.query_include_reference_chunks
            and str(payload.get("content_kind") or "body") == "reference"
        ):
            return False
        if not self.settings.enforce_embedding_metadata:
            return True
        return (
            payload.get("embedding_model_id") == self.settings.embedding_model_id
            and int(payload.get("embedding_dimension") or 0) == int(self.settings.embedding_dim)
            and payload.get("embedding_prompt_version") == self.settings.embedding_prompt_version
            and payload.get("embedding_schema_version") == self.settings.embedding_version
            and payload.get("active") is True
        )

    def _queryable_version_ids(self) -> list[str] | None:
        if self.store is None:
            return None
        return self.store.get_queryable_document_version_ids()
