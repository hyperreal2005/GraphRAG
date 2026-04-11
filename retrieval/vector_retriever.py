"""Vector retrieval over Qdrant chunk embeddings."""

from __future__ import annotations

from typing import Any

from config import Settings
from ingestion.embedder import GeminiEmbedder
from models.query import RetrieverResult


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
            return
        try:
            from qdrant_client import QdrantClient
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("qdrant-client package is required for vector retrieval.") from exc
        self._client = QdrantClient(
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
        query_vector = self.embedder.embed_texts(
            [clean_query],
            task_type=self.settings.query_embedding_task_type,
        )[0]
        hits = self._search_chunks(query_vector=query_vector, limit=limit)
        results: list[RetrieverResult] = []
        for hit in hits:
            payload = self._extract_payload(hit)
            point_id = self._extract_point_id(hit, payload)
            score = float(self._extract_score(hit))
            text = str(payload.get("text") or "")
            results.append(
                RetrieverResult(
                    source_type="vector",
                    source_id=point_id,
                    text_or_path=text,
                    score_raw=max(0.0, min(1.0, score)),
                    provenance={
                        "chunk_id": str(payload.get("chunk_id") or point_id),
                        "doc_id": payload.get("doc_id"),
                        "section_heading": payload.get("section_heading"),
                        "page": payload.get("page"),
                    },
                )
            )
        return results

    def _search_chunks(self, query_vector: list[float], limit: int) -> list[Any]:
        if hasattr(self._client, "query_points"):
            try:
                response = self._client.query_points(
                    collection_name=self.settings.qdrant_chunks_collection,
                    query=query_vector,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                )
                points = self._extract_points(response)
                if points:
                    return points
            except Exception:
                pass

        if hasattr(self._client, "search"):
            response = self._client.search(
                collection_name=self.settings.qdrant_chunks_collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
            return self._extract_points(response)

        if hasattr(self._client, "search_points"):
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

    def _extract_point_id(self, point: Any, payload: dict[str, Any]) -> str:
        if isinstance(point, dict):
            point_id = point.get("id")
        else:
            point_id = getattr(point, "id", None)
        return str(payload.get("chunk_id") or point_id or "")
