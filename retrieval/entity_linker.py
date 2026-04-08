"""Entity linking for query mentions using Qdrant entity vectors."""

from __future__ import annotations

from typing import Any

from config import Settings
from ingestion.embedder import GeminiEmbedder
from models.query import LinkedEntity


class EntityLinker:
    """Resolve query entity mentions against canonical graph entity vectors."""

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
            raise RuntimeError("qdrant-client package is required for entity linking.") from exc

        self._client = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=self.settings.qdrant_timeout_seconds,
        )

    def link_entities(self, mentions: list[str]) -> list[LinkedEntity]:
        """Resolve mention strings to graph entity IDs with thresholded decisions."""
        if not mentions:
            return []
        results: list[LinkedEntity] = []
        for mention in mentions:
            mention_text = mention.strip()
            if not mention_text:
                continue
            results.append(self._link_single_mention(mention_text))
        return results

    def _link_single_mention(self, mention: str) -> LinkedEntity:
        query_vector = self.embedder.embed_texts(
            [mention],
            task_type=self.settings.query_embedding_task_type,
        )[0]
        points = self._search_entities(query_vector, limit=3)
        if not points:
            return LinkedEntity(mention=mention, status="unresolved", similarity=0.0)

        best = points[0]
        best_score = float(self._extract_score(best))
        payload = self._extract_payload(best)
        point_id = self._extract_point_id(best, payload)
        candidate_ids = [
            self._extract_point_id(point, self._extract_payload(point))
            for point in points
        ]
        candidate_ids = [item for item in candidate_ids if item]

        if best_score >= float(self.settings.entity_link_resolve_threshold):
            return LinkedEntity(
                mention=mention,
                status="resolved",
                similarity=best_score,
                entity_id=point_id,
                entity_name=str(payload.get("name") or mention),
                entity_type=str(payload.get("type") or "Entity"),
                candidate_ids=candidate_ids[:3],
            )
        if best_score >= float(self.settings.entity_link_ambiguous_threshold):
            return LinkedEntity(
                mention=mention,
                status="ambiguous",
                similarity=best_score,
                entity_id=point_id,
                entity_name=str(payload.get("name") or mention),
                entity_type=str(payload.get("type") or "Entity"),
                candidate_ids=candidate_ids[:3],
            )
        return LinkedEntity(
            mention=mention,
            status="unresolved",
            similarity=best_score,
            candidate_ids=candidate_ids[:3],
        )

    def _search_entities(self, query_vector: list[float], limit: int) -> list[Any]:
        try:
            response = self._client.query_points(
                collection_name=self.settings.qdrant_entities_collection,
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
        response = self._client.search(
            collection_name=self.settings.qdrant_entities_collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )
        return self._extract_points(response)

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

    def _extract_point_id(self, point: Any, payload: dict[str, Any]) -> str | None:
        if isinstance(point, dict):
            point_id = point.get("id")
        else:
            point_id = getattr(point, "id", None)
        candidate = payload.get("neo4j_node_id") or point_id
        if candidate is None:
            return None
        return str(candidate)

