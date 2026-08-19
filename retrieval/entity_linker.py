"""Entity linking for query mentions using Qdrant entity vectors."""

from __future__ import annotations

import asyncio
from typing import Any

from config import Settings
from ingestion.embedder import GeminiEmbedder
from ingestion.state_store import MetadataStore
from ingestion.vector_indexer import build_active_version_filter
from models.query import LinkedEntity


class EntityLinker:
    """Resolve query entity mentions against canonical graph entity vectors."""

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

        if client is not None:
            self._client = client
            self._async_client = None
            return
        try:
            from qdrant_client import AsyncQdrantClient, QdrantClient
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("qdrant-client package is required for entity linking.") from exc

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

    def link_entities(
        self,
        mentions: list[str],
        query_context: str = "",
    ) -> list[LinkedEntity]:
        """Resolve mention strings to graph entity IDs with thresholded decisions."""
        if not mentions:
            return []
        clean_mentions = [mention.strip() for mention in mentions if mention.strip()]
        if not clean_mentions:
            return []
        active_version_ids = self._queryable_version_ids()
        if active_version_ids == []:
            return [
                LinkedEntity(mention=mention, status="unresolved", similarity=0.0)
                for mention in clean_mentions
            ]
        embedding_inputs = self._embedding_inputs(clean_mentions, query_context)
        vectors = self.embedder.embed_texts(
            embedding_inputs,
            task_type=self.settings.query_embedding_task_type,
        )
        if len(vectors) != len(embedding_inputs):
            raise RuntimeError("Entity-link embedding output length mismatch.")
        point_groups = [
            self._search_entities(
                vector,
                limit=int(self.settings.entity_link_candidate_k),
                active_version_ids=active_version_ids,
            )
            for vector in vectors
        ]
        mention_groups = point_groups[: len(clean_mentions)]
        context_groups = point_groups[len(clean_mentions) :]
        return [
            self._link_from_points(
                mention,
                self._merge_candidate_points(mention, mention_points, context_groups[index])
                if context_groups
                else mention_points,
                active_version_ids,
            )
            for index, (mention, mention_points) in enumerate(zip(clean_mentions, mention_groups))
        ]

    def _link_single_mention(
        self,
        mention: str,
        query_vector: list[float],
        active_version_ids: list[str] | None,
    ) -> LinkedEntity:
        points = self._search_entities(
            query_vector,
            limit=int(self.settings.entity_link_candidate_k),
            active_version_ids=active_version_ids,
        )
        return self._link_from_points(mention, points, active_version_ids)

    def _link_from_points(
        self,
        mention: str,
        points: list[Any],
        active_version_ids: list[str] | None = None,
    ) -> LinkedEntity:
        points = [
            point
            for point in points
            if self._embedding_payload_matches(
                self._extract_payload(point), active_version_ids
            )
        ]
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
        second_score = float(self._extract_score(points[1])) if len(points) > 1 else 0.0
        score_margin = best_score - second_score

        if (
            best_score >= float(self.settings.entity_link_resolve_threshold)
            and score_margin >= float(self.settings.entity_link_resolve_margin)
        ):
            return LinkedEntity(
                mention=mention,
                status="resolved",
                similarity=best_score,
                entity_id=point_id,
                entity_name=str(payload.get("name") or mention),
                entity_type=str(payload.get("type") or "Entity"),
                candidate_ids=candidate_ids[: int(self.settings.entity_link_candidate_k)],
            )
        if best_score >= float(self.settings.entity_link_ambiguous_threshold):
            return LinkedEntity(
                mention=mention,
                status="ambiguous",
                similarity=best_score,
                entity_id=point_id,
                entity_name=str(payload.get("name") or mention),
                entity_type=str(payload.get("type") or "Entity"),
                candidate_ids=candidate_ids[: int(self.settings.entity_link_candidate_k)],
            )
        return LinkedEntity(
            mention=mention,
            status="unresolved",
            similarity=best_score,
            candidate_ids=candidate_ids[: int(self.settings.entity_link_candidate_k)],
        )

    async def alink_entities(
        self,
        mentions: list[str],
        query_context: str = "",
    ) -> list[LinkedEntity]:
        """Link mentions through Gemini/Qdrant async clients when available."""
        if self._async_client is None:
            return await asyncio.to_thread(self.link_entities, mentions, query_context)
        clean_mentions = [mention.strip() for mention in mentions if mention.strip()]
        if not clean_mentions:
            return []
        active_version_ids = self._queryable_version_ids()
        if active_version_ids == []:
            return [
                LinkedEntity(mention=mention, status="unresolved", similarity=0.0)
                for mention in clean_mentions
            ]
        async_embed = getattr(self.embedder, "aembed_texts", None)
        embedding_inputs = self._embedding_inputs(clean_mentions, query_context)
        if callable(async_embed):
            vectors = await async_embed(
                embedding_inputs,
                task_type=self.settings.query_embedding_task_type,
            )
        else:
            vectors = await asyncio.to_thread(
                self.embedder.embed_texts,
                embedding_inputs,
                self.settings.query_embedding_task_type,
            )
        if len(vectors) != len(embedding_inputs):
            raise RuntimeError("Entity-link embedding output length mismatch.")
        point_groups = await asyncio.gather(
            *(
                self._asearch_entities(
                    vector,
                    limit=int(self.settings.entity_link_candidate_k),
                    active_version_ids=active_version_ids,
                )
                for vector in vectors
            )
        )
        mention_groups = point_groups[: len(clean_mentions)]
        context_groups = point_groups[len(clean_mentions) :]
        return [
            self._link_from_points(
                mention,
                self._merge_candidate_points(mention, mention_points, context_groups[index])
                if context_groups
                else mention_points,
                active_version_ids,
            )
            for index, (mention, mention_points) in enumerate(zip(clean_mentions, mention_groups))
        ]

    @staticmethod
    def _embedding_inputs(mentions: list[str], query_context: str) -> list[str]:
        context = str(query_context or "").strip()
        if not context:
            return list(mentions)
        bounded_context = context[:2000]
        context_inputs = [
            f"{mention}\nQuery context: {bounded_context}"
            for mention in mentions
        ]
        return [*mentions, *context_inputs]

    def _merge_candidate_points(
        self,
        mention: str,
        mention_points: list[Any],
        context_points: list[Any],
    ) -> list[dict[str, Any]]:
        """Fuse mention identity with query-context relevance before thresholding."""
        by_id: dict[str, dict[str, Any]] = {}
        for source, points in (("mention", mention_points), ("context", context_points)):
            for point in points:
                payload = self._extract_payload(point)
                candidate_id = self._extract_point_id(point, payload)
                if not candidate_id:
                    continue
                entry = by_id.setdefault(
                    candidate_id,
                    {
                        "id": candidate_id,
                        "payload": payload,
                        "mention_score": None,
                        "context_score": None,
                    },
                )
                score_key = f"{source}_score"
                score = self._extract_score(point)
                existing = entry.get(score_key)
                if existing is None or score > float(existing):
                    entry[score_key] = score
                    entry["payload"] = payload

        mention_weight = float(self.settings.entity_link_mention_weight)
        context_weight = 1.0 - mention_weight
        normalized_mention = self._normalize_name(mention)
        merged: list[dict[str, Any]] = []
        for candidate_id, entry in by_id.items():
            mention_score = entry.get("mention_score")
            context_score = entry.get("context_score")
            if mention_score is not None and context_score is not None:
                score = (mention_weight * float(mention_score)) + (
                    context_weight * float(context_score)
                )
            else:
                available_score = float(
                    mention_score if mention_score is not None else context_score or 0.0
                )
                candidate_name = str(entry["payload"].get("name") or "")
                score = (
                    available_score
                    if normalized_mention
                    and self._normalize_name(candidate_name) == normalized_mention
                    else available_score * max(mention_weight, context_weight)
                )
            merged.append(
                {
                    "id": candidate_id,
                    "payload": entry["payload"],
                    "score": max(0.0, min(1.0, score)),
                }
            )
        return sorted(merged, key=self._extract_score, reverse=True)[
            : int(self.settings.entity_link_candidate_k)
        ]

    @staticmethod
    def _normalize_name(value: str) -> str:
        return "".join(character for character in value.casefold() if character.isalnum())

    async def _asearch_entities(
        self,
        query_vector: list[float],
        limit: int,
        active_version_ids: list[str] | None,
    ) -> list[Any]:
        response = await self._async_client.query_points(
            collection_name=self.settings.qdrant_entities_collection,
            query=query_vector,
            query_filter=build_active_version_filter(active_version_ids),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return self._extract_points(response)

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.close()

    def _search_entities(
        self,
        query_vector: list[float],
        limit: int,
        active_version_ids: list[str] | None = None,
    ) -> list[Any]:
        query_filter = build_active_version_filter(active_version_ids)
        query_points_error: Exception | None = None
        if hasattr(self._client, "query_points"):
            try:
                try:
                    response = self._client.query_points(
                        collection_name=self.settings.qdrant_entities_collection,
                        query=query_vector,
                        query_filter=query_filter,
                        limit=limit,
                        with_payload=True,
                        with_vectors=False,
                    )
                except TypeError:
                    response = self._client.query_points(
                        collection_name=self.settings.qdrant_entities_collection,
                        query=query_vector,
                        limit=limit,
                        with_payload=True,
                        with_vectors=False,
                    )
                # An empty result is a successful query, not a signal to call
                # legacy methods that may not exist on modern clients.
                return self._extract_points(response)
            except Exception as exc:
                query_points_error = exc

        if hasattr(self._client, "search"):
            response = self._client.search(
                collection_name=self.settings.qdrant_entities_collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return self._extract_points(response)

        if hasattr(self._client, "search_points"):
            response = self._client.search_points(
                collection_name=self.settings.qdrant_entities_collection,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return self._extract_points(response)

        if query_points_error is not None:
            raise RuntimeError(f"Qdrant entity search failed: {query_points_error}") from query_points_error
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

    def _extract_point_id(self, point: Any, payload: dict[str, Any]) -> str | None:
        if isinstance(point, dict):
            point_id = point.get("id")
        else:
            point_id = getattr(point, "id", None)
        candidate = payload.get("neo4j_node_id") or point_id
        if candidate is None:
            return None
        return str(candidate)

    def _embedding_payload_matches(
        self,
        payload: dict[str, Any],
        active_version_ids: list[str] | None = None,
    ) -> bool:
        if active_version_ids is not None and str(payload.get("document_version_id") or "") not in set(
            active_version_ids
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
