"""Qdrant vector indexing for chunks and entities."""

from __future__ import annotations

import json
import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any

from config import Settings
from ingestion.embedder import GeminiEmbedder


@dataclass
class ChunkVectorInput:
    chunk_id: str
    doc_id: str
    text: str
    section_heading: str | None
    page: int | None
    entity_ids_mentioned: list[str] = field(default_factory=list)
    document_version_id: str | None = None
    active: bool = True


@dataclass
class EntityVectorInput:
    neo4j_node_id: str
    name: str
    entity_type: str
    summary: str
    document_version_id: str | None = None
    doc_id: str | None = None
    active: bool = True


def _coerce_properties(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def build_entity_summary(name: str, entity_type: str, properties: Any) -> str:
    """Build a stable, concise summary string for entity vector embedding."""
    props = _coerce_properties(properties)
    if not props:
        return f"{name} ({entity_type})"
    parts = []
    for key in sorted(props.keys()):
        value = props[key]
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}: {value}")
    props_text = "; ".join(parts[:8])
    if not props_text:
        return f"{name} ({entity_type})"
    return f"{name} ({entity_type}) - {props_text}"


class QdrantVectorIndexer:
    """Owns collection lifecycle and point upserts/deletes in Qdrant."""

    def __init__(self, settings: Settings, embedder: GeminiEmbedder | None = None, client: Any | None = None) -> None:
        self.settings = settings
        self.embedder = embedder or GeminiEmbedder(settings)
        self._models = None

        if client is not None:
            self._client = client
        else:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.http import models
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("qdrant-client package is required for vector indexing.") from exc

            self._models = models
            self._client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=self.settings.qdrant_timeout_seconds,
            )

        self.ensure_collections()

    def ensure_collections(self) -> None:
        """Create collections when missing and verify vector dimensions."""
        self._ensure_collection(self.settings.qdrant_chunks_collection)
        self._ensure_collection(self.settings.qdrant_entities_collection)

    def upsert_chunk_vectors(self, inputs: list[ChunkVectorInput]) -> None:
        """Embed and upsert chunk vectors with deterministic point IDs."""
        if not inputs:
            return
        vectors = self.embedder.embed_texts([item.text for item in inputs])
        if len(vectors) != len(inputs):
            raise RuntimeError("Embedding output length mismatch for chunk indexing.")
        points = []
        for item, vector in zip(inputs, vectors):
            points.append(
                {
                    "id": item.chunk_id,
                    "vector": vector,
                    "payload": {
                        "doc_id": item.doc_id,
                        "chunk_id": item.chunk_id,
                        "text": item.text,
                        "section_heading": item.section_heading,
                        "page": item.page,
                        "entity_ids_mentioned": sorted(set(item.entity_ids_mentioned)),
                        "document_version_id": item.document_version_id,
                        "active": bool(item.active),
                        "embedding_model_id": self.settings.embedding_model_id,
                        "embedding_dimension": int(self.settings.embedding_dim),
                        "embedding_normalization_policy": self.settings.embedding_normalization_policy,
                        "embedding_prompt_version": self.settings.embedding_prompt_version,
                        "embedding_schema_version": self.settings.embedding_version,
                        "content_sha256": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
                    },
                }
            )
        self._client.upsert(
            collection_name=self.settings.qdrant_chunks_collection,
            points=points,
            wait=True,
        )

    def upsert_entity_vectors(self, inputs: list[EntityVectorInput]) -> None:
        """Embed and upsert entity vectors with deterministic point IDs."""
        if not inputs:
            return
        embed_texts = [f"{item.name}\nType: {item.entity_type}\nSummary: {item.summary}" for item in inputs]
        vectors = self.embedder.embed_texts(embed_texts)
        if len(vectors) != len(inputs):
            raise RuntimeError("Embedding output length mismatch for entity indexing.")
        points = []
        for item, vector in zip(inputs, vectors):
            points.append(
                {
                    "id": self._entity_point_id(item),
                    "vector": vector,
                    "payload": {
                        "neo4j_node_id": item.neo4j_node_id,
                        "name": item.name,
                        "type": item.entity_type,
                        "summary": item.summary,
                        "doc_id": item.doc_id,
                        "document_version_id": item.document_version_id,
                        "active": bool(item.active),
                        "embedding_model_id": self.settings.embedding_model_id,
                        "embedding_dimension": int(self.settings.embedding_dim),
                        "embedding_normalization_policy": self.settings.embedding_normalization_policy,
                        "embedding_prompt_version": self.settings.embedding_prompt_version,
                        "embedding_schema_version": self.settings.embedding_version,
                    },
                }
            )
        self._client.upsert(
            collection_name=self.settings.qdrant_entities_collection,
            points=points,
            wait=True,
        )

    def delete_chunk_vectors(self, chunk_ids: list[str]) -> None:
        """Delete stale chunk vector points by chunk IDs."""
        self._delete_points(self.settings.qdrant_chunks_collection, chunk_ids)

    def delete_entity_vectors(self, entity_ids: list[str]) -> None:
        """Delete stale entity vector points by entity IDs."""
        self._delete_points(self.settings.qdrant_entities_collection, entity_ids)

    def delete_version_entity_vectors(
        self,
        document_version_id: str,
        entity_ids: list[str],
    ) -> None:
        """Delete selected entity points within exactly one document version."""
        point_ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"graphrag:entity-vector:{document_version_id}:{entity_id}",
                )
            )
            for entity_id in sorted(set(entity_ids))
            if entity_id
        ]
        self._delete_points(self.settings.qdrant_entities_collection, point_ids)

    def set_document_version_active(self, document_version_id: str, active: bool) -> None:
        """Flip one staged version in both authoritative vector collections."""
        payload = {"active": bool(active)}
        for collection in (
            self.settings.qdrant_chunks_collection,
            self.settings.qdrant_entities_collection,
        ):
            self._client.set_payload(
                collection_name=collection,
                payload=payload,
                points=self._version_filter(document_version_id),
                wait=True,
            )

    def delete_document_version(self, document_version_id: str) -> None:
        """Delete vector points owned by exactly one immutable document version."""
        selector = self._version_filter(document_version_id)
        for collection in (
            self.settings.qdrant_chunks_collection,
            self.settings.qdrant_entities_collection,
        ):
            self._client.delete(
                collection_name=collection,
                points_selector=selector,
                wait=True,
            )

    def count_document_version(self, document_version_id: str) -> dict[str, int]:
        """Return exact staged counts for activation integrity checks."""
        return {
            "chunks": self._count(self.settings.qdrant_chunks_collection, document_version_id),
            "entities": self._count(self.settings.qdrant_entities_collection, document_version_id),
        }

    def _count(self, collection_name: str, document_version_id: str) -> int:
        response = self._client.count(
            collection_name=collection_name,
            count_filter=self._version_filter(document_version_id),
            exact=True,
        )
        if isinstance(response, dict):
            return int(response.get("count") or 0)
        return int(getattr(response, "count", 0) or 0)

    def _entity_point_id(self, item: EntityVectorInput) -> str:
        if not item.document_version_id:
            return item.neo4j_node_id
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"graphrag:entity-vector:{item.document_version_id}:{item.neo4j_node_id}",
            )
        )

    def _version_filter(self, document_version_id: str) -> Any:
        if self._models is not None:
            return self._models.Filter(
                must=[
                    self._models.FieldCondition(
                        key="document_version_id",
                        match=self._models.MatchValue(value=document_version_id),
                    )
                ]
            )
        return {
            "must": [
                {"key": "document_version_id", "match": {"value": document_version_id}}
            ]
        }

    def _delete_points(self, collection_name: str, point_ids: list[str]) -> None:
        if not point_ids:
            return
        if self._models is not None:
            selector = self._models.PointIdsList(points=point_ids)
        else:
            selector = {"points": point_ids}
        self._client.delete(
            collection_name=collection_name,
            points_selector=selector,
            wait=True,
        )

    def _ensure_collection(self, name: str) -> None:
        collections = self._client.get_collections()
        existing = {entry.name for entry in getattr(collections, "collections", [])}
        if name not in existing:
            self._create_collection(name)
            return

        info = self._client.get_collection(name)
        size = self._extract_vector_size(info)
        if size is not None and int(size) != int(self.settings.embedding_dim):
            raise RuntimeError(
                f"Qdrant collection '{name}' has vector size {size}, expected {self.settings.embedding_dim}."
            )
        distance = self._extract_vector_distance(info)
        if distance is not None and str(distance).lower().split(".")[-1] != "cosine":
            raise RuntimeError(
                f"Qdrant collection '{name}' uses distance {distance}, expected cosine."
            )

    def _create_collection(self, name: str) -> None:
        if self._models is None:
            # Test doubles may not expose qdrant models; caller handles collection setup.
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=self._models.VectorParams(
                size=self.settings.embedding_dim,
                distance=self._models.Distance.COSINE,
            ),
        )

    def _extract_vector_size(self, info: Any) -> int | None:
        config = getattr(info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if vectors is None:
            return None
        size = getattr(vectors, "size", None)
        if size is not None:
            return int(size)
        if isinstance(vectors, dict):
            default = vectors.get("")
            if isinstance(default, dict):
                default_size = default.get("size")
                if default_size is not None:
                    return int(default_size)
            default_size = getattr(default, "size", None)
            if default_size is not None:
                return int(default_size)
        return None

    def _extract_vector_distance(self, info: Any) -> str | None:
        config = getattr(info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if vectors is None:
            return None
        distance = getattr(vectors, "distance", None)
        if distance is not None:
            return str(distance)
        if isinstance(vectors, dict):
            default = vectors.get("")
            if isinstance(default, dict) and default.get("distance") is not None:
                return str(default["distance"])
            nested = getattr(default, "distance", None)
            if nested is not None:
                return str(nested)
        return None
