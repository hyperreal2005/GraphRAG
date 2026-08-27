"""Qdrant vector indexing for chunks and entities."""

from __future__ import annotations

import json
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from config import Settings
from ingestion.embedder import GeminiEmbedder
from ingestion.sparse_encoder import SPARSE_TOKENIZER_VERSION, encode_document

logger = logging.getLogger(__name__)

# Name of the sparse channel inside the chunks collection. The dense vector stays unnamed
# so existing dense-only queries keep working unchanged.
SPARSE_VECTOR_NAME = "text_sparse"


def build_active_version_filter(
    active_version_ids: list[str] | None,
    *,
    exclude_reference_chunks: bool = False,
) -> Any:
    """Build the shared Qdrant read boundary owned by SQLite activation state."""
    must_fallback: list[dict[str, Any]] = [
        {"key": "active", "match": {"value": True}}
    ]
    if active_version_ids is not None:
        must_fallback.append(
            {"key": "document_version_id", "match": {"any": active_version_ids}}
        )
    must_not_fallback: list[dict[str, Any]] = []
    if exclude_reference_chunks:
        must_not_fallback.append(
            {"key": "content_kind", "match": {"value": "reference"}}
        )
    try:
        from qdrant_client.http import models
    except Exception:
        payload: dict[str, Any] = {"must": must_fallback}
        if must_not_fallback:
            payload["must_not"] = must_not_fallback
        return payload

    must = [
        models.FieldCondition(key="active", match=models.MatchValue(value=True))
    ]
    if active_version_ids is not None:
        must.append(
            models.FieldCondition(
                key="document_version_id",
                match=models.MatchAny(any=active_version_ids),
            )
        )
    must_not = []
    if exclude_reference_chunks:
        must_not.append(
            models.FieldCondition(
                key="content_kind",
                match=models.MatchValue(value="reference"),
            )
        )
    return models.Filter(must=must, must_not=must_not or None)


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
    content_kind: str = "body"


@dataclass
class EntityVectorInput:
    neo4j_node_id: str
    name: str
    entity_type: str
    summary: str
    aliases: list[str] = field(default_factory=list)
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


def build_entity_embed_text(item: "EntityVectorInput") -> str:
    """Text embedded for one entity.

    WP2-B removed the raw `Type:` line. The model assigns that type per mention and it is
    highly unstable — 313 distinct strings across 3,370 mentions — so including it made
    the vector itself wobble with the same noise that was already corrupting the identity
    gate (REVIEW1 finding A2). Aliases replace it: they are stable, and they are what a
    query mention is most likely to look like.
    """
    parts = [item.name]
    aliases = [
        alias for alias in dict.fromkeys(item.aliases)
        if alias and alias.strip() and alias.strip() != item.name.strip()
    ]
    if aliases:
        parts.append("Also known as: " + ", ".join(aliases[:8]))
    if item.summary and item.summary.strip():
        parts.append("Summary: " + item.summary.strip())
    return "\n".join(parts)


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
        self._ensure_payload_indexes()

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
                    "vector": self._chunk_vector_payload(vector, item.text),
                    "payload": {
                        "doc_id": item.doc_id,
                        "chunk_id": item.chunk_id,
                        "text": item.text,
                        "section_heading": item.section_heading,
                        "page": item.page,
                        "entity_ids_mentioned": sorted(set(item.entity_ids_mentioned)),
                        "document_version_id": item.document_version_id,
                        "active": bool(item.active),
                        "content_kind": item.content_kind,
                        "embedding_model_id": self.settings.embedding_model_id,
                        "embedding_dimension": int(self.settings.embedding_dim),
                        "embedding_normalization_policy": self.settings.embedding_normalization_policy,
                        "embedding_prompt_version": self.settings.embedding_prompt_version,
                        "embedding_schema_version": self.settings.embedding_version,
                        "content_sha256": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
                        # WP6: part of the fence. A chunk indexed by one tokenizer must
                        # never be scored against a query encoded by another.
                        "sparse_tokenizer_version": (
                            SPARSE_TOKENIZER_VERSION if self.settings.query_sparse_enabled else None
                        ),
                    },
                }
            )
        self._upsert_in_batches(self.settings.qdrant_chunks_collection, points)

    def _upsert_in_batches(self, collection_name: str, points: list[dict]) -> None:
        """Upsert in bounded batches.

        A single request carrying every point of a document is unbounded in size: a
        987-chunk document produced a 47.7 MB body against Qdrant's 33.5 MB limit and
        failed permanently, *after* the graph stage had already committed. Retries could
        never help, because the payload was the same size every time. Batching makes the
        request size a function of the batch, not of the document.
        """
        batch_size = max(1, int(self.settings.qdrant_upsert_batch_size))
        for start in range(0, len(points), batch_size):
            self._client.upsert(
                collection_name=collection_name,
                points=points[start : start + batch_size],
                wait=True,
            )

    def _chunk_vector_payload(self, dense: list[float], text: str):  # noqa: ANN201
        """Dense alone, or dense plus the local sparse channel when WP6 is enabled."""
        if not self.settings.query_sparse_enabled or self._models is None:
            return dense
        sparse = encode_document(
            text, average_length=float(self.settings.sparse_average_chunk_tokens)
        )
        if not sparse:
            return dense
        return {
            "": dense,
            SPARSE_VECTOR_NAME: self._models.SparseVector(
                indices=sparse.indices, values=sparse.values
            ),
        }

    def upsert_entity_vectors(self, inputs: list[EntityVectorInput]) -> None:
        """Embed and upsert entity vectors with deterministic point IDs."""
        if not inputs:
            return
        embed_texts = [build_entity_embed_text(item) for item in inputs]
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
        self._upsert_in_batches(self.settings.qdrant_entities_collection, points)

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

    # Every read is fenced by `active` and `document_version_id` (see
    # `build_active_version_filter`), and retraction deletes by `document_version_id`.
    # Without payload indexes Qdrant evaluates those as a full payload scan on every
    # query, which is what put vector retrieval on its timeout (REVIEW1 finding B8).
    PAYLOAD_INDEXES: dict[str, list[tuple[str, str]]] = {
        "chunks": [
            ("active", "bool"),
            ("document_version_id", "keyword"),
            ("doc_id", "keyword"),
            ("content_kind", "keyword"),
        ],
        "entities": [
            ("active", "bool"),
            ("document_version_id", "keyword"),
            ("doc_id", "keyword"),
            # The entity payload keys its graph identity as `neo4j_node_id`, not
            # `entity_id`; `EntityLinker` dedupes on it after every lookup.
            ("neo4j_node_id", "keyword"),
        ],
    }

    def _ensure_payload_indexes(self) -> None:
        """Create the filter indexes, idempotently.

        Qdrant returns an error when an index already exists with the same schema; that is
        the normal steady state here, so it is swallowed. A genuine failure is logged and
        left alone rather than blocking ingestion — a missing index costs latency, not
        correctness.
        """
        creator = getattr(self._client, "create_payload_index", None)
        if creator is None:  # test doubles do not implement payload indexes
            return
        targets = (
            (self.settings.qdrant_chunks_collection, self.PAYLOAD_INDEXES["chunks"]),
            (self.settings.qdrant_entities_collection, self.PAYLOAD_INDEXES["entities"]),
        )
        for collection_name, fields in targets:
            for field_name, schema in fields:
                try:
                    creator(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=schema,
                        wait=True,
                    )
                except Exception as exc:  # noqa: BLE001 - see docstring
                    message = str(exc).lower()
                    if "already exists" in message or "already_exists" in message:
                        continue
                    logger.warning(
                        "qdrant_payload_index_failed collection=%s field=%s error=%s",
                        collection_name,
                        field_name,
                        exc,
                    )

    def _create_collection(self, name: str) -> None:
        if self._models is None:
            # Test doubles may not expose qdrant models; caller handles collection setup.
            return
        kwargs = {
            "collection_name": name,
            "vectors_config": self._models.VectorParams(
                size=self.settings.embedding_dim,
                distance=self._models.Distance.COSINE,
            ),
        }
        if name == self.settings.qdrant_chunks_collection and self.settings.query_sparse_enabled:
            # IDF is computed by Qdrant from the collection it actually holds. Computing
            # it here would need a corpus pass that goes stale on the next ingest.
            kwargs["sparse_vectors_config"] = {
                SPARSE_VECTOR_NAME: self._models.SparseVectorParams(
                    modifier=self._models.Modifier.IDF
                )
            }
        self._client.create_collection(**kwargs)

    def chunks_collection_has_sparse(self) -> bool:
        """Whether the live chunks collection carries the WP6 sparse channel.

        An existing dense-only collection cannot gain a sparse vector in place; it has to
        be recreated and re-indexed. The retriever asks this so it degrades to dense-only
        rather than issuing a query the collection cannot answer.
        """
        try:
            info = self._client.get_collection(self.settings.qdrant_chunks_collection)
        except Exception:
            return False
        config = getattr(getattr(info, "config", None), "params", None)
        sparse = getattr(config, "sparse_vectors", None) or {}
        try:
            return SPARSE_VECTOR_NAME in sparse
        except TypeError:
            return False

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
