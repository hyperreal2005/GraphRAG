"""Within-document entity disambiguation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Settings
from models.extraction import ExtractedEntity


@dataclass
class CanonicalEntity:
    entity_id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    entity_types: set[str] = field(default_factory=set)
    member_count: int = 1


@dataclass
class DisambiguationResult:
    canonical_entities: list[CanonicalEntity]
    entity_id_map: dict[str, str]


class EntityDisambiguator:
    """Cluster entity name variants within a document."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = SentenceTransformer(self.settings.disamb_model)

    def disambiguate(
        self,
        entities: list[ExtractedEntity],
        existing_entities: list[dict[str, object]],
    ) -> DisambiguationResult:
        """Return canonical entities and mapping from raw IDs to canonical IDs."""
        canonical_clusters = self._load_existing_clusters(existing_entities)
        entity_id_map: dict[str, str] = {}
        usable_entities = [entity for entity in entities if entity.name.strip()]
        embeddings = self._embed_many([entity.name.strip() for entity in usable_entities])

        for entity, embedding in zip(usable_entities, embeddings):
            name = entity.name.strip()
            matched = self._match_exact_alias(name, entity.entity_type, canonical_clusters)
            if matched is None:
                matched = self._match_cluster(embedding, canonical_clusters, entity.entity_type)
            if matched is None:
                new_cluster = CanonicalEntity(
                    entity_id=str(uuid.uuid4()),
                    canonical_name=name,
                    aliases=[name],
                    embedding=embedding,
                    entity_types={self._normalize_type(entity.entity_type)} if entity.entity_type.strip() else set(),
                )
                canonical_clusters.append(new_cluster)
                entity_id_map[entity.entity_id] = new_cluster.entity_id
            else:
                if name not in matched.aliases:
                    matched.aliases.append(name)
                if self._canonical_name_key(name) > self._canonical_name_key(matched.canonical_name):
                    matched.canonical_name = name
                normalized_type = self._normalize_type(entity.entity_type)
                if normalized_type:
                    matched.entity_types.add(normalized_type)
                self._update_centroid(matched, embedding)
                entity_id_map[entity.entity_id] = matched.entity_id

        referenced_cluster_ids = set(entity_id_map.values())
        return DisambiguationResult(
            canonical_entities=[
                cluster
                for cluster in canonical_clusters
                if cluster.entity_id in referenced_cluster_ids
            ],
            entity_id_map=entity_id_map,
        )

    def _load_existing_clusters(self, existing_entities: list[dict[str, object]]) -> list[CanonicalEntity]:
        clusters: list[CanonicalEntity] = []
        for entry in existing_entities:
            canonical_name = str(entry.get("canonical_name", "")).strip()
            if not canonical_name:
                continue
            aliases = [canonical_name]
            stored_aliases = entry.get("aliases") or []
            for alias in stored_aliases:
                if alias not in aliases:
                    aliases.append(alias)
            clusters.append(
                CanonicalEntity(
                    entity_id=str(entry.get("entity_id")),
                    canonical_name=canonical_name,
                    aliases=aliases,
                    embedding=self._embed(canonical_name),
                    entity_types={self._normalize_type(str(entry.get("entity_type") or ""))}
                    if str(entry.get("entity_type") or "").strip()
                    else set(),
                    member_count=max(1, len(aliases)),
                )
            )
        return clusters

    def _embed(self, text: str) -> np.ndarray:
        return self._embed_many([text])[0]

    def _embed_many(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [np.asarray(vector) for vector in vectors]

    def _match_cluster(
        self,
        embedding: np.ndarray,
        clusters: list[CanonicalEntity],
        entity_type: str,
    ) -> CanonicalEntity | None:
        best = None
        best_score = -1.0
        normalized_type = self._normalize_type(entity_type)
        for cluster in clusters:
            if cluster.embedding is None:
                continue
            if normalized_type and cluster.entity_types and normalized_type not in cluster.entity_types:
                continue
            score = float(np.dot(embedding, cluster.embedding))
            if score > best_score:
                best_score = score
                best = cluster
        if best is not None and best_score >= self.settings.disamb_threshold:
            return best
        return None

    def _match_exact_alias(
        self,
        name: str,
        entity_type: str,
        clusters: list[CanonicalEntity],
    ) -> CanonicalEntity | None:
        normalized_name = self._normalize_name(name)
        normalized_type = self._normalize_type(entity_type)
        for cluster in clusters:
            if normalized_type and cluster.entity_types and normalized_type not in cluster.entity_types:
                continue
            if any(self._normalize_name(alias) == normalized_name for alias in cluster.aliases):
                return cluster
        return None

    def _update_centroid(self, cluster: CanonicalEntity, embedding: np.ndarray) -> None:
        if cluster.embedding is None:
            cluster.embedding = embedding
            cluster.member_count = 1
            return
        count = max(1, int(cluster.member_count))
        centroid = (cluster.embedding * count + embedding) / (count + 1)
        norm = float(np.linalg.norm(centroid))
        cluster.embedding = centroid / norm if norm else centroid
        cluster.member_count = count + 1

    def _normalize_name(self, value: str) -> str:
        return " ".join(value.casefold().split())

    def _normalize_type(self, value: str) -> str:
        normalized = " ".join(str(value or "").casefold().split())
        aliases = {
            "org": "organization",
            "organisation": "organization",
            "company": "organization",
            "business": "organization",
            "human": "person",
        }
        return aliases.get(normalized, normalized)

    def _canonical_name_key(self, value: str) -> tuple[int, int, str]:
        normalized = " ".join(value.split())
        return (len(normalized.split()), len(normalized), normalized.casefold())
