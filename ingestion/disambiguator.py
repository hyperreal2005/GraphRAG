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

        for entity in entities:
            name = entity.name.strip()
            if not name:
                continue
            embedding = self._embed(name)
            matched = self._match_cluster(embedding, canonical_clusters)
            if matched is None:
                new_cluster = CanonicalEntity(
                    entity_id=str(uuid.uuid4()),
                    canonical_name=name,
                    aliases=[name],
                    embedding=embedding,
                )
                canonical_clusters.append(new_cluster)
                entity_id_map[entity.entity_id] = new_cluster.entity_id
            else:
                if name not in matched.aliases:
                    matched.aliases.append(name)
                entity_id_map[entity.entity_id] = matched.entity_id

        return DisambiguationResult(canonical_entities=canonical_clusters, entity_id_map=entity_id_map)

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
                )
            )
        return clusters

    def _embed(self, text: str) -> np.ndarray:
        vector = self._model.encode([text], normalize_embeddings=True)
        return np.asarray(vector[0])

    def _match_cluster(self, embedding: np.ndarray, clusters: list[CanonicalEntity]) -> CanonicalEntity | None:
        best = None
        best_score = -1.0
        for cluster in clusters:
            if cluster.embedding is None:
                continue
            score = float(np.dot(embedding, cluster.embedding))
            if score > best_score:
                best_score = score
                best = cluster
        if best is not None and best_score >= self.settings.disamb_threshold:
            return best
        return None

