"""Within-document entity disambiguation.

WP2-B makes identity name-first (REVIEW1 findings A2/B2). Previously both the exact-alias
and the embedding paths refused to consider a cluster whose type set did not already
contain the incoming mention's free-form type. Because the model invents that type per
mention — 313 distinct strings over 3,370 mentions — this shattered entities that share a
byte-identical name: 91 of 265 active entities (34%) were exact-name duplicates inside a
single document version, and coreference B-cubed on multi-mention entities sat at 56.2%.

Now:
  * an exact normalized name or alias match merges unconditionally;
  * the embedding path blocks only on a confident coarse-class disagreement, and gives a
    small bonus when the raw types agree, so type still informs ranking;
  * observed types accumulate as a multiset — the emergent schema the architecture wants
    to observe — instead of acting as a gate.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import Settings
from ingestion.entity_types import (
    coarse_class,
    coarse_classes_compatible,
    display_type,
    dominant_coarse_class,
)
from models.extraction import ExtractedEntity


@dataclass
class CanonicalEntity:
    entity_id: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None
    # Every raw type ever seen for this entity, with counts. Evidence, not a gate.
    entity_types: Counter[str] = field(default_factory=Counter)
    member_count: int = 1

    @property
    def coarse_class(self) -> str:
        """Decided once per entity from every observed type, never per mention."""
        return dominant_coarse_class(self.entity_types.elements())

    @property
    def display_type(self) -> str:
        return display_type(self.entity_types.elements())


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
                    entity_types=Counter(
                        [entity.entity_type.strip()] if entity.entity_type.strip() else []
                    ),
                )
                canonical_clusters.append(new_cluster)
                entity_id_map[entity.entity_id] = new_cluster.entity_id
            else:
                if name not in matched.aliases:
                    matched.aliases.append(name)
                if self._canonical_name_key(name) > self._canonical_name_key(matched.canonical_name):
                    matched.canonical_name = name
                if entity.entity_type.strip():
                    matched.entity_types[entity.entity_type.strip()] += 1
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
        usable: list[tuple[dict[str, object], str, list[str]]] = []
        for entry in existing_entities:
            canonical_name = str(entry.get("canonical_name", "")).strip()
            if not canonical_name:
                continue
            aliases = [canonical_name]
            for alias in entry.get("aliases") or []:
                if alias not in aliases:
                    aliases.append(alias)
            usable.append((entry, canonical_name, aliases))

        # One batched encode rather than one call per prior entity (REVIEW1 B12).
        embeddings = self._embed_many([name for _, name, _ in usable])
        return [
            CanonicalEntity(
                entity_id=str(entry.get("entity_id")),
                canonical_name=canonical_name,
                aliases=aliases,
                embedding=embedding,
                entity_types=Counter(
                    [str(entry.get("entity_type")).strip()]
                    if str(entry.get("entity_type") or "").strip()
                    else []
                ),
                member_count=max(1, len(aliases)),
            )
            for (entry, canonical_name, aliases), embedding in zip(usable, embeddings)
        ]

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
        """Nearest compatible cluster above the threshold.

        Type is demoted from a gate to a ranking signal: it can only block on a confident
        coarse-class disagreement (a person against a place), and otherwise contributes a
        small bonus when the raw types agree.
        """
        best = None
        best_score = -1.0
        incoming_class = coarse_class(entity_type)
        normalized_type = self._normalize_type(entity_type)
        blocking = bool(self.settings.entity_coarse_class_blocking)

        for cluster in clusters:
            if cluster.embedding is None:
                continue
            if blocking and not coarse_classes_compatible(incoming_class, cluster.coarse_class):
                continue
            score = float(np.dot(embedding, cluster.embedding))
            if normalized_type and normalized_type in {
                self._normalize_type(item) for item in cluster.entity_types
            }:
                score = min(1.0, score + self.settings.entity_same_type_bonus)
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
        """Identical name or alias is identity, unless the coarse classes disagree.

        This removes the 34% duplicate rate: two mentions of "RAG" are one entity even
        when the model typed one System and the other Concept, because both reduce to
        MISC. But it must not flatten genuine homonyms — "Atlas" the person and "Atlas"
        the organisation are different entities, and PER against ORG is exactly the
        confident disagreement worth blocking on.
        """
        normalized_name = self._normalize_name(name)
        incoming_class = coarse_class(entity_type)
        blocking = bool(self.settings.entity_coarse_class_blocking)
        for cluster in clusters:
            if blocking and not coarse_classes_compatible(incoming_class, cluster.coarse_class):
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
