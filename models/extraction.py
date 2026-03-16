"""Models for extraction outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExtractedEntity:
    entity_id: str
    name: str
    entity_type: str
    properties: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ExtractedEntity":
        return ExtractedEntity(
            entity_id=str(payload.get("id") or payload.get("entity_id") or ""),
            name=str(payload.get("name") or "").strip(),
            entity_type=str(payload.get("type") or payload.get("entity_type") or "").strip(),
            properties=dict(payload.get("properties") or {}),
        )


@dataclass
class ExtractedRelation:
    head_id: str
    head_type: str
    relation: str
    tail_id: str
    tail_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ExtractedRelation":
        return ExtractedRelation(
            head_id=str(payload.get("head") or ""),
            head_type=str(payload.get("head_type") or "").strip(),
            relation=str(payload.get("relation") or "").strip(),
            tail_id=str(payload.get("tail") or ""),
            tail_type=str(payload.get("tail_type") or "").strip(),
            properties=dict(payload.get("properties") or {}),
            confidence=float(payload.get("confidence") or 0.0),
        )


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ExtractionResult":
        entities_raw = payload.get("entities") or []
        relations_raw = payload.get("relations") or []
        entities = [ExtractedEntity.from_dict(item) for item in entities_raw if isinstance(item, dict)]
        relations = [ExtractedRelation.from_dict(item) for item in relations_raw if isinstance(item, dict)]
        return ExtractionResult(entities=entities, relations=relations)

