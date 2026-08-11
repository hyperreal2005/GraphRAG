"""Validated models for provider extraction outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.refactor import normalize_predicate


class ExtractionModel(BaseModel):
    """Strict provider-boundary model with assignment validation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ExtractedEntity(ExtractionModel):
    entity_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ExtractedEntity":
        return ExtractedEntity.model_validate(
            {
                "entity_id": payload.get("id") or payload.get("entity_id") or "",
                "name": payload.get("name") or "",
                "entity_type": payload.get("type") or payload.get("entity_type") or "",
                "properties": payload.get("properties") or {},
                "confidence": payload.get("confidence", 0.75),
            }
        )


class ExtractedRelation(ExtractionModel):
    head_id: str = Field(min_length=1)
    head_type: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    raw_relation: str = Field(min_length=1)
    tail_id: str = Field(min_length=1)
    tail_type: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    valid_from: str | None = None
    valid_until: str | None = None

    @field_validator("relation")
    @classmethod
    def normalize_relation(cls, value: str) -> str:
        return normalize_predicate(value)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ExtractedRelation":
        raw_relation = str(payload.get("raw_relation") or payload.get("relation") or "").strip()
        return ExtractedRelation.model_validate(
            {
                "head_id": payload.get("head") or payload.get("head_id") or "",
                "head_type": payload.get("head_type") or "Entity",
                "relation": payload.get("relation") or raw_relation,
                "raw_relation": raw_relation,
                "tail_id": payload.get("tail") or payload.get("tail_id") or "",
                "tail_type": payload.get("tail_type") or "Entity",
                "properties": payload.get("properties") or {},
                "confidence": payload.get("confidence", 0.0),
                "valid_from": payload.get("valid_from"),
                "valid_until": payload.get("valid_until"),
            }
        )


class ExtractionResult(ExtractionModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ExtractionResult":
        entities_raw = payload.get("entities") or []
        relations_raw = payload.get("relations") or []
        return ExtractionResult(
            entities=[ExtractedEntity.from_dict(item) for item in entities_raw if isinstance(item, dict)],
            relations=[ExtractedRelation.from_dict(item) for item in relations_raw if isinstance(item, dict)],
        )
