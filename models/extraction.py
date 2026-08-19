"""Validated models for provider extraction outputs."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models.refactor import normalize_predicate


class ExtractionModel(BaseModel):
    """Strict provider-boundary model with assignment validation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _validate_primitive_properties(value: dict[str, Any]) -> dict[str, Any]:
    """Keep provider properties safe for deterministic literal assertions."""
    validated: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError("Property names must be non-empty strings.")
        if item is None or isinstance(item, (dict, list, tuple, set)):
            raise ValueError(f"Property {key!r} must contain a non-null primitive value.")
        if not isinstance(item, (str, int, float, bool)):
            raise ValueError(f"Property {key!r} contains an unsupported value type.")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"Property {key!r} must contain a finite numeric value.")
        validated[key] = item
    return validated


def _coerce_properties(value: Any) -> dict[str, Any]:
    """Accept legacy dictionaries and Gemini-safe fixed property records."""
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, list):
        raise ValueError("Properties must be an object or an array of property records.")

    properties: dict[str, Any] = {}
    for raw_item in value:
        if not isinstance(raw_item, dict):
            raise ValueError("Every property record must be an object.")
        name = str(raw_item.get("name") or "").strip()
        value_type = str(raw_item.get("value_type") or "").strip().lower()
        value_text = str(raw_item.get("value_text") or "").strip()
        if not name:
            raise ValueError("Property records require a non-empty name.")
        if name in properties:
            raise ValueError(f"Duplicate property name {name!r}.")
        if value_type == "string":
            parsed: Any = value_text
        elif value_type == "integer":
            parsed = int(value_text)
        elif value_type == "number":
            parsed = float(value_text)
        elif value_type == "boolean":
            normalized = value_text.lower()
            if normalized not in {"true", "false"}:
                raise ValueError(f"Boolean property {name!r} must be true or false.")
            parsed = normalized == "true"
        else:
            raise ValueError(f"Property {name!r} has unsupported value_type {value_type!r}.")
        properties[name] = parsed
    return properties


class ExtractedEntity(ExtractionModel):
    entity_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)

    @field_validator("properties")
    @classmethod
    def validate_properties(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_primitive_properties(value)

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ExtractedEntity":
        return ExtractedEntity.model_validate(
            {
                "entity_id": payload.get("id") or payload.get("entity_id") or "",
                "name": payload.get("name") or "",
                "entity_type": payload.get("type") or payload.get("entity_type") or "",
                "properties": _coerce_properties(payload.get("properties")),
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

    @field_validator("properties")
    @classmethod
    def validate_properties(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_primitive_properties(value)

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
                "properties": _coerce_properties(payload.get("properties")),
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
