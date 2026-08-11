"""Validated Refactor 1 contracts and deterministic assertion identity helpers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def canonical_json(value: Any) -> str:
    """Serialize JSON data deterministically or fail for unsupported values."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_json_object(value: str | dict[str, Any] | None) -> str:
    """Validate and canonicalize a JSON object."""
    if value is None:
        return "{}"
    decoded: Any = value
    if isinstance(value, str):
        decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("Expected a JSON object.")
    return canonical_json(decoded)


def canonical_json_value(value: str | Any) -> str:
    """Validate and canonicalize any JSON literal/object/list value."""
    decoded = value
    if isinstance(value, str):
        decoded = json.loads(value)
    return canonical_json(decoded)


def normalize_predicate(value: str) -> str:
    """Mechanically normalize a raw predicate without applying an ontology."""
    normalized = re.sub(r"[^A-Z0-9]+", "_", value.strip().upper()).strip("_")
    return normalized or "RELATED_TO"


def build_claim_key(
    *,
    subject_entity_id: str,
    predicate_normalized: str,
    object_kind: str,
    object_entity_id: str | None,
    literal_value_json: str | None,
    properties_json: str,
    valid_from: str | None,
    valid_until: str | None,
) -> str:
    """Build the stable semantic identity shared by corroborating assertions."""
    object_value = object_entity_id if object_kind == "entity" else literal_value_json
    payload = "|".join(
        [
            subject_entity_id,
            predicate_normalized,
            object_kind,
            object_value or "",
            properties_json,
            valid_from or "",
            valid_until or "",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_assertion_id(document_version_id: str, chunk_id: str, claim_key: str) -> str:
    """Build a source-specific assertion identity."""
    return hashlib.sha256(f"{document_version_id}|{chunk_id}|{claim_key}".encode("utf-8")).hexdigest()


class RefactorModel(BaseModel):
    """Strict mutable model used at provider and public API boundaries."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class AssertionRecord(RefactorModel):
    """One source-specific, evidence-bearing graph assertion."""

    id: str = Field(min_length=64, max_length=64)
    claim_key: str = Field(min_length=64, max_length=64)
    subject_entity_id: str = Field(min_length=1)
    subject_local_entity_id: str | None = None
    predicate_raw: str = Field(min_length=1)
    predicate_normalized: str = Field(min_length=1)
    object_kind: Literal["entity", "literal"]
    object_entity_id: str | None = None
    object_local_entity_id: str | None = None
    literal_value_json: str | None = None
    properties_json: str = "{}"
    confidence: float = Field(ge=0.0, le=1.0)
    doc_id: str = Field(min_length=1)
    document_version_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    extractor_model: str = Field(min_length=1)
    extraction_prompt_version: str = Field(min_length=1)
    valid_from: str | None = None
    valid_until: str | None = None
    active: bool = False
    review_status: Literal["accepted", "flagged", "rejected"] = "accepted"
    created_at: AwareDatetime = Field(default_factory=utc_now)
    updated_at: AwareDatetime = Field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        subject_entity_id: str,
        subject_local_entity_id: str | None = None,
        predicate_raw: str,
        object_kind: Literal["entity", "literal"],
        object_entity_id: str | None,
        object_local_entity_id: str | None = None,
        literal_value: Any | None,
        properties: dict[str, Any] | None,
        confidence: float,
        doc_id: str,
        document_version_id: str,
        chunk_id: str,
        extractor_model: str,
        extraction_prompt_version: str,
        valid_from: str | None = None,
        valid_until: str | None = None,
        active: bool = False,
        review_status: Literal["accepted", "flagged", "rejected"] = "accepted",
    ) -> "AssertionRecord":
        predicate_normalized = normalize_predicate(predicate_raw)
        properties_json = canonical_json_object(properties or {})
        literal_value_json = None
        if object_kind == "literal":
            literal_value_json = canonical_json(literal_value)
        claim_key = build_claim_key(
            subject_entity_id=subject_entity_id,
            predicate_normalized=predicate_normalized,
            object_kind=object_kind,
            object_entity_id=object_entity_id,
            literal_value_json=literal_value_json,
            properties_json=properties_json,
            valid_from=valid_from,
            valid_until=valid_until,
        )
        return cls(
            id=build_assertion_id(document_version_id, chunk_id, claim_key),
            claim_key=claim_key,
            subject_entity_id=subject_entity_id,
            subject_local_entity_id=subject_local_entity_id,
            predicate_raw=predicate_raw,
            predicate_normalized=predicate_normalized,
            object_kind=object_kind,
            object_entity_id=object_entity_id,
            object_local_entity_id=object_local_entity_id,
            literal_value_json=literal_value_json,
            properties_json=properties_json,
            confidence=confidence,
            doc_id=doc_id,
            document_version_id=document_version_id,
            chunk_id=chunk_id,
            source_ref=f"{doc_id}:{document_version_id}:{chunk_id}",
            extractor_model=extractor_model,
            extraction_prompt_version=extraction_prompt_version,
            valid_from=valid_from,
            valid_until=valid_until,
            active=active,
            review_status=review_status,
        )

    @model_validator(mode="after")
    def validate_identity_and_object(self) -> "AssertionRecord":
        object.__setattr__(self, "properties_json", canonical_json_object(self.properties_json))
        if self.object_kind == "entity":
            if not self.object_entity_id:
                raise ValueError("Entity assertions require object_entity_id.")
            if self.literal_value_json is not None:
                raise ValueError("Entity assertions cannot contain literal_value_json.")
        else:
            if self.object_entity_id is not None:
                raise ValueError("Literal assertions cannot contain object_entity_id.")
            if self.literal_value_json is None:
                raise ValueError("Literal assertions require literal_value_json.")
            object.__setattr__(self, "literal_value_json", canonical_json_value(self.literal_value_json))
        if self.valid_from and self.valid_until and self.valid_from > self.valid_until:
            raise ValueError("valid_from cannot be after valid_until.")
        expected_predicate = normalize_predicate(self.predicate_raw)
        if self.predicate_normalized != expected_predicate:
            raise ValueError("predicate_normalized does not match predicate_raw.")
        expected_claim_key = build_claim_key(
            subject_entity_id=self.subject_entity_id,
            predicate_normalized=self.predicate_normalized,
            object_kind=self.object_kind,
            object_entity_id=self.object_entity_id,
            literal_value_json=self.literal_value_json,
            properties_json=self.properties_json,
            valid_from=self.valid_from,
            valid_until=self.valid_until,
        )
        if self.claim_key != expected_claim_key:
            raise ValueError("claim_key does not match assertion content.")
        expected_id = build_assertion_id(self.document_version_id, self.chunk_id, self.claim_key)
        if self.id != expected_id:
            raise ValueError("assertion id does not match its source and claim key.")
        expected_source_ref = f"{self.doc_id}:{self.document_version_id}:{self.chunk_id}"
        if self.source_ref != expected_source_ref:
            raise ValueError("source_ref does not match the assertion source.")
        return self


class DocumentVersion(RefactorModel):
    """Immutable input/config generation for one logical document."""

    version_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    checksum: str = Field(min_length=64, max_length=64)
    pipeline_fingerprint: str = Field(min_length=1)
    generative_model: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_dim: int = Field(gt=0)
    embedding_prompt_version: str = Field(min_length=1)
    embedding_schema_version: str = "legacy"
    embedding_normalization_policy: str = "provider-default"
    document_embedding_instruction: str = "legacy"
    query_embedding_instruction: str = "legacy"
    extraction_prompt_version: str = "legacy"
    graph_schema_version: int = Field(gt=0)
    status: Literal["building", "staged", "active", "failed", "superseded"] = "building"
    last_completed_stage: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    activated_at: AwareDatetime | None = None
    superseded_at: AwareDatetime | None = None
    last_error: str | None = None


OutboxEventType = Literal[
    "GRAPH_STAGE_VERSION",
    "VECTOR_STAGE_VERSION",
    "ACTIVATE_VERSION",
    "RETRACT_VERSION",
    "REFRESH_ENTITY_VECTOR",
]


class OutboxEvent(RefactorModel):
    """Durable idempotent work item for external stores."""

    event_id: str = Field(min_length=1)
    event_type: OutboxEventType
    run_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    document_version_id: str = Field(min_length=1)
    payload_json: str = "{}"
    status: Literal["pending", "processing", "completed", "failed", "dead_letter"] = "pending"
    attempts: int = Field(default=0, ge=0)
    available_at: AwareDatetime = Field(default_factory=utc_now)
    last_error: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    completed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def validate_payload(self) -> "OutboxEvent":
        object.__setattr__(self, "payload_json", canonical_json_object(self.payload_json))
        return self


class MergeDecision(RefactorModel):
    """Auditable identity decision made before graph mutation."""

    decision_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    document_version_id: str = Field(min_length=1)
    incoming_local_entity_id: str = Field(min_length=1)
    incoming_name: str = Field(min_length=1)
    incoming_type: str = Field(min_length=1)
    candidate_entity_id: str | None = None
    candidate_name: str | None = None
    candidate_type: str | None = None
    top1_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    top2_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    score_margin: float | None = None
    type_compatible: bool = False
    decision: Literal["exact_merge", "auto_merge", "llm_merge", "new_entity", "rollback"]
    rationale: str = Field(min_length=1)
    model_id: str | None = None
    prompt_version: str | None = None
    adjudication_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reverses_decision_id: str | None = None
    created_at: AwareDatetime = Field(default_factory=utc_now)
    reversed_at: AwareDatetime | None = None
