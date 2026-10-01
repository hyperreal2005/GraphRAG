"""Convert validated extraction output into source-specific assertions."""

from __future__ import annotations

import logging

from typing import Any, Iterable

from config import Settings
from models.extraction import ExtractedEntity
from models.refactor import AssertionRecord


logger = logging.getLogger(__name__)


class AssertionBuilder:
    """Build and deduplicate entity-object and literal-object assertions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build(
        self,
        *,
        doc_id: str,
        document_version_id: str,
        mapped_relations: Iterable[dict[str, Any]],
        extracted_entities: Iterable[ExtractedEntity],
        raw_to_final_entity_id: dict[str, str],
        raw_to_local_entity_id: dict[str, str] | None = None,
        raw_entity_chunk_ids: dict[str, str],
    ) -> list[AssertionRecord]:
        """Build all assertions and keep the strongest duplicate per source claim.

        An assertion the contract rejects is dropped and counted, never raised: one
        impossible value from the model must not discard every other fact in the document
        (REVIEW1 finding `B11`, downstream of the WP3-C extraction guard). `rejected` is
        reported by the caller so a systemic problem still surfaces as a number.
        """
        assertions: dict[str, AssertionRecord] = {}
        local_map = raw_to_local_entity_id or {}
        self.rejected: list[str] = []

        for relation in mapped_relations:
            confidence = float(relation.get("weight", relation.get("confidence", 0.0)))
            assertion = self._safe_create(
                f"relation {relation.get('head_id')} -> {relation.get('tail_id')}",
                AssertionRecord.create,
                subject_entity_id=str(relation["head_id"]),
                subject_local_entity_id=self._optional_text(relation.get("head_local_id")),
                predicate_raw=str(relation.get("raw_relation") or relation.get("relation") or ""),
                predicate_canonical=str(relation.get("relation") or relation.get("raw_relation") or ""),
                object_kind="entity",
                object_entity_id=str(relation["tail_id"]),
                object_local_entity_id=self._optional_text(relation.get("tail_local_id")),
                literal_value=None,
                properties=dict(relation.get("properties") or {}),
                confidence=confidence,
                doc_id=doc_id,
                document_version_id=document_version_id,
                chunk_id=str(relation["source_chunk"]),
                extractor_model=self.settings.ingestion_model,
                extraction_prompt_version=self.settings.extraction_prompt_version,
                valid_from=self._optional_text(relation.get("valid_from")),
                valid_until=self._optional_text(relation.get("valid_until")),
                active=False,
                review_status=self._review_status(confidence),
            )
            if assertion is not None:
                self._keep_strongest(assertions, assertion)

        for entity in extracted_entities:
            final_entity_id = raw_to_final_entity_id.get(entity.entity_id)
            chunk_id = raw_entity_chunk_ids.get(entity.entity_id)
            if not final_entity_id or not chunk_id:
                continue
            for property_name, property_value in sorted(entity.properties.items(), key=lambda item: str(item[0])):
                raw_predicate = str(property_name).strip()
                if not raw_predicate:
                    continue
                assertion = self._safe_create(
                    f"property {raw_predicate} of {final_entity_id}",
                    AssertionRecord.create,
                subject_entity_id=final_entity_id,
                subject_local_entity_id=local_map.get(entity.entity_id),
                    predicate_raw=raw_predicate,
                    predicate_canonical=raw_predicate,
                    object_kind="literal",
                object_entity_id=None,
                object_local_entity_id=None,
                    literal_value=property_value,
                    properties={},
                    confidence=float(entity.confidence),
                    doc_id=doc_id,
                    document_version_id=document_version_id,
                    chunk_id=chunk_id,
                    extractor_model=self.settings.ingestion_model,
                    extraction_prompt_version=self.settings.extraction_prompt_version,
                    active=False,
                    review_status=self._review_status(float(entity.confidence)),
                )
                if assertion is not None:
                    self._keep_strongest(assertions, assertion)

        return sorted(assertions.values(), key=lambda item: (item.chunk_id, item.claim_key, item.id))

    def _safe_create(self, label: str, factory, **kwargs):  # noqa: ANN001, ANN201
        """Build one assertion, recording a contract rejection instead of raising."""
        try:
            return factory(**kwargs)
        except Exception as exc:
            # Pydantic's first line is only "N validation errors for AssertionRecord",
            # which says nothing about what was actually wrong. The reason lives on the
            # following lines, so the message is flattened rather than truncated.
            reason = " ".join(part.strip() for part in str(exc).splitlines() if part.strip())[:200]
            self.rejected.append(f"{label}: {reason}")
            logger.warning("assertion_rejected %s: %s", label, reason)
            return None

    @staticmethod
    def _keep_strongest(target: dict[str, AssertionRecord], assertion: AssertionRecord) -> None:
        existing = target.get(assertion.id)
        if existing is None or assertion.confidence > existing.confidence:
            target[assertion.id] = assertion

    def _review_status(self, confidence: float) -> str:
        if confidence < float(self.settings.assertion_review_confidence_threshold):
            return "flagged"
        return "accepted"

    @staticmethod
    def _optional_text(value: Any) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None
