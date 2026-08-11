"""Cross-document entity merging for ingestion Phase 5."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from config import Settings
from generation.model_adapter import build_generate_config
from ingestion.embedder import GeminiEmbedder
from ingestion.vector_indexer import build_entity_summary
from models.refactor import MergeDecision


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class CrossDocMergeDecision:
    local_entity_id: str
    final_entity_id: str
    decision: str  # AUTO_MERGE | LLM_MERGE | NEW_NODE
    similarity: float = 0.0
    candidate_entity_id: str | None = None
    candidate_name: str | None = None
    candidate_type: str | None = None
    top2_similarity: float | None = None
    score_margin: float | None = None
    type_compatible: bool = False
    adjudication_confidence: float | None = None
    rationale: str = ""


@dataclass
class CrossDocMergeResult:
    local_to_final_id: dict[str, str] = field(default_factory=dict)
    decisions: list[CrossDocMergeDecision] = field(default_factory=list)

    @property
    def auto_merged(self) -> int:
        return sum(1 for item in self.decisions if item.decision == "AUTO_MERGE")

    @property
    def llm_merged(self) -> int:
        return sum(1 for item in self.decisions if item.decision == "LLM_MERGE")

    @property
    def new_nodes(self) -> int:
        return sum(1 for item in self.decisions if item.decision == "NEW_NODE")

    @property
    def merged_total(self) -> int:
        return self.auto_merged + self.llm_merged


class CrossDocumentMerger:
    """Resolve document-local canonical entities to corpus-level canonical IDs."""

    def __init__(
        self,
        settings: Settings,
        embedder: GeminiEmbedder | None = None,
        qdrant_client: Any | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or GeminiEmbedder(settings)
        self._qdrant_client = qdrant_client
        self._llm_client = llm_client
        self._types = None
        self._llm_schema = None

        if self._qdrant_client is None:
            try:
                from qdrant_client import QdrantClient
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("qdrant-client package is required for cross-document merging.") from exc
            self._qdrant_client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
                timeout=self.settings.qdrant_timeout_seconds,
            )

        if self._llm_client is None and self.settings.cross_doc_llm_enabled and self.settings.gemini_api_key:
            try:
                from google import genai
                from google.genai import types
            except Exception:
                self._llm_client = None
            else:
                self._types = types
                self._llm_client = genai.Client(api_key=self.settings.gemini_api_key)
                self._llm_schema = types.Schema(
                    type="OBJECT",
                    required=["same_entity", "confidence", "rationale"],
                    properties={
                        "same_entity": types.Schema(type="BOOLEAN"),
                        "confidence": types.Schema(type="NUMBER"),
                        "rationale": types.Schema(type="STRING"),
                    },
                )

    def resolve(
        self,
        canonical_nodes: list[dict[str, Any]],
        *,
        run_id: str | None = None,
        doc_id: str | None = None,
        document_version_id: str | None = None,
        store: Any | None = None,
    ) -> CrossDocMergeResult:
        """Resolve identities and persist each decision before graph mutation."""
        result = CrossDocMergeResult()
        for node in canonical_nodes:
            decision = self._resolve_one(node)
            local_id = str(node.get("id") or "")
            if not local_id:
                continue
            if store is not None:
                if not run_id or not doc_id or not document_version_id:
                    raise ValueError("Audited merge resolution requires run, document, and version IDs.")
                self._record_decision(
                    store=store,
                    run_id=run_id,
                    doc_id=doc_id,
                    document_version_id=document_version_id,
                    node=node,
                    decision=decision,
                )
            result.local_to_final_id[local_id] = decision.final_entity_id
            result.decisions.append(decision)
        return result

    def _resolve_one(self, node: dict[str, Any]) -> CrossDocMergeDecision:
        local_id = str(node.get("id") or "")
        name = str(node.get("name") or "").strip()
        entity_type = str(node.get("type") or "Entity").strip() or "Entity"
        if not local_id:
            return CrossDocMergeDecision(local_entity_id="", final_entity_id="", decision="NEW_NODE")

        if not name:
            return CrossDocMergeDecision(
                local_entity_id=local_id,
                final_entity_id=local_id,
                decision="NEW_NODE",
            )

        query_text = self._build_entity_query_text(node)
        query_vector = self.embedder.embed_texts(
            [query_text],
            task_type=self.settings.query_embedding_task_type,
        )[0]
        candidates = self._get_candidates(query_vector=query_vector)
        if not candidates:
            return CrossDocMergeDecision(
                local_entity_id=local_id,
                final_entity_id=local_id,
                decision="NEW_NODE",
            )

        usable = []
        for candidate in candidates:
            payload = candidate.get("payload") or {}
            candidate_id = str(payload.get("neo4j_node_id") or candidate.get("id") or "").strip()
            if not candidate_id or candidate_id == local_id:
                continue
            usable.append(candidate)
        if not usable:
            return CrossDocMergeDecision(
                local_entity_id=local_id,
                final_entity_id=local_id,
                decision="NEW_NODE",
                rationale="No non-self active candidate was available.",
            )

        best = usable[0]
        payload = best.get("payload") or {}
        score = float(best.get("score") or 0.0)
        top2 = float(usable[1].get("score") or 0.0) if len(usable) > 1 else 0.0
        margin = score - top2
        candidate_id = str(payload.get("neo4j_node_id") or best.get("id") or "").strip()
        candidate_type = str(payload.get("type") or "Entity").strip() or "Entity"
        candidate_name = str(payload.get("name") or "").strip() or None
        compatible = self._types_compatible(entity_type=entity_type, candidate_type=candidate_type)

        exact_name = self._normalized_names(node) & self._normalized_names(payload)
        if exact_name and compatible:
            return CrossDocMergeDecision(
                local_entity_id=local_id,
                final_entity_id=candidate_id,
                decision="AUTO_MERGE",
                similarity=score,
                candidate_entity_id=candidate_id,
                candidate_name=candidate_name,
                candidate_type=candidate_type,
                top2_similarity=top2,
                score_margin=margin,
                type_compatible=True,
                rationale="Exact normalized canonical/alias match with compatible types.",
            )

        if not compatible:
            return CrossDocMergeDecision(
                local_entity_id=local_id,
                final_entity_id=local_id,
                decision="NEW_NODE",
                similarity=score,
                candidate_entity_id=candidate_id,
                candidate_name=candidate_name,
                candidate_type=candidate_type,
                top2_similarity=top2,
                score_margin=margin,
                type_compatible=False,
                rationale="Strong type conflict; merge rejected.",
            )

        auto_threshold = float(self.settings.cross_doc_auto_merge_threshold)
        required_margin = float(self.settings.cross_doc_auto_merge_margin)
        if score >= auto_threshold and margin >= required_margin and not self._weak_type(entity_type, candidate_type):
            return CrossDocMergeDecision(
                local_entity_id=local_id,
                final_entity_id=candidate_id,
                decision="AUTO_MERGE",
                similarity=score,
                candidate_entity_id=candidate_id,
                candidate_name=candidate_name,
                candidate_type=candidate_type,
                top2_similarity=top2,
                score_margin=margin,
                type_compatible=True,
                rationale="Similarity, compatible type, and candidate margin passed strict thresholds.",
            )

        llm_low = float(self.settings.cross_doc_llm_min_threshold)
        llm_high = float(self.settings.cross_doc_llm_max_threshold)
        should_adjudicate = score >= llm_low and (
            score <= llm_high or margin < required_margin or self._weak_type(entity_type, candidate_type)
        )
        if should_adjudicate:
            approved, confidence, rationale = self._llm_should_merge(
                incoming_name=name,
                incoming_type=entity_type,
                incoming_properties=node.get("properties"),
                candidate_name=str(candidate_name or ""),
                candidate_type=candidate_type,
                candidate_summary=payload.get("summary"),
                similarity=score,
            )
            if approved and confidence >= float(self.settings.cross_doc_llm_confidence_threshold):
                return CrossDocMergeDecision(
                    local_entity_id=local_id,
                    final_entity_id=candidate_id,
                    decision="LLM_MERGE",
                    similarity=score,
                    candidate_entity_id=candidate_id,
                    candidate_name=candidate_name,
                    candidate_type=candidate_type,
                    top2_similarity=top2,
                    score_margin=margin,
                    type_compatible=True,
                    adjudication_confidence=confidence,
                    rationale=rationale or "Structured adjudication approved the merge.",
                )

        return CrossDocMergeDecision(
            local_entity_id=local_id,
            final_entity_id=local_id,
            decision="NEW_NODE",
            similarity=score,
            candidate_entity_id=candidate_id,
            candidate_name=candidate_name,
            candidate_type=candidate_type,
            top2_similarity=top2,
            score_margin=margin,
            type_compatible=compatible,
            rationale="Strict automatic/LLM merge requirements were not met.",
        )

    def _build_entity_query_text(self, node: dict[str, Any]) -> str:
        name = str(node.get("name") or "")
        entity_type = str(node.get("type") or "Entity")
        summary = build_entity_summary(
            name=name,
            entity_type=entity_type,
            properties=node.get("properties"),
        )
        return f"{name}\nType: {entity_type}\nSummary: {summary}"

    def _get_candidates(self, query_vector: list[float]) -> list[dict[str, Any]]:
        limit = max(1, int(self.settings.cross_doc_candidate_k))
        points = self._search_points(query_vector=query_vector, limit=limit)
        return sorted(points, key=lambda item: float(item.get("score") or 0.0), reverse=True)

    def _search_points(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        collection = self.settings.qdrant_entities_collection
        active_filter: Any = {
            "must": [{"key": "active", "match": {"value": True}}]
        }
        try:
            from qdrant_client.http import models as qdrant_models
        except Exception:
            pass
        else:
            active_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="active",
                        match=qdrant_models.MatchValue(value=True),
                    )
                ]
            )
        if hasattr(self._qdrant_client, "query_points"):
            try:
                try:
                    response = self._qdrant_client.query_points(
                        collection_name=collection,
                        query=query_vector,
                        query_filter=active_filter,
                        limit=limit,
                        with_payload=True,
                        with_vectors=False,
                    )
                except TypeError:
                    response = self._qdrant_client.query_points(
                        collection_name=collection,
                        query=query_vector,
                        limit=limit,
                        with_payload=True,
                        with_vectors=False,
                    )
                # Successful query_points call may validly return zero points
                # (e.g., first entity in an empty collection). Treat that as
                # "no candidate" instead of falling through to API-shape errors.
                return self._extract_points(response)
            except Exception:
                pass

        if hasattr(self._qdrant_client, "search"):
            response = self._qdrant_client.search(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=active_filter,
                limit=limit,
                with_payload=True,
            )
            return self._extract_points(response)

        if hasattr(self._qdrant_client, "search_points"):
            response = self._qdrant_client.search_points(
                collection_name=collection,
                query_vector=query_vector,
                query_filter=active_filter,
                limit=limit,
                with_payload=True,
            )
            return self._extract_points(response)

        raise RuntimeError("Qdrant client does not expose a compatible search API.")

    def _extract_points(self, response: Any) -> list[dict[str, Any]]:
        if isinstance(response, list):
            return [self._point_to_dict(item) for item in response]
        points = getattr(response, "points", None)
        if isinstance(points, list):
            return [self._point_to_dict(item) for item in points]
        result = getattr(response, "result", None)
        if isinstance(result, list):
            return [self._point_to_dict(item) for item in result]
        if result is not None and hasattr(result, "points"):
            nested = getattr(result, "points", None)
            if isinstance(nested, list):
                return [self._point_to_dict(item) for item in nested]
        return []

    def _point_to_dict(self, point: Any) -> dict[str, Any]:
        if isinstance(point, dict):
            payload = point.get("payload")
            return {
                "id": point.get("id"),
                "score": float(point.get("score") or 0.0),
                "payload": payload if isinstance(payload, dict) else {},
            }
        payload = getattr(point, "payload", None)
        return {
            "id": getattr(point, "id", None),
            "score": float(getattr(point, "score", 0.0) or 0.0),
            "payload": payload if isinstance(payload, dict) else {},
        }

    def _types_compatible(self, entity_type: str, candidate_type: str) -> bool:
        if not self.settings.cross_doc_same_type_required_for_auto:
            return True
        aliases = {
            "org": "organization",
            "organisation": "organization",
            "company": "organization",
            "business": "organization",
            "institution": "organization",
            "university": "organization",
            "human": "person",
        }
        incoming = entity_type.strip().lower()
        candidate = candidate_type.strip().lower()
        if incoming == "entity" or candidate == "entity":
            return True
        return aliases.get(incoming, incoming) == aliases.get(candidate, candidate)

    @staticmethod
    def _weak_type(entity_type: str, candidate_type: str) -> bool:
        return entity_type.strip().lower() == "entity" or candidate_type.strip().lower() == "entity"

    @staticmethod
    def _normalized_names(value: dict[str, Any]) -> set[str]:
        names = [value.get("name")]
        aliases = value.get("aliases")
        if isinstance(aliases, list):
            names.extend(aliases)
        return {
            re.sub(r"[^a-z0-9]+", "", str(item).strip().lower())
            for item in names
            if str(item or "").strip()
        }

    def _llm_should_merge(
        self,
        incoming_name: str,
        incoming_type: str,
        incoming_properties: Any,
        candidate_name: str,
        candidate_type: str,
        candidate_summary: Any,
        similarity: float,
    ) -> tuple[bool, float, str]:
        if not self.settings.cross_doc_llm_enabled:
            return False, 0.0, "LLM adjudication is disabled."
        if self._llm_client is None:
            return False, 0.0, "LLM adjudicator is unavailable."

        incoming_props = incoming_properties
        if isinstance(incoming_props, str):
            try:
                incoming_props = json.loads(incoming_props)
            except json.JSONDecodeError:
                incoming_props = {}
        if not isinstance(incoming_props, dict):
            incoming_props = {}

        prompt = (
            "You decide if two entities refer to the same real-world entity.\n"
            "Return strict JSON: {\"same_entity\": true|false, \"confidence\": 0.0-1.0, "
            "\"rationale\": \"brief reason\"}.\n"
            "Be conservative. If uncertain, return false.\n"
            f"Incoming entity: name={incoming_name}, type={incoming_type}, properties={json.dumps(incoming_props)}\n"
            f"Candidate entity: name={candidate_name}, type={candidate_type}, summary={candidate_summary}\n"
            f"Embedding similarity: {similarity:.4f}"
        )

        try:
            config = None
            if self._types is not None:
                config = build_generate_config(
                    self._types,
                    self.settings.gemini_model,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=self._llm_schema,
                )
            response = self._llm_client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config=config,
            )
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, dict):
                return self._parse_adjudication(parsed)
            if parsed is not None and hasattr(parsed, "model_dump"):
                dumped = parsed.model_dump()
                if isinstance(dumped, dict):
                    return self._parse_adjudication(dumped)
            text = str(getattr(response, "text", "") or "").strip()
            if not text:
                return False, 0.0, "Adjudicator returned an empty response."
            return self._parse_adjudication_text(text)
        except Exception:
            return False, 0.0, "Adjudication request or validation failed."

    def _parse_adjudication_text(self, text: str) -> tuple[bool, float, str]:
        cleaned = text.strip().strip("`")
        try:
            decoded = json.loads(cleaned)
            if isinstance(decoded, dict):
                return self._parse_adjudication(decoded)
        except json.JSONDecodeError:
            match = JSON_BLOCK_PATTERN.search(cleaned)
            if match:
                try:
                    decoded = json.loads(match.group(0))
                    if isinstance(decoded, dict):
                        return self._parse_adjudication(decoded)
                except json.JSONDecodeError:
                    return False, 0.0, "Adjudication JSON was malformed."
        return False, 0.0, "Adjudication JSON was malformed."

    @staticmethod
    def _parse_adjudication(payload: dict[str, Any]) -> tuple[bool, float, str]:
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return (
            bool(payload.get("same_entity")),
            confidence,
            str(payload.get("rationale") or "").strip(),
        )

    def _record_decision(
        self,
        *,
        store: Any,
        run_id: str,
        doc_id: str,
        document_version_id: str,
        node: dict[str, Any],
        decision: CrossDocMergeDecision,
    ) -> None:
        audit_decision = {
            "AUTO_MERGE": "exact_merge" if decision.rationale.startswith("Exact normalized") else "auto_merge",
            "LLM_MERGE": "llm_merge",
            "NEW_NODE": "new_entity",
        }[decision.decision]
        identity = hashlib.sha256(
            (
                f"{document_version_id}|{decision.local_entity_id}|"
                f"{decision.candidate_entity_id or ''}|{audit_decision}"
            ).encode("utf-8")
        ).hexdigest()
        store.record_merge_decision(
            MergeDecision(
                decision_id=identity,
                run_id=run_id,
                doc_id=doc_id,
                document_version_id=document_version_id,
                incoming_local_entity_id=decision.local_entity_id,
                incoming_name=str(node.get("name") or "Entity"),
                incoming_type=str(node.get("type") or "Entity"),
                candidate_entity_id=decision.candidate_entity_id,
                candidate_name=decision.candidate_name,
                candidate_type=decision.candidate_type,
                top1_similarity=decision.similarity,
                top2_similarity=decision.top2_similarity,
                score_margin=decision.score_margin,
                type_compatible=decision.type_compatible,
                decision=audit_decision,
                rationale=decision.rationale or "No rationale supplied.",
                model_id=self.settings.gemini_model if decision.decision == "LLM_MERGE" else None,
                prompt_version="merge-r1-v1" if decision.decision == "LLM_MERGE" else None,
                adjudication_confidence=decision.adjudication_confidence,
            )
        )
