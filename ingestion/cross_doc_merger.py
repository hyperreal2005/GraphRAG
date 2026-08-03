"""Cross-document entity merging for ingestion Phase 5."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from config import Settings
from ingestion.embedder import GeminiEmbedder
from ingestion.vector_indexer import build_entity_summary


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
                    required=["same_entity"],
                    properties={"same_entity": types.Schema(type="BOOLEAN")},
                )

    def resolve(self, canonical_nodes: list[dict[str, Any]]) -> CrossDocMergeResult:
        """Resolve each local canonical node to final graph identity."""
        result = CrossDocMergeResult()
        for node in canonical_nodes:
            decision = self._resolve_one(node)
            local_id = str(node.get("id") or "")
            if not local_id:
                continue
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

        auto_threshold = float(self.settings.cross_doc_auto_merge_threshold)
        llm_low = float(self.settings.cross_doc_llm_min_threshold)
        best_observed: tuple[float, str, str | None, str] | None = None

        for candidate in candidates:
            score = float(candidate.get("score") or 0.0)
            payload = candidate.get("payload") or {}
            candidate_id = str(payload.get("neo4j_node_id") or candidate.get("id") or "").strip()
            if not candidate_id or candidate_id == local_id:
                continue
            candidate_type = str(payload.get("type") or "Entity").strip() or "Entity"
            candidate_name = str(payload.get("name") or "").strip() or None
            if best_observed is None:
                best_observed = (score, candidate_id, candidate_name, candidate_type)

            types_compatible = self._types_compatible(entity_type=entity_type, candidate_type=candidate_type)
            if score >= auto_threshold and types_compatible:
                return CrossDocMergeDecision(
                    local_entity_id=local_id,
                    final_entity_id=candidate_id,
                    decision="AUTO_MERGE",
                    similarity=score,
                    candidate_entity_id=candidate_id,
                    candidate_name=candidate_name,
                    candidate_type=candidate_type,
                )

            # High-scoring type mismatches are not silently discarded. They
            # enter the conservative LLM review path along with the ambiguity
            # band, while lower-ranked compatible candidates are still checked.
            if score >= llm_low and self._llm_should_merge(
                incoming_name=name,
                incoming_type=entity_type,
                incoming_properties=node.get("properties"),
                candidate_name=str(candidate_name or ""),
                candidate_type=candidate_type,
                candidate_summary=payload.get("summary"),
                similarity=score,
            ):
                return CrossDocMergeDecision(
                    local_entity_id=local_id,
                    final_entity_id=candidate_id,
                    decision="LLM_MERGE",
                    similarity=score,
                    candidate_entity_id=candidate_id,
                    candidate_name=candidate_name,
                    candidate_type=candidate_type,
                )

        score, candidate_id, candidate_name, candidate_type = best_observed or (0.0, "", None, "Entity")
        return CrossDocMergeDecision(
            local_entity_id=local_id,
            final_entity_id=local_id,
            decision="NEW_NODE",
            similarity=score,
            candidate_entity_id=candidate_id,
            candidate_name=candidate_name,
            candidate_type=candidate_type,
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
        if hasattr(self._qdrant_client, "query_points"):
            try:
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
                limit=limit,
                with_payload=True,
            )
            return self._extract_points(response)

        if hasattr(self._qdrant_client, "search_points"):
            response = self._qdrant_client.search_points(
                collection_name=collection,
                query_vector=query_vector,
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
        return aliases.get(incoming, incoming) == aliases.get(candidate, candidate)

    def _llm_should_merge(
        self,
        incoming_name: str,
        incoming_type: str,
        incoming_properties: Any,
        candidate_name: str,
        candidate_type: str,
        candidate_summary: Any,
        similarity: float,
    ) -> bool:
        if not self.settings.cross_doc_llm_enabled:
            return False
        if self._llm_client is None:
            return False

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
            "Return strict JSON: {\"same_entity\": true|false}.\n"
            "Be conservative. If uncertain, return false.\n"
            f"Incoming entity: name={incoming_name}, type={incoming_type}, properties={json.dumps(incoming_props)}\n"
            f"Candidate entity: name={candidate_name}, type={candidate_type}, summary={candidate_summary}\n"
            f"Embedding similarity: {similarity:.4f}"
        )

        try:
            config = None
            if self._types is not None:
                config = self._types.GenerateContentConfig(
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
                return bool(parsed.get("same_entity"))
            if parsed is not None and hasattr(parsed, "model_dump"):
                dumped = parsed.model_dump()
                if isinstance(dumped, dict):
                    return bool(dumped.get("same_entity"))
            text = str(getattr(response, "text", "") or "").strip()
            if not text:
                return False
            return self._parse_same_entity(text)
        except Exception:
            return False

    def _parse_same_entity(self, text: str) -> bool:
        cleaned = text.strip().strip("`")
        try:
            decoded = json.loads(cleaned)
            if isinstance(decoded, dict):
                return bool(decoded.get("same_entity"))
        except json.JSONDecodeError:
            match = JSON_BLOCK_PATTERN.search(cleaned)
            if match:
                try:
                    decoded = json.loads(match.group(0))
                    if isinstance(decoded, dict):
                        return bool(decoded.get("same_entity"))
                except json.JSONDecodeError:
                    return False
        return False
