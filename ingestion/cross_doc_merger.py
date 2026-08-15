"""Cross-document entity merging for ingestion Phase 5."""

from __future__ import annotations

import json
import hashlib
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from config import Settings
from generation.model_adapter import build_generate_config
from generation.usage import extract_usage_metadata
from ingestion.embedder import GeminiEmbedder
from ingestion.vector_indexer import build_active_version_filter, build_entity_summary
from models.refactor import MergeDecision


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class MergeProviderStats:
    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


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
        self._stats = MergeProviderStats()
        self._stats_lock = threading.Lock()

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
                self._llm_client = genai.Client(
                    api_key=self.settings.gemini_api_key,
                    http_options=types.HttpOptions(
                        timeout=max(
                            1,
                            int(float(self.settings.ingestion_provider_timeout_seconds) * 1000),
                        )
                    ),
                )
                self._llm_schema = types.Schema(
                    type="OBJECT",
                    required=["same_entity", "confidence", "rationale"],
                    properties={
                        "same_entity": types.Schema(type="BOOLEAN"),
                        "confidence": types.Schema(type="NUMBER"),
                        "rationale": types.Schema(type="STRING"),
                    },
                )

    def snapshot_stats(self) -> MergeProviderStats:
        """Return an atomic provider-usage snapshot for stage telemetry."""
        with self._stats_lock:
            return MergeProviderStats(**self._stats.__dict__)

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
        active_version_ids = (
            store.get_queryable_document_version_ids() if store is not None else None
        )
        if store is not None and doc_id and active_version_ids is not None:
            # This stage is explicitly cross-document. The within-document
            # disambiguator has already compared a replacement with its prior
            # active mappings, so searching the same document's old vectors is
            # redundant, costly, and can create circular merge evidence.
            active_doc_version = store.get_active_document_version(doc_id)
            if active_doc_version is not None:
                active_version_ids = [
                    version_id
                    for version_id in active_version_ids
                    if version_id != active_doc_version.version_id
                ]

        eligible_nodes = [
            node
            for node in canonical_nodes
            if str(node.get("id") or "").strip() and str(node.get("name") or "").strip()
        ]
        candidates_by_local_id: dict[str, list[dict[str, Any]]] = {}
        if active_version_ids != [] and eligible_nodes:
            query_vectors = self.embedder.embed_texts(
                [self._build_entity_query_text(node) for node in eligible_nodes],
                task_type=self.settings.query_embedding_task_type,
            )
            if len(query_vectors) != len(eligible_nodes):
                raise RuntimeError("Embedding output length mismatch for cross-document resolution.")
            candidate_sets = self._get_candidates_batch(
                query_vectors=query_vectors,
                active_version_ids=active_version_ids,
            )
            for node, candidates in zip(eligible_nodes, candidate_sets):
                candidates_by_local_id[str(node.get("id") or "")] = candidates

        for node in canonical_nodes:
            local_id = str(node.get("id") or "")
            precomputed_candidates = (
                candidates_by_local_id.get(local_id)
                if local_id in candidates_by_local_id
                else None
            )
            decision = self._resolve_one(
                node,
                active_version_ids=active_version_ids,
                candidates=precomputed_candidates,
            )
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

    def _resolve_one(
        self,
        node: dict[str, Any],
        active_version_ids: list[str] | None = None,
        *,
        candidates: list[dict[str, Any]] | None = None,
    ) -> CrossDocMergeDecision:
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

        if active_version_ids == []:
            return CrossDocMergeDecision(
                local_entity_id=local_id,
                final_entity_id=local_id,
                decision="NEW_NODE",
                rationale="No SQLite-active corpus version exists yet.",
            )

        if candidates is None:
            query_text = self._build_entity_query_text(node)
            query_vector = self.embedder.embed_texts(
                [query_text],
                task_type=self.settings.query_embedding_task_type,
            )[0]
            candidates = self._get_candidates(
                query_vector=query_vector,
                active_version_ids=active_version_ids,
            )
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

        if self._is_document_local_reference(name) or self._is_document_local_reference(
            str(candidate_name or "")
        ):
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
                rationale="Document-local/deictic entity names are never cross-document merged.",
            )

        if self._is_context_dependent_generic(name) or self._is_context_dependent_generic(
            str(candidate_name or "")
        ):
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
                rationale=(
                    "Unqualified context-dependent identity labels are never "
                    "cross-document merged."
                ),
            )

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
            lexical_support = self._names_support_identity(name, str(candidate_name or ""))
            if (
                approved
                and confidence >= float(self.settings.cross_doc_llm_confidence_threshold)
                and lexical_support
            ):
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
            if approved and confidence >= float(self.settings.cross_doc_llm_confidence_threshold):
                rationale = (
                    "Structured adjudication approved the pair, but deterministic lexical "
                    "identity support was absent; strict merge rejected."
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
            rationale=rationale if should_adjudicate and rationale else "Strict automatic/LLM merge requirements were not met.",
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

    def _get_candidates(
        self,
        query_vector: list[float],
        active_version_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if active_version_ids == []:
            return []
        limit = max(1, int(self.settings.cross_doc_candidate_k))
        points = self._search_points(
            query_vector=query_vector,
            limit=limit,
            active_version_ids=active_version_ids,
        )
        return self._authorize_candidates(points, active_version_ids)

    def _get_candidates_batch(
        self,
        *,
        query_vectors: list[list[float]],
        active_version_ids: list[str] | None,
    ) -> list[list[dict[str, Any]]]:
        """Retrieve candidate sets in one Qdrant request when supported."""
        if not query_vectors:
            return []
        if active_version_ids == []:
            return [[] for _ in query_vectors]
        limit = max(1, int(self.settings.cross_doc_candidate_k))
        batches = self._search_points_batch(
            query_vectors=query_vectors,
            limit=limit,
            active_version_ids=active_version_ids,
        )
        if len(batches) != len(query_vectors):
            raise RuntimeError("Qdrant batch candidate response length mismatch.")
        return [
            self._authorize_candidates(points, active_version_ids)
            for points in batches
        ]

    @staticmethod
    def _authorize_candidates(
        points: list[dict[str, Any]],
        active_version_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        if active_version_ids is not None:
            authorized = set(active_version_ids)
            points = [
                point
                for point in points
                if str((point.get("payload") or {}).get("document_version_id") or "")
                in authorized
                and (point.get("payload") or {}).get("active") is True
            ]
        return sorted(points, key=lambda item: float(item.get("score") or 0.0), reverse=True)

    def _search_points_batch(
        self,
        *,
        query_vectors: list[list[float]],
        limit: int,
        active_version_ids: list[str] | None,
    ) -> list[list[dict[str, Any]]]:
        collection = self.settings.qdrant_entities_collection
        active_filter = build_active_version_filter(active_version_ids)
        if hasattr(self._qdrant_client, "query_batch_points"):
            try:
                from qdrant_client.http import models

                requests = [
                    models.QueryRequest(
                        query=query_vector,
                        filter=active_filter,
                        limit=limit,
                        with_payload=True,
                        with_vector=False,
                    )
                    for query_vector in query_vectors
                ]
                responses = self._qdrant_client.query_batch_points(
                    collection_name=collection,
                    requests=requests,
                )
                return [self._extract_points(response) for response in responses]
            except Exception:
                # Compatibility fallback for older clients and lightweight
                # fakes. Each fallback query retains the same active fence.
                pass
        return [
            self._search_points(
                query_vector=query_vector,
                limit=limit,
                active_version_ids=active_version_ids,
            )
            for query_vector in query_vectors
        ]

    def _search_points(
        self,
        query_vector: list[float],
        limit: int,
        active_version_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        collection = self.settings.qdrant_entities_collection
        active_filter = build_active_version_filter(active_version_ids)
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
    def _is_document_local_reference(name: str) -> bool:
        tokens = re.findall(r"[a-z0-9]+", str(name).casefold())
        if not tokens:
            return False
        deictic = {"this", "our", "current", "present", "the"}
        local_heads = {
            "study",
            "paper",
            "work",
            "research",
            "document",
            "article",
            "analysis",
            "experiment",
            "survey",
            "project",
        }
        return tokens[0] in deictic and any(token in local_heads for token in tokens[1:])

    @classmethod
    def _is_context_dependent_generic(cls, name: str) -> bool:
        """Reject unqualified labels whose identity is supplied only by context."""
        tokens = cls._identity_tokens(name)
        local_modifiers = {
            "a",
            "an",
            "the",
            "this",
            "our",
            "current",
            "present",
            "proposed",
            "developed",
            "existing",
            "target",
            "source",
        }
        while tokens and tokens[0] in local_modifiers:
            tokens.pop(0)
        generic_heads = {
            "analysis",
            "application",
            "approach",
            "component",
            "database",
            "dataset",
            "document",
            "experiment",
            "feature",
            "framework",
            "interface",
            "method",
            "model",
            "paper",
            "pipeline",
            "platform",
            "project",
            "research",
            "solution",
            "study",
            "survey",
            "system",
            "tool",
        }
        return len(tokens) == 1 and tokens[0] in generic_heads

    @classmethod
    def _names_support_identity(cls, incoming_name: str, candidate_name: str) -> bool:
        """Require deterministic lexical support before accepting an LLM merge."""
        left = cls._identity_tokens(incoming_name)
        right = cls._identity_tokens(candidate_name)
        if not left or not right:
            return False
        if cls._identity_token_sequences_match(left, right):
            return True
        suffixes = {
            "system",
            "model",
            "architecture",
            "method",
            "technology",
            "tool",
            "application",
            "concept",
        }
        left_base = list(left)
        right_base = list(right)
        while len(left_base) > 1 and left_base[-1] in suffixes:
            left_base.pop()
        while len(right_base) > 1 and right_base[-1] in suffixes:
            right_base.pop()
        return cls._identity_token_sequences_match(left_base, right_base)

    @staticmethod
    def _identity_token_sequences_match(left: list[str], right: list[str]) -> bool:
        if left == right:
            return True
        left_compact = "".join(left)
        right_compact = "".join(right)
        if left_compact == right_compact:
            return True
        if len(left) == 1 and left_compact == "".join(token[0] for token in right):
            return True
        if len(right) == 1 and right_compact == "".join(token[0] for token in left):
            return True
        return False

    @staticmethod
    def _identity_tokens(name: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", str(name).casefold())
        if not tokens:
            return []
        normalized = [token[:-1] if len(token) > 3 and token.endswith("s") else token for token in tokens]
        if len(normalized) > 1 and normalized[-1] == "".join(token[0] for token in normalized[:-1]):
            normalized = normalized[:-1]
        return normalized

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
            with self._stats_lock:
                self._stats.call_count += 1
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
            usage = extract_usage_metadata(response)
            with self._stats_lock:
                self._stats.input_tokens += usage["input_tokens"]
                self._stats.output_tokens += usage["output_tokens"]
                self._stats.total_tokens += usage["total_tokens"]
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
