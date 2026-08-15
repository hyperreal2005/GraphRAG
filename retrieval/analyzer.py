"""Query analysis and routing decisions."""

from __future__ import annotations

import json
import re
from typing import Any

from config import Settings
from generation.model_adapter import build_generate_config
from generation.usage import extract_usage_metadata
from models.query import QueryAnalysis, QueryType


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

GLOBAL_HINTS = (
    "overall",
    "main themes",
    "main topics",
    "across the corpus",
    "in this corpus",
    "what does this corpus cover",
    "high level summary",
    "big picture",
)
COMPARATIVE_HINTS = ("compare", "difference between", "versus", " vs ")
MULTIHOP_HINTS = (
    "how is",
    "how are",
    "connected to",
    "relationship between",
    "path between",
    "lead to",
    "caused by",
)


class QueryAnalyzer:
    """Classify query type and retrieval flags with Gemini + heuristic fallback."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client
        self._types = None
        self._schema = None
        self.last_usage: dict[str, int] = {}

        if self._client is not None:
            return
        if not self.settings.gemini_api_key:
            return
        try:
            from google import genai
            from google.genai import types
        except Exception:
            return
        self._types = types
        self._client = genai.Client(
            api_key=self.settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=max(
                    1,
                    int(float(self.settings.query_analysis_timeout_seconds) * 1000),
                )
            ),
        )
        self._schema = self._build_schema(types)

    def analyze(self, user_query: str) -> QueryAnalysis:
        """Return structured query analysis for retrieval orchestration."""
        clean_query = user_query.strip()
        if not clean_query:
            return self._heuristic_analysis(clean_query)

        if self._client is None:
            return self._heuristic_analysis(clean_query)

        try:
            payload = self._analyze_with_model(clean_query)
            return self._analysis_from_payload(payload, fallback_query=clean_query)
        except Exception:
            return self._heuristic_analysis(clean_query)

    def _build_schema(self, types: Any) -> Any:
        return types.Schema(
            type="OBJECT",
            required=[
                "query_type",
                "mentioned_entities",
                "retrieval_flags",
                "sub_queries",
            ],
            properties={
                "query_type": types.Schema(
                    type="STRING",
                    enum=[item.value for item in QueryType],
                ),
                "mentioned_entities": types.Schema(
                    type="ARRAY",
                    items=types.Schema(type="STRING"),
                ),
                "retrieval_flags": types.Schema(
                    type="OBJECT",
                    properties={
                        "graph": types.Schema(type="BOOLEAN"),
                        "vector": types.Schema(type="BOOLEAN"),
                        "community": types.Schema(type="BOOLEAN"),
                    },
                ),
                "sub_queries": types.Schema(
                    type="ARRAY",
                    items=types.Schema(type="STRING"),
                ),
                "temporal_filter": types.Schema(type="STRING"),
            },
        )

    def _analyze_with_model(self, user_query: str) -> dict[str, Any]:
        prompt = (
            "Classify this query for a GraphRAG retrieval system.\n"
            "Return strict JSON with keys:\n"
            "- query_type: one of local/multihop/comparative/global/hybrid\n"
            "- mentioned_entities: list[str]\n"
            "- retrieval_flags: {graph: bool, vector: bool, community: bool}\n"
            "- sub_queries: list[str]\n"
            "- temporal_filter: optional string or empty\n"
            "Rules:\n"
            "- local: factual single-entity or direct lookup\n"
            "- multihop: requires reasoning across multiple relationships\n"
            "- comparative: compares two or more entities/options\n"
            "- global: asks for themes/summary across corpus\n"
            "- hybrid: needs global synthesis and specific entities together\n"
            f"QUERY: {user_query}"
        )
        config = None
        if self._types is not None:
            config = build_generate_config(
                self._types,
                self.settings.gemini_model,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=self._schema,
            )
        response = self._client.models.generate_content(
            model=self.settings.gemini_model,
            contents=prompt,
            config=config,
        )
        self.last_usage = extract_usage_metadata(response)
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        if parsed is not None and hasattr(parsed, "model_dump"):
            dumped = parsed.model_dump()
            if isinstance(dumped, dict):
                return dumped
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return self._parse_json(text)
        raise RuntimeError("Analyzer model returned no parseable payload.")

    def _analysis_from_payload(self, payload: dict[str, Any], fallback_query: str) -> QueryAnalysis:
        query_type = QueryType.from_value(payload.get("query_type"))
        mentioned_entities = self._normalize_mentions(payload.get("mentioned_entities"))
        if not mentioned_entities:
            mentioned_entities = self._extract_mentions_heuristic(fallback_query)
        flags = payload.get("retrieval_flags")
        retrieval_flags = self._normalize_flags(flags, query_type, mentioned_entities)
        temporal_filter = str(payload.get("temporal_filter") or "").strip() or self._extract_temporal_filter(
            fallback_query
        )
        sub_queries = [
            str(item).strip()
            for item in payload.get("sub_queries", [])
            if isinstance(item, str) and item.strip()
        ]
        if not sub_queries:
            sub_queries = self._derive_sub_queries(fallback_query, query_type, mentioned_entities)
        return QueryAnalysis(
            query_type=query_type,
            mentioned_entities=mentioned_entities,
            retrieval_flags=retrieval_flags,
            sub_queries=sub_queries,
            temporal_filter=temporal_filter,
        )

    def _heuristic_analysis(self, query: str) -> QueryAnalysis:
        query_type = self._detect_query_type(query)
        mentions = self._extract_mentions_heuristic(query)
        flags = self._default_retrieval_flags(query_type, mentions)
        return QueryAnalysis(
            query_type=query_type,
            mentioned_entities=mentions,
            retrieval_flags=flags,
            sub_queries=self._derive_sub_queries(query, query_type, mentions),
            temporal_filter=self._extract_temporal_filter(query),
        )

    def _detect_query_type(self, query: str) -> QueryType:
        lowered = query.lower()
        has_global = any(hint in lowered for hint in GLOBAL_HINTS)
        has_comparative = any(hint in lowered for hint in COMPARATIVE_HINTS)
        has_multihop = any(hint in lowered for hint in MULTIHOP_HINTS)
        mentions = self._extract_mentions_heuristic(query)

        if has_global and mentions:
            return QueryType.HYBRID
        if has_global:
            return QueryType.GLOBAL
        if has_comparative:
            return QueryType.COMPARATIVE
        if has_multihop:
            return QueryType.MULTIHOP
        return QueryType.LOCAL

    def _normalize_mentions(self, payload: Any) -> list[str]:
        if not isinstance(payload, list):
            return []
        seen: set[str] = set()
        mentions: list[str] = []
        for item in payload:
            if not isinstance(item, str):
                continue
            value = item.strip()
            key = value.lower()
            if not value or key in seen:
                continue
            seen.add(key)
            mentions.append(value)
        return mentions

    def _extract_mentions_heuristic(self, query: str) -> list[str]:
        candidates: list[str] = []
        title_case_matches = re.findall(r"\b(?:[A-Z][\w-]*)(?:\s+[A-Z][\w-]*)+\b", query)
        candidates.extend(title_case_matches)
        acronym_matches = re.findall(r"\b[A-Z]{2,}\b", query)
        candidates.extend(acronym_matches)
        quoted_matches = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
        for left, right in quoted_matches:
            value = left or right
            if value:
                candidates.append(value)

        seen: set[str] = set()
        mentions: list[str] = []
        for candidate in candidates:
            value = candidate.strip()
            key = value.lower()
            if len(value) < 2 or key in seen:
                continue
            seen.add(key)
            mentions.append(value)
        return mentions[:8]

    def _normalize_flags(self, payload: Any, query_type: QueryType, mentions: list[str]) -> dict[str, bool]:
        if not isinstance(payload, dict):
            return self._default_retrieval_flags(query_type, mentions)
        defaults = self._default_retrieval_flags(query_type, mentions)
        return {
            "graph": bool(payload.get("graph", defaults["graph"])),
            "vector": bool(payload.get("vector", defaults["vector"])),
            "community": bool(payload.get("community", defaults["community"])),
        }

    def _default_retrieval_flags(self, query_type: QueryType, mentions: list[str]) -> dict[str, bool]:
        if query_type is QueryType.GLOBAL:
            return {"graph": False, "vector": True, "community": True}
        if query_type is QueryType.HYBRID:
            return {"graph": bool(mentions), "vector": True, "community": True}
        if query_type is QueryType.COMPARATIVE:
            return {"graph": bool(mentions), "vector": True, "community": False}
        if query_type is QueryType.MULTIHOP:
            return {"graph": True, "vector": True, "community": False}
        return {"graph": bool(mentions), "vector": True, "community": False}

    def _derive_sub_queries(self, query: str, query_type: QueryType, mentions: list[str]) -> list[str]:
        if query_type is not QueryType.COMPARATIVE:
            return []
        parts = re.split(r"\b(?:vs|versus|and)\b", query, flags=re.IGNORECASE)
        normalized = [part.strip(" ?.") for part in parts if part.strip(" ?.")]
        if len(normalized) >= 2:
            return normalized[:4]
        return mentions[:2]

    def _extract_temporal_filter(self, query: str) -> str | None:
        if not query:
            return None
        range_match = re.search(r"\b((?:19|20)\d{2})\s*(?:-|to|through|until)\s*((?:19|20)\d{2})\b", query)
        if range_match:
            return f"{range_match.group(1)}-{range_match.group(2)}"
        year_match = re.search(r"\b((?:19|20)\d{2})\b", query)
        if year_match:
            return year_match.group(1)
        return None

    def _parse_json(self, raw_text: str) -> dict[str, Any]:
        stripped = raw_text.strip().strip("`")
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            pass

        match = JSON_BLOCK_PATTERN.search(stripped)
        if not match:
            raise RuntimeError("No JSON object found in analyzer response.")
        decoded = json.loads(match.group(0))
        if not isinstance(decoded, dict):
            raise RuntimeError("Analyzer response JSON was not an object.")
        return decoded
