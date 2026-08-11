"""Dynamic Cypher generation with schema-aware prompting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from config import Settings
from generation.model_adapter import build_generate_config
from models.query import LinkedEntity, QueryAnalysis, QueryType


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SchemaSnapshot:
    entity_types: list[str]
    node_property_keys: list[str]
    relationship_types: list[str]
    relationship_property_keys: list[str]
    generated_at: str

    def to_prompt_dict(self) -> dict[str, object]:
        return {
            "entity_types": self.entity_types,
            "node_property_keys": self.node_property_keys,
            "relationship_types": self.relationship_types,
            "relationship_property_keys": self.relationship_property_keys,
            "generated_at": self.generated_at,
        }


@dataclass
class DynamicCypherPlan:
    """Validated logical graph plan; it never contains executable model output."""

    mode: str
    params: dict[str, object]
    schema_generated_at: str
    relationship_types: list[str]
    temporal_filter_hint: str | None = None


class DynamicCypherPlanner:
    """Generate a constrained logical plan that application code compiles."""

    def __init__(
        self,
        settings: Settings,
        driver: Any | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self._driver = driver
        self._client = client
        self._types = None
        self._schema_cache: SchemaSnapshot | None = None
        self._schema_cache_ts: datetime | None = None

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
        self._client = genai.Client(api_key=self.settings.gemini_api_key)

    def generate_plan(
        self,
        user_query: str,
        analysis: QueryAnalysis,
        anchors: list[LinkedEntity],
    ) -> DynamicCypherPlan:
        """Build one dynamic Cypher plan for graph retrieval."""
        if self._client is None:
            raise RuntimeError("Dynamic Cypher model client is unavailable.")
        if not anchors:
            raise RuntimeError("Dynamic Cypher requires at least one linked anchor.")

        schema = self.get_schema_snapshot()
        mode, params = self._build_mode_and_params(analysis, anchors)
        temporal_hint = analysis.temporal_filter
        prompt = self._build_prompt(
            user_query=user_query,
            analysis=analysis,
            anchors=anchors,
            schema=schema,
            mode=mode,
            params=params,
            temporal_hint=temporal_hint,
        )

        attempts = max(1, 1 + int(self.settings.query_graph_dynamic_max_retries))
        last_error: Exception | None = None
        for _ in range(attempts):
            try:
                raw_payload = self._generate_payload(prompt)
                relationship_types = self._extract_relationship_types(raw_payload, schema)
                return DynamicCypherPlan(
                    mode=mode,
                    params=params,
                    schema_generated_at=schema.generated_at,
                    relationship_types=relationship_types,
                    temporal_filter_hint=temporal_hint,
                )
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise RuntimeError(f"Dynamic Cypher plan generation failed: {last_error}") from last_error
        raise RuntimeError("Dynamic Cypher plan generation failed.")

    def get_schema_snapshot(self) -> SchemaSnapshot:
        """Return schema info with TTL-based cache."""
        if self._driver is None:
            raise RuntimeError("Neo4j driver unavailable for schema introspection.")

        now = datetime.now(timezone.utc)
        ttl_seconds = max(0, int(self.settings.query_graph_schema_cache_ttl_seconds))
        if (
            self._schema_cache is not None
            and self._schema_cache_ts is not None
            and (now - self._schema_cache_ts).total_seconds() <= ttl_seconds
        ):
            return self._schema_cache

        with self._driver.session() as session:
            entity_types = self._safe_list(
                session.run(
                    "MATCH (n:Entity) RETURN collect(DISTINCT coalesce(n.type, 'Entity')) AS values"
                ).single()
            )
            node_keys = self._safe_list(
                session.run(
                    "MATCH (n:Entity) UNWIND keys(n) AS key RETURN collect(DISTINCT key) AS values"
                ).single()
            )
            rel_types = self._safe_list(
                session.run(
                    "MATCH (a:Assertion {active: true}) "
                    "RETURN collect(DISTINCT coalesce(a.predicate_normalized, 'RELATED_TO')) AS values"
                ).single()
            )
            rel_keys = self._safe_list(
                session.run(
                    "MATCH (a:Assertion {active: true}) UNWIND keys(a) AS key "
                    "RETURN collect(DISTINCT key) AS values"
                ).single()
            )

        snapshot = SchemaSnapshot(
            entity_types=sorted(entity_types),
            node_property_keys=sorted(node_keys),
            relationship_types=sorted(rel_types),
            relationship_property_keys=sorted(rel_keys),
            generated_at=utc_now_iso(),
        )
        self._schema_cache = snapshot
        self._schema_cache_ts = now
        return snapshot

    def _build_mode_and_params(
        self,
        analysis: QueryAnalysis,
        anchors: list[LinkedEntity],
    ) -> tuple[str, dict[str, object]]:
        anchor_ids = [str(item.entity_id) for item in anchors if item.entity_id]
        if not anchor_ids:
            raise RuntimeError("No anchor IDs available for dynamic graph planning.")

        if analysis.query_type is QueryType.COMPARATIVE and len(anchor_ids) >= 2:
            return (
                "comparative",
                {
                    "anchor_ids": anchor_ids[:2],
                    "max_hops": int(self.settings.query_graph_local_hops),
                    "path_limit": int(self.settings.query_graph_path_limit),
                },
            )
        if analysis.query_type is QueryType.MULTIHOP and len(anchor_ids) >= 2:
            return (
                "multihop_pair",
                {
                    "anchor_a": anchor_ids[0],
                    "anchor_b": anchor_ids[1],
                    "max_hops": int(self.settings.query_graph_multihop_hops),
                    "path_limit": int(self.settings.query_graph_path_limit),
                },
            )
        hops = (
            int(self.settings.query_graph_multihop_hops)
            if analysis.query_type is QueryType.MULTIHOP
            else int(self.settings.query_graph_local_hops)
        )
        return (
            "neighborhood",
            {
                "anchor_ids": anchor_ids,
                "max_hops": hops,
                "path_limit": int(self.settings.query_graph_path_limit),
            },
        )

    def _build_prompt(
        self,
        user_query: str,
        analysis: QueryAnalysis,
        anchors: list[LinkedEntity],
        schema: SchemaSnapshot,
        mode: str,
        params: dict[str, object],
        temporal_hint: str | None,
    ) -> str:
        anchor_payload = [
            {
                "entity_id": str(item.entity_id),
                "mention": item.mention,
                "status": item.status,
                "similarity": float(item.similarity),
                "entity_name": item.entity_name,
                "entity_type": item.entity_type,
            }
            for item in anchors
            if item.entity_id
        ]
        mode_instructions = {
            "neighborhood": (
                "Select zero or more relationship types useful for a neighborhood around the anchors."
            ),
            "multihop_pair": (
                "Select zero or more relationship types useful for connecting the two anchors."
            ),
            "comparative": (
                "Select zero or more relationship types useful for separate evidence around each anchor."
            ),
        }
        safety_rules = (
            "Return data only. Never output Cypher, database clauses, comments, markdown, or explanations. "
            "Every relationship type must exactly match a type in the supplied schema. "
            "Use an empty list when no safe filter is useful."
        )
        output_rules = (
            'Output one strict JSON object with key "relationship_types" only, for example:\n'
            '{"relationship_types":["WORKS_AT","FOUNDED"]}'
        )

        schema_json = json.dumps(schema.to_prompt_dict(), ensure_ascii=True)
        anchors_json = json.dumps(anchor_payload, ensure_ascii=True)
        params_json = json.dumps(params, ensure_ascii=True)
        return (
            "You select schema-backed relationship filters for a GraphRAG retriever.\n"
            f"User question: {user_query}\n"
            f"Query type: {analysis.query_type.value}\n"
            f"Planning mode: {mode}\n"
            f"Temporal filter hint (do not enforce unless clearly safe): {temporal_hint or ''}\n"
            f"Anchors: {anchors_json}\n"
            f"Schema snapshot: {schema_json}\n"
            f"Required parameters: {params_json}\n"
            f"Mode instructions: {mode_instructions.get(mode, mode_instructions['neighborhood'])}\n"
            f"{safety_rules}\n"
            f"{output_rules}"
        )

    def _generate_payload(self, prompt: str) -> str:
        config = None
        if self._types is not None:
            model_id = str(self.settings.query_graph_dynamic_model or self.settings.gemini_model)
            config = build_generate_config(
                self._types,
                model_id,
                temperature=float(self.settings.query_graph_dynamic_temperature),
                response_mime_type="application/json",
            )
        response = self._client.models.generate_content(
            model=str(self.settings.query_graph_dynamic_model or self.settings.gemini_model),
            contents=prompt,
            config=config,
        )
        text = str(getattr(response, "text", "") or "").strip()
        if text:
            return text
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, dict):
            return json.dumps(parsed)
        raise RuntimeError("Dynamic Cypher model returned empty payload.")

    def _extract_relationship_types(self, raw_payload: str, schema: SchemaSnapshot) -> list[str]:
        payload = self._parse_json(raw_payload)
        if set(payload) != {"relationship_types"}:
            raise RuntimeError("Logical graph plan must contain only 'relationship_types'.")
        values = payload.get("relationship_types")
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise RuntimeError("Logical graph plan 'relationship_types' must be a list of strings.")
        allowed = set(schema.relationship_types)
        normalized = list(dict.fromkeys(item.strip() for item in values if item.strip()))
        unknown = [item for item in normalized if item not in allowed]
        if unknown:
            raise RuntimeError(f"Logical graph plan used relationship types outside the schema: {unknown}")
        return normalized

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
            raise RuntimeError("No JSON object found in dynamic Cypher response.")
        decoded = json.loads(match.group(0))
        if not isinstance(decoded, dict):
            raise RuntimeError("Dynamic Cypher response JSON was not an object.")
        return decoded

    def _safe_list(self, record: Any) -> list[str]:
        if record is None:
            return []
        if isinstance(record, dict):
            value = record.get("values")
        else:
            value = record["values"] if "values" in record else None
        if not isinstance(value, list):
            return []
        return [str(item) for item in value if item is not None and str(item).strip()]
