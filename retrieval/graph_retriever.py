"""Graph retrieval using dynamic Cypher with strict read-only safety."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from config import Settings
from models.query import LinkedEntity, QueryAnalysis, QueryType, RetrieverResult
from retrieval.dynamic_cypher import DynamicCypherPlan, DynamicCypherPlanner


WRITE_CLAUSE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|LOAD|FOREACH|APOC|DBMS)\b",
    re.IGNORECASE,
)
READ_START_PATTERN = re.compile(r"^\s*(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN)\b", re.IGNORECASE)
STRING_LITERAL_PATTERN = re.compile(r"'(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\"|`(?:\\.|[^`])*`")


class GraphRetriever:
    """Execute safe graph reads over Neo4j for linked entities."""

    def __init__(
        self,
        settings: Settings,
        driver: Any | None = None,
        planner: DynamicCypherPlanner | None = None,
    ) -> None:
        self.settings = settings
        self._driver = driver
        if self._driver is None:
            try:
                from neo4j import GraphDatabase
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("neo4j driver is required for graph retrieval.") from exc
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )
        self._planner = planner or DynamicCypherPlanner(settings=self.settings, driver=self._driver)

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def retrieve(
        self,
        analysis: QueryAnalysis,
        linked_entities: list[LinkedEntity],
        user_query: str = "",
    ) -> list[RetrieverResult]:
        """Retrieve graph paths from linked anchors with dynamic + static fallback."""
        anchors = self._select_graph_anchors(linked_entities)
        if not anchors:
            return []

        use_dynamic = bool(self.settings.query_graph_dynamic_enabled) and analysis.query_type is not QueryType.GLOBAL
        if use_dynamic and user_query.strip():
            try:
                return self._retrieve_dynamic(user_query=user_query, analysis=analysis, anchors=anchors)
            except Exception as exc:
                if not bool(self.settings.query_graph_dynamic_fallback_to_static):
                    return []
                reason = f"{type(exc).__name__}: {exc}"
                return self._retrieve_static(analysis=analysis, anchors=anchors, query_mode="static_fallback", fallback_reason=reason)

        return self._retrieve_static(analysis=analysis, anchors=anchors, query_mode="static")

    def _retrieve_dynamic(
        self,
        user_query: str,
        analysis: QueryAnalysis,
        anchors: list[LinkedEntity],
    ) -> list[RetrieverResult]:
        plan = self._planner.generate_plan(user_query=user_query, analysis=analysis, anchors=anchors)
        self._assert_read_only(plan.cypher)
        rows = self._execute_read(plan.cypher, plan.params)
        return self._rows_to_results(
            rows=rows,
            analysis=analysis,
            anchors=anchors,
            query_mode="dynamic",
            fallback_reason=None,
            plan=plan,
        )

    def _retrieve_static(
        self,
        analysis: QueryAnalysis,
        anchors: list[LinkedEntity],
        query_mode: str,
        fallback_reason: str | None = None,
    ) -> list[RetrieverResult]:
        if analysis.query_type is QueryType.COMPARATIVE and len(anchors) >= 2:
            return self._retrieve_comparative(
                linked_entities=anchors[:2],
                analysis=analysis,
                query_mode=query_mode,
                fallback_reason=fallback_reason,
            )
        max_hops = (
            int(self.settings.query_graph_multihop_hops)
            if analysis.query_type is QueryType.MULTIHOP
            else int(self.settings.query_graph_local_hops)
        )
        return self._retrieve_neighborhood(
            linked_entities=anchors,
            max_hops=max_hops,
            query_type=analysis.query_type,
            query_mode=query_mode,
            fallback_reason=fallback_reason,
        )

    def _select_graph_anchors(self, linked_entities: list[LinkedEntity]) -> list[LinkedEntity]:
        resolved = [item for item in linked_entities if item.status == "resolved" and item.entity_id]
        if resolved:
            return resolved
        if not bool(self.settings.query_graph_allow_ambiguous_fallback):
            return []
        min_similarity = float(self.settings.query_graph_ambiguous_min_similarity)
        ambiguous = [
            item
            for item in linked_entities
            if item.status == "ambiguous" and item.entity_id and float(item.similarity) >= min_similarity
        ]
        return ambiguous

    def _retrieve_neighborhood(
        self,
        linked_entities: list[LinkedEntity],
        max_hops: int,
        query_type: QueryType,
        query_mode: str,
        fallback_reason: str | None,
    ) -> list[RetrieverResult]:
        query = self._build_neighbor_query(max_hops=max_hops)
        all_results: list[RetrieverResult] = []
        for linked in linked_entities:
            params = {
                "entity_id": str(linked.entity_id),
                "path_limit": int(self.settings.query_graph_path_limit),
            }
            rows = self._execute_read(query, params)
            all_results.extend(
                self._rows_to_results(
                    rows=rows,
                    analysis=QueryAnalysis(query_type=query_type),
                    anchors=[linked],
                    query_mode=query_mode,
                    fallback_reason=fallback_reason,
                )
            )
        return all_results

    def _retrieve_comparative(
        self,
        linked_entities: list[LinkedEntity],
        analysis: QueryAnalysis,
        query_mode: str,
        fallback_reason: str | None,
    ) -> list[RetrieverResult]:
        query = self._build_comparative_query(max_hops=int(self.settings.query_graph_multihop_hops))
        params = {
            "entity_a": str(linked_entities[0].entity_id),
            "entity_b": str(linked_entities[1].entity_id),
            "path_limit": int(self.settings.query_graph_path_limit),
        }
        rows = self._execute_read(query, params)
        return self._rows_to_results(
            rows=rows,
            analysis=analysis,
            anchors=linked_entities[:2],
            query_mode=query_mode,
            fallback_reason=fallback_reason,
        )

    def _build_neighbor_query(self, max_hops: int) -> str:
        hops = max(1, int(max_hops))
        return (
            "MATCH p=(start:Entity {id: $entity_id})-[*1.."
            f"{hops}"
            "]-(nbr:Entity) "
            "RETURN "
            "$entity_id AS anchor_entity_id, "
            "[n IN nodes(p) | coalesce(n.name, n.id)] AS node_names, "
            "[rel IN relationships(p) | coalesce(rel.type, type(rel))] AS rel_types, "
            "[rel IN relationships(p) | coalesce(rel.weight, 0.0)] AS rel_weights "
            "LIMIT $path_limit"
        )

    def _build_comparative_query(self, max_hops: int) -> str:
        hops = max(1, int(max_hops))
        return (
            "MATCH p=shortestPath((a:Entity {id: $entity_a})-[*1.."
            f"{hops}"
            "]-(b:Entity {id: $entity_b})) "
            "RETURN "
            "$entity_a + '|' + $entity_b AS anchor_entity_id, "
            "[n IN nodes(p) | coalesce(n.name, n.id)] AS node_names, "
            "[rel IN relationships(p) | coalesce(rel.type, type(rel))] AS rel_types, "
            "[rel IN relationships(p) | coalesce(rel.weight, 0.0)] AS rel_weights "
            "LIMIT $path_limit"
        )

    def _execute_read(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self._assert_read_only(query)
        with self._driver.session() as session:
            records = session.run(query, params)
            return [dict(record) for record in records]

    def _assert_read_only(self, query: str) -> None:
        normalized = query.strip()
        if not normalized:
            raise ValueError("Graph retriever received an empty Cypher query.")

        if normalized.endswith(";"):
            normalized = normalized[:-1].strip()
        if ";" in normalized:
            raise ValueError("Graph retriever only allows single-statement Cypher queries.")

        stripped = STRING_LITERAL_PATTERN.sub(" ", normalized)
        if WRITE_CLAUSE_PATTERN.search(stripped):
            raise ValueError("Graph retriever only allows read-only Cypher queries.")
        if not READ_START_PATTERN.match(stripped):
            raise ValueError("Graph retriever requires a read-only starting clause.")

    def _rows_to_results(
        self,
        rows: list[dict[str, Any]],
        analysis: QueryAnalysis,
        anchors: list[LinkedEntity],
        query_mode: str,
        fallback_reason: str | None,
        plan: DynamicCypherPlan | None = None,
    ) -> list[RetrieverResult]:
        results: list[RetrieverResult] = []
        status_by_anchor = {str(item.entity_id): item.status for item in anchors if item.entity_id}
        similarity_by_anchor = {str(item.entity_id): float(item.similarity) for item in anchors if item.entity_id}
        default_anchor = self._default_anchor_id(anchors, plan)

        for row in rows:
            row_anchor = str(row.get("anchor_entity_id") or default_anchor)
            anchor_status, anchor_similarity = self._resolve_anchor_state(
                anchor_key=row_anchor,
                status_by_anchor=status_by_anchor,
                similarity_by_anchor=similarity_by_anchor,
                anchors=anchors,
            )
            result = self._build_result_from_row(
                row=row,
                anchor_entity_id=row_anchor,
                query_type=analysis.query_type,
                anchor_status=anchor_status,
                anchor_similarity=anchor_similarity,
                query_mode=query_mode,
                fallback_reason=fallback_reason,
                planner_mode=plan.mode if plan is not None else None,
            )
            if result is None:
                continue
            if plan is not None:
                result.provenance["schema_generated_at"] = plan.schema_generated_at
                if plan.temporal_filter_hint:
                    result.provenance["temporal_filter_hint"] = plan.temporal_filter_hint
            results.append(result)
        return results

    def _default_anchor_id(self, anchors: list[LinkedEntity], plan: DynamicCypherPlan | None) -> str:
        if plan is not None and plan.mode == "multihop_pair" and len(anchors) >= 2:
            return f"{anchors[0].entity_id}|{anchors[1].entity_id}"
        if anchors:
            return str(anchors[0].entity_id or "")
        return ""

    def _resolve_anchor_state(
        self,
        anchor_key: str,
        status_by_anchor: dict[str, str],
        similarity_by_anchor: dict[str, float],
        anchors: list[LinkedEntity],
    ) -> tuple[str, float]:
        if "|" in anchor_key:
            parts = [part.strip() for part in anchor_key.split("|") if part.strip()]
            statuses = [status_by_anchor.get(part, "ambiguous") for part in parts]
            similarities = [similarity_by_anchor.get(part, 0.0) for part in parts]
            status = "resolved" if statuses and all(item == "resolved" for item in statuses) else "ambiguous"
            similarity = min(similarities) if similarities else 0.0
            return status, similarity

        if anchor_key in status_by_anchor:
            return status_by_anchor[anchor_key], similarity_by_anchor.get(anchor_key, 0.0)
        if anchors:
            default = anchors[0]
            return default.status, float(default.similarity)
        return "ambiguous", 0.0

    def _build_result_from_row(
        self,
        row: dict[str, Any],
        anchor_entity_id: str,
        query_type: QueryType,
        anchor_status: str,
        anchor_similarity: float,
        query_mode: str,
        fallback_reason: str | None,
        planner_mode: str | None,
    ) -> RetrieverResult | None:
        node_names, rel_types, rel_weights = self._extract_path_parts(row)
        if len(node_names) < 2 or not rel_types:
            return None
        path = self._format_path(node_names, rel_types)
        score = self._score_path(rel_weights)
        if anchor_status == "ambiguous":
            penalty = max(0.0, min(1.0, float(self.settings.query_graph_ambiguous_score_penalty)))
            score = score * penalty
        source_id = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]

        provenance = {
            "anchor_entity_id": anchor_entity_id,
            "anchor_status": anchor_status,
            "anchor_similarity": anchor_similarity,
            "query_type": query_type.value,
            "hop_count": len(rel_types),
            "node_names": node_names,
            "rel_types": rel_types,
            "query_mode": query_mode,
        }
        if fallback_reason:
            provenance["fallback_reason"] = fallback_reason
        if planner_mode:
            provenance["planner_mode"] = planner_mode

        return RetrieverResult(
            source_type="graph",
            source_id=source_id,
            text_or_path=path,
            score_raw=score,
            provenance=provenance,
        )

    def _extract_path_parts(self, row: dict[str, Any]) -> tuple[list[str], list[str], list[float]]:
        node_names = row.get("node_names")
        rel_types = row.get("rel_types")
        rel_weights = row.get("rel_weights")
        if isinstance(node_names, list) and isinstance(rel_types, list):
            safe_nodes = [str(item) for item in node_names]
            safe_rels = [str(item) for item in rel_types]
            safe_weights = [float(item) for item in rel_weights] if isinstance(rel_weights, list) else []
            return safe_nodes, safe_rels, safe_weights

        path = row.get("p")
        if path is None:
            return [], [], []
        nodes = getattr(path, "nodes", [])
        relationships = getattr(path, "relationships", [])
        safe_nodes = [str(node.get("name") or node.get("id") or "") for node in nodes]
        safe_rels = [str(rel.get("type") or getattr(rel, "type", "RELATED_TO")) for rel in relationships]
        safe_weights = [float(rel.get("weight") or 0.0) for rel in relationships]
        return safe_nodes, safe_rels, safe_weights

    def _format_path(self, node_names: list[str], rel_types: list[str]) -> str:
        parts: list[str] = [node_names[0]]
        for relation, tail in zip(rel_types, node_names[1:]):
            parts.append(f"-[{relation}]->")
            parts.append(tail)
        return " ".join(parts)

    def _score_path(self, rel_weights: list[float]) -> float:
        if not rel_weights:
            return 0.5
        avg = sum(rel_weights) / len(rel_weights)
        return max(0.0, min(1.0, float(avg)))
