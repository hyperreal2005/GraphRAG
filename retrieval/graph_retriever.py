"""Graph retrieval using static, read-only Cypher templates."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from config import Settings
from models.query import LinkedEntity, QueryAnalysis, QueryType, RetrieverResult


WRITE_CLAUSE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|LOAD|FOREACH)\b",
    re.IGNORECASE,
)


class GraphRetriever:
    """Execute safe graph reads over Neo4j for linked entities."""

    def __init__(self, settings: Settings, driver: Any | None = None) -> None:
        self.settings = settings
        self._driver = driver
        if self._driver is not None:
            return
        try:
            from neo4j import GraphDatabase
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("neo4j driver is required for graph retrieval.") from exc
        self._driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_user, self.settings.neo4j_password),
        )

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def retrieve(self, analysis: QueryAnalysis, linked_entities: list[LinkedEntity]) -> list[RetrieverResult]:
        """Retrieve graph paths from resolved anchors with ambiguous fallback policy."""
        anchors = self._select_graph_anchors(linked_entities)
        if not anchors:
            return []

        if analysis.query_type is QueryType.COMPARATIVE and len(anchors) >= 2:
            return self._retrieve_comparative(anchors[:2], analysis)
        max_hops = (
            int(self.settings.query_graph_multihop_hops)
            if analysis.query_type is QueryType.MULTIHOP
            else int(self.settings.query_graph_local_hops)
        )
        return self._retrieve_neighborhood(anchors, max_hops=max_hops, query_type=analysis.query_type)

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
    ) -> list[RetrieverResult]:
        query = self._build_neighbor_query(max_hops=max_hops)
        all_results: list[RetrieverResult] = []
        for linked in linked_entities:
            params = {
                "entity_id": str(linked.entity_id),
                "path_limit": int(self.settings.query_graph_path_limit),
            }
            rows = self._execute_read(query, params)
            for row in rows:
                result = self._build_result_from_row(
                    row=row,
                    anchor_entity_id=str(linked.entity_id),
                    query_type=query_type,
                    anchor_status=linked.status,
                    anchor_similarity=float(linked.similarity),
                )
                if result is not None:
                    all_results.append(result)
        return all_results

    def _retrieve_comparative(
        self,
        linked_entities: list[LinkedEntity],
        analysis: QueryAnalysis,
    ) -> list[RetrieverResult]:
        query = self._build_comparative_query(max_hops=int(self.settings.query_graph_multihop_hops))
        params = {
            "entity_a": str(linked_entities[0].entity_id),
            "entity_b": str(linked_entities[1].entity_id),
        }
        rows = self._execute_read(query, params)
        results: list[RetrieverResult] = []
        anchor = f"{linked_entities[0].entity_id}|{linked_entities[1].entity_id}"
        anchor_status = (
            "resolved"
            if all(item.status == "resolved" for item in linked_entities[:2])
            else "ambiguous"
        )
        anchor_similarity = min(float(item.similarity) for item in linked_entities[:2])
        for row in rows:
            result = self._build_result_from_row(
                row=row,
                anchor_entity_id=anchor,
                query_type=analysis.query_type,
                anchor_status=anchor_status,
                anchor_similarity=anchor_similarity,
            )
            if result is not None:
                results.append(result)
        return results

    def _build_neighbor_query(self, max_hops: int) -> str:
        hops = max(1, int(max_hops))
        return (
            "MATCH p=(start:Entity {id: $entity_id})-[*1.."
            f"{hops}"
            "]-(nbr:Entity) "
            "RETURN "
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
            "[n IN nodes(p) | coalesce(n.name, n.id)] AS node_names, "
            "[rel IN relationships(p) | coalesce(rel.type, type(rel))] AS rel_types, "
            "[rel IN relationships(p) | coalesce(rel.weight, 0.0)] AS rel_weights"
        )

    def _execute_read(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self._assert_read_only(query)
        with self._driver.session() as session:
            records = session.run(query, params)
            return [dict(record) for record in records]

    def _assert_read_only(self, query: str) -> None:
        stripped = re.sub(r"'[^']*'|\"[^\"]*\"", " ", query)
        if WRITE_CLAUSE_PATTERN.search(stripped):
            raise ValueError("Graph retriever only allows read-only Cypher queries.")

    def _build_result_from_row(
        self,
        row: dict[str, Any],
        anchor_entity_id: str,
        query_type: QueryType,
        anchor_status: str,
        anchor_similarity: float,
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
        return RetrieverResult(
            source_type="graph",
            source_id=source_id,
            text_or_path=path,
            score_raw=score,
            provenance={
                "anchor_entity_id": anchor_entity_id,
                "anchor_status": anchor_status,
                "anchor_similarity": anchor_similarity,
                "query_type": query_type.value,
                "hop_count": len(rel_types),
                "node_names": node_names,
                "rel_types": rel_types,
            },
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
