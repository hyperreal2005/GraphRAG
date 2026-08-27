"""Graph retrieval using dynamic Cypher with strict read-only safety."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import re
from typing import Any

from config import Settings
from ingestion.state_store import MetadataStore
from models.query import LinkedEntity, QueryAnalysis, QueryType, RetrieverResult
from retrieval import graph_traversal
from retrieval.dynamic_cypher import DynamicCypherPlan, DynamicCypherPlanner


WRITE_CLAUSE_PATTERN = re.compile(
    r"\b(CREATE|INSERT|MERGE|DELETE|DETACH|SET|REMOVE|DROP|ALTER|RENAME|GRANT|DENY|REVOKE|"
    r"CALL|LOAD|FOREACH|APOC|DBMS)\b|\bSTART\s+(DATABASE|TRANSACTION)\b|"
    r"\bSTOP\s+DATABASE\b|\bTERMINATE\s+TRANSACTIONS?\b",
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
        store: MetadataStore | None = None,
    ) -> None:
        self.settings = settings
        self._driver = driver
        self._async_driver = None
        if self._driver is None:
            try:
                from neo4j import AsyncGraphDatabase, GraphDatabase
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("neo4j driver is required for graph retrieval.") from exc
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(
                    self.settings.neo4j_query_user or self.settings.neo4j_user,
                    self.settings.neo4j_query_password or self.settings.neo4j_password,
                ),
            )
            self._async_driver = AsyncGraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(
                    self.settings.neo4j_query_user or self.settings.neo4j_user,
                    self.settings.neo4j_query_password or self.settings.neo4j_password,
                ),
            )
        self._planner = planner or DynamicCypherPlanner(settings=self.settings, driver=self._driver)
        self.store = store

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    async def aclose(self) -> None:
        if self._async_driver is not None:
            await self._async_driver.close()

    async def aretrieve(
        self,
        analysis: QueryAnalysis,
        linked_entities: list[LinkedEntity],
        user_query: str = "",
    ) -> list[RetrieverResult]:
        """Retrieve using Neo4j's async driver while preserving static fallback."""
        if self._async_driver is None:
            return await asyncio.to_thread(
                self.retrieve,
                analysis,
                linked_entities,
                user_query,
            )
        anchors = self._select_graph_anchors(linked_entities)
        if not anchors or analysis.query_type is QueryType.GLOBAL:
            return []
        if self._queryable_version_ids() == []:
            return []
        if self.settings.query_graph_dynamic_enabled and user_query.strip():
            try:
                plan = await asyncio.to_thread(
                    self._planner.generate_plan,
                    user_query,
                    analysis,
                    anchors,
                )
                anchor_ids, max_hops, targets, params = self._compile_logical_plan(plan)
                rows = await self._abfs(
                    anchor_ids=anchor_ids,
                    max_hops=max_hops,
                    params=params,
                    targets=targets,
                )
                default_anchor = (
                    "|".join(sorted(targets | set(anchor_ids))) if targets else anchor_ids[0]
                )
                for row in rows:
                    row.setdefault("anchor_entity_id", default_anchor)
                return self._rows_to_results(
                    rows,
                    analysis,
                    anchors,
                    "dynamic",
                    None,
                    plan,
                )
            except Exception as exc:
                if not self.settings.query_graph_dynamic_fallback_to_static:
                    return []
                return await self._aretrieve_static(
                    analysis,
                    anchors,
                    query_mode="static_fallback",
                    fallback_reason=f"{type(exc).__name__}: {exc}",
                )
        return await self._aretrieve_static(analysis, anchors, query_mode="static")

    async def _aretrieve_static(
        self,
        analysis: QueryAnalysis,
        anchors: list[LinkedEntity],
        *,
        query_mode: str,
        fallback_reason: str | None = None,
    ) -> list[RetrieverResult]:
        if analysis.query_type is QueryType.MULTIHOP and len(anchors) >= 2:
            pairs = self._anchor_pairs(anchors)
            row_groups = await asyncio.gather(
                *(
                    self._abfs(
                        anchor_ids=[str(pair[0].entity_id)],
                        max_hops=int(self.settings.query_graph_multihop_hops),
                        params=self._traversal_params(analysis.temporal_filter),
                        targets={str(pair[1].entity_id)},
                    )
                    for pair in pairs
                )
            )
            connecting_results: list[RetrieverResult] = []
            for pair, rows in zip(pairs, row_groups):
                for row in rows:
                    row.setdefault(
                        "anchor_entity_id", f"{pair[0].entity_id}|{pair[1].entity_id}"
                    )
                connecting_results.extend(
                    self._rows_to_results(
                        rows,
                        analysis,
                        list(pair),
                        query_mode,
                        fallback_reason,
                    )
                )
            if connecting_results:
                return self._dedupe_results(connecting_results)
            fallback_reason = fallback_reason or "no_connecting_path"
        hops = (
            int(self.settings.query_graph_multihop_hops)
            if analysis.query_type is QueryType.MULTIHOP
            else int(self.settings.query_graph_local_hops)
        )
        row_groups = await asyncio.gather(
            *(
                self._abfs(
                    anchor_ids=[str(anchor.entity_id)],
                    max_hops=hops,
                    params=self._traversal_params(analysis.temporal_filter),
                )
                for anchor in anchors
            )
        )
        results: list[RetrieverResult] = []
        for anchor, rows in zip(anchors, row_groups):
            for row in rows:
                row.setdefault("anchor_entity_id", str(anchor.entity_id))
            results.extend(
                self._rows_to_results(
                    rows,
                    QueryAnalysis(query_type=analysis.query_type),
                    [anchor],
                    query_mode,
                    fallback_reason,
                )
            )
        return self._dedupe_results(results)

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
        if self._queryable_version_ids() == []:
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
        anchor_ids, max_hops, targets, params = self._compile_logical_plan(plan)
        rows = self._bfs(
            anchor_ids=anchor_ids,
            max_hops=max_hops,
            params=params,
            targets=targets,
        )
        default_anchor = "|".join(sorted(targets | set(anchor_ids))) if targets else anchor_ids[0]
        for row in rows:
            row.setdefault("anchor_entity_id", default_anchor)
        return self._rows_to_results(
            rows=rows,
            analysis=analysis,
            anchors=anchors,
            query_mode="dynamic",
            fallback_reason=None,
            plan=plan,
        )

    def _compile_logical_plan(
        self,
        plan: DynamicCypherPlan,
    ) -> tuple[list[str], int, set[str] | None, dict[str, Any]]:
        """Turn validated model choices into bounded traversal inputs.

        The model contributes only relationship-type filters and hop/limit numbers, all
        of which are clamped here. Since WP4-B the traversal itself is level-wise BFS, so
        there is no generated Cypher string to validate — the query is a constant in
        `retrieval/graph_traversal.py`.
        """
        max_hops = max(1, min(6, int(plan.params.get("max_hops", 1))))
        path_limit = max(1, min(100, int(plan.params.get("path_limit", self.settings.query_graph_path_limit))))
        params = {
            "path_limit": path_limit,
            "relationship_types": list(plan.relationship_types),
            **self._temporal_params(plan.temporal_filter_hint),
        }

        if plan.mode in {"neighborhood", "comparative"}:
            anchor_ids = plan.params.get("anchor_ids")
            if not isinstance(anchor_ids, list) or not anchor_ids:
                raise ValueError(f"Logical graph plan mode {plan.mode} requires anchor_ids.")
            return [str(item) for item in anchor_ids], max_hops, None, params

        if plan.mode == "multihop_pair":
            anchor_a = str(plan.params.get("anchor_a") or "")
            anchor_b = str(plan.params.get("anchor_b") or "")
            if not anchor_a or not anchor_b:
                raise ValueError("Logical graph plan mode multihop_pair requires anchor_a and anchor_b.")
            return [anchor_a], max_hops, {anchor_b}, params

        raise ValueError(f"Unsupported logical graph plan mode: {plan.mode}")

    def _retrieve_static(
        self,
        analysis: QueryAnalysis,
        anchors: list[LinkedEntity],
        query_mode: str,
        fallback_reason: str | None = None,
    ) -> list[RetrieverResult]:
        if analysis.query_type is QueryType.GLOBAL:
            return []
        if analysis.query_type is QueryType.MULTIHOP and len(anchors) >= 2:
            connecting_results: list[RetrieverResult] = []
            for pair in self._anchor_pairs(anchors):
                connecting_results.extend(
                    self._retrieve_connecting_path(
                        linked_entities=list(pair),
                        analysis=analysis,
                        query_mode=query_mode,
                        fallback_reason=fallback_reason,
                    )
                )
            if connecting_results:
                return self._dedupe_results(connecting_results)
            fallback_reason = fallback_reason or "no_connecting_path"
        if analysis.query_type is QueryType.COMPARATIVE:
            return self._retrieve_neighborhood(
                linked_entities=anchors,
                max_hops=int(self.settings.query_graph_local_hops),
                query_type=analysis.query_type,
                temporal_filter=analysis.temporal_filter,
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
            temporal_filter=analysis.temporal_filter,
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
        temporal_filter: str | None,
        query_mode: str,
        fallback_reason: str | None,
    ) -> list[RetrieverResult]:
        all_results: list[RetrieverResult] = []
        for linked in linked_entities:
            rows = self._bfs(
                anchor_ids=[str(linked.entity_id)],
                max_hops=max_hops,
                params=self._traversal_params(temporal_filter),
            )
            for row in rows:
                row.setdefault("anchor_entity_id", str(linked.entity_id))
            all_results.extend(
                self._rows_to_results(
                    rows=rows,
                    analysis=QueryAnalysis(query_type=query_type),
                    anchors=[linked],
                    query_mode=query_mode,
                    fallback_reason=fallback_reason,
                )
            )
        return self._dedupe_results(all_results)

    @staticmethod
    def _anchor_pairs(
        anchors: list[LinkedEntity],
        max_pairs: int = 6,
    ) -> list[tuple[LinkedEntity, LinkedEntity]]:
        return list(itertools.islice(itertools.combinations(anchors, 2), max(1, max_pairs)))

    @staticmethod
    def _dedupe_results(results: list[RetrieverResult]) -> list[RetrieverResult]:
        deduped: dict[str, RetrieverResult] = {}
        for item in results:
            key = item.dedupe_key()
            existing = deduped.get(key)
            if existing is None or float(item.score_raw) > float(existing.score_raw):
                deduped[key] = item
        return list(deduped.values())

    def _retrieve_connecting_path(
        self,
        linked_entities: list[LinkedEntity],
        analysis: QueryAnalysis,
        query_mode: str,
        fallback_reason: str | None,
    ) -> list[RetrieverResult]:
        entity_a = str(linked_entities[0].entity_id)
        entity_b = str(linked_entities[1].entity_id)
        rows = self._bfs(
            anchor_ids=[entity_a],
            max_hops=int(self.settings.query_graph_multihop_hops),
            params=self._traversal_params(analysis.temporal_filter),
            targets={entity_b},
        )
        for row in rows:
            row.setdefault("anchor_entity_id", f"{entity_a}|{entity_b}")
        return self._rows_to_results(
            rows=rows,
            analysis=analysis,
            anchors=linked_entities[:2],
            query_mode=query_mode,
            fallback_reason=fallback_reason,
        )

    def _traversal_params(
        self,
        temporal_filter: str | None,
        relationship_types: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "relationship_types": list(relationship_types or []),
            **self._temporal_params(temporal_filter),
        }

    def _bfs(
        self,
        *,
        anchor_ids: list[str],
        max_hops: int,
        params: dict[str, Any],
        targets: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronous level-wise expansion (WP4-B)."""
        result = graph_traversal.expand(
            self._execute_read,
            anchor_ids=anchor_ids,
            max_hops=max_hops,
            level_limit=self._level_limit(),
            params=params,
            targets=targets,
        )
        return result.limited(int(self.settings.query_graph_path_limit))

    async def _abfs(
        self,
        *,
        anchor_ids: list[str],
        max_hops: int,
        params: dict[str, Any],
        targets: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Async level-wise expansion. Levels are sequential by nature: each one's
        frontier is the previous one's output."""
        result = graph_traversal.TraversalResult()
        frontier = [str(item) for item in anchor_ids if str(item)]
        visited: set[str] = set(frontier)
        result.reached.update(frontier)
        parents: dict[str, tuple[dict[str, Any], str]] = {}
        seen: set[str] = set()

        for level in range(1, max(1, int(max_hops)) + 1):
            if not frontier:
                break
            rows = await self._aexecute_read(
                graph_traversal.build_level_query(),
                {**params, "frontier": frontier, "level_limit": self._level_limit()},
            )
            result.levels_executed = level
            next_frontier: list[str] = []
            for row in rows:
                source_entity_id = str(row.get("source_entity_id") or "")
                annotated = {**row, "hop_count": level}
                identity = str(row.get("assertion_id") or "") or graph_traversal._fallback_identity(row)
                if identity not in seen:
                    seen.add(identity)
                    result.rows.append(annotated)
                for neighbour in graph_traversal.neighbours(row, source_entity_id):
                    result.reached.add(neighbour)
                    if neighbour not in parents and neighbour not in visited:
                        parents[neighbour] = (annotated, source_entity_id)
                    if neighbour not in visited:
                        visited.add(neighbour)
                        next_frontier.append(neighbour)

            if targets:
                hit = next((item for item in targets if item in result.reached), None)
                if hit is not None:
                    result.target_found = True
                    result.rows = graph_traversal._path_rows(parents, hit)
                    return result.limited(int(self.settings.query_graph_path_limit))
            frontier = next_frontier

        if targets and not result.target_found:
            return []
        return result.limited(int(self.settings.query_graph_path_limit))

    def _level_limit(self) -> int:
        return max(1, int(self.settings.query_graph_level_fanout))

    @staticmethod
    def _temporal_params(value: str | None) -> dict[str, str | None]:
        clean = str(value or "").strip()
        if not clean:
            return {"temporal_start": None, "temporal_end": None}
        match = re.fullmatch(r"((?:19|20)\d{2})-((?:19|20)\d{2})", clean)
        if match:
            return {
                "temporal_start": f"{match.group(1)}-01-01",
                "temporal_end": f"{match.group(2)}-12-31",
            }
        if re.fullmatch(r"(?:19|20)\d{2}", clean):
            return {"temporal_start": f"{clean}-01-01", "temporal_end": f"{clean}-12-31"}
        return {"temporal_start": None, "temporal_end": None}

    def _execute_read(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self._assert_read_only(query)
        params = {**params, **self._version_fence_params()}
        with self._driver.session() as session:
            records = session.run(self._timed_query(query), params)
            return [dict(record) for record in records]

    async def _aexecute_read(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self._assert_read_only(query)
        params = {**params, **self._version_fence_params()}
        if self._async_driver is None:
            return await asyncio.to_thread(self._execute_read, query, params)

        async with self._async_driver.session() as session:
            result = await session.run(self._timed_query(query), params)
            return [record.data() async for record in result]

    def _timed_query(self, query: str) -> Any:
        """Attach the server transaction timeout to a session-level read query."""
        try:
            from neo4j import Query

            return Query(
                query,
                timeout=max(
                    0.1,
                    float(self.settings.query_graph_transaction_timeout_seconds),
                ),
                metadata={"component": "graphrag_retrieval"},
            )
        except ImportError:  # pragma: no cover - guarded by the runtime dependency
            return query

    def _queryable_version_ids(self) -> list[str] | None:
        if self.store is None:
            return None
        return self.store.get_queryable_document_version_ids()

    def _version_fence_params(self) -> dict[str, Any]:
        active_version_ids = self._queryable_version_ids()
        return {
            "enforce_version_fence": active_version_ids is not None,
            "active_version_ids": active_version_ids or [],
        }

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
        if row.get("assertion_id"):
            return self._build_assertion_result(
                row=row,
                anchor_entity_id=anchor_entity_id,
                query_type=query_type,
                anchor_status=anchor_status,
                anchor_similarity=anchor_similarity,
                query_mode=query_mode,
                fallback_reason=fallback_reason,
                planner_mode=planner_mode,
            )
        node_names, rel_types, rel_weights, rel_directions, rel_source_refs = self._extract_path_parts(row)
        if len(node_names) < 2 or not rel_types:
            return None
        path = self._format_path(node_names, rel_types, rel_directions)
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
            "rel_directions": rel_directions,
            "source_refs": list(
                dict.fromkeys(
                    str(ref)
                    for refs in rel_source_refs
                    for ref in refs
                    if str(ref).strip()
                )
            ),
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

    def _build_assertion_result(
        self,
        *,
        row: dict[str, Any],
        anchor_entity_id: str,
        query_type: QueryType,
        anchor_status: str,
        anchor_similarity: float,
        query_mode: str,
        fallback_reason: str | None,
        planner_mode: str | None,
    ) -> RetrieverResult | None:
        assertion_id = str(row.get("assertion_id") or "").strip()
        chunk_id = str(row.get("chunk_id") or "").strip()
        version_id = str(row.get("document_version_id") or "").strip()
        if not assertion_id or not chunk_id or not version_id:
            return None
        chunk = None
        if self.store is not None:
            chunk = self.store.get_chunk_by_id(
                chunk_id,
                document_version_id=version_id,
                active_only=True,
            )
            if chunk is None:
                return None

        subject_name = str(row.get("subject_name") or row.get("subject_entity_id") or "Entity")
        predicate = str(row.get("predicate_raw") or row.get("predicate_normalized") or "related to")
        if str(row.get("object_kind") or "entity") == "literal":
            raw_literal = str(row.get("literal_value_json") or "null")
            try:
                import json

                object_value = json.loads(raw_literal)
            except (TypeError, ValueError):
                object_value = raw_literal
            object_text = str(object_value)
        else:
            object_text = str(row.get("object_name") or row.get("object_entity_id") or "Entity")
        graph_path = f"{subject_name} -[{predicate}]-> {object_text}"
        evidence_text = str(chunk["text"] or "") if chunk is not None else ""
        text = graph_path
        if evidence_text:
            text = f"{graph_path}\nExact source chunk: {evidence_text}"

        hop_count = max(1, int(row.get("hop_count") or 1))
        score = max(0.0, min(1.0, float(row.get("confidence") or 0.0)))
        score /= 1.0 + (0.15 * float(hop_count - 1))
        if anchor_status == "ambiguous":
            score *= max(0.0, min(1.0, float(self.settings.query_graph_ambiguous_score_penalty)))
        evidence_id = f"assertion:{assertion_id}"
        provenance: dict[str, Any] = {
            "evidence_id": evidence_id,
            "assertion_id": assertion_id,
            "claim_key": row.get("claim_key"),
            "anchor_entity_id": anchor_entity_id,
            "anchor_status": anchor_status,
            "anchor_similarity": anchor_similarity,
            "query_type": query_type.value,
            "hop_count": hop_count,
            "subject_entity_id": row.get("subject_entity_id"),
            "object_entity_id": row.get("object_entity_id"),
            "object_kind": row.get("object_kind"),
            "literal_value_json": row.get("literal_value_json"),
            "properties_json": row.get("properties_json"),
            "predicate_raw": row.get("predicate_raw"),
            "predicate_canonical": row.get("predicate_canonical"),
            "predicate_normalized": row.get("predicate_normalized"),
            "direction": "subject_to_object",
            "chunk_id": chunk_id,
            "doc_id": row.get("doc_id") or (chunk["doc_id"] if chunk is not None else None),
            "document_version_id": version_id,
            "source_ref": row.get("source_ref"),
            "valid_from": row.get("valid_from"),
            "valid_until": row.get("valid_until"),
            "section_heading": chunk["section"] if chunk is not None else None,
            "page": chunk["page"] if chunk is not None else None,
            "evidence_text": evidence_text,
            "graph_path": graph_path,
            "query_mode": query_mode,
        }
        if fallback_reason:
            provenance["fallback_reason"] = fallback_reason
        if planner_mode:
            provenance["planner_mode"] = planner_mode
        return RetrieverResult(
            source_type="graph",
            source_id=assertion_id,
            text_or_path=text,
            score_raw=score,
            provenance=provenance,
        )

    def _extract_path_parts(
        self,
        row: dict[str, Any],
    ) -> tuple[list[str], list[str], list[float], list[str], list[list[str]]]:
        node_names = row.get("node_names")
        rel_types = row.get("rel_types")
        rel_weights = row.get("rel_weights")
        rel_directions = row.get("rel_directions")
        rel_source_refs = row.get("rel_source_refs")
        if isinstance(node_names, list) and isinstance(rel_types, list):
            safe_nodes = [str(item) for item in node_names]
            safe_rels = [str(item) for item in rel_types]
            safe_weights = [float(item) for item in rel_weights] if isinstance(rel_weights, list) else []
            safe_directions = (
                ["in" if str(item).lower() == "in" else "out" for item in rel_directions]
                if isinstance(rel_directions, list)
                else ["out"] * len(safe_rels)
            )
            safe_source_refs = []
            if isinstance(rel_source_refs, list):
                for item in rel_source_refs:
                    safe_source_refs.append([str(ref) for ref in item] if isinstance(item, list) else [])
            while len(safe_source_refs) < len(safe_rels):
                safe_source_refs.append([])
            return safe_nodes, safe_rels, safe_weights, safe_directions, safe_source_refs

        path = row.get("p")
        if path is None:
            return [], [], [], [], []
        nodes = getattr(path, "nodes", [])
        relationships = getattr(path, "relationships", [])
        safe_nodes = [str(node.get("name") or node.get("id") or "") for node in nodes]
        safe_rels = [str(rel.get("type") or getattr(rel, "type", "RELATED_TO")) for rel in relationships]
        safe_weights = [float(rel.get("weight") or 0.0) for rel in relationships]
        safe_source_refs = [
            [str(ref) for ref in (rel.get("source_refs") or [])]
            for rel in relationships
        ]
        return safe_nodes, safe_rels, safe_weights, ["out"] * len(safe_rels), safe_source_refs

    def _format_path(self, node_names: list[str], rel_types: list[str], rel_directions: list[str]) -> str:
        parts: list[str] = [node_names[0]]
        for index, (relation, tail) in enumerate(zip(rel_types, node_names[1:])):
            direction = rel_directions[index] if index < len(rel_directions) else "out"
            parts.append(f"<-[{relation}]-" if direction == "in" else f"-[{relation}]->")
            parts.append(tail)
        return " ".join(parts)

    def _score_path(self, rel_weights: list[float]) -> float:
        if not rel_weights:
            return 0.5
        avg = sum(rel_weights) / len(rel_weights)
        return max(0.0, min(1.0, float(avg)))
