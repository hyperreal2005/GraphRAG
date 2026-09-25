"""Phase 8: Louvain community detection over the entity graph, with LLM summaries.

Phase 8 was never built. `CommunityRetriever` shipped as a stub returning `[]`, no
`communities` collection was ever created, and `SOURCE_WEIGHTS["community"]` has therefore
never been exercised - GLOBAL and HYBRID queries have been running with two of their three
retrieval sources silently absent.

Two properties of this corpus shape the design:

**The entity graph is sparse.** 9,141 active entities are joined by 3,819 distinct pairs,
so 1,424 entities are isolated and Louvain returns 973 communities of which 296 have one or
two members. A two-entity "community" carries no theme, so `min_size` is a correctness
control before it is a cost one - though it is also the difference between 973 summary
calls and 218.

**Communities are derived, not source data.** They are rebuilt wholesale rather than
versioned: each build gets an id, writes its own nodes and points, and the previous build
is removed once the new one is in place. That deliberately avoids the immutable-version
machinery the ingestion plane needs, because nothing cites a community and a stale summary
is discarded rather than retracted.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from config import Settings
from generation.model_adapter import build_generate_config
from ingestion.embedder import GeminiEmbedder

logger = logging.getLogger(__name__)

JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

SUMMARY_INSTRUCTION = (
    "You describe a cluster of related entities from a knowledge graph. "
    "Use only the entities and relations supplied. Do not add outside knowledge. "
    "Return only the requested JSON."
)


@dataclass
class Community:
    """One detected cluster, before or after summarisation."""

    community_id: str
    member_ids: list[str]
    member_names: list[str]
    relations: list[str]
    doc_ids: list[str] = field(default_factory=list)
    title: str = ""
    summary: str = ""

    @property
    def size(self) -> int:
        return len(self.member_ids)

    def embed_text(self) -> str:
        return f"{self.title}\n{self.summary}" if self.title else self.summary


@dataclass
class CommunityBuildReport:
    build_id: str
    detected: int
    kept: int
    summarised: int
    indexed: int
    failed: int
    entities_covered: int
    largest: int
    provider_calls: int


class CommunityBuilder:
    """Detect entity communities, summarise them, and publish them for retrieval."""

    def __init__(
        self,
        settings: Settings,
        driver: Any | None = None,
        embedder: GeminiEmbedder | None = None,
        qdrant_client: Any | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self._embedder = embedder
        self._qdrant = qdrant_client
        self._client = client
        self._types = None
        self.last_usage: dict[str, int] = {}

        if driver is not None:
            self._driver = driver
        else:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(
                    settings.neo4j_query_user or settings.neo4j_user,
                    settings.neo4j_query_password or settings.neo4j_password,
                ),
            )

    # ------------------------------------------------------------------ detection
    def detect(self, min_size: int | None = None) -> list[Community]:
        """Louvain over the entity-to-entity projection of active assertions."""
        import networkx as nx

        floor = int(self.settings.community_min_size if min_size is None else min_size)
        query = """
        MATCH (x:Entity)-[:SUBJECT_OF]->(a:Assertion {active: true})-[:OBJECT]->(y:Entity)
        WHERE x.id <> y.id
        RETURN x.id AS a_id, x.name AS a_name, y.id AS b_id, y.name AS b_name,
               a.predicate_normalized AS predicate, a.doc_id AS doc_id
        """
        graph = nx.Graph()
        names: dict[str, str] = {}
        edge_facts: dict[tuple[str, str], list[str]] = {}
        edge_docs: dict[tuple[str, str], set[str]] = {}

        with self._driver.session() as session:
            for record in session.run(query):
                left, right = str(record["a_id"]), str(record["b_id"])
                names.setdefault(left, str(record["a_name"] or left))
                names.setdefault(right, str(record["b_name"] or right))
                if graph.has_edge(left, right):
                    graph[left][right]["weight"] += 1
                else:
                    graph.add_edge(left, right, weight=1)
                key = (left, right) if left < right else (right, left)
                fact = f"{names[left]} -[{record['predicate']}]-> {names[right]}"
                facts = edge_facts.setdefault(key, [])
                if fact not in facts:
                    facts.append(fact)
                if record["doc_id"]:
                    edge_docs.setdefault(key, set()).add(str(record["doc_id"]))

        if graph.number_of_nodes() == 0:
            return []

        partitions = nx.community.louvain_communities(
            graph, weight="weight", seed=int(self.settings.community_seed)
        )
        communities: list[Community] = []
        for members in partitions:
            if len(members) < floor:
                continue
            member_ids = sorted(str(m) for m in members)
            member_set = set(member_ids)
            relations: list[str] = []
            docs: set[str] = set()
            for (left, right), facts in edge_facts.items():
                if left in member_set and right in member_set:
                    relations.extend(facts)
                    docs.update(edge_docs.get((left, right), set()))
            communities.append(
                Community(
                    community_id=str(uuid.uuid4()),
                    member_ids=member_ids,
                    member_names=[names.get(m, m) for m in member_ids],
                    relations=relations,
                    doc_ids=sorted(docs),
                )
            )
        communities.sort(key=lambda c: c.size, reverse=True)
        return communities

    # --------------------------------------------------------------- summarisation
    def summarise(self, community: Community) -> bool:
        """Attach a title and summary. False if the provider could not produce one."""
        client = self._ensure_client()
        if client is None:
            return False

        max_names = max(1, int(self.settings.community_summary_max_entities))
        max_relations = max(1, int(self.settings.community_summary_max_relations))
        prompt = (
            "Summarise this cluster of related entities.\n\n"
            f"ENTITIES ({community.size} total, showing {min(community.size, max_names)}):\n"
            + "\n".join(f"- {name}" for name in community.member_names[:max_names])
            + "\n\nRELATIONS:\n"
            + "\n".join(f"- {rel}" for rel in community.relations[:max_relations])
            + '\n\nReturn JSON: {"title": "<six words or fewer>", '
            '"summary": "<two to four sentences on what connects these entities>"}'
        )

        # Everything after the request is inside the guard too. One community returning a
        # JSON array rather than an object raised out of `build`, discarding every summary
        # already paid for in that run - a whole build lost to one malformed response.
        try:
            response = client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config=build_generate_config(
                    self._types,
                    self.settings.gemini_model,
                    system_instruction=SUMMARY_INSTRUCTION,
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            payload = self._parse_json(getattr(response, "text", "") or "")
            title = str(payload.get("title") or "").strip()
            summary = str(payload.get("summary") or "").strip()
        except Exception as exc:
            logger.warning(
                "community_summary_failed community=%s error_type=%s",
                community.community_id,
                type(exc).__name__,
            )
            return False

        if not summary:
            return False
        community.title = title or ", ".join(community.member_names[:3])
        community.summary = summary
        return True

    # ------------------------------------------------------------------ publishing
    def build(
        self,
        min_size: int | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> CommunityBuildReport:
        """Detect, summarise and publish. `limit` caps provider spend for a trial run."""
        detected = self.detect(min_size=min_size)
        kept = detected[: limit] if limit else detected
        build_id = str(uuid.uuid4())

        if dry_run:
            return CommunityBuildReport(
                build_id=build_id,
                detected=len(detected),
                kept=len(kept),
                summarised=0,
                indexed=0,
                failed=0,
                entities_covered=sum(c.size for c in kept),
                largest=max((c.size for c in kept), default=0),
                provider_calls=0,
            )

        # Each community is a paid call, so one unexpected failure must not discard the
        # ones already bought. `summarise` guards its own provider call; this guards
        # against anything it did not anticipate.
        summarised: list[Community] = []
        for community in kept:
            try:
                if self.summarise(community):
                    summarised.append(community)
            except Exception as exc:
                logger.warning(
                    "community_summary_error community=%s error_type=%s",
                    community.community_id,
                    type(exc).__name__,
                )
        failed = len(kept) - len(summarised)

        indexed = 0
        if summarised:
            self._write_graph(build_id, summarised)
            indexed = self._write_vectors(build_id, summarised)

        return CommunityBuildReport(
            build_id=build_id,
            detected=len(detected),
            kept=len(kept),
            summarised=len(summarised),
            indexed=indexed,
            failed=failed,
            entities_covered=sum(c.size for c in summarised),
            largest=max((c.size for c in summarised), default=0),
            provider_calls=len(kept),
        )

    def _write_graph(self, build_id: str, communities: list[Community]) -> None:
        """Replace the previous build. Communities are derived, so this is a swap."""
        created_at = datetime.now(timezone.utc).isoformat()
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                CREATE (c:Community {
                    id: row.id, build_id: $build_id, title: row.title,
                    summary: row.summary, size: row.size, doc_ids: row.doc_ids,
                    created_at: $created_at, algorithm: 'louvain'
                })
                WITH c, row
                UNWIND row.member_ids AS member_id
                MATCH (e:Entity {id: member_id})
                MERGE (e)-[:IN_COMMUNITY]->(c)
                """,
                {
                    "build_id": build_id,
                    "created_at": created_at,
                    "rows": [
                        {
                            "id": c.community_id,
                            "title": c.title,
                            "summary": c.summary,
                            "size": c.size,
                            "doc_ids": c.doc_ids,
                            "member_ids": c.member_ids,
                        }
                        for c in communities
                    ],
                },
            )
            session.run(
                """
                MATCH (c:Community) WHERE c.build_id <> $build_id
                DETACH DELETE c
                """,
                {"build_id": build_id},
            )

    def _write_vectors(self, build_id: str, communities: list[Community]) -> int:
        client = self._ensure_qdrant()
        if client is None:
            return 0
        from qdrant_client import models

        collection = self.settings.qdrant_communities_collection
        # Recreated per build: whatever the collection holds *is* the active build, so
        # the retriever needs no fence and a partial build cannot be half-read.
        try:
            client.delete_collection(collection)
        except Exception:
            pass
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=self.settings.embedding_dim, distance=models.Distance.COSINE
            ),
        )

        embedder = self._ensure_embedder()
        vectors = embedder.embed_texts(
            [c.embed_text() for c in communities],
            task_type=self.settings.gemini_embedding_task_type,
        )
        points = [
            models.PointStruct(
                id=c.community_id,
                vector=vector,
                payload={
                    "community_id": c.community_id,
                    "build_id": build_id,
                    "title": c.title,
                    "summary": c.summary,
                    "size": c.size,
                    "doc_ids": c.doc_ids,
                    "member_names": c.member_names[:50],
                },
            )
            for c, vector in zip(communities, vectors)
        ]
        batch = max(1, int(self.settings.qdrant_upsert_batch_size))
        for start in range(0, len(points), batch):
            client.upsert(collection_name=collection, points=points[start : start + batch])
        return len(points)

    # ----------------------------------------------------------------- plumbing
    def _ensure_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        if not self.settings.gemini_api_key:
            return None
        try:
            from google import genai
            from google.genai import types
        except Exception:
            return None
        self._types = types
        self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def _ensure_embedder(self) -> GeminiEmbedder:
        if self._embedder is None:
            self._embedder = GeminiEmbedder(self.settings)
        return self._embedder

    def _ensure_qdrant(self) -> Any | None:
        if self._qdrant is not None:
            return self._qdrant
        try:
            from qdrant_client import QdrantClient
        except Exception:
            return None
        self._qdrant = QdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key,
            timeout=self.settings.qdrant_timeout_seconds,
        )
        return self._qdrant

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Always a mapping. A model asked for one object sometimes returns a list of one."""
        text = (raw or "").strip()
        if not text:
            return {}
        parsed: Any = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = JSON_BLOCK_PATTERN.search(text)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = None
        if isinstance(parsed, list):
            parsed = next((item for item in parsed if isinstance(item, dict)), None)
        return parsed if isinstance(parsed, dict) else {}

    def close(self) -> None:
        driver = getattr(self, "_driver", None)
        if driver is not None:
            driver.close()
