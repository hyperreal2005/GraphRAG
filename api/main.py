"""The GraphRAG API (ROADMAP D3; architecture: `api/main.py`, FastAPI, localhost:8000).

    python main_api.py                       # http://127.0.0.1:8000, docs at /docs

Endpoints:

* `POST /query` - answer a question. Omit `session_id` to start a conversation; pass back the
  one returned to ask a follow-up (D2's session memory).
* `GET /sessions/{session_id}` - the turns a session remembers.
* `GET /documents`, `GET /documents/{doc_id}` - the corpus.
* `DELETE /documents/{doc_id}` - remove a document (D1), as a job.
* `POST /ingest` - ingest a file or directory under an allowed root, as a job.
* `GET /jobs`, `GET /jobs/{job_id}` - job state.
* `POST /jobs/communities`, `POST /jobs/heal-report` - run a scheduled job now (D4).
* `GET /schedule` - what is due: community staleness and the healer report.
* `GET /graph/overview` - the schema emergence dashboard (architecture 9.4) for the console.
* `GET /` - the web console (D5), a static page that calls the endpoints above.
* `GET /health` - the stores, the embedding contract, models and the corpus of record. It
  reports state and versions, never a key or a password.

Built by `create_app`, whose parts can be injected, so it is tested against fakes.
"""

from __future__ import annotations

import hmac
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api.jobs import JobRunner
from api.overview import GraphOverview
from observability.usage_ledger import summary as usage_summary
from background.community_job import run_community_build, staleness
from background.healer_job import run_heal_report
from background.scheduler import Scheduler, due_jobs
from config import Settings, default_metadata_db
from ingestion.state_store import MetadataStore
from models.query import GraphRAGResponse

logger = logging.getLogger(__name__)

SESSION_PATTERN = r"^[A-Za-z0-9_.:-]{1,100}$"
STATIC = Path(__file__).resolve().parent / "static"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, pattern=SESSION_PATTERN)


class QueryReply(GraphRAGResponse):
    session_id: str
    # What each cited document is - title, publisher, date - so a reader can judge a source
    # without a second request. Keyed by doc_id.
    documents: dict[str, dict[str, str]] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)
    recursive: bool = True


class Operations:
    """The write-side work behind the job endpoints, against the real stores."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def ingest(self, path: Path, recursive: bool) -> dict[str, Any]:
        from ingestion.pipeline import PhaseOnePipeline

        pipeline = PhaseOnePipeline(settings=self.settings)
        try:
            summary = pipeline.ingest_path(path, recursive=recursive).to_dict()
        finally:
            pipeline.close()
        return {
            key: summary[key] for key in ("processed", "changed", "skipped", "failed", "missing_sources")
        } | {
            "results": [
                {k: r.get(k) for k in ("source_path", "action", "doc_id", "chunk_count", "error")}
                for r in summary["results"]
            ]
        }

    def communities(self) -> dict[str, Any]:
        return run_community_build(self.settings, MetadataStore(self.settings.metadata_db_path))

    def heal_report(self) -> dict[str, Any]:
        return run_heal_report(self.settings, MetadataStore(self.settings.metadata_db_path))

    def remove(self, doc_id: str) -> dict[str, Any]:
        from ingestion.graph_writer import GraphWriter
        from ingestion.outbox import IngestionOutboxProcessor
        from ingestion.vector_indexer import QdrantVectorIndexer

        store = MetadataStore(self.settings.metadata_db_path)
        graph_writer = GraphWriter(self.settings)
        vector_indexer = QdrantVectorIndexer(self.settings)
        try:
            processor = IngestionOutboxProcessor(
                settings=self.settings, store=store, graph_writer=graph_writer, vector_indexer=vector_indexer
            )
            version_id = processor.remove_document(doc_id, run_id=f"api-{uuid.uuid4().hex[:8]}")
            event = next(
                (e for e in store.list_outbox_events(document_version_id=version_id) if e.event_type == "REMOVE_VERSION"),
                None,
            )
            return {
                "doc_id": doc_id,
                "removed_version_id": version_id,
                "retraction": event.status if event else "missing",
                "left_in_neo4j": graph_writer.verify_document_version(version_id),
                "left_in_qdrant": vector_indexer.count_document_version(version_id),
                "note": "Community summaries still reflect the document until they are rebuilt.",
            }
        finally:
            graph_writer.close()


def default_probes(settings: Settings) -> dict[str, Any]:
    """Each store's state. A probe that fails reports its error type, never the call's details."""
    stores: dict[str, Any] = {}
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        try:
            driver.verify_connectivity()
            stores["neo4j"] = {"ok": True}
        finally:
            driver.close()
    except Exception as exc:
        stores["neo4j"] = {"ok": False, "error": type(exc).__name__}
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=5)
        collections = {}
        for name in (
            settings.qdrant_chunks_collection,
            settings.qdrant_entities_collection,
            settings.qdrant_communities_collection,
        ):
            if not client.collection_exists(name):
                collections[name] = {"ok": False, "error": "missing"}
                continue
            points, _ = client.scroll(name, limit=1, with_payload=True, with_vectors=False)
            payload = (points[0].payload or {}) if points else {}
            stamp = payload.get("embedding_model_id")
            collections[name] = {
                "ok": stamp in (None, settings.embedding_model_id),
                "points": client.count(name, exact=False).count,
                "embedding_model_id": stamp,
            }
        stores["qdrant"] = {"ok": all(c["ok"] for c in collections.values()), "collections": collections}
    except Exception as exc:
        stores["qdrant"] = {"ok": False, "error": type(exc).__name__}
    try:
        import redis

        redis.Redis.from_url(settings.redis_url, socket_timeout=1.0, socket_connect_timeout=1.0).ping()
        stores["redis"] = {"ok": True}
    except Exception as exc:
        # Not a failure: the in-process cache stands in (D2). Reported so it is visible.
        stores["redis"] = {"ok": True, "fallback": "in-process cache", "error": type(exc).__name__}
    return stores


def create_app(
    settings: Settings | None = None,
    *,
    orchestrator: Any | None = None,
    store: MetadataStore | None = None,
    runner: JobRunner | None = None,
    operations: Any | None = None,
    probes: Callable[[Settings], dict[str, Any]] | None = None,
    scheduler: Scheduler | None = None,
    overview: GraphOverview | None = None,
) -> FastAPI:
    settings = settings or Settings(metadata_db_path=Path(default_metadata_db()))
    store = store or MetadataStore(settings.metadata_db_path)
    runner = runner or JobRunner()
    operations = operations or Operations(settings)
    probes = probes or default_probes
    state: dict[str, Any] = {"orchestrator": orchestrator}
    scheduled_jobs = {"communities": lambda: operations.communities(), "heal-report": lambda: operations.heal_report()}
    scheduler = scheduler or Scheduler(settings, store, runner, scheduled_jobs)
    overview = overview or GraphOverview(settings, store)

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # noqa: ANN202
        if state["orchestrator"] is None:
            from retrieval.query_orchestrator import QueryOrchestrator

            state["orchestrator"] = QueryOrchestrator(settings=settings, store=store)
        if settings.scheduler_enabled:
            scheduler.start()
        try:
            yield
        finally:
            scheduler.stop()
            closer = getattr(state["orchestrator"], "aclose", None)
            if closer is not None:
                await closer()
            runner.stop()

    app = FastAPI(
        title="GraphRAG",
        version="1.0",
        description="Question answering over a versioned knowledge graph and vector index.",
        lifespan=lifespan,
    )

    def authorised(request: Request) -> None:
        if not settings.api_token:
            return
        header = request.headers.get("authorization", "")
        if not hmac.compare_digest(header.encode(), f"Bearer {settings.api_token}".encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "A valid bearer token is required.")

    guard = [Depends(authorised)]

    @app.middleware("http")
    async def console_revalidates(request: Request, call_next):  # noqa: ANN001, ANN202
        # The console's files change with the code; without this a browser keeps serving an
        # old page after an update (seen: a stale index.html beside a fresh stylesheet).
        response = await call_next(request)
        if request.url.path.startswith("/console"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("api_unhandled error_type=%s", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": f"Internal error: {type(exc).__name__}"})

    @app.post("/query", response_model=QueryReply, dependencies=guard)
    async def query(body: QueryRequest) -> QueryReply:
        session_id = body.session_id or f"api-{uuid.uuid4().hex[:12]}"
        response = await state["orchestrator"].query(user_query=body.question, session_id=session_id)
        cards = store.get_document_cards(c.doc_id for c in response.citations if c.doc_id)
        documents = {
            doc_id: {"title": card.title, "publisher": card.publisher, "published": card.published}
            for doc_id, card in cards.items()
        }
        return QueryReply(**response.to_dict(), session_id=session_id, documents=documents)

    @app.get("/sessions/{session_id}", dependencies=guard)
    async def session(session_id: str) -> dict[str, Any]:
        cache = getattr(state["orchestrator"], "cache", None)
        turns = cache.get_session_entries(session_id, limit=int(settings.session_max_turns)) if cache else []
        return {
            "session_id": session_id,
            "turns": [
                {k: t.get(k) for k in ("query", "standalone_query", "answer", "timestamp")} for t in turns
            ],
        }

    @app.get("/documents", dependencies=guard)
    async def documents(
        status_: str | None = Query(default=None, alias="status"),
        search: str | None = Query(default=None, max_length=200),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        total, rows = store.document_overview(status=status_, search=search, limit=limit, offset=offset)
        return {"total": total, "limit": limit, "offset": offset, "documents": [dict(row) for row in rows]}

    @app.get("/documents/{doc_id}", dependencies=guard)
    async def document(doc_id: str) -> dict[str, Any]:
        row = store.get_document_by_id(doc_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document.")
        versions = store.list_document_versions(doc_id)
        active = next((v for v in versions if v.status == "active"), None)
        return {
            "doc_id": doc_id,
            "title": row["title"],
            "publisher": row["publisher"],
            "published": row["published"],
            "format": row["format"],
            "status": row["status"],
            "source_path": row["source_path"],
            "source_exists": Path(str(row["source_path"])).exists(),
            "active_version_id": active.version_id if active else None,
            "active_chunks": len(store.get_chunks_for_doc(doc_id, active.version_id, active_only=True)) if active else 0,
            "coverage": store.get_document_coverage(doc_id),
            "last_error": row["last_error"],
            "versions": [
                {"version_id": v.version_id, "status": v.status, "created_at": str(v.created_at)} for v in versions
            ],
        }

    @app.delete("/documents/{doc_id}", status_code=status.HTTP_202_ACCEPTED, dependencies=guard)
    async def remove_document(doc_id: str) -> dict[str, Any]:
        if store.get_document_by_id(doc_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document.")
        if store.get_active_document_version(doc_id) is None:
            raise HTTPException(status.HTTP_409_CONFLICT, "The document has no active version to remove.")
        job = runner.submit("remove", {"doc_id": doc_id}, lambda: operations.remove(doc_id))
        return job.to_dict()

    @app.post("/ingest", status_code=status.HTTP_202_ACCEPTED, dependencies=guard)
    async def ingest(body: IngestRequest) -> dict[str, Any]:
        path = Path(body.path).resolve()
        roots = [Path(root).resolve() for root in settings.api_ingest_roots]
        if not any(path == root or path.is_relative_to(root) for root in roots):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Ingest is limited to paths under: {', '.join(settings.api_ingest_roots)}.",
            )
        if not path.exists():
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No such file or directory.")
        job = runner.submit(
            "ingest",
            {"path": body.path, "recursive": body.recursive},
            lambda: operations.ingest(path, body.recursive),
        )
        return job.to_dict()

    @app.get("/jobs", dependencies=guard)
    async def jobs(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        return {"jobs": [job.to_dict() for job in runner.list(limit)]}

    @app.get("/jobs/{job_id}", dependencies=guard)
    async def job(job_id: str) -> dict[str, Any]:
        found = runner.get(job_id)
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "No such job.")
        return found.to_dict()

    @app.post("/jobs/{kind}", status_code=status.HTTP_202_ACCEPTED, dependencies=guard)
    async def run_job(kind: str) -> dict[str, Any]:
        if kind not in scheduled_jobs:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown job; one of: {', '.join(scheduled_jobs)}.")
        return runner.submit(kind, {"trigger": "on_demand"}, scheduled_jobs[kind]).to_dict()

    @app.get("/schedule", dependencies=guard)
    async def schedule() -> dict[str, Any]:
        from datetime import datetime

        due = due_jobs(settings, store, datetime.now())
        return {"scheduler_enabled": settings.scheduler_enabled, "tick_seconds": settings.scheduler_tick_seconds, **due}

    @app.get("/graph/overview", dependencies=guard)
    async def graph_overview() -> dict[str, Any]:
        import asyncio

        return await asyncio.to_thread(overview.get)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/console/")

    # The console holds no data - every figure it shows comes through the guarded endpoints -
    # so its files are served without the token, which it asks for when a call is refused.
    if STATIC.is_dir():
        app.mount("/console", StaticFiles(directory=STATIC, html=True), name="console")

    @app.get("/health", dependencies=guard)
    async def health() -> dict[str, Any]:
        total, _ = store.document_overview(limit=1)
        active, _ = store.document_overview(status="complete", limit=1)
        stores = {"sqlite": {"ok": True, "documents": total, "complete": active}} | probes(settings)
        return {
            "status": "ok" if all(s.get("ok") for s in stores.values()) else "degraded",
            "corpus": {
                "metadata_db": str(settings.metadata_db_path),
                "collections": {
                    "chunks": settings.qdrant_chunks_collection,
                    "entities": settings.qdrant_entities_collection,
                    "communities": settings.qdrant_communities_collection,
                },
            },
            "stores": stores,
            "communities": {
                k: v for k, v in staleness(store, settings.community_rebuild_every_n_documents).items()
                if k != "versions"
            },
            "coverage": store.coverage_summary(),
            "cohere": usage_summary(settings),
            "models": {
                "generation": settings.gemini_model,
                "ingestion": settings.ingestion_model,
                "embedding": settings.embedding_model_id,
                "embedding_version": settings.embedding_version,
                "rerank": settings.query_rerank_model if settings.query_rerank_enabled else None,
            },
            "prompts": {
                "generation": settings.generation_prompt_version,
                "extraction": settings.extraction_prompt_version,
            },
            "features": {
                "session_memory": settings.session_memory_enabled,
                "query_cache": settings.query_cache_enabled,
                "polar_verdicts": settings.generation_polar_verdicts,
            },
        }

    return app
