"""CLI entrypoint for  query pipeline.

    python main_query.py --query "Who founded Acme?"           # one question, JSON out
    python main_query.py --chat                                # a conversation (ROADMAP D2)
    python main_query.py --chat --session-id my-session        # resume a conversation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv

from config import Settings, default_metadata_db
from models.query import GraphRAGResponse
from retrieval.query_orchestrator import QueryOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG query pipeline")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", help="User question to answer")
    mode.add_argument(
        "--chat",
        action="store_true",
        help="Ask questions in turn; a follow-up is read in the light of the earlier ones",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session identifier; reuse one to continue its conversation (default: a new session)",
    )
    parser.add_argument(
        "--metadata-db",
        default=default_metadata_db(),
        help="SQLite metadata database path",
    )
    return parser


async def _run_query(orchestrator: QueryOrchestrator, query: str, session_id: str) -> dict[str, object]:
    try:
        response = await orchestrator.query(user_query=query, session_id=session_id)
        return response.to_dict()
    finally:
        await orchestrator.aclose()


def render(response: GraphRAGResponse) -> str:
    """A response as a reader wants it in a terminal: the answer, then where it came from."""
    lines = []
    if response.interpreted_query:
        lines.append(f"(read as: {response.interpreted_query})")
    lines.append(response.answer)
    for citation in response.citations:
        where = citation.doc_name or citation.doc_id or citation.chunk_id
        page = f", p. {citation.page}" if citation.page else ""
        lines.append(f"  [{citation.citation_id.removeprefix('citation:')[:8]}] {where}{page}")
    notes = [w for w in response.warnings if w != "insufficient_evidence"]
    if response.cached:
        notes.append("from cache")
    if notes:
        lines.append(f"  ({'; '.join(notes)})")
    return "\n".join(lines)


async def _chat(orchestrator: QueryOrchestrator, session_id: str) -> None:
    print(f"Session {session_id}. Ask a question; an empty line or 'exit' ends the chat.")
    try:
        while True:
            try:
                question = await asyncio.to_thread(input, "\n> ")
            except EOFError:
                break
            if question.strip().lower() in {"", "exit", "quit"}:
                break
            response = await orchestrator.query(user_query=question, session_id=session_id)
            print(render(response))
    finally:
        await orchestrator.aclose()


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings(metadata_db_path=Path(args.metadata_db))
    # ROADMAP D2, measured and adopted: a follow-up is read in the light of its session, and a
    # repeated question is answered from the cache until the corpus or configuration changes.
    # On here, where conversations happen; the library default stays off, so tests and
    # evaluations never touch Redis.
    settings.session_memory_enabled = True
    settings.query_cache_enabled = True
    if args.chat:
        orchestrator = QueryOrchestrator(settings=settings)
        asyncio.run(_chat(orchestrator, args.session_id or f"chat-{uuid.uuid4().hex[:8]}"))
        return 0

    orchestrator = QueryOrchestrator(settings=settings)
    # A one-shot question starts its own session unless told to continue one.
    session_id = args.session_id or f"cli-{uuid.uuid4().hex[:8]}"
    payload = asyncio.run(_run_query(orchestrator, query=args.query, session_id=session_id))

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
