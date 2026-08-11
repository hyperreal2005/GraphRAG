"""CLI entrypoint for  query pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from config import Settings
from retrieval.query_orchestrator import QueryOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG query pipeline")
    parser.add_argument("--query", required=True, help="User question to answer")
    parser.add_argument(
        "--session-id",
        default="default",
        help="Session identifier for query-chain tracking",
    )
    parser.add_argument(
        "--metadata-db",
        default="data/metadata.db",
        help="SQLite metadata database path",
    )
    return parser


async def _run_query(orchestrator: QueryOrchestrator, query: str, session_id: str) -> dict[str, object]:
    try:
        response = await orchestrator.query(user_query=query, session_id=session_id)
        return response.to_dict()
    finally:
        await orchestrator.aclose()


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings(metadata_db_path=Path(args.metadata_db))
    orchestrator = QueryOrchestrator(settings=settings)
    payload = asyncio.run(_run_query(orchestrator, query=args.query, session_id=args.session_id))

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
