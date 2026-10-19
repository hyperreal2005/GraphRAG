"""Serve the GraphRAG API (ROADMAP D3).

    python main_api.py                                  # http://127.0.0.1:8000, docs at /docs
    python main_api.py --port 8080 --metadata-db data/refactor3_wp24/metadata.db

It binds to localhost. Binding anywhere else needs API_TOKEN in the environment, and then every
request must send `Authorization: Bearer <token>`.
"""

from __future__ import annotations

import argparse
import ipaddress
import sys
from pathlib import Path

from dotenv import load_dotenv

from config import Settings, default_metadata_db


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the GraphRAG API")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument("--metadata-db", default=default_metadata_db(), help="SQLite metadata database path")
    return parser


def main() -> int:
    load_dotenv()
    defaults = Settings()
    args = build_parser(defaults).parse_args()
    settings = Settings(metadata_db_path=Path(args.metadata_db))
    if not is_loopback(args.host) and not settings.api_token:
        print(f"Refusing to bind {args.host} without API_TOKEN: the API would be open to the network.")
        return 2
    # ROADMAP D2, adopted: the API is where conversations happen.
    settings.session_memory_enabled = True
    settings.query_cache_enabled = True
    # ROADMAP D4: community rebuilds and the nightly healer report run inside the API process.
    settings.scheduler_enabled = True

    import uvicorn

    from api.main import create_app

    # On Windows uvicorn defaults to the Proactor loop, whose accept loop dies on WinError 64
    # when a client drops a connection mid-request: the process lives on but stops listening.
    # Seen live (a browser killed during a page load). The selector loop has no such failure.
    loop = "asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto"
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info", loop=loop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
