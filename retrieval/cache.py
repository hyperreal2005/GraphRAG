"""Cache abstraction for query pipeline."""

from __future__ import annotations

from typing import Any


class NoOpCache:
    """Cache implementation: always miss, ignore writes."""

    def get_query(self, query_hash: str) -> dict[str, Any] | None:
        _ = query_hash
        return None

    def set_query(self, query_hash: str, response_payload: dict[str, Any], ttl_seconds: int) -> None:
        _ = query_hash
        _ = response_payload
        _ = ttl_seconds

    def append_session_entry(self, session_id: str, entry: dict[str, Any]) -> None:
        _ = session_id
        _ = entry

    def get_session_entries(self, session_id: str, limit: int = 5) -> list[dict[str, Any]]:
        _ = session_id
        _ = limit
        return []

