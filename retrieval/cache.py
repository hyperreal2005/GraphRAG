"""Query cache and session chains (architecture 4.3, Phase 9; ROADMAP D2).

Three implementations of one interface:

* `NoOpCache` - always misses, keeps nothing. The default, and what every evaluation uses.
* `RedisCache` - the architecture's store: `query_cache:{key}` holds a response as JSON with a
  TTL, and `session:{id}:chain` the session's last turns.
* `MemoryCache` - the same behaviour inside one process, used when Redis cannot be reached, so
  a chat still remembers its turns and a query never fails over its cache.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)


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


class MemoryCache:
    """Process-local cache with the Redis cache's semantics: TTLs and a bounded chain."""

    def __init__(self, max_turns: int = 10, session_ttl_seconds: int = 86400) -> None:
        self.max_turns = max(1, int(max_turns))
        self.session_ttl_seconds = max(1, int(session_ttl_seconds))
        self._queries: dict[str, tuple[float, dict[str, Any]]] = {}
        self._sessions: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._lock = threading.Lock()

    def get_query(self, query_hash: str) -> dict[str, Any] | None:
        with self._lock:
            hit = self._queries.get(query_hash)
            if hit is None or hit[0] < time.monotonic():
                self._queries.pop(query_hash, None)
                return None
            return json.loads(json.dumps(hit[1]))

    def set_query(self, query_hash: str, response_payload: dict[str, Any], ttl_seconds: int) -> None:
        with self._lock:
            self._queries[query_hash] = (
                time.monotonic() + max(1, int(ttl_seconds)),
                json.loads(json.dumps(response_payload, default=str)),
            )

    def append_session_entry(self, session_id: str, entry: dict[str, Any]) -> None:
        with self._lock:
            _, chain = self._live_chain(session_id)
            chain = [*chain, {**entry, "timestamp": entry.get("timestamp") or time.time()}][-self.max_turns:]
            self._sessions[session_id] = (time.monotonic() + self.session_ttl_seconds, chain)

    def get_session_entries(self, session_id: str, limit: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            _, chain = self._live_chain(session_id)
            return [dict(item) for item in chain[-max(1, int(limit)):]]

    def _live_chain(self, session_id: str) -> tuple[float, list[dict[str, Any]]]:
        expires, chain = self._sessions.get(session_id, (0.0, []))
        if expires < time.monotonic():
            self._sessions.pop(session_id, None)
            return 0.0, []
        return expires, chain


class RedisCache:
    """The architecture's Redis store. Any Redis failure reads as a miss and logs a warning:
    a cache that is down must cost recomputation, never an answer."""

    def __init__(self, client: Any, max_turns: int = 10, session_ttl_seconds: int = 86400) -> None:
        self._client = client
        self.max_turns = max(1, int(max_turns))
        self.session_ttl_seconds = max(1, int(session_ttl_seconds))

    def get_query(self, query_hash: str) -> dict[str, Any] | None:
        try:
            raw = self._client.get(f"query_cache:{query_hash}")
        except Exception as exc:
            logger.warning("redis_cache_get_failed error_type=%s", type(exc).__name__)
            return None
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def set_query(self, query_hash: str, response_payload: dict[str, Any], ttl_seconds: int) -> None:
        try:
            self._client.setex(
                f"query_cache:{query_hash}",
                max(1, int(ttl_seconds)),
                json.dumps(response_payload, default=str),
            )
        except Exception as exc:
            logger.warning("redis_cache_set_failed error_type=%s", type(exc).__name__)

    def append_session_entry(self, session_id: str, entry: dict[str, Any]) -> None:
        key = f"session:{session_id}:chain"
        record = json.dumps({**entry, "timestamp": entry.get("timestamp") or time.time()}, default=str)
        try:
            self._client.rpush(key, record)
            self._client.ltrim(key, -self.max_turns, -1)
            self._client.expire(key, self.session_ttl_seconds)
        except Exception as exc:
            logger.warning("redis_session_append_failed error_type=%s", type(exc).__name__)

    def get_session_entries(self, session_id: str, limit: int = 5) -> list[dict[str, Any]]:
        try:
            raw = self._client.lrange(f"session:{session_id}:chain", -max(1, int(limit)), -1)
        except Exception as exc:
            logger.warning("redis_session_read_failed error_type=%s", type(exc).__name__)
            return []
        entries = []
        for item in raw or []:
            try:
                value = json.loads(item)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                entries.append(value)
        return entries


def build_cache(settings: Any) -> NoOpCache | MemoryCache | RedisCache:
    """The cache the settings ask for: none unless caching or session memory is on; Redis when
    it answers a ping; otherwise the in-process cache, with a warning."""
    if not (settings.query_cache_enabled or settings.session_memory_enabled):
        return NoOpCache()
    max_turns = int(settings.session_max_turns)
    session_ttl = int(settings.session_ttl_seconds)
    url = str(settings.redis_url or "").strip()
    if url:
        try:
            import redis

            client = redis.Redis.from_url(url, socket_timeout=1.0, socket_connect_timeout=1.0, decode_responses=True)
            client.ping()
            return RedisCache(client, max_turns=max_turns, session_ttl_seconds=session_ttl)
        except Exception as exc:
            logger.warning("redis_unavailable error_type=%s; using the in-process cache", type(exc).__name__)
    return MemoryCache(max_turns=max_turns, session_ttl_seconds=session_ttl)
