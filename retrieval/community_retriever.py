"""Community retrieval placeholder for Phase 4."""

from __future__ import annotations

from config import Settings
from models.query import RetrieverResult


class CommunityRetriever:
    """Stub: community retrieval placeholder."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrieverResult]:
        _ = query
        _ = top_k
        return []

