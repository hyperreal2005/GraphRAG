"""Query reranking adapter with Cohere and deterministic fallback."""

from __future__ import annotations

from typing import Any

from config import Settings
from models.query import RetrieverResult


class QueryReranker:
    """Apply reranking over fused retrieval candidates."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client
        self._cohere_module = None

        if self._client is not None:
            return
        if not self.settings.query_rerank_enabled or not self.settings.cohere_api_key:
            return

        try:
            import cohere
        except Exception:
            return
        self._cohere_module = cohere
        self._client = self._build_client()

    def rerank(self, query: str, items: list[RetrieverResult]) -> tuple[list[RetrieverResult], str]:
        """Return reranked items plus mode marker."""
        if not self.settings.query_rerank_enabled:
            return list(items), "disabled"

        clean_query = query.strip()
        if len(items) <= 1 or not clean_query:
            return list(items), "disabled"

        if self._client is None:
            return list(items), "deterministic_fallback"

        top_n = min(len(items), max(1, int(self.settings.query_rerank_top_n)))
        documents = [self._document_text(item) for item in items]
        try:
            response = self._client.rerank(
                model=self.settings.query_rerank_model,
                query=clean_query,
                documents=documents,
                top_n=top_n,
            )
            ranking = self._extract_ranking(response)
        except Exception:
            return list(items), "deterministic_fallback"

        if not ranking:
            return list(items), "deterministic_fallback"

        reranked: list[RetrieverResult] = []
        seen_indices: set[int] = set()
        for index, score in ranking:
            if index < 0 or index >= len(items) or index in seen_indices:
                continue
            item = items[index]
            item.score_reranked = float(score)
            reranked.append(item)
            seen_indices.add(index)

        remaining = [items[index] for index in range(len(items)) if index not in seen_indices]
        reranked.extend(remaining)
        return reranked, "cohere"

    def _build_client(self) -> Any | None:
        if self._cohere_module is None or not self.settings.cohere_api_key:
            return None
        api_key = self.settings.cohere_api_key

        client_v2 = getattr(self._cohere_module, "ClientV2", None)
        if client_v2 is not None:
            try:
                return client_v2(api_key=api_key)
            except TypeError:
                try:
                    return client_v2(api_key)
                except Exception:
                    pass
            except Exception:
                pass

        client_v1 = getattr(self._cohere_module, "Client", None)
        if client_v1 is not None:
            try:
                return client_v1(api_key=api_key)
            except TypeError:
                try:
                    return client_v1(api_key)
                except Exception:
                    pass
            except Exception:
                pass

        return None

    def _extract_ranking(self, response: Any) -> list[tuple[int, float]]:
        results = None
        if isinstance(response, dict):
            results = response.get("results") or response.get("data")
        if results is None:
            results = getattr(response, "results", None)
        if results is None:
            results = getattr(response, "data", None)
        if not isinstance(results, list):
            return []

        ranking: list[tuple[int, float]] = []
        for item in results:
            if isinstance(item, dict):
                index = item.get("index")
                score = item.get("relevance_score", item.get("score"))
            else:
                index = getattr(item, "index", None)
                score = getattr(item, "relevance_score", None)
                if score is None:
                    score = getattr(item, "score", None)
            if index is None or score is None:
                continue
            try:
                ranking.append((int(index), float(score)))
            except (TypeError, ValueError):
                continue
        return ranking

    def _document_text(self, item: RetrieverResult) -> str:
        if item.source_type == "vector":
            doc_id = item.provenance.get("doc_id")
            chunk_id = item.provenance.get("chunk_id")
            section = item.provenance.get("section_heading")
            return (
                "[VECTOR]\n"
                f"doc_id={doc_id} chunk_id={chunk_id} section={section}\n"
                f"{item.text_or_path}"
            )
        if item.source_type == "graph":
            return f"[GRAPH]\n{item.text_or_path}"
        if item.source_type == "community":
            return f"[COMMUNITY]\n{item.text_or_path}"
        return item.text_or_path
