"""Query reranking: Cohere, a local cross-encoder, or nothing.

The third case used to be called "deterministic fallback", which returned the candidate
list in the order it arrived - no reranking at all. Since no Cohere key has ever been
configured on this project, that meant Phase 6's reranking stage never executed, and its
name hid the fact in every log line it wrote. The local cross-encoder is the architecture's
own stated alternative and needs no key, so the no-op is now a last resort rather than the
normal path.
"""

from __future__ import annotations

import logging
from typing import Any

from config import Settings
from models.query import RetrieverResult

logger = logging.getLogger(__name__)

# Loading a cross-encoder costs seconds, and a FusionEngine is constructed per query in
# some paths. Keyed by model name so a settings change is still honoured.
_LOCAL_MODELS: dict[str, Any] = {}
_LOCAL_FAILED: set[str] = set()


class QueryReranker:
    """Apply reranking over fused retrieval candidates."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client
        self._cohere_module = None
        self._llm = None
        self._llm_types = None

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

    def _local_model(self) -> Any | None:
        """The cross-encoder, loaded once per process. None if it cannot be had.

        Loaded lazily rather than in __init__ because the first call downloads weights,
        and a FusionEngine is built in contexts that never rerank anything.
        """
        name = str(self.settings.query_rerank_local_model or "").strip()
        if not name or name in _LOCAL_FAILED:
            return None
        if name in _LOCAL_MODELS:
            return _LOCAL_MODELS[name]
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(name, max_length=512)
        except Exception as exc:
            # No weights cached and no network, or the package is absent. Recorded once;
            # retrying per query would add the failure cost to every request.
            logger.warning(
                "local_reranker_unavailable model=%s error_type=%s", name, type(exc).__name__
            )
            _LOCAL_FAILED.add(name)
            return None
        _LOCAL_MODELS[name] = model
        return model

    def _rerank_local(
        self, query: str, items: list[RetrieverResult]
    ) -> tuple[list[RetrieverResult], str] | None:
        model = self._local_model()
        if model is None:
            return None

        limit = max(1, int(self.settings.query_rerank_local_max_items))
        scored, overflow = items[:limit], items[limit:]
        pairs = [(query, self._document_text(item)[:2000]) for item in scored]
        try:
            scores = model.predict(pairs)
        except Exception as exc:
            logger.warning("local_rerank_failed error_type=%s", type(exc).__name__)
            return None

        for item, score in zip(scored, scores):
            item.score_reranked = float(score)
        ordered = sorted(scored, key=lambda i: float(i.score_reranked or 0.0), reverse=True)
        # Items past the cap keep score_reranked None and sort after, by fused score.
        return [*ordered, *overflow], "local_cross_encoder"

    def _rerank_llm(
        self, query: str, items: list[RetrieverResult]
    ) -> tuple[list[RetrieverResult], str] | None:
        """Ask the generation model to order the top candidates. One call per query.

        A cross-encoder trained on short web queries measured *worse* than cosine on this
        corpus's long compositional questions. The generation model already reads these
        questions well enough to answer them, so it is the obvious candidate to order
        evidence for them, at the cost of one extra call. Candidates the model omits keep
        their fused order after the ones it ranked; a malformed reply falls through.
        """
        client = self._llm_client()
        if client is None:
            return None
        top_n = max(1, int(self.settings.query_rerank_llm_top_n))
        head, tail = list(items[:top_n]), list(items[top_n:])
        if len(head) <= 1:
            return None

        snippet_chars = max(80, int(self.settings.query_rerank_llm_snippet_chars))
        lines = []
        for index, item in enumerate(head):
            text = " ".join(self._document_text(item).split())[:snippet_chars]
            lines.append(f"[{index}] {text}")
        prompt = (
            "Rank these candidate passages by how directly they help answer the question. "
            "Consider every candidate. Return JSON only: "
            '{"ranking": [<candidate numbers, most helpful first>]}\n\n'
            f"QUESTION: {query}\n\nCANDIDATES:\n" + "\n".join(lines)
        )
        try:
            from generation.model_adapter import build_generate_config

            response = client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config=build_generate_config(
                    self._llm_types,
                    self.settings.gemini_model,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            import json

            payload = json.loads(getattr(response, "text", "") or "{}")
            order = [int(i) for i in (payload.get("ranking") or []) if str(i).lstrip("-").isdigit()]
        except Exception as exc:
            logger.warning("llm_rerank_failed error_type=%s", type(exc).__name__)
            return None

        seen: set[int] = set()
        ranked: list[RetrieverResult] = []
        for position, index in enumerate(order):
            if 0 <= index < len(head) and index not in seen:
                seen.add(index)
                head[index].score_reranked = 1.0 / (position + 1)
                ranked.append(head[index])
        if not ranked:
            return None
        rest = [item for index, item in enumerate(head) if index not in seen]
        return [*ranked, *rest, *tail], "llm"

    def _llm_client(self) -> Any | None:
        if getattr(self, "_llm", None) is not None:
            return self._llm
        if not self.settings.query_rerank_llm_enabled or not self.settings.gemini_api_key:
            return None
        try:
            from google import genai
            from google.genai import types
        except Exception:
            return None
        self._llm_types = types
        self._llm = genai.Client(api_key=self.settings.gemini_api_key)
        return self._llm

    def _degrade(
        self, query: str, items: list[RetrieverResult]
    ) -> tuple[list[RetrieverResult], str]:
        """Cohere refused or failed: try the remaining tiers before giving up on ordering."""
        if self.settings.query_rerank_llm_enabled:
            ranked = self._rerank_llm(query, list(items))
            if ranked is not None:
                return ranked
        if self.settings.query_rerank_local_enabled:
            local = self._rerank_local(query, list(items))
            if local is not None:
                return local
        return list(items), "unranked_passthrough"

    def rerank(self, query: str, items: list[RetrieverResult]) -> tuple[list[RetrieverResult], str]:
        """Return reranked items plus mode marker."""
        if not self.settings.query_rerank_enabled:
            return list(items), "disabled"

        clean_query = query.strip()
        if len(items) <= 1 or not clean_query:
            return list(items), "disabled"

        if self._client is None:
            return self._degrade(clean_query, items)

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
            return self._degrade(clean_query, items)

        if not ranking:
            return self._degrade(clean_query, items)

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
