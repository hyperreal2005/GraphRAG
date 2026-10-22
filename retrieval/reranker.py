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
import threading
import time
from typing import Any

from config import Settings
from observability.usage_ledger import record_cohere_call
from models.query import RetrieverResult

logger = logging.getLogger(__name__)

_COHERE_PACE_LOCK = threading.Lock()
_COHERE_LAST_CALL_AT = 0.0


def _pace_cohere(min_interval: float, *, max_wait: float = float("inf")) -> bool:
    """Process-wide minimum gap between Cohere calls. Off (0) by default.

    Trial keys are rate-limited per minute. Pacing keeps a run under the limit instead of
    relying on 429s, each of which would otherwise degrade that query to no reranking.
    Returns False - without sleeping or taking the slot - when the required wait exceeds
    `max_wait`, so a caller with a deadline can degrade instead of overrunning it.
    """
    if min_interval <= 0:
        return max_wait >= 0
    global _COHERE_LAST_CALL_AT
    with _COHERE_PACE_LOCK:
        wait = _COHERE_LAST_CALL_AT + min_interval - time.monotonic()
        if wait > max_wait:
            return False
        if wait > 0:
            time.sleep(wait)
        _COHERE_LAST_CALL_AT = time.monotonic()
        return True


def _is_rate_limited(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    text = f"{type(exc).__name__} {exc}"
    return "429" in text or "TooManyRequests" in text or "rate limit" in text.lower()


def _is_stall(exc: Exception) -> bool:
    """A request that hung or dropped, not one that was refused. A stall is usually gone on
    a second try: 5 of 50 questions in one run lost their reranking to single stalls that a
    retry inside the same deadline would have absorbed (REFACTOR3 WP23)."""
    name = type(exc).__name__
    return "Timeout" in name or name in {"ReadError", "RemoteProtocolError", "ConnectError"}

# Loading a cross-encoder costs seconds, and a FusionEngine is constructed per query in
# some paths. Keyed by model name so a settings change is still honoured.
_LOCAL_MODELS: dict[str, Any] = {}
_LOCAL_FAILED: set[str] = set()


class QueryReranker:
    """Apply reranking over fused retrieval candidates."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client
        self.last_timing: dict[str, float] = {}
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
        # ROADMAP F1/F2: how much of the stage is the trial key's pacing and how much the call.
        timing = {"rerank_pacing_wait": 0.0, "rerank_call": 0.0}
        self.last_timing = timing
        if not self.settings.query_rerank_enabled:
            return list(items), "disabled"

        clean_query = query.strip()
        if len(items) <= 1 or not clean_query:
            return list(items), "disabled"

        if self._client is None:
            return self._degrade(clean_query, items)

        top_n = min(len(items), max(1, int(self.settings.query_rerank_top_n)))
        documents = [self._document_text(item) for item in items]
        # A throttled call is retried inside a small total budget, then degrades. The
        # budget must sit well under the fusion stage's deadline: overrunning it loses the
        # whole fused context, which is far worse than one unreranked query. The same
        # mistake was made once already, in the embedder (docs/ERRORS.md 2026-09-05).
        budget = max(0.0, float(self.settings.query_rerank_rate_limit_max_total_seconds))
        request_timeout = max(1.0, float(self.settings.query_rerank_request_timeout_seconds))
        # One deadline for pacing, requests and retries together, fixed at entry. The
        # fusion stage that contains this call has its own deadline, and overrunning it
        # discards the whole fused context: better to return fused order unreranked than
        # to lose the answer. Each step checks it can still finish before starting.
        deadline = time.monotonic() + max(1.0, float(self.settings.query_rerank_total_budget_seconds))
        spent = 0.0
        response = None
        for attempt in range(4):
            remaining = deadline - time.monotonic()
            paced_from = time.perf_counter()
            paced = _pace_cohere(
                float(self.settings.query_rerank_min_interval_seconds),
                max_wait=remaining - request_timeout,
            )
            timing["rerank_pacing_wait"] += (time.perf_counter() - paced_from) * 1000.0
            if not paced:
                logger.warning("cohere_rerank_degraded reason=pacing_would_exceed_deadline")
                return self._degrade(clean_query, items)
            # ROADMAP F2: every attempt that reaches Cohere is counted against the key.
            record_cohere_call(self.settings)
            called_from = time.perf_counter()
            try:
                response = self._client.rerank(
                    model=self.settings.query_rerank_model,
                    query=clean_query,
                    documents=documents,
                    top_n=top_n,
                )
                timing["rerank_call"] += (time.perf_counter() - called_from) * 1000.0
                break
            except Exception as exc:
                timing["rerank_call"] += (time.perf_counter() - called_from) * 1000.0
                stall = _is_stall(exc)
                # A throttle backs off exponentially; a stall retries almost at once.
                delay = 0.5 if stall else 1.0 * (2**attempt)
                too_late = time.monotonic() + delay + request_timeout > deadline
                retryable = _is_rate_limited(exc) or stall
                if not retryable or spent + delay > budget or too_late:
                    logger.warning(
                        "cohere_rerank_degraded error_type=%s rate_limited=%s",
                        type(exc).__name__,
                        _is_rate_limited(exc),
                    )
                    return self._degrade(clean_query, items)
                time.sleep(delay)
                spent += delay
        if response is None:
            return self._degrade(clean_query, items)
        try:
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
                # A per-request ceiling. Without one a stalled call waits on the SDK's own
                # default, the fusion stage's deadline fires first, and the whole fused
                # context is discarded - seven questions in one judged run, 2026-09-26.
                return client_v2(
                    api_key=api_key,
                    timeout=max(1.0, float(self.settings.query_rerank_request_timeout_seconds)),
                )
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
        # The document card, when fusion attached one: a question naming "the TechCrunch
        # report of October 31" can only be ranked against passages that say whose report
        # they are (WP23).
        document = str(item.provenance.get("document") or "")
        document_line = f"Document: {document}\n" if document else ""
        if item.source_type == "vector":
            doc_id = item.provenance.get("doc_id")
            chunk_id = item.provenance.get("chunk_id")
            section = item.provenance.get("section_heading")
            return (
                "[VECTOR]\n"
                f"{document_line}"
                f"doc_id={doc_id} chunk_id={chunk_id} section={section}\n"
                f"{item.text_or_path}"
            )
        if item.source_type == "graph":
            return f"[GRAPH]\n{document_line}{item.text_or_path}"
        if item.source_type == "community":
            return f"[COMMUNITY]\n{item.text_or_path}"
        return item.text_or_path
