"""Gemini embedding service for vector indexing.

WP2-E (REVIEW1 finding B10): behaviour is selected by explicit capability flags rather
than by comparing the configured model id against a literal. The previous code branched
on `embedding_model_id != "gemini-embedding-2"`, so setting the id Google actually
documents — `gemini-embedding-2-preview` — silently reverted the pipeline to legacy
behaviour: it began sending a `task_type` the model rejects and stopped applying the
instruction prefixes the model requires. A config change that looks correct should not
corrupt the corpus.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from typing import Any

from config import Settings

logger = logging.getLogger(__name__)

_PACE_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0


def _pace(min_interval: float) -> None:
    """Hold a process-wide minimum interval between embedding requests.

    Lowering concurrency does not help against a *per-minute* quota, and on the query path
    there is no worker count to lower anyway: entity linking, vector retrieval and
    community retrieval are dispatched together by one `asyncio.gather`, and
    `gemini_max_workers` governs ingestion only. Pacing is the lever that actually reduces
    requests per minute, and it applies wherever the call comes from.
    """
    if min_interval <= 0:
        return
    global _LAST_REQUEST_AT
    with _PACE_LOCK:
        wait = _LAST_REQUEST_AT + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST_AT = time.monotonic()


class GeminiEmbedder:
    """Generate embeddings using one configured Gemini embedding model."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        if client is not None:
            self._client = client
            self._types = None
            return

        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY is required for vector indexing.")

        try:
            from google import genai
            from google.genai import types
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("google-genai package is required for vector indexing.") from exc

        self._types = types
        self._client = genai.Client(
            api_key=self.settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=max(
                    1,
                    int(float(self.settings.ingestion_provider_timeout_seconds) * 1000),
                )
            ),
        )
        if self.settings.embedding_probe_on_startup:
            self.probe()

    def probe(self) -> None:
        """Fail closed before any corpus write if the embedding contract is wrong.

        A wrong model id or dimensionality otherwise surfaces thousands of calls later,
        or worse, silently produces vectors in a different space.
        """
        try:
            vectors = self.embed_texts(["probe"], task_type=self.settings.query_embedding_task_type)
        except Exception as exc:
            raise RuntimeError(
                f"Embedding probe failed for model {self.settings.embedding_model_id!r}: {exc}. "
                "Google documents this model as 'gemini-embedding-2-preview'; check "
                "EMBEDDING_MODEL_ID and the EMBEDDING_MODEL_* capability flags."
            ) from exc
        if not vectors or len(vectors[0]) != int(self.settings.embedding_dim):
            actual = len(vectors[0]) if vectors else 0
            raise RuntimeError(
                f"Embedding probe returned {actual} dimensions, expected "
                f"{self.settings.embedding_dim} for {self.settings.embedding_model_id!r}."
            )

    def embed_texts(
        self,
        texts: list[str],
        task_type: str | None = None,
        *,
        title: str | None = None,
    ) -> list[list[float]]:
        """Embed texts in deterministic batches using the configured retrieval instruction."""
        if not texts:
            return []

        prepared = self._prepare_texts(texts, task_type=task_type, title=title)
        batch_size = max(1, int(self.settings.embedding_batch_size))
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            vectors.extend(self._embed_batch(batch, task_type=task_type))
        return vectors

    def _prepare_texts(
        self,
        texts: list[str],
        *,
        task_type: str | None,
        title: str | None,
    ) -> list[str]:
        """Apply the asymmetric retrieval instruction when the model expects one."""
        if not self.settings.embedding_model_uses_instruction_prefix:
            return list(texts)
        query_mode = str(task_type or "").upper() == str(self.settings.query_embedding_task_type).upper()
        if query_mode:
            prefix = self.settings.embedding_query_prefix
        else:
            prefix = self.settings.embedding_document_prefix.format(title=title or "untitled")
        return [f"{prefix}{text}" for text in texts]

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        """A throttle, as opposed to a real failure.

        Matched on the response rather than the exception class because the SDK raises the
        same `ClientError` for a bad request as for a throttle, and retrying a bad request
        forever is worse than not retrying a throttle.
        """
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status == 429:
            return True
        text = str(exc)
        return "429" in text or "RESOURCE_EXHAUSTED" in text

    def _embed_batch(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        config = self._build_config(task_type)

        # A throttled batch used to raise straight out, failing the whole VECTOR_STAGE
        # event and burning one of the outbox's three coarse attempts - which re-embeds
        # every chunk in the document to recover one rate-limited call. A rebuild hit 26
        # of these in six documents and lost a document to it (docs/ERRORS.md 2026-09-04).
        # Retry the call, not the document.
        # The backoff is bounded in *total*, not just in attempts. This same embedder is
        # called from the query path, where the vector stage has a 15s deadline and entity
        # linking 20s: an unbounded 2+4+8+16s backoff turns a recoverable 429 into a stage
        # timeout, which is strictly worse than one failed call. Measured: a judged run
        # with attempt-only limits failed retrieval on 38% of questions, 8 of them as
        # outright TimeoutErrors. Slow retries belong at the outbox, which has no deadline.
        attempts = max(1, int(self.settings.embedding_rate_limit_attempts))
        base = max(0.0, float(self.settings.embedding_rate_limit_base_seconds))
        budget = max(0.0, float(self.settings.embedding_rate_limit_max_total_seconds))
        spent = 0.0
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                _pace(float(self.settings.embedding_min_interval_seconds))
                response = self._client.models.embed_content(
                    model=self.settings.embedding_model_id,
                    contents=texts,
                    config=config,
                )
                return self._parse_response(response)
            except Exception as exc:
                last = exc
                delay = base * (2**attempt) + random.uniform(0.0, base)
                if (
                    not self._is_rate_limited(exc)
                    or attempt + 1 >= attempts
                    or spent + delay > budget
                ):
                    raise RuntimeError(f"Gemini embedding request failed: {exc}") from exc
                logger.warning(
                    "embedding_rate_limited attempt=%d/%d sleeping=%.1fs spent=%.1f/%.1fs",
                    attempt + 1,
                    attempts,
                    delay,
                    spent,
                    budget,
                )
                time.sleep(delay)
                spent += delay
        raise RuntimeError(f"Gemini embedding request failed: {last}") from last

    async def aembed_texts(
        self,
        texts: list[str],
        task_type: str | None = None,
        *,
        title: str | None = None,
    ) -> list[list[float]]:
        """Asynchronously embed batches when the Gemini async client is available."""
        if not texts:
            return []
        prepared = self._prepare_texts(texts, task_type=task_type, title=title)
        aio = getattr(self._client, "aio", None)
        models = getattr(aio, "models", None)
        if models is None:
            return await asyncio.to_thread(
                self.embed_texts,
                texts,
                task_type,
                title=title,
            )
        batch_size = max(1, int(self.settings.embedding_batch_size))
        vectors: list[list[float]] = []
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            try:
                response = await models.embed_content(
                    model=self.settings.embedding_model_id,
                    contents=batch,
                    config=self._build_config(task_type),
                )
            except Exception as exc:
                raise RuntimeError(f"Gemini embedding request failed: {exc}") from exc
            vectors.extend(self._parse_response(response))
        return vectors

    def _build_config(self, task_type: str | None) -> Any:
        if self._types is None:
            return None
        kwargs: dict[str, Any] = {
            "output_dimensionality": int(self.settings.embedding_dim),
        }
        # Sent only to models that accept it. gemini-embedding-2 rejects task_type and
        # takes its instruction from the content instead.
        if self.settings.embedding_model_supports_task_type:
            kwargs["task_type"] = task_type or self.settings.gemini_embedding_task_type
        return self._types.EmbedContentConfig(**kwargs)

    def _parse_response(self, response: Any) -> list[list[float]]:
        embeddings = getattr(response, "embeddings", None)
        if not isinstance(embeddings, list):
            raise RuntimeError("Gemini embeddings response is malformed.")

        vectors: list[list[float]] = []
        for item in embeddings:
            values = getattr(item, "values", None)
            if not isinstance(values, list):
                raise RuntimeError("Embedding vector missing from Gemini response.")
            if len(values) != self.settings.embedding_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {self.settings.embedding_dim}, got {len(values)}."
                )
            vectors.append([float(value) for value in values])
        return vectors
