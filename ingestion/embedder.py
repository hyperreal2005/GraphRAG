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
from typing import Any

from config import Settings


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

    def _embed_batch(self, texts: list[str], task_type: str | None = None) -> list[list[float]]:
        config = self._build_config(task_type)

        try:
            response = self._client.models.embed_content(
                model=self.settings.embedding_model_id,
                contents=texts,
                config=config,
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini embedding request failed: {exc}") from exc

        return self._parse_response(response)

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
