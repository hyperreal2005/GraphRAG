"""Gemini embedding service for vector indexing."""

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
        self._client = genai.Client(api_key=self.settings.gemini_api_key)

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
        if self.settings.embedding_model_id != "gemini-embedding-2":
            return list(texts)
        query_mode = str(task_type or "").upper() == str(self.settings.query_embedding_task_type).upper()
        if query_mode:
            prefix = self.settings.embedding_query_prefix
        else:
            prefix = self.settings.embedding_document_prefix.format(title=title or "none")
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
        # gemini-embedding-2 uses instructions in the content; the legacy
        # task_type parameter is only sent to models that support it.
        if self.settings.embedding_model_id != "gemini-embedding-2":
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
