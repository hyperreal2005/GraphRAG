"""Gemini embedding service for vector indexing."""

from __future__ import annotations

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

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in deterministic batching order."""
        if not texts:
            return []

        batch_size = max(1, int(self.settings.embedding_batch_size))
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        config = None
        if self._types is not None:
            config = self._types.EmbedContentConfig(
                output_dimensionality=int(self.settings.embedding_dim),
                task_type=self.settings.gemini_embedding_task_type,
            )

        try:
            response = self._client.models.embed_content(
                model=self.settings.embedding_model_id,
                contents=texts,
                config=config,
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini embedding request failed: {exc}") from exc

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
