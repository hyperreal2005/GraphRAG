"""Answer generation using Gemini with deterministic confidence/citations."""

from __future__ import annotations

from typing import Any

from config import Settings
from generation.prompt_builder import PromptBuilder
from models.query import FusionOutput, GraphRAGResponse, QueryAnalysis


class AnswerGenerator:
    """Generate grounded final response from fused retrieval context."""

    def __init__(
        self,
        settings: Settings,
        prompt_builder: PromptBuilder | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.prompt_builder = prompt_builder or PromptBuilder(settings)
        self._client = client
        self._types = None

        if self._client is not None:
            return
        if not self.settings.gemini_api_key:
            return
        try:
            from google import genai
            from google.genai import types
        except Exception:
            return
        self._types = types
        self._client = genai.Client(api_key=self.settings.gemini_api_key)

    def generate(self, question: str, analysis: QueryAnalysis, fusion: FusionOutput) -> GraphRAGResponse:
        """Generate answer text, then package with deterministic metadata."""
        if not fusion.context_items:
            answer = "I do not have enough retrieved context to answer this question."
        else:
            answer = self._generate_answer_text(question=question, analysis=analysis, fusion=fusion)

        confidence = self._compute_confidence(fusion.top_scores)
        if fusion.community_fallback_used:
            answer = (
                f"{answer}\n\n"
                "Note: community retrieval is not available yet in this phase; semantic chunk fallback was used."
            )

        return GraphRAGResponse(
            answer=answer.strip(),
            citations=fusion.citations,
            graph_paths_used=fusion.graph_paths,
            confidence=confidence,
            query_type=analysis.query_type.value,
            retrieval_sources=fusion.retrieval_sources,
        )

    def _generate_answer_text(self, question: str, analysis: QueryAnalysis, fusion: FusionOutput) -> str:
        prompt, temperature = self.prompt_builder.build_prompt(question=question, analysis=analysis, fusion=fusion)
        if self._client is None:
            return self._fallback_answer(fusion)

        config = None
        if self._types is not None:
            config = self._types.GenerateContentConfig(temperature=temperature)
        try:
            response = self._client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config=config,
            )
            text = str(getattr(response, "text", "") or "").strip()
            if text:
                return text
        except Exception:
            pass
        return self._fallback_answer(fusion)

    def _fallback_answer(self, fusion: FusionOutput) -> str:
        top = fusion.context_items[0] if fusion.context_items else None
        if top is None:
            return "I do not have enough retrieved context to answer this question."
        if top.source_type == "graph":
            return f"Retrieved graph evidence indicates: {top.text_or_path}"
        excerpt = " ".join(top.text_or_path.split())
        if len(excerpt) > 320:
            excerpt = excerpt[:317].rstrip() + "..."
        return f"Based on retrieved context: {excerpt}"

    def _compute_confidence(self, top_scores: list[float]) -> float:
        if not top_scores:
            return 0.0
        avg = sum(float(score) for score in top_scores) / len(top_scores)
        return max(0.0, min(1.0, avg))

