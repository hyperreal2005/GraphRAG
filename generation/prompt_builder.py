"""Prompt assembly for grounded answer generation."""

from __future__ import annotations

from config import Settings
from models.query import FusionOutput, QueryAnalysis, QueryType


class PromptBuilder:
    """Build structured generation prompt with query-type controls."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def build_prompt(self, question: str, analysis: QueryAnalysis, fusion: FusionOutput) -> tuple[str, float]:
        """Return prompt + temperature tuned for query type."""
        temperature = self._temperature_for_type(analysis.query_type)
        context = fusion.context_text.strip() or "No retrieved evidence."
        prompt = (
            f"QUERY TYPE: {analysis.query_type.value}\n\n"
            "<retrieved_evidence untrusted=\"true\">\n"
            f"{context}\n"
            "</retrieved_evidence>\n\n"
            f"QUESTION: {question}"
        )
        return prompt, temperature

    def _temperature_for_type(self, query_type: QueryType) -> float:
        if query_type is QueryType.GLOBAL:
            return float(self.settings.query_temperature_global)
        if query_type is QueryType.MULTIHOP:
            return float(self.settings.query_temperature_multihop)
        if query_type is QueryType.COMPARATIVE:
            return float(self.settings.query_temperature_comparative)
        if query_type is QueryType.HYBRID:
            return float(self.settings.query_temperature_hybrid)
        return float(self.settings.query_temperature_local)
