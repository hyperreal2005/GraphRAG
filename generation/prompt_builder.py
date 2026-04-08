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

        graph_lines = [item.text_or_path for item in fusion.context_items if item.source_type == "graph"]
        chunk_lines = []
        community_lines = []
        for item in fusion.context_items:
            if item.source_type == "vector":
                doc_id = item.provenance.get("doc_id")
                chunk_id = item.provenance.get("chunk_id")
                page = item.provenance.get("page")
                chunk_lines.append(
                    f'Text: "{item.text_or_path}" (Source: doc_id={doc_id}, chunk_id={chunk_id}, page={page})'
                )
            if item.source_type == "community":
                community_lines.append(f"Summary: {item.text_or_path}")

        sections: list[str] = []
        sections.append(
            "SYSTEM:\n"
            "You are a precise reasoning engine.\n"
            "Answer the question using ONLY the provided context.\n"
            "Do not use prior knowledge not present in the context.\n"
            "If context is insufficient, say so explicitly.\n"
            "Cite the source context when making claims."
        )
        sections.append(f"QUERY TYPE: {analysis.query_type.value}")
        sections.append("[GRAPH PATHS]\n" + ("\n".join(graph_lines) if graph_lines else "None"))
        sections.append("[DOCUMENT CHUNKS]\n" + ("\n".join(chunk_lines) if chunk_lines else "None"))
        sections.append("[COMMUNITY CONTEXT]\n" + ("\n".join(community_lines) if community_lines else "None"))
        sections.append(f"QUESTION: {question}")

        return "\n\n".join(sections), temperature

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

