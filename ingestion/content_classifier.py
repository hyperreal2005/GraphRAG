"""Deterministic document-content classification for ingestion and retrieval."""

from __future__ import annotations

import re


REFERENCE_HEADING_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s*)?"
    r"(?:references|bibliography|works\s+cited|literature\s+cited|references\s+and\s+notes)"
    r"\s*[:.]?\s*$",
    re.IGNORECASE,
)

POST_REFERENCE_CONTENT_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s*)?"
    r"(?:appendix|appendices|supplement|supplementary\s+(?:material|materials|information))"
    r"(?:\s+[A-Z0-9]+)?\s*[:.]?\s*$",
    re.IGNORECASE,
)


def classify_content_kind(section_heading: str | None) -> str:
    """Classify a chunk without asking an LLM or imposing graph ontology."""
    heading = " ".join(str(section_heading or "").split())
    if heading and REFERENCE_HEADING_PATTERN.fullmatch(heading):
        return "reference"
    return "body"


def is_graph_extractable(content_kind: str) -> bool:
    """Return whether content should contribute factual graph assertions."""
    return str(content_kind).strip().lower() != "reference"


def classify_content_sequence(section_headings: list[str | None]) -> list[str]:
    """Classify ordered chunks while preserving bibliography continuation state."""
    in_references = False
    classifications: list[str] = []
    for section_heading in section_headings:
        heading = " ".join(str(section_heading or "").split())
        direct = classify_content_kind(heading)
        if direct == "reference":
            in_references = True
        elif in_references and heading and POST_REFERENCE_CONTENT_PATTERN.fullmatch(heading):
            in_references = False
        classifications.append("reference" if in_references else "body")
    return classifications
