"""Which questions ask for a verdict rather than a name (ROADMAP C1).

A polar question - "Was there consistency between the TechCrunch and Fortune reports?",
"Does the Verge article suggest...?" - is answered by yes or no. Before C1 the generator
treated it like any other question and often described both sources without committing,
which A1 measured: on news, 2 of 18 yes/no answers gave the right verdict.

Deterministic on purpose, and measured before use. On all 2,556 MultiHop-RAG questions,
against the questions whose gold answer is Yes or No: precision 0.942, recall 0.991. The
"false positives" are polar questions with other one-word golds ("Consistent", "True",
"Agree"), which a verdict answer still fits; the misses open with a long "After the report
..., which described ..." clause.
"""

from __future__ import annotations

import re

_AUXILIARY = r"(?:does|did|do|is|was|were|are|has|have|had|can|could|would|will|should)"
_LEADING_AUXILIARY = re.compile(rf"^\s*{_AUXILIARY}\b", re.IGNORECASE)
_CLAUSE_AUXILIARY = re.compile(rf"[,;]\s*{_AUXILIARY}\s+\w+", re.IGNORECASE)
_WH_CLAUSE = re.compile(r"(?:^|[,;:]\s*)(?:what|which|who|whom|whose|where|when|how|why)\b", re.IGNORECASE)


def is_polar_question(question: str) -> bool:
    """True when the question asks for a yes/no verdict."""
    text = question or ""
    if _LEADING_AUXILIARY.match(text):
        return True
    if _WH_CLAUSE.search(text):
        return False
    return bool(_CLAUSE_AUXILIARY.search(text))
