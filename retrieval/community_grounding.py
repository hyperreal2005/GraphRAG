"""Ground community summaries in passages from their member documents (ROADMAP C2).

A community summary is synthesised from many documents and has no chunk, so a claim resting
on it reaches the reader uncited: 13-19% of GLOBAL claims on arm v2's baseline. And a summary
seldom survives the rerank into the context - 9 of 23 answerable questions - so most GLOBAL
answers are built from whichever few chunks vector search found, from few documents of the
theme.

For each retrieved community this adds its most central passages: in each member document,
the chunk naming the most community members, and of those, the documents whose chunk names
the most. They enter as ordinary vector evidence with no measured relevance - fusion ranks
them last among vector items, and the reranker decides whether they reach the context - and
they are citable like any chunk.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from models.query import RetrieverResult


def _names_pattern(names: Iterable[str]) -> re.Pattern[str] | None:
    """One alternation over the member names, longest first, matched as whole words.

    One pass per chunk: a pattern per name cost 2-4 s a question on wp24 (up to 40 names over
    a few thousand chunks), measured on the first live run.
    """
    unique = {str(n).strip().casefold() for n in names if len(str(n).strip()) >= 3}
    if not unique:
        return None
    body = "|".join(re.escape(name) for name in sorted(unique, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)")


def central_passages(
    community: RetrieverResult, rows: list[Any], per_community: int
) -> list[RetrieverResult]:
    """The community's most central passages among `rows`, at most one per document."""
    pattern = _names_pattern(community.provenance.get("member_names") or [])
    if pattern is None or per_community <= 0:
        return []
    best: dict[str, tuple[int, Any]] = {}
    for row in rows:
        text = str(row["text"] or "")
        named = len(set(pattern.findall(text.casefold())))
        if named and named > best.get(row["doc_id"], (0, None))[0]:
            best[row["doc_id"]] = (named, row)
    ranked = sorted(best.values(), key=lambda pair: (-pair[0], str(pair[1]["chunk_id"])))
    community_id = str(community.provenance.get("community_id") or community.source_id)
    out = []
    for named, row in ranked[:per_community]:
        chunk_id = str(row["chunk_id"])
        out.append(
            RetrieverResult(
                source_type="vector",
                source_id=chunk_id,
                text_or_path=str(row["text"] or ""),
                score_raw=0.0,
                provenance={
                    "chunk_id": chunk_id,
                    "doc_id": row["doc_id"],
                    "section_heading": row["section"],
                    "page": row["page"],
                    "document_version_id": row["document_version_id"],
                    "evidence_id": f"chunk:{chunk_id}",
                    "via_community": community_id,
                    "community_members_named": named,
                },
            )
        )
    return out


def ground_communities(
    communities: list[RetrieverResult], store: Any, per_community: int
) -> list[RetrieverResult]:
    """Central passages for every community, without repeating a chunk."""
    if per_community <= 0 or not communities or store is None:
        return []
    doc_ids = [d for c in communities for d in (c.provenance.get("doc_ids") or [])]
    by_doc: dict[str, list[Any]] = {}
    for row in store.get_queryable_chunks_for_docs(doc_ids):
        by_doc.setdefault(row["doc_id"], []).append(row)
    seen: set[str] = set()
    grounded: list[RetrieverResult] = []
    for community in communities:
        rows = [row for d in (community.provenance.get("doc_ids") or []) for row in by_doc.get(d, [])]
        for item in central_passages(community, rows, per_community):
            if item.source_id not in seen:
                seen.add(item.source_id)
                grounded.append(item)
    return grounded
