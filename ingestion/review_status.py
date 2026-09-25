"""Decide which assertions are flagged for review, from signals the system computes.

`review_status` was written at ingest from the model's self-reported confidence and read by
nothing. Refactor 3 WP3 established that confidence is unusable for this: 0 of 1,100
assertions fell below the review threshold, and instructing the model to lower it made it
*more* assertive - mean 0.957 to 0.992, 91% at exactly 1.0. A field that never varies
cannot mark anything.

**Predicate rarity does vary, and it separates.** Measured against Re-DocRED gold on the
accepted corpus, the share of assertions matching a gold relation by predicate frequency:

    predicate seen 1 time     1.2%
    seen 2-4 times            0.4%
    seen 5-19 times          16.1%
    seen 20+ times           43.6%

A 36-fold spread. An assertion whose predicate the corpus uses once is close to worthless;
one whose predicate is used constantly is the system's most reliable output. That is a
usable signal precisely because it is computed from the corpus rather than reported by the
model that produced the assertion.

**Corroboration overrides rarity.** A claim asserted by two independent documents is
supported whatever its predicate looks like. It is rare - 0.7% of assertions on this corpus
- so it cannot be the primary signal, but where it exists it is the stronger one.

This runs as a maintenance pass rather than at ingest, because the frequency of a predicate
is a property of the whole corpus and changes as documents arrive. Newly ingested
assertions keep the ingest-time default until the pass runs.

**Corpus-size dependence, stated plainly:** `review_rare_predicate_below` is an absolute
count, so on a small corpus almost every predicate is rare and almost everything is
flagged. That is arguably correct - a small corpus supports few conclusions - but it is a
property to be aware of rather than a bug to be surprised by.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any

from config import Settings

logger = logging.getLogger(__name__)

COUNT_QUERY = """
MATCH (a:Assertion)
WHERE a.active = true AND a.document_version_id IN $version_ids
RETURN a.id AS id, a.doc_id AS doc_id, a.claim_key AS claim_key,
       a.predicate_normalized AS predicate, a.review_status AS review_status
"""

UPDATE_QUERY = """
UNWIND $rows AS row
MATCH (a:Assertion {id: row.id})
SET a.review_status = row.review_status
"""


@dataclass
class ReviewStatusReport:
    total: int = 0
    accepted: int = 0
    flagged: int = 0
    changed: int = 0
    rare_predicate: int = 0
    corroborated_override: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "flagged": self.flagged,
            "changed": self.changed,
            "rare_predicate": self.rare_predicate,
            "corroborated_override": self.corroborated_override,
            "flagged_share": round(self.flagged / self.total, 4) if self.total else None,
        }


def recompute(
    settings: Settings,
    *,
    version_ids: list[str],
    driver: Any,
    dry_run: bool = False,
    batch_size: int = 500,
) -> ReviewStatusReport:
    """Recompute `review_status` for the active assertions of the given versions."""
    threshold = max(1, int(settings.review_rare_predicate_below))
    report = ReviewStatusReport()
    if not version_ids:
        return report

    with driver.session() as session:
        rows = [dict(record) for record in session.run(COUNT_QUERY, {"version_ids": version_ids})]
    report.total = len(rows)
    if not rows:
        return report

    predicate_counts = Counter(str(row["predicate"] or "") for row in rows)
    documents_per_claim: dict[str, set[str]] = {}
    for row in rows:
        documents_per_claim.setdefault(str(row["claim_key"] or ""), set()).add(str(row["doc_id"] or ""))

    updates: list[dict[str, str]] = []
    for row in rows:
        predicate = str(row["predicate"] or "")
        rare = predicate_counts[predicate] < threshold
        corroborated = len(documents_per_claim[str(row["claim_key"] or "")]) > 1

        if rare and not corroborated:
            status = "flagged"
            report.rare_predicate += 1
        else:
            status = "accepted"
            if rare and corroborated:
                report.corroborated_override += 1

        if status == "flagged":
            report.flagged += 1
        else:
            report.accepted += 1
        if status != str(row["review_status"] or "accepted"):
            report.changed += 1
            updates.append({"id": str(row["id"]), "review_status": status})

    if updates and not dry_run:
        with driver.session() as session:
            for start in range(0, len(updates), batch_size):
                session.run(UPDATE_QUERY, {"rows": updates[start : start + batch_size]}).consume()
    logger.info(
        "review_status_recomputed total=%d flagged=%d changed=%d dry_run=%s",
        report.total,
        report.flagged,
        report.changed,
        dry_run,
    )
    return report


def render_markdown(report: ReviewStatusReport, *, threshold: int) -> str:
    share = report.to_dict()["flagged_share"]
    return "\n".join(
        [
            "## Review status",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Active assertions | {report.total} |",
            f"| Accepted | {report.accepted} |",
            f"| **Flagged for review** | **{report.flagged}** |",
            f"| Flagged share | {'n/a' if share is None else f'{100 * share:.1f}%'} |",
            f"| Flagged: rare predicate (< {threshold} uses) | {report.rare_predicate} |",
            f"| Kept by cross-document corroboration | {report.corroborated_override} |",
            f"| Rows changed | {report.changed} |",
            "",
            "A flagged assertion stays queryable but is ranked below confident evidence, and "
            "an answer resting only on flagged evidence says so. Rarity is measured over the "
            "whole active corpus, so it shifts as documents are added.",
        ]
    )
