"""Phase 9: graph self-healing — orphans, contradictions and confidence decay.

Architecture section 6 specifies four healer tasks and a weekly decay job. None were built.
Two of them do not survive contact with this corpus, and saying so is the point of the
measurements recorded here rather than shipping machinery that runs and finds nothing:

**Contradiction detection reuses `GraphWriter.find_value_conflicts`** rather than
repeating it. **Property contradictions cannot be detected as specified.** Section 9.3 says to find
"entity nodes with conflicting property values from different source documents". Entity
`properties` is `{}` for every multi-source entity in the corpus - the extractor records
relationships, not attributes. What does exist is *type* disagreement across sources (161
entities) and *functional-predicate* disagreement (a subject with two different objects for
a predicate that is single-valued everywhere else). Both are implemented; the property
check is not, because there is nothing to check.

**Confidence decay is inert at the specified threshold.** Section 6.2 prunes edges below
0.50 confidence with no corroboration. **Zero of 12,676 active assertions are below 0.50**;
ten are below 0.80. The extractor's confidence is not calibrated - it means "the model was
asked for a number", not "the model was unsure" - which Refactor 3 WP3 already found when
instructing for lower confidence made the model *more* assertive. The job is built, and
reports honestly that it flags nothing at the documented threshold.

Near-duplicate merging is proposed, never applied. Gate 8 still shows 10 false merges
against gold, so an unattended job that merges on cosine alone would make identity worse.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Contradiction:
    kind: str
    subject_id: str
    subject_name: str
    detail: str


@dataclass
class HealReport:
    orphans_found: int = 0
    orphans_pruned: int = 0
    type_conflicts: int = 0
    functional_conflicts: int = 0
    contradictions_flagged: int = 0
    low_confidence_found: int = 0
    low_confidence_flagged: int = 0
    dry_run: bool = True
    contradictions: list[Contradiction] = field(default_factory=list)


class GraphHealer:
    """Maintenance passes over the knowledge graph. Read-only unless told otherwise."""

    def __init__(self, settings: Settings, driver: Any | None = None) -> None:
        self.settings = settings
        if driver is not None:
            self._driver = driver
        else:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )

    # -------------------------------------------------------------------- orphans
    def find_orphans(self) -> list[tuple[str, str]]:
        """Entities no active assertion references.

        These accumulate from extraction errors and one-off mentions, and from merges that
        left a node behind. They are still reachable by entity-vector search, so they can
        surface as linkable entities that lead nowhere.
        """
        query = """
        MATCH (e:Entity)
        WHERE NOT (e)--(:Assertion)
        RETURN e.id AS id, e.name AS name
        ORDER BY e.name
        """
        with self._driver.session() as session:
            return [(str(r["id"]), str(r["name"] or "")) for r in session.run(query)]

    def prune_orphans(self, dry_run: bool = True) -> tuple[int, int]:
        orphans = self.find_orphans()
        if dry_run or not orphans:
            return len(orphans), 0
        with self._driver.session() as session:
            session.run(
                "UNWIND $ids AS id MATCH (e:Entity {id: id}) WHERE NOT (e)--(:Assertion) "
                "DETACH DELETE e",
                {"ids": [oid for oid, _ in orphans]},
            )
        return len(orphans), len(orphans)

    # -------------------------------------------------------------- contradictions
    def find_type_conflicts(self) -> list[Contradiction]:
        """One entity given different types by different documents."""
        query = """
        MATCH (e:Entity)
        WHERE size(e.types) > 1
        RETURN e.id AS id, e.name AS name, e.types AS types
        """
        with self._driver.session() as session:
            return [
                Contradiction(
                    kind="type",
                    subject_id=str(r["id"]),
                    subject_name=str(r["name"] or ""),
                    detail=" | ".join(sorted(str(t) for t in (r["types"] or []))),
                )
                for r in session.run(query)
            ]

    def find_functional_conflicts(self) -> list[Contradiction]:
        """Delegates to `GraphWriter.find_value_conflicts`.

        That query already infers cardinality from the corpus, resolves object names rather
        than ids, and returns the assertion ids behind each conflict. Reimplementing it here
        produced a worse version of the same thing - which is the defect this Phase 9 audit
        was looking for, reintroduced by the audit itself.
        """
        from ingestion.graph_writer import GraphWriter

        writer = GraphWriter(self.settings, driver=self._driver)
        rows = writer.find_value_conflicts(
            min_single_ratio=float(self.settings.heal_functional_threshold),
            min_subjects=int(self.settings.heal_functional_min_support),
        )
        return [
            Contradiction(
                kind="functional_predicate",
                subject_id=str(row["subject_id"]),
                subject_name=str(row["subject_name"] or ""),
                detail=(
                    f"{row['predicate']}: "
                    + ", ".join(str(v) for v in (row.get("values") or [])[:4])
                ),
            )
            for row in rows
        ]

    def flag_contradictions(self, dry_run: bool = True) -> tuple[list[Contradiction], int]:
        """Only functional-predicate conflicts are flagged on the node.

        Type disagreement is reported but never flagged. Inspecting all 161 of them showed
        they are overwhelmingly benign variance in a schema-free graph - "Australia:
        Country | Location", "GAT: Method | Model", "Neo4j: Database | Graph Database |
        Tool" - which is the design working as intended, not a factual conflict. Setting
        `contradicted` on those would mark a sixth of the multi-source entities as suspect
        and teach a reader to ignore the flag.

        The functional conflicts are the real thing: one birthplace per person, one release
        date per model.
        """
        conflicts = self.find_functional_conflicts()
        if dry_run or not conflicts:
            return conflicts, 0
        with self._driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (e:Entity {id: row.id})
                SET e.contradicted = true, e.contradiction_kinds = row.kinds
                """,
                {
                    "rows": [
                        {"id": subject_id, "kinds": sorted({c.kind for c in group})}
                        for subject_id, group in _group_by_subject(conflicts).items()
                    ]
                },
            )
        return conflicts, len(_group_by_subject(conflicts))

    # ----------------------------------------------------------- confidence decay
    def find_low_confidence(self) -> list[tuple[str, float, str]]:
        """Low-confidence assertions no second document corroborates.

        Corroboration is by `claim_key`, which is shared across documents making the same
        claim - the same test `review_status` uses for rare predicates.
        """
        query = """
        MATCH (a:Assertion {active: true})
        WHERE a.confidence < $threshold
        WITH a.claim_key AS claim_key, collect(a) AS group
        WITH claim_key, group, size(apoc.coll.toSet([x IN group | x.doc_id])) AS doc_count
        WHERE doc_count < 2
        UNWIND group AS a
        RETURN a.id AS id, a.confidence AS confidence, a.predicate_normalized AS predicate
        """
        fallback = """
        MATCH (a:Assertion {active: true})
        WHERE a.confidence < $threshold
        WITH a.claim_key AS claim_key, collect(DISTINCT a.doc_id) AS docs, collect(a) AS group
        WHERE size(docs) < 2
        UNWIND group AS a
        RETURN a.id AS id, a.confidence AS confidence, a.predicate_normalized AS predicate
        """
        threshold = float(self.settings.assertion_review_confidence_threshold)
        with self._driver.session() as session:
            try:
                rows = list(session.run(query, {"threshold": threshold}))
            except Exception:
                # APOC is not installed on Neo4j Community by default.
                rows = list(session.run(fallback, {"threshold": threshold}))
        return [
            (str(r["id"]), float(r["confidence"] or 0.0), str(r["predicate"] or ""))
            for r in rows
        ]

    def flag_low_confidence(self, dry_run: bool = True) -> tuple[int, int]:
        rows = self.find_low_confidence()
        if dry_run or not rows:
            return len(rows), 0
        with self._driver.session() as session:
            session.run(
                "UNWIND $ids AS id MATCH (a:Assertion {id: id}) "
                "SET a.review_status = 'low_confidence_uncorroborated'",
                {"ids": [row[0] for row in rows]},
            )
        return len(rows), len(rows)

    # ------------------------------------------------------------------- driver
    def heal(self, dry_run: bool = True) -> HealReport:
        report = HealReport(dry_run=dry_run)
        report.orphans_found, report.orphans_pruned = self.prune_orphans(dry_run=dry_run)
        conflicts, flagged = self.flag_contradictions(dry_run=dry_run)
        variance = self.find_type_conflicts()
        report.contradictions = [*conflicts, *variance]
        report.type_conflicts = len(variance)
        report.functional_conflicts = len(conflicts)
        report.contradictions_flagged = flagged
        report.low_confidence_found, report.low_confidence_flagged = self.flag_low_confidence(
            dry_run=dry_run
        )
        return report

    def close(self) -> None:
        driver = getattr(self, "_driver", None)
        if driver is not None:
            driver.close()


def _group_by_subject(conflicts: list[Contradiction]) -> dict[str, list[Contradiction]]:
    grouped: dict[str, list[Contradiction]] = defaultdict(list)
    for conflict in conflicts:
        grouped[conflict.subject_id].append(conflict)
    return grouped
