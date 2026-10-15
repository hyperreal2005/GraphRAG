"""Operational CLI for corpus maintenance (docs/REFACTOR2.md WP3).

Two Refactor 1 capabilities existed in code but had no caller, so their exit gates could
not be met:

  * merge rollback (`REVIEW1` B3) — implemented, untested, and unreachable
  * the outbox reconciler (`REVIEW1` B6) — only ever invoked scoped to the version being
    built, so a crashed run was never revisited

    python main_admin.py reconcile      --db data/corpus.db
    python main_admin.py sweep-orphans  --db data/corpus.db --older-than-hours 24
    python main_admin.py list-merges    --db data/corpus.db --doc-id <id>
    python main_admin.py rollback-merge --db data/corpus.db --decision-id <id> --dry-run
    python main_admin.py review-status  --db data/corpus.db --dry-run
    python main_admin.py corpus-health  --db data/corpus.db
    python main_admin.py contradictions --db data/corpus.db --limit 40
    python main_admin.py communities    --db data/corpus.db --min-size 10 --dry-run
    python main_admin.py heal           --db data/corpus.db            # dry run; --apply writes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from config import Settings, default_metadata_db
from ingestion.state_store import MetadataStore


def _settings(args: argparse.Namespace) -> Settings:
    return Settings(metadata_db_path=Path(args.db))


def _components(settings: Settings):  # noqa: ANN201
    """Build the write-side components. Imported lazily so read-only commands are cheap."""
    from ingestion.graph_writer import GraphWriter
    from ingestion.outbox import IngestionOutboxProcessor
    from ingestion.vector_indexer import QdrantVectorIndexer

    store = MetadataStore(settings.metadata_db_path)
    graph_writer = GraphWriter(settings)
    vector_indexer = QdrantVectorIndexer(settings)
    processor = IngestionOutboxProcessor(
        settings=settings,
        store=store,
        graph_writer=graph_writer,
        vector_indexer=vector_indexer,
    )
    return store, graph_writer, vector_indexer, processor


def command_reconcile(args: argparse.Namespace) -> int:
    settings = _settings(args)
    store, graph_writer, _, processor = _components(settings)

    pending = [
        event
        for event in store.list_outbox_events()
        if event.status in {"pending", "processing", "failed", "dead_letter"}
    ]
    if not pending:
        print("Nothing to reconcile: every outbox event is completed.")
        return 0

    print(f"{len(pending)} incomplete event(s) before reconciliation:")
    for event in pending[:20]:
        print(f"  {event.status:<11} {event.event_type:<22} attempts={event.attempts} {event.document_version_id}")

    processed = processor.reconcile(
        document_version_id=args.document_version_id,
        wait_for_retries=True,
        max_events=args.max_events,
    )
    remaining = [
        event
        for event in store.list_outbox_events()
        if event.status in {"pending", "processing", "failed", "dead_letter"}
    ]
    print(f"\nProcessed {processed} event(s); {len(remaining)} still incomplete.")
    for event in remaining[:20]:
        print(f"  {event.status:<11} {event.event_type:<22} {event.last_error or ''}"[:150])
    graph_writer.close()
    return 0 if not remaining else 1


def command_sweep_orphans(args: argparse.Namespace) -> int:
    settings = _settings(args)
    _, graph_writer, _, processor = _components(settings)
    swept = processor.sweep_orphans(
        older_than_hours=args.older_than_hours,
        dry_run=args.dry_run,
    )
    verb = "Would sweep" if args.dry_run else "Swept"
    print(f"{verb} {len(swept)} abandoned version(s).")
    for version_id in swept:
        print(f"  {version_id}")
    if not swept:
        print("  (an active version is never swept, and one with pending events is left"
              " for the reconciler)")
    graph_writer.close()
    return 0


def command_list_merges(args: argparse.Namespace) -> int:
    settings = _settings(args)
    store = MetadataStore(settings.metadata_db_path)
    rows = store.list_merge_decisions(
        doc_id=args.doc_id,
        decision=args.decision,
        limit=args.limit,
    )
    if not rows:
        print("No merge decisions matched.")
        return 0
    # The full id is printed, not a prefix: it is the argument `rollback-merge` takes,
    # and a truncated one silently fails to match.
    print(f"{'decision_id':<34} {'decision':<12} {'incoming':<26} {'candidate':<26} sim")
    for row in rows:
        reversed_marker = " [reversed]" if row["reversed_at"] else ""
        similarity = row["top1_similarity"]
        print(
            f"{str(row['decision_id']):<34} {str(row['decision']):<12} "
            f"{str(row['incoming_name'])[:25]:<26} {str(row['candidate_name'] or '-')[:25]:<26} "
            f"{similarity if similarity is None else round(float(similarity), 3)}{reversed_marker}"
        )
    return 0


def command_rollback_merge(args: argparse.Namespace) -> int:
    import uuid

    from ingestion.merge_rollback import MergeRollbackService

    settings = _settings(args)
    store, graph_writer, vector_indexer, processor = _components(settings)

    decision = store.get_merge_decision(args.decision_id)
    if decision is None:
        print(f"No merge decision with id {args.decision_id!r}.")
        return 1

    print("Decision to reverse:")
    print(f"  kind      : {decision.decision}")
    print(f"  incoming  : {decision.incoming_name!r} ({decision.incoming_type})")
    print(f"  merged to : {decision.candidate_name!r} ({decision.candidate_type}) = {decision.candidate_entity_id}")
    print(f"  similarity: {decision.top1_similarity}  margin: {decision.score_margin}")
    print(f"  rationale : {decision.rationale[:160]}")

    if args.dry_run:
        affected = graph_writer.preview_rollback(
            document_version_id=decision.document_version_id,
            local_entity_id=decision.incoming_local_entity_id,
            merged_entity_id=str(decision.candidate_entity_id),
        )
        print(f"\nDry run: {len(affected)} assertion(s) would be rewritten onto a restored entity.")
        for assertion_id in affected[:20]:
            print(f"  {assertion_id}")
        graph_writer.close()
        return 0

    service = MergeRollbackService(
        store=store,
        graph_writer=graph_writer,
        vector_indexer=vector_indexer,
        outbox_processor=processor,
    )
    try:
        reversal = service.rollback(args.decision_id, run_id=str(uuid.uuid4()))
    except Exception as exc:
        print(f"\nRollback failed: {type(exc).__name__}: {exc}")
        graph_writer.close()
        return 1
    print(f"\nReversed. New decision {reversal.decision_id} records the reversal;"
          " the original row is preserved.")
    graph_writer.close()
    return 0


def command_review_status(args: argparse.Namespace) -> int:
    """Recompute which assertions are flagged for review."""
    from ingestion.review_status import recompute, render_markdown

    settings = _settings(args)
    store = MetadataStore(settings.metadata_db_path)
    version_ids = [
        version.version_id
        for row in store.list_documents()
        for version in [store.get_active_document_version(str(row["doc_id"]))]
        if version is not None
    ]
    if not version_ids:
        print("No active document versions; nothing to recompute.")
        return 0

    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(
            settings.neo4j_query_user or settings.neo4j_user,
            settings.neo4j_query_password or settings.neo4j_password,
        ),
    )
    try:
        report = recompute(
            settings, version_ids=version_ids, driver=driver, dry_run=args.dry_run
        )
    finally:
        driver.close()

    print(render_markdown(report, threshold=settings.review_rare_predicate_below))
    if args.dry_run:
        print("\nDry run: no rows were written.")
    return 0


def command_corpus_health(args: argparse.Namespace) -> int:
    from eval import corpus_health

    print(corpus_health.render_markdown(
        corpus_health.collect(_settings(args), include_graph=not args.no_graph)
    ))
    return 0


def command_contradictions(args: argparse.Namespace) -> int:
    """Surface subjects carrying conflicting values for one predicate.

    `GraphWriter.find_contradictions` was written and then never called by anything - the
    same shape as the merge rollback and the outbox reconciler above. Architecture 9.3
    wants contradictions surfaced rather than silently resolved; this is the read-only half
    of that, and it is what makes the query reachable at all.
    """
    from ingestion.graph_writer import GraphWriter

    settings = _settings(args)
    writer = GraphWriter(settings)
    # Fenced to this store's versions: Neo4j holds several corpora (ROADMAP D0).
    version_ids = MetadataStore(settings.metadata_db_path).get_queryable_document_version_ids()
    try:
        rows = (
            writer.find_contradictions(doc_id=args.doc_id, version_ids=version_ids)
            if args.raw
            else writer.find_value_conflicts(
                doc_id=args.doc_id,
                min_single_ratio=args.min_single_ratio,
                min_subjects=args.min_subjects,
                version_ids=version_ids,
            )
        )
    finally:
        close = getattr(writer, "close", None)
        if callable(close):
            close()

    if not rows:
        print("No contradictions found.")
        return 0

    limit = max(1, int(args.limit))
    print("# Contradictions")
    print()
    print(f"{len(rows)} subject/predicate pairs carry more than one value.")
    print()
    for row in rows[:limit]:
        values = row.get("values") or []
        subject = row.get("subject_name") or row.get("subject_id")
        print(f"- **{subject}** -[{row.get('predicate')}]-> {len(values)} values")
        for value in values[:6]:
            print(f"    - {value}")
        if len(values) > 6:
            print(f"    - ... and {len(values) - 6} more")
    if len(rows) > limit:
        print()
        print(f"... and {len(rows) - limit} more pairs.")
    return 0


def command_communities(args: argparse.Namespace) -> int:
    """Detect entity communities, summarise them, and publish them for GLOBAL queries.

    Phase 8. `--dry-run` costs nothing and reports what a real build would summarise, which
    matters because each kept community is one provider call.
    """
    from ingestion.community_builder import CommunityBuilder

    settings = _settings(args)
    builder = CommunityBuilder(settings)
    try:
        report = builder.build(
            min_size=args.min_size, limit=args.limit, dry_run=args.dry_run
        )
    finally:
        builder.close()

    print("# Community build")
    print()
    print(f"- build id: `{report.build_id}`")
    print(f"- detected at min size {args.min_size or settings.community_min_size}: {report.detected}")
    print(f"- kept: {report.kept}")
    print(f"- entities covered: {report.entities_covered}")
    print(f"- largest community: {report.largest}")
    if args.dry_run:
        print()
        print(f"Dry run. A real build would make {report.kept} provider calls.")
        return 0
    print(f"- summarised: {report.summarised} (failed {report.failed})")
    print(f"- indexed to Qdrant: {report.indexed}")
    print(f"- provider calls: {report.provider_calls}")
    return 0 if report.summarised else 1


def command_heal(args: argparse.Namespace) -> int:
    """Phase 9 graph self-healing: orphans, contradictions, confidence decay.

    Read-only unless --apply is given. Nothing here merges entities: gate 8 still shows
    false merges against gold, so an unattended cosine-only merge would worsen identity.
    """
    from ingestion.healer import GraphHealer

    settings = _settings(args)
    healer = GraphHealer(settings)
    try:
        report = healer.heal(dry_run=not args.apply)
    finally:
        healer.close()

    mode = "APPLIED" if args.apply else "DRY RUN"
    print(f"# Graph heal ({mode})")
    print()
    print(f"- orphan entities found : {report.orphans_found}  (pruned {report.orphans_pruned})")
    print(f"- type variance         : {report.type_conflicts}  (reported, never flagged)")
    print(f"- functional conflicts  : {report.functional_conflicts}")
    print(f"- entities flagged      : {report.contradictions_flagged}")
    print(
        f"- low confidence (<{settings.assertion_review_confidence_threshold}), uncorroborated: "
        f"{report.low_confidence_found}  (flagged {report.low_confidence_flagged})"
    )
    if report.contradictions:
        print()
        print("Sample contradictions:")
        for item in report.contradictions[:14]:
            print(f"  [{item.kind:20}] {item.subject_name[:40]:40} {item.detail[:60]}")
    if not report.low_confidence_found:
        print()
        print(
            "No assertion falls below the confidence threshold. The extractor's confidence"
            " is not calibrated for this - see ingestion/healer.py."
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG corpus administration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--db", default=default_metadata_db(), help="Metadata database path")

    reconcile = subparsers.add_parser("reconcile", help="Drive incomplete outbox events to completion")
    common(reconcile)
    reconcile.add_argument("--document-version-id", default=None, help="Limit to one version")
    reconcile.add_argument("--max-events", type=int, default=500)
    reconcile.set_defaults(func=command_reconcile)

    sweep = subparsers.add_parser("sweep-orphans", help="Retract rows left by abandoned versions")
    common(sweep)
    sweep.add_argument(
        "--older-than-hours",
        type=float,
        default=None,
        help="Override OUTBOX_ORPHAN_SWEEP_HOURS for this run",
    )
    sweep.add_argument("--dry-run", action="store_true")
    sweep.set_defaults(func=command_sweep_orphans)

    merges = subparsers.add_parser("list-merges", help="Audit recorded merge decisions")
    common(merges)
    merges.add_argument("--doc-id", default=None)
    merges.add_argument(
        "--decision",
        default=None,
        choices=["exact_merge", "auto_merge", "llm_merge", "new_entity", "rollback"],
    )
    merges.add_argument("--limit", type=int, default=50)
    merges.set_defaults(func=command_list_merges)

    rollback = subparsers.add_parser("rollback-merge", help="Reverse one audited merge")
    common(rollback)
    rollback.add_argument("--decision-id", required=True)
    rollback.add_argument("--dry-run", action="store_true", help="Show the effect, change nothing")
    rollback.set_defaults(func=command_rollback_merge)

    review = subparsers.add_parser(
        "review-status", help="Recompute which assertions are flagged for review"
    )
    common(review)
    review.add_argument("--dry-run", action="store_true", help="Report, change nothing")
    review.set_defaults(func=command_review_status)

    health = subparsers.add_parser("corpus-health", help="Measure the corpus")
    common(health)
    health.add_argument("--no-graph", action="store_true")
    health.set_defaults(func=command_corpus_health)

    contradictions = subparsers.add_parser(
        "contradictions", help="List subjects with conflicting values for one predicate"
    )
    common(contradictions)
    contradictions.add_argument("--doc-id", default=None, help="Restrict to one document")
    contradictions.add_argument("--limit", type=int, default=40, help="Pairs to print")
    contradictions.add_argument(
        "--min-single-ratio",
        type=float,
        default=0.9,
        help="Treat a predicate as single-valued when this share of its subjects carry one value",
    )
    contradictions.add_argument(
        "--min-subjects",
        type=int,
        default=5,
        help="Ignore predicates with fewer subjects than this",
    )
    contradictions.add_argument(
        "--raw",
        action="store_true",
        help="Every multi-valued pair, without inferring cardinality (mostly not conflicts)",
    )
    contradictions.set_defaults(func=command_contradictions)

    communities = subparsers.add_parser(
        "communities", help="Detect and summarise entity communities (Phase 8)"
    )
    common(communities)
    communities.add_argument("--min-size", type=int, default=None, help="Smallest community to keep")
    communities.add_argument("--limit", type=int, default=None, help="Cap communities summarised")
    communities.add_argument("--dry-run", action="store_true", help="Detect only; no provider calls")
    communities.set_defaults(func=command_communities)

    heal = subparsers.add_parser("heal", help="Graph self-healing (Phase 9)")
    common(heal)
    heal.add_argument("--apply", action="store_true", help="Write changes (default is a dry run)")
    heal.set_defaults(func=command_heal)

    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
