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
    python main_admin.py corpus-health  --db data/corpus.db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from config import Settings
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
    print(f"{'decision_id'[:16]:<18} {'decision':<12} {'incoming':<28} {'candidate':<28} sim")
    for row in rows:
        reversed_marker = " [reversed]" if row["reversed_at"] else ""
        similarity = row["top1_similarity"]
        print(
            f"{str(row['decision_id'])[:16]:<18} {str(row['decision']):<12} "
            f"{str(row['incoming_name'])[:27]:<28} {str(row['candidate_name'] or '-')[:27]:<28} "
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


def command_corpus_health(args: argparse.Namespace) -> int:
    from eval import corpus_health

    print(corpus_health.render_markdown(
        corpus_health.collect(_settings(args), include_graph=not args.no_graph)
    ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG corpus administration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--db", default="data/metadata.db", help="Metadata database path")

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

    health = subparsers.add_parser("corpus-health", help="Measure the corpus")
    common(health)
    health.add_argument("--no-graph", action="store_true")
    health.set_defaults(func=command_corpus_health)

    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
