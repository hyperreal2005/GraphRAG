"""CLI for the Refactor 2 evaluation harness (docs/REFACTOR2.md WP1).

    python main_eval.py resolve        --db data/corpus.db
    python main_eval.py health         --db data/corpus.db
    python main_eval.py run            --db data/corpus.db --limit 60
    python main_eval.py baseline       --db data/corpus.db --limit 60

Read-only against SQLite, Neo4j and Qdrant. `resolve` and `health` make no provider
calls at all; `run` calls the query pipeline, and `--judge` adds the faithfulness judge.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from config import Settings, default_metadata_db
from eval import corpus_health, report
from eval.goldset import load_goldset, sample
from eval.judge import FaithfulnessJudge
from eval.metrics import summarize
from eval.resolve import CorpusIndex, format_unresolved, resolve_goldset
from ingestion.state_store import MetadataStore

DEFAULT_GOLDSET = Path("eval/goldset/multihoprag.jsonl")
DEFAULT_REPORT_ROOT = Path("artifacts/eval")


def _settings(args: argparse.Namespace) -> Settings:
    return Settings(metadata_db_path=Path(args.db))


def _goldset_paths(args: argparse.Namespace) -> list[Path]:
    if args.goldset:
        return [Path(item) for item in args.goldset]
    paths = [DEFAULT_GOLDSET]
    for extra in ("eval/goldset/fixtures.jsonl", "eval/goldset/docred.jsonl"):
        candidate = Path(extra)
        if candidate.exists():
            paths.append(candidate)
    return paths


def _report_dir(args: argparse.Namespace, label: str) -> Path:
    if args.report:
        return Path(args.report)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_REPORT_ROOT / f"{stamp}-{label}"


def _load_and_resolve(args: argparse.Namespace, settings: Settings):
    paths = _goldset_paths(args)
    records = load_goldset(*paths)
    print(f"Loaded {len(records)} gold records from {', '.join(str(item) for item in paths)}")

    store = MetadataStore(settings.metadata_db_path)
    index = CorpusIndex.from_store(store)
    print(f"Indexed {len(index)} active chunks from {settings.metadata_db_path}")
    if not len(index):
        raise SystemExit(
            "No active chunks found. Ingest the corpus before evaluating, or point --db "
            "at the right metadata database."
        )

    resolution = resolve_goldset(index, records)
    print(
        f"Resolved {resolution.resolved_count}/{resolution.checked_count} gold phrases "
        f"({resolution.resolution_rate:.1%})"
    )
    return records, resolution


def command_resolve(args: argparse.Namespace) -> int:
    settings = _settings(args)
    _, resolution = _load_and_resolve(args, settings)
    print()
    print(format_unresolved(resolution))
    # A stale gold record deflates recall silently, so it fails the command.
    return 0 if not resolution.unresolved else 1


def command_health(args: argparse.Namespace) -> int:
    settings = _settings(args)
    health = corpus_health.collect(settings, include_graph=not args.no_graph)
    print(corpus_health.render_markdown(health))
    if args.report:
        directory = Path(args.report)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "corpus_health.json").write_text(
            corpus_health.to_json(health), encoding="utf-8"
        )
        print(f"\nWrote {directory / 'corpus_health.json'}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    from eval import harness

    settings = _settings(args)
    started = time.perf_counter()

    problems = harness.preflight(settings)
    if problems:
        print("Preflight failed; refusing to spend provider budget on an unusable corpus:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nAn unreachable store does not error at query time: every stage times out "
            "and every metric reads 0.000, which is indistinguishable from a real regression."
        )
        return 2

    records, resolution = _load_and_resolve(args, settings)

    if resolution.unresolved and not args.allow_unresolved:
        print()
        print(format_unresolved(resolution))
        print("\nRefusing to run against a stale judged set. Pass --allow-unresolved to override.")
        return 1

    selected = sample(
        records,
        limit=args.limit,
        query_types=set(args.query_type) if args.query_type else None,
        seed=args.seed,
    )
    print(f"Running {len(selected)} of {len(records)} questions ...")

    result = harness.run(settings, selected, resolution, progress_every=args.progress_every)

    faithfulness = None
    if args.judge:
        print("Judging claim faithfulness ...")
        faithfulness = FaithfulnessJudge(settings).judge(result.outcomes, max_claims=args.max_claims)
        print(
            f"  judged {len(faithfulness.verdicts)} claims, "
            f"skipped {faithfulness.skipped}"
        )

    metrics = summarize(
        result.outcomes,
        linked_by_record=result.linked_by_record,
        faithfulness=faithfulness.summary if faithfulness else None,
        store=MetadataStore(settings.metadata_db_path),
    )
    health = corpus_health.collect(settings, include_graph=not args.no_graph) if args.health else None

    payload = report.build(
        settings=settings,
        label=args.label,
        metrics=metrics,
        health=health,
        faithfulness=faithfulness,
        extraction=None,
        resolution=resolution,
        outcomes=result.outcomes,
        wall_seconds=time.perf_counter() - started,
    )
    directory = _report_dir(args, args.label)
    json_path, markdown_path = report.write(directory, payload, health, None)

    print()
    print(report.render_markdown(payload, health, None))
    print(f"\nWrote {json_path}\nWrote {markdown_path}")
    return 0


def command_extraction(args: argparse.Namespace) -> int:
    """Score the live corpus against Re-DocRED gold (docs/CORPUS.md section 7.2)."""
    from eval import extraction as extraction_module
    from eval import extraction_run

    settings = _settings(args)
    started = time.perf_counter()
    report_data = extraction_run.run(settings, split=args.split, limit=args.limit)
    print(extraction_module.render_markdown(report_data))

    if args.report:
        payload = report.build(
            settings=settings,
            label="extraction",
            metrics=None,
            health=None,
            faithfulness=None,
            extraction=report_data,
            resolution=None,
            outcomes=None,
            wall_seconds=time.perf_counter() - started,
        )
        json_path, markdown_path = report.write(Path(args.report), payload, None, report_data)
        print(f"\nWrote {json_path}\nWrote {markdown_path}")
    return 0 if report_data.scores else 1


def command_baseline(args: argparse.Namespace) -> int:
    """The pre-WP2 baseline: query metrics plus corpus health, in one artifact."""
    args.health = True
    args.label = args.label if args.label != "eval" else "baseline"
    return command_run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG evaluation harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--db", default=default_metadata_db(), help="Metadata database path")
        sub.add_argument(
            "--goldset",
            action="append",
            help="Judged-set JSONL (repeatable). Defaults to eval/goldset/*.jsonl",
        )
        sub.add_argument("--report", help="Report output directory")
        sub.add_argument("--no-graph", action="store_true", help="Skip Neo4j-backed metrics")

    resolve_parser = subparsers.add_parser("resolve", help="Check every gold phrase resolves")
    common(resolve_parser)
    resolve_parser.set_defaults(func=command_resolve)

    health_parser = subparsers.add_parser("health", help="Measure corpus health only")
    common(health_parser)
    health_parser.set_defaults(func=command_health)

    extraction_parser = subparsers.add_parser(
        "extraction", help="Score extraction and coreference against Re-DocRED gold"
    )
    common(extraction_parser)
    extraction_parser.add_argument("--split", default="dev", help="Re-DocRED split")
    extraction_parser.add_argument("--limit", type=int, default=None, help="Gold document cap")
    extraction_parser.set_defaults(func=command_extraction)

    for name, handler in (("run", command_run), ("baseline", command_baseline)):
        run_parser = subparsers.add_parser(name, help=f"{name} the judged set end to end")
        common(run_parser)
        run_parser.add_argument("--limit", type=int, default=None, help="Question cap")
        run_parser.add_argument(
            "--query-type",
            action="append",
            choices=["local", "multihop", "comparative", "hybrid", "global"],
        )
        run_parser.add_argument("--seed", type=int, default=17, help="Sampling seed")
        run_parser.add_argument("--label", default="eval", help="Report label")
        run_parser.add_argument("--judge", action="store_true", help="Run the faithfulness judge")
        run_parser.add_argument("--max-claims", type=int, default=400, help="Judge call cap")
        run_parser.add_argument("--health", action="store_true", help="Include corpus health")
        run_parser.add_argument("--progress-every", type=int, default=10)
        run_parser.add_argument(
            "--allow-unresolved",
            action="store_true",
            help="Run even when some gold phrases no longer resolve",
        )
        run_parser.set_defaults(func=handler)

    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
