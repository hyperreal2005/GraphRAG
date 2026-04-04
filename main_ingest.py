"""CLI entrypoint for Phase 1 document ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from config import Settings
from ingestion.pipeline import PhaseOnePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG Phase 1 ingestion pipeline")
    parser.add_argument("--input", required=True, help="Input file or directory path")
    parser.add_argument(
        "--metadata-db",
        default="data/metadata.db",
        help="SQLite metadata database path",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/phase1",
        help="Directory for parsed/chunked artifacts",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively ingest files when input is a directory",
    )
    parser.add_argument(
        "--strict-coref",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail file ingestion when coreference model is unavailable or fails",
    )
    parser.add_argument(
        "--coref-profile",
        default="balanced",
        choices=["speed", "balanced", "quality"],
        help="FastCoref profile preference",
    )
    parser.add_argument(
        "--phase1-only",
        action="store_true",
        help="Run only Phase 1 stages (parse/coref/chunk) without graph/vector writes",
    )
    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings(
        metadata_db_path=Path(args.metadata_db),
        artifact_dir=Path(args.output_dir),
        strict_coref=args.strict_coref,
        coref_profile=args.coref_profile,
        enable_graph=not args.phase1_only,
    )
    pipeline = PhaseOnePipeline(settings=settings)
    summary = pipeline.ingest_path(Path(args.input), recursive=bool(args.recursive))
    print(json.dumps(summary.to_dict(), indent=2))
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
