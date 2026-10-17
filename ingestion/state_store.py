"""SQLite metadata store for ingestion pipeline state."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from ingestion.document_cards import DocumentCard
from models.ingestion import Chunk
from models.refactor import DocumentVersion, MergeDecision, OutboxEvent, OutboxEventType


SCHEMA_VERSION = 5


def utc_now_iso() -> str:
    """Current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


class MetadataStore:
    """Encapsulates all SQLite reads/writes for Phase 1."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        return conn

    @contextmanager
    def _connection(self) -> sqlite3.Connection:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    format TEXT,
                    word_count INTEGER,
                    ingested_at TEXT,
                    status TEXT,
                    pipeline_stage TEXT,
                    last_error TEXT,
                    active_checksum TEXT,
                    desired_checksum TEXT,
                    last_completed_stage TEXT,
                    target_stage TEXT,
                    active_pipeline_fingerprint TEXT,
                    desired_pipeline_fingerprint TEXT,
                    current_run_id TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT REFERENCES documents(doc_id),
                    text TEXT,
                    token_count INTEGER,
                    position INTEGER,
                    section TEXT,
                    page INTEGER,
                    chunk_hash TEXT,
                    embedding_version TEXT,
                    content_kind TEXT NOT NULL DEFAULT 'body'
                );

                CREATE TABLE IF NOT EXISTS entity_mentions (
                    entity_id TEXT,
                    chunk_id TEXT,
                    doc_id TEXT,
                    confidence REAL
                );

                CREATE TABLE IF NOT EXISTS doc_entities (
                    doc_id TEXT,
                    entity_id TEXT,
                    canonical_name TEXT,
                    aliases TEXT,
                    PRIMARY KEY (doc_id, entity_id)
                );

                CREATE TABLE IF NOT EXISTS system_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE TABLE IF NOT EXISTS ingestion_log (
                    run_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    entities_created INTEGER,
                    relations_created INTEGER,
                    entities_merged INTEGER,
                    status TEXT
                );

                CREATE TABLE IF NOT EXISTS ingestion_stage_metrics (
                    metric_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    doc_id TEXT NOT NULL,
                    document_version_id TEXT,
                    duration_ms REAL NOT NULL,
                    status TEXT NOT NULL,
                    provider TEXT,
                    model_id TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    call_count INTEGER,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS document_versions (
                    version_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
                    checksum TEXT NOT NULL,
                    pipeline_fingerprint TEXT NOT NULL,
                    generative_model TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    embedding_prompt_version TEXT NOT NULL,
                    embedding_schema_version TEXT NOT NULL DEFAULT 'legacy',
                    embedding_normalization_policy TEXT NOT NULL DEFAULT 'provider-default',
                    document_embedding_instruction TEXT NOT NULL DEFAULT 'legacy',
                    query_embedding_instruction TEXT NOT NULL DEFAULT 'legacy',
                    extraction_prompt_version TEXT NOT NULL DEFAULT 'legacy',
                    graph_schema_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    last_completed_stage TEXT,
                    created_at TEXT NOT NULL,
                    activated_at TEXT,
                    superseded_at TEXT,
                    last_error TEXT
                );

                CREATE TABLE IF NOT EXISTS outbox_events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    document_version_id TEXT NOT NULL REFERENCES document_versions(version_id),
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at TEXT NOT NULL,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS merge_decisions (
                    decision_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    document_version_id TEXT NOT NULL REFERENCES document_versions(version_id),
                    incoming_local_entity_id TEXT NOT NULL,
                    incoming_name TEXT NOT NULL,
                    incoming_type TEXT NOT NULL,
                    candidate_entity_id TEXT,
                    candidate_name TEXT,
                    candidate_type TEXT,
                    top1_similarity REAL,
                    top2_similarity REAL,
                    score_margin REAL,
                    type_compatible INTEGER NOT NULL,
                    decision TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    model_id TEXT,
                    prompt_version TEXT,
                    adjudication_confidence REAL,
                    reverses_decision_id TEXT,
                    created_at TEXT NOT NULL,
                    reversed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS chunk_extraction_failures (
                    document_version_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (document_version_id, chunk_id)
                );

                CREATE TABLE IF NOT EXISTS version_doc_entities (
                    document_version_id TEXT NOT NULL REFERENCES document_versions(version_id),
                    doc_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    aliases TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (document_version_id, entity_id)
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_hash
                    ON chunks (doc_id, chunk_hash);

                CREATE INDEX IF NOT EXISTS idx_entity_mentions_doc
                    ON entity_mentions (doc_id);

                CREATE INDEX IF NOT EXISTS idx_doc_entities_name
                    ON doc_entities (doc_id, canonical_name);

                CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_document_version
                    ON document_versions (doc_id) WHERE status = 'active';

                CREATE INDEX IF NOT EXISTS idx_document_versions_doc_status
                    ON document_versions (doc_id, status);

                CREATE INDEX IF NOT EXISTS idx_outbox_ready
                    ON outbox_events (status, available_at);

                CREATE INDEX IF NOT EXISTS idx_outbox_version_type
                    ON outbox_events (document_version_id, event_type);

                CREATE INDEX IF NOT EXISTS idx_merge_incoming
                    ON merge_decisions (document_version_id, incoming_local_entity_id);

                CREATE INDEX IF NOT EXISTS idx_merge_candidate
                    ON merge_decisions (candidate_entity_id);

                CREATE INDEX IF NOT EXISTS idx_version_doc_entities_active
                    ON version_doc_entities (doc_id, active);

                CREATE INDEX IF NOT EXISTS idx_ingestion_stage_metrics_version
                    ON ingestion_stage_metrics (document_version_id, stage);
                """
            )
            self._migrate_documents_table(conn)
            self._migrate_version_columns(conn)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _migrate_documents_table(self, conn: sqlite3.Connection) -> None:
        """Add resumable-ingestion columns to databases created by earlier phases."""
        columns = {str(row["name"]) for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
        additions = {
            "active_checksum": "TEXT",
            "desired_checksum": "TEXT",
            "last_completed_stage": "TEXT",
            "target_stage": "TEXT",
            "active_pipeline_fingerprint": "TEXT",
            "desired_pipeline_fingerprint": "TEXT",
            "current_run_id": "TEXT",
            # WP23 document cards: shown with every passage from the document.
            "title": "TEXT",
            "publisher": "TEXT",
            "published": "TEXT",
        }
        for name, sql_type in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE documents ADD COLUMN {name} {sql_type}")

        if "checksum" in columns:
            # Legacy rows carried the content hash in `documents.checksum`; move it into
            # the generation columns before the column is dropped below.
            conn.execute(
                """
                UPDATE documents
                SET active_checksum = CASE
                        WHEN active_checksum IS NULL AND status = 'complete' THEN checksum
                        ELSE active_checksum
                    END,
                    desired_checksum = coalesce(desired_checksum, checksum)
                """
            )
        conn.execute(
            """
            UPDATE documents
            SET last_completed_stage = CASE
                    WHEN last_completed_stage IS NULL AND pipeline_stage <> 'FAILED' THEN pipeline_stage
                    ELSE last_completed_stage
                END
            """
        )
        self._drop_legacy_checksum_column(conn, columns)

    def _drop_legacy_checksum_column(self, conn: sqlite3.Connection, columns: set[str]) -> None:
        """Remove `documents.checksum` (WP5-A, REVIEW1 finding `B8`).

        The column was written as NULL by every current code path while
        `_is_active_generation_current` still read it as a fallback, so a legacy row and a
        fresh row answered "is this generation current?" by different rules. SQLite cannot
        `DROP COLUMN` a UNIQUE column, so the table is rebuilt; foreign keys are suspended
        for the swap because `chunks.doc_id` references it by name.
        """
        if "checksum" not in columns:
            return
        # `PRAGMA foreign_keys` is silently ignored while a transaction is open, and this
        # runs inside `_initialize`'s transaction. The rebuild therefore needs its own
        # autocommit connection, where the pragma actually takes effect — without it
        # `DROP TABLE documents` fails on every child row in `chunks`.
        conn.commit()
        rebuild = sqlite3.connect(self.db_path, timeout=5.0)
        rebuild.row_factory = sqlite3.Row
        try:
            rebuild.isolation_level = None
            rebuild.execute("PRAGMA foreign_keys = OFF")
            self._rebuild_documents_without_checksum(rebuild)
            rebuild.execute("PRAGMA foreign_keys = ON")
            violations = rebuild.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise RuntimeError(
                    f"documents rebuild left {len(violations)} dangling reference(s); "
                    "the legacy checksum column was not dropped."
                )
        finally:
            rebuild.close()

    def _rebuild_documents_without_checksum(self, conn: sqlite3.Connection) -> None:
        """Copy `documents` into a table without `checksum`, then swap it in."""
        keep = [
            "doc_id", "source_path", "format", "word_count", "ingested_at", "status",
            "pipeline_stage", "last_error", "active_checksum", "desired_checksum",
            "last_completed_stage", "target_stage", "active_pipeline_fingerprint",
            "desired_pipeline_fingerprint", "current_run_id", "title", "publisher", "published",
        ]
        column_list = ", ".join(keep)
        # The caller has already disabled foreign keys, which SQLite only permits outside
        # a transaction. Everything below is one transaction so a failure cannot leave the
        # database with `documents` dropped and nothing in its place.
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                """
                CREATE TABLE documents_wp5a (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    format TEXT,
                    word_count INTEGER,
                    ingested_at TEXT,
                    status TEXT,
                    pipeline_stage TEXT,
                    last_error TEXT,
                    active_checksum TEXT,
                    desired_checksum TEXT,
                    last_completed_stage TEXT,
                    target_stage TEXT,
                    active_pipeline_fingerprint TEXT,
                    desired_pipeline_fingerprint TEXT,
                    current_run_id TEXT,
                    title TEXT,
                    publisher TEXT,
                    published TEXT
                )
                """
            )
            conn.execute(
                f"INSERT INTO documents_wp5a ({column_list}) SELECT {column_list} FROM documents"
            )
            conn.execute("DROP TABLE documents")
            conn.execute("ALTER TABLE documents_wp5a RENAME TO documents")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def _migrate_version_columns(self, conn: sqlite3.Connection) -> None:
        """Add nullable Refactor 1 version keys without rewriting legacy rows."""
        additions = {
            "chunks": {
                "document_version_id": "TEXT",
                "active": "INTEGER NOT NULL DEFAULT 1",
                "content_kind": "TEXT NOT NULL DEFAULT 'body'",
                # WP2-D: change detection that survives a position shift.
                "content_hash": "TEXT",
                # WP20: what extraction read, when it was not `text` (coreference annotations).
                # Extraction reuse compares inputs, so it needs the old one.
                "extraction_text": "TEXT",
            },
            "entity_mentions": {"document_version_id": "TEXT", "active": "INTEGER NOT NULL DEFAULT 1"},
            "doc_entities": {"document_version_id": "TEXT", "active": "INTEGER NOT NULL DEFAULT 1"},
            "ingestion_log": {"document_version_id": "TEXT"},
        }
        additions["entity_mentions"]["local_entity_id"] = "TEXT"
        additions["version_doc_entities"] = {
            # WP2-B: the emergent schema, kept as evidence rather than as a gate.
            "entity_types_json": "TEXT",
            "coarse_class": "TEXT",
        }
        additions["document_versions"] = {
            "embedding_schema_version": "TEXT NOT NULL DEFAULT 'legacy'",
            "embedding_normalization_policy": "TEXT NOT NULL DEFAULT 'provider-default'",
            "document_embedding_instruction": "TEXT NOT NULL DEFAULT 'legacy'",
            "query_embedding_instruction": "TEXT NOT NULL DEFAULT 'legacy'",
            "extraction_prompt_version": "TEXT NOT NULL DEFAULT 'legacy'",
        }
        for table, columns_to_add in additions.items():
            existing = {
                str(row["name"])
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for name, sql_type in columns_to_add.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")

        conn.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_version_active
                ON chunks (document_version_id, active);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_content_hash
                ON chunks (doc_id, content_hash);
            CREATE INDEX IF NOT EXISTS idx_mentions_version_active
                ON entity_mentions (document_version_id, active);
            CREATE INDEX IF NOT EXISTS idx_doc_entities_version_active
                ON doc_entities (document_version_id, active);
            DROP INDEX IF EXISTS idx_mentions_version_entity_chunk;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_mentions_version_local_chunk
                ON entity_mentions (document_version_id, local_entity_id, chunk_id)
                WHERE document_version_id IS NOT NULL AND local_entity_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_mentions_version_final_chunk_legacy
                ON entity_mentions (document_version_id, entity_id, chunk_id)
                WHERE document_version_id IS NOT NULL AND local_entity_id IS NULL;
            """
        )

    @staticmethod
    def build_outbox_event_id(
        document_version_id: str,
        event_type: OutboxEventType,
        discriminator: str = "",
    ) -> str:
        """Build a deterministic event identity for idempotent enqueue."""
        payload = f"{document_version_id}|{event_type}|{discriminator}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def create_document_version(self, version: DocumentVersion) -> None:
        """Persist an immutable document version before processing begins."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO document_versions (
                    version_id, doc_id, checksum, pipeline_fingerprint,
                    generative_model, embedding_model, embedding_dim,
                    embedding_prompt_version, embedding_schema_version,
                    embedding_normalization_policy, document_embedding_instruction,
                    query_embedding_instruction, extraction_prompt_version,
                    graph_schema_version, status,
                    last_completed_stage, created_at, activated_at,
                    superseded_at, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version.version_id,
                    version.doc_id,
                    version.checksum,
                    version.pipeline_fingerprint,
                    version.generative_model,
                    version.embedding_model,
                    version.embedding_dim,
                    version.embedding_prompt_version,
                    version.embedding_schema_version,
                    version.embedding_normalization_policy,
                    version.document_embedding_instruction,
                    version.query_embedding_instruction,
                    version.extraction_prompt_version,
                    version.graph_schema_version,
                    version.status,
                    version.last_completed_stage,
                    version.created_at.isoformat(),
                    version.activated_at.isoformat() if version.activated_at else None,
                    version.superseded_at.isoformat() if version.superseded_at else None,
                    version.last_error,
                ),
            )

    def get_document_version(self, version_id: str) -> DocumentVersion | None:
        """Fetch one validated document version."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM document_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
        return self._document_version_from_row(row) if row else None

    def get_active_document_version(self, doc_id: str) -> DocumentVersion | None:
        """Fetch the active immutable version for a document."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM document_versions WHERE doc_id = ? AND status = 'active'",
                (doc_id,),
            ).fetchone()
        return self._document_version_from_row(row) if row else None

    def get_queryable_document_version_ids(self) -> list[str] | None:
        """Return SQLite-authorized corpus versions, or ``None`` for a legacy corpus.

        An empty list is intentionally distinct from ``None``: it means versioned
        records exist but none is active, so query-plane stores must return no data.
        ``None`` keeps read compatibility only for databases that predate immutable
        document versions entirely.
        """
        with self._connection() as conn:
            versioned = conn.execute("SELECT 1 FROM document_versions LIMIT 1").fetchone()
            if versioned is None:
                return None
            rows = conn.execute(
                "SELECT version_id FROM document_versions WHERE status = 'active' ORDER BY version_id"
            ).fetchall()
        return [str(row["version_id"]) for row in rows]

    def get_all_document_version_ids(self) -> list[str] | None:
        """Every version this store has recorded, in any status; ``None`` for a legacy store.

        Entities carry the versions that produced them (`source_versions`), including
        superseded ones, so deciding which graph nodes belong to this store needs all of them,
        not only the active set that fences reads.
        """
        with self._connection() as conn:
            rows = conn.execute("SELECT version_id FROM document_versions ORDER BY version_id").fetchall()
        return [str(row["version_id"]) for row in rows] or None

    def list_document_versions(self, doc_id: str) -> list[DocumentVersion]:
        """List document versions newest first."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM document_versions WHERE doc_id = ? ORDER BY created_at DESC",
                (doc_id,),
            ).fetchall()
        return [self._document_version_from_row(row) for row in rows]

    def list_abandoned_document_versions(self, created_before: str) -> list[DocumentVersion]:
        """Versions stuck mid-build, old enough that no run is plausibly still working.

        An `active` version is excluded by the status filter and can never be returned.
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM document_versions
                WHERE status IN ('building', 'staged', 'failed')
                  AND created_at < ?
                ORDER BY created_at ASC
                """,
                (created_before,),
            ).fetchall()
        return [self._document_version_from_row(row) for row in rows]

    def supersede_abandoned_version(self, version_id: str, reason: str) -> None:
        """Retire a swept version without ever disturbing the active one."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE document_versions
                SET status = 'superseded', superseded_at = ?, last_error = ?
                WHERE version_id = ? AND status <> 'active'
                """,
                (utc_now_iso(), reason, version_id),
            )

    def stage_document_version(self, version_id: str, last_completed_stage: str) -> None:
        """Record a successfully staged external-store checkpoint."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE document_versions
                SET status = 'staged', last_completed_stage = ?, last_error = NULL
                WHERE version_id = ? AND status IN ('building', 'staged')
                """,
                (last_completed_stage, version_id),
            )

    def fail_document_version(self, version_id: str, error_message: str) -> None:
        """Fail only the candidate version; never replace the active version."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE document_versions
                SET status = 'failed', last_error = ?
                WHERE version_id = ? AND status <> 'active'
                """,
                (error_message, version_id),
            )

    def activate_document_version(self, version_id: str) -> str | None:
        """Atomically swap the active SQLite version and return the superseded ID."""
        now = utc_now_iso()
        with self._connection() as conn:
            candidate = conn.execute(
                "SELECT doc_id FROM document_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            if candidate is None:
                raise KeyError(f"Unknown document version: {version_id}")
            doc_id = str(candidate["doc_id"])
            previous = conn.execute(
                """
                SELECT version_id FROM document_versions
                WHERE doc_id = ? AND status = 'active' AND version_id <> ?
                """,
                (doc_id, version_id),
            ).fetchone()
            previous_id = str(previous["version_id"]) if previous else None
            conn.execute(
                """
                UPDATE document_versions
                SET status = 'superseded', superseded_at = ?
                WHERE doc_id = ? AND status = 'active' AND version_id <> ?
                """,
                (now, doc_id, version_id),
            )
            updated = conn.execute(
                """
                UPDATE document_versions
                SET status = 'active', activated_at = ?, superseded_at = NULL,
                    last_completed_stage = 'ACTIVE', last_error = NULL
                WHERE version_id = ? AND status IN ('building', 'staged')
                """,
                (now, version_id),
            )
            if updated.rowcount != 1:
                raise RuntimeError(f"Document version {version_id} is not activatable.")
            conn.execute(
                "UPDATE chunks SET active = 0 WHERE doc_id = ? AND document_version_id <> ?",
                (doc_id, version_id),
            )
            conn.execute(
                "UPDATE chunks SET active = 1 WHERE document_version_id = ?",
                (version_id,),
            )
            conn.execute(
                "UPDATE entity_mentions SET active = 0 WHERE doc_id = ? AND document_version_id <> ?",
                (doc_id, version_id),
            )
            conn.execute(
                "UPDATE entity_mentions SET active = 1 WHERE document_version_id = ?",
                (version_id,),
            )
            conn.execute(
                "UPDATE version_doc_entities SET active = 0 WHERE doc_id = ? AND document_version_id <> ?",
                (doc_id, version_id),
            )
            conn.execute(
                "UPDATE version_doc_entities SET active = 1 WHERE document_version_id = ?",
                (version_id,),
            )
            # Keep the legacy convenience table as a projection of the active
            # immutable version while all query code is migrated.
            conn.execute("DELETE FROM doc_entities WHERE doc_id = ?", (doc_id,))
            conn.execute(
                """
                INSERT INTO doc_entities (
                    doc_id, entity_id, canonical_name, aliases,
                    document_version_id, active
                )
                SELECT doc_id, entity_id, canonical_name, aliases,
                       document_version_id, 1
                FROM version_doc_entities
                WHERE document_version_id = ?
                """,
                (version_id,),
            )
        return previous_id

    def complete_document_version(self, version_id: str) -> None:
        """Record completion only after activation and required retraction finish."""
        with self._connection() as conn:
            updated = conn.execute(
                """
                UPDATE document_versions
                SET last_completed_stage = 'DONE', last_error = NULL
                WHERE version_id = ? AND status = 'active'
                """,
                (version_id,),
            )
            if updated.rowcount != 1:
                raise RuntimeError(f"Document version {version_id} is not active.")

    def record_document_version_error(self, version_id: str, error_message: str) -> None:
        """Attach reconciliation failure details without corrupting lifecycle status."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE document_versions SET last_error = ? WHERE version_id = ?",
                (error_message, version_id),
            )

    def enqueue_outbox_event(self, event: OutboxEvent) -> None:
        """Insert an event once; replaying the same enqueue is a no-op."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO outbox_events (
                    event_id, event_type, run_id, doc_id, document_version_id,
                    payload_json, status, attempts, available_at, last_error,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.run_id,
                    event.doc_id,
                    event.document_version_id,
                    event.payload_json,
                    event.status,
                    event.attempts,
                    event.available_at.isoformat(),
                    event.last_error,
                    event.created_at.isoformat(),
                    event.completed_at.isoformat() if event.completed_at else None,
                ),
            )

    def claim_next_outbox_event(
        self,
        max_attempts: int = 5,
        *,
        document_version_id: str | None = None,
        lease_seconds: float = 60.0,
    ) -> OutboxEvent | None:
        """Claim ready work, including a processing event whose lease expired."""
        now = utc_now_iso()
        lease_until = (
            datetime.now(timezone.utc) + timedelta(seconds=max(1.0, lease_seconds))
        ).isoformat()
        version_clause = "AND document_version_id = ?" if document_version_id is not None else ""
        select_params: list[object] = [now, max_attempts]
        if document_version_id is not None:
            select_params.append(document_version_id)
        with self._connection() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM outbox_events
                WHERE status IN ('pending', 'failed', 'processing')
                  AND available_at <= ?
                  AND attempts < ?
                  {version_clause}
                ORDER BY created_at ASC
                LIMIT 1
                """,
                select_params,
            ).fetchone()
            if row is None:
                return None
            event_id = str(row["event_id"])
            updated = conn.execute(
                """
                UPDATE outbox_events
                SET status = 'processing', attempts = attempts + 1, available_at = ?
                WHERE event_id = ? AND status IN ('pending', 'failed', 'processing')
                """,
                (lease_until, event_id),
            )
            if updated.rowcount != 1:
                return None
            claimed = conn.execute(
                "SELECT * FROM outbox_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return self._outbox_event_from_row(claimed)

    def complete_outbox_event(self, event_id: str) -> None:
        """Mark an idempotent external-store event complete."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE outbox_events
                SET status = 'completed', completed_at = ?, last_error = NULL
                WHERE event_id = ?
                """,
                (utc_now_iso(), event_id),
            )

    def fail_outbox_event(
        self,
        event_id: str,
        error_message: str,
        *,
        max_attempts: int = 5,
        retry_base_seconds: float = 1.0,
    ) -> str:
        """Schedule retry or dead-letter an exhausted event; return new status."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT attempts FROM outbox_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown outbox event: {event_id}")
            attempts = int(row["attempts"] or 0)
            if attempts >= max_attempts:
                status = "dead_letter"
                available_at = utc_now_iso()
            else:
                status = "failed"
                delay = max(0.0, retry_base_seconds) * (2 ** max(0, attempts - 1))
                available_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat()
            conn.execute(
                """
                UPDATE outbox_events
                SET status = ?, available_at = ?, last_error = ?, completed_at = NULL
                WHERE event_id = ?
                """,
                (status, available_at, error_message, event_id),
            )
        return status

    def list_outbox_events(
        self,
        *,
        document_version_id: str | None = None,
        status: str | None = None,
    ) -> list[OutboxEvent]:
        """List validated outbox events for reconciliation/observability."""
        clauses: list[str] = []
        params: list[str] = []
        if document_version_id is not None:
            clauses.append("document_version_id = ?")
            params.append(document_version_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM outbox_events {where} ORDER BY created_at ASC",
                params,
            ).fetchall()
        return [self._outbox_event_from_row(row) for row in rows]

    def has_pending_corpus_update(self) -> bool:
        """Return whether activation/reconciliation work remains incomplete."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM outbox_events
                WHERE status IN ('pending', 'processing', 'failed', 'dead_letter')
                LIMIT 1
                """
            ).fetchone()
        return row is not None

    def record_merge_decision(self, decision: MergeDecision) -> None:
        """Persist a merge decision before applying its graph mapping."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO merge_decisions (
                    decision_id, run_id, doc_id, document_version_id,
                    incoming_local_entity_id, incoming_name, incoming_type,
                    candidate_entity_id, candidate_name, candidate_type,
                    top1_similarity, top2_similarity, score_margin,
                    type_compatible, decision, rationale, model_id,
                    prompt_version, adjudication_confidence,
                    reverses_decision_id, created_at, reversed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.run_id,
                    decision.doc_id,
                    decision.document_version_id,
                    decision.incoming_local_entity_id,
                    decision.incoming_name,
                    decision.incoming_type,
                    decision.candidate_entity_id,
                    decision.candidate_name,
                    decision.candidate_type,
                    decision.top1_similarity,
                    decision.top2_similarity,
                    decision.score_margin,
                    int(decision.type_compatible),
                    decision.decision,
                    decision.rationale,
                    decision.model_id,
                    decision.prompt_version,
                    decision.adjudication_confidence,
                    decision.reverses_decision_id,
                    decision.created_at.isoformat(),
                    decision.reversed_at.isoformat() if decision.reversed_at else None,
                ),
            )

    def list_merge_decisions(
        self,
        *,
        doc_id: str | None = None,
        decision: str | None = None,
        limit: int = 50,
    ) -> list[sqlite3.Row]:
        """List recorded merge decisions for audit (WP3-A)."""
        clauses: list[str] = []
        params: list[object] = []
        if doc_id:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if decision:
            clauses.append("decision = ?")
            params.append(decision)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))
        with self._connection() as conn:
            return conn.execute(
                f"""SELECT * FROM merge_decisions {where}
                    ORDER BY created_at DESC LIMIT ?""",
                params,
            ).fetchall()

    def get_merge_decision(self, decision_id: str) -> MergeDecision | None:
        """Fetch one merge decision for audit or rollback."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM merge_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        return self._merge_decision_from_row(row) if row else None

    def get_local_entities_for_version(
        self,
        doc_id: str,
        document_version_id: str,
    ) -> list[dict[str, object]]:
        """Rebuild prior document-local identities without global merge leakage."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT incoming_local_entity_id, incoming_name, incoming_type
                FROM merge_decisions
                WHERE doc_id = ? AND document_version_id = ?
                  AND reverses_decision_id IS NULL
                ORDER BY created_at ASC, decision_id ASC
                """,
                (doc_id, document_version_id),
            ).fetchall()
        by_local_id: dict[str, dict[str, object]] = {}
        for row in rows:
            local_id = str(row["incoming_local_entity_id"] or "").strip()
            name = str(row["incoming_name"] or "").strip()
            if not local_id or not name:
                continue
            by_local_id[local_id] = {
                "entity_id": local_id,
                "canonical_name": name,
                "aliases": [name],
                "entity_type": str(row["incoming_type"] or "Entity"),
            }
        if by_local_id:
            return [by_local_id[key] for key in sorted(by_local_id)]
        return self.get_doc_entities(doc_id, document_version_id, active_only=True)

    def mark_merge_decision_reversed(self, decision_id: str, reversed_at: str | None = None) -> None:
        """Mark the original decision reversed without deleting its history."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE merge_decisions SET reversed_at = ? WHERE decision_id = ?",
                (reversed_at or utc_now_iso(), decision_id),
            )

    def stage_version_records(
        self,
        *,
        document_version_id: str,
        chunks: Sequence[Chunk],
        mentions: Sequence[tuple[str, str, str, float] | tuple[str, str, str, float, str]],
        doc_entities: Sequence[dict[str, object]],
        initial_event: OutboxEvent,
    ) -> None:
        """Atomically stage SQLite records and the first external-store event."""
        if initial_event.document_version_id != document_version_id:
            raise ValueError("Initial outbox event belongs to a different document version.")
        chunk_ids = {chunk.chunk_id for chunk in chunks}
        mention_entity_ids = {str(mention[0]) for mention in mentions}
        mention_chunk_ids = {str(mention[1]) for mention in mentions}
        doc_entity_ids = {str(entity["entity_id"]) for entity in doc_entities}
        entities_without_mentions = doc_entity_ids - mention_entity_ids
        mentions_without_entities = mention_entity_ids - doc_entity_ids
        mentions_without_chunks = mention_chunk_ids - chunk_ids
        if entities_without_mentions:
            raise ValueError(
                "Document-version entities without current mentions: "
                f"{sorted(entities_without_mentions)}"
            )
        if mentions_without_entities:
            raise ValueError(
                "Document-version mentions without entity mappings: "
                f"{sorted(mentions_without_entities)}"
            )
        if mentions_without_chunks:
            raise ValueError(
                "Document-version mentions reference missing chunks: "
                f"{sorted(mentions_without_chunks)}"
            )
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, text, token_count, position, section, page,
                    chunk_hash, embedding_version, content_kind, content_hash,
                    document_version_id, extraction_text, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    text = excluded.text,
                    token_count = excluded.token_count,
                    position = excluded.position,
                    section = excluded.section,
                    page = excluded.page,
                    chunk_hash = excluded.chunk_hash,
                    embedding_version = excluded.embedding_version,
                    content_kind = excluded.content_kind,
                    content_hash = excluded.content_hash,
                    document_version_id = excluded.document_version_id,
                    extraction_text = excluded.extraction_text,
                    active = 0
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.text,
                        chunk.token_count,
                        chunk.position,
                        chunk.section_heading,
                        chunk.page_number,
                        chunk.chunk_hash,
                        None,
                        chunk.content_kind,
                        chunk.content_hash,
                        document_version_id,
                        chunk.extraction_text,
                    )
                    for chunk in chunks
                ],
            )
            conn.executemany(
                """
                INSERT INTO entity_mentions (
                    entity_id, chunk_id, doc_id, confidence, document_version_id, active,
                    local_entity_id
                ) VALUES (?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT DO UPDATE SET
                    entity_id = excluded.entity_id,
                    confidence = excluded.confidence,
                    doc_id = excluded.doc_id,
                    active = 0
                """,
                [
                    (
                        mention[0], mention[1], mention[2], mention[3], document_version_id,
                        mention[4] if len(mention) > 4 else mention[0],
                    )
                    for mention in mentions
                ],
            )

            conn.executemany(
                """
                INSERT INTO version_doc_entities (
                    document_version_id, doc_id, entity_id, canonical_name, aliases, active,
                    entity_types_json, coarse_class
                ) VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                ON CONFLICT(document_version_id, entity_id) DO UPDATE SET
                    canonical_name = excluded.canonical_name,
                    aliases = excluded.aliases,
                    entity_types_json = excluded.entity_types_json,
                    coarse_class = excluded.coarse_class,
                    active = 0
                """,
                [
                    (
                        document_version_id,
                        str(entity["doc_id"]),
                        str(entity["entity_id"]),
                        str(entity["canonical_name"]),
                        json.dumps(entity.get("aliases", [])),
                        json.dumps(entity.get("entity_types", {}), sort_keys=True),
                        str(entity.get("coarse_class") or "UNKNOWN"),
                    )
                    for entity in doc_entities
                ],
            )
            conn.execute(
                """
                INSERT INTO outbox_events (
                    event_id, event_type, run_id, doc_id, document_version_id,
                    payload_json, status, attempts, available_at, last_error,
                    created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (
                    initial_event.event_id,
                    initial_event.event_type,
                    initial_event.run_id,
                    initial_event.doc_id,
                    initial_event.document_version_id,
                    initial_event.payload_json,
                    initial_event.status,
                    initial_event.attempts,
                    initial_event.available_at.isoformat(),
                    initial_event.last_error,
                    initial_event.created_at.isoformat(),
                    initial_event.completed_at.isoformat() if initial_event.completed_at else None,
                ),
            )

    def rollback_version_entity_mapping(
        self,
        *,
        document_version_id: str,
        local_entity_id: str,
        merged_entity_id: str,
        restored_entity_id: str,
        canonical_name: str,
        aliases: Sequence[str],
    ) -> int:
        """Remap only mentions owned by one audited local entity identity."""
        with self._connection() as conn:
            version = conn.execute(
                "SELECT doc_id, status FROM document_versions WHERE version_id = ?",
                (document_version_id,),
            ).fetchone()
            if version is None:
                raise KeyError(document_version_id)
            doc_id = str(version["doc_id"])
            updated = conn.execute(
                """
                UPDATE entity_mentions
                SET entity_id = ?
                WHERE document_version_id = ?
                  AND local_entity_id = ?
                  AND entity_id = ?
                """,
                (restored_entity_id, document_version_id, local_entity_id, merged_entity_id),
            )
            if updated.rowcount == 0:
                raise RuntimeError("No affected entity mentions were found for merge rollback.")
            conn.execute(
                """
                INSERT INTO version_doc_entities (
                    document_version_id, doc_id, entity_id, canonical_name, aliases, active
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_version_id, entity_id) DO UPDATE SET
                    canonical_name = excluded.canonical_name,
                    aliases = excluded.aliases,
                    active = excluded.active
                """,
                (
                    document_version_id,
                    doc_id,
                    restored_entity_id,
                    canonical_name,
                    json.dumps(list(dict.fromkeys(aliases))),
                    int(str(version["status"]) == "active"),
                ),
            )
            remaining = conn.execute(
                """
                SELECT 1 FROM entity_mentions
                WHERE document_version_id = ? AND entity_id = ?
                LIMIT 1
                """,
                (document_version_id, merged_entity_id),
            ).fetchone()
            if remaining is None:
                conn.execute(
                    """
                    DELETE FROM version_doc_entities
                    WHERE document_version_id = ? AND entity_id = ?
                    """,
                    (document_version_id, merged_entity_id),
                )
            if str(version["status"]) == "active":
                conn.execute("DELETE FROM doc_entities WHERE doc_id = ?", (doc_id,))
                conn.execute(
                    """
                    INSERT INTO doc_entities (
                        doc_id, entity_id, canonical_name, aliases,
                        document_version_id, active
                    )
                    SELECT doc_id, entity_id, canonical_name, aliases,
                           document_version_id, 1
                    FROM version_doc_entities
                    WHERE document_version_id = ?
                    """,
                    (document_version_id,),
                )
        return int(updated.rowcount)

    @staticmethod
    def _document_version_from_row(row: sqlite3.Row) -> DocumentVersion:
        return DocumentVersion.model_validate(dict(row))

    @staticmethod
    def _outbox_event_from_row(row: sqlite3.Row) -> OutboxEvent:
        return OutboxEvent.model_validate(dict(row))

    @staticmethod
    def _merge_decision_from_row(row: sqlite3.Row) -> MergeDecision:
        payload = dict(row)
        payload["type_compatible"] = bool(payload.get("type_compatible"))
        return MergeDecision.model_validate(payload)

    def get_document_by_source(self, source_path: str) -> sqlite3.Row | None:
        """Fetch document row by source path."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        return row

    def get_document_by_id(self, doc_id: str) -> sqlite3.Row | None:
        """Fetch document row by document id."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        return row

    def set_document_card(self, doc_id: str, card: DocumentCard) -> None:
        """Record what the document is - title, publisher, date. Metadata only: it changes
        no chunk and no extraction, so it is not part of the pipeline fingerprint."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE documents SET title = ?, publisher = ?, published = ? WHERE doc_id = ?",
                (card.title or None, card.publisher or None, card.published or None, doc_id),
            )

    def get_document_cards(self, doc_ids: Iterable[str]) -> dict[str, DocumentCard]:
        ids = sorted({str(doc_id) for doc_id in doc_ids if doc_id})
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT doc_id, title, publisher, published FROM documents "
                f"WHERE doc_id IN ({placeholders})",
                ids,
            ).fetchall()
        cards = {
            str(row["doc_id"]): DocumentCard(
                title=str(row["title"] or ""),
                publisher=str(row["publisher"] or ""),
                published=str(row["published"] or ""),
            )
            for row in rows
        }
        return {doc_id: card for doc_id, card in cards.items() if card}

    def list_documents(self, status: str | None = None, pipeline_stage: str | None = None) -> list[sqlite3.Row]:
        """List documents filtered by optional status/stage."""
        clauses = []
        params: list[str] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if pipeline_stage is not None:
            clauses.append("pipeline_stage = ?")
            params.append(pipeline_stage)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT * FROM documents {where_clause} ORDER BY ingested_at ASC"
        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return rows

    def start_document(
        self,
        doc_id: str,
        source_path: str,
        doc_format: str,
        checksum: str,
        target_stage: str = "PARSING",
        pipeline_fingerprint: str = "legacy",
        run_id: str | None = None,
    ) -> None:
        """Record desired state without replacing the last active generation."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, source_path, format, word_count,
                    ingested_at, status, pipeline_stage, last_error,
                    active_checksum, desired_checksum, last_completed_stage,
                    target_stage, active_pipeline_fingerprint,
                    desired_pipeline_fingerprint, current_run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    format = excluded.format,
                    desired_checksum = excluded.desired_checksum,
                    desired_pipeline_fingerprint = excluded.desired_pipeline_fingerprint,
                    target_stage = excluded.target_stage,
                    current_run_id = excluded.current_run_id,
                    status = excluded.status,
                    pipeline_stage = excluded.pipeline_stage,
                    last_error = excluded.last_error
                """,
                (
                    doc_id,
                    source_path,
                    doc_format,
                    0,
                    utc_now_iso(),
                    "processing",
                    "PARSING",
                    None,
                    None,
                    checksum,
                    None,
                    target_stage,
                    None,
                    pipeline_fingerprint,
                    run_id,
                ),
            )

    def finish_document(self, doc_id: str, word_count: int, pipeline_stage: str) -> None:
        """Promote the desired generation to active after its target stage succeeds."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET word_count = ?,
                    ingested_at = ?,
                    status = ?,
                    pipeline_stage = ?,
                    last_completed_stage = ?,
                    active_checksum = coalesce(desired_checksum, active_checksum),
                    active_pipeline_fingerprint = coalesce(
                        desired_pipeline_fingerprint,
                        active_pipeline_fingerprint
                    ),
                    last_error = NULL
                WHERE doc_id = ?
                """,
                (word_count, utc_now_iso(), "complete", pipeline_stage, pipeline_stage, doc_id),
            )

    def record_chunk_extraction_failure(
        self,
        *,
        document_version_id: str,
        chunk_id: str,
        attempts: int,
        error: str,
    ) -> None:
        """Quarantine one chunk that could not be extracted (WP3-C)."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO chunk_extraction_failures (
                    document_version_id, chunk_id, attempts, error, created_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(document_version_id, chunk_id) DO UPDATE SET
                    attempts = excluded.attempts,
                    error = excluded.error
                """,
                (document_version_id, chunk_id, max(0, int(attempts)), error, utc_now_iso()),
            )

    def list_chunk_extraction_failures(self, document_version_id: str) -> list[sqlite3.Row]:
        """Chunks withheld from the graph for one version."""
        with self._connection() as conn:
            return conn.execute(
                "SELECT * FROM chunk_extraction_failures WHERE document_version_id = ?"
                " ORDER BY created_at",
                (document_version_id,),
            ).fetchall()

    def record_ingestion_stage_metric(
        self,
        *,
        run_id: str,
        stage: str,
        attempt: int,
        doc_id: str,
        document_version_id: str | None,
        duration_ms: float,
        status: str,
        provider: str | None = None,
        model_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        call_count: int | None = None,
        details: dict[str, object] | None = None,
        metric_id: str | None = None,
    ) -> None:
        """Persist one idempotent per-stage attempt for latency/usage acceptance."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_stage_metrics (
                    metric_id, run_id, stage, attempt, doc_id, document_version_id,
                    duration_ms, status, provider, model_id, input_tokens,
                    output_tokens, total_tokens, call_count, details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(metric_id) DO UPDATE SET
                    duration_ms = excluded.duration_ms,
                    status = excluded.status,
                    provider = excluded.provider,
                    model_id = excluded.model_id,
                    input_tokens = excluded.input_tokens,
                    output_tokens = excluded.output_tokens,
                    total_tokens = excluded.total_tokens,
                    call_count = excluded.call_count,
                    details_json = excluded.details_json
                """,
                (
                    metric_id
                    or hashlib.sha256(
                        f"{run_id}|{stage}|{max(1, int(attempt))}".encode("utf-8")
                    ).hexdigest(),
                    run_id,
                    stage,
                    max(1, int(attempt)),
                    doc_id,
                    document_version_id,
                    max(0.0, float(duration_ms)),
                    status,
                    provider,
                    model_id,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    call_count,
                    json.dumps(details or {}, sort_keys=True, separators=(",", ":"), default=str),
                    utc_now_iso(),
                ),
            )

    def list_ingestion_stage_metrics(self, run_id: str) -> list[sqlite3.Row]:
        """List stage timings and provider usage for one ingestion run."""
        with self._connection() as conn:
            return conn.execute(
                """
                SELECT * FROM ingestion_stage_metrics
                WHERE run_id = ? ORDER BY created_at, stage, attempt
                """,
                (run_id,),
            ).fetchall()

    def update_pipeline_stage(self, doc_id: str, stage: str, word_count: int | None = None) -> None:
        """Record a durable successful checkpoint without promoting the generation."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET pipeline_stage = ?,
                    last_completed_stage = ?,
                    word_count = coalesce(?, word_count),
                    status = 'processing'
                WHERE doc_id = ?
                """,
                (stage, stage, word_count, doc_id),
            )

    def fail_document(self, doc_id: str, error_message: str) -> None:
        """Mark document as failed for current run."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, pipeline_stage = ?, last_error = ?
                WHERE doc_id = ?
                """,
                ("failed", "FAILED", error_message, doc_id),
            )

    def get_extraction_inputs(self, doc_id: str, document_version_id: str) -> dict[str, str]:
        """Map chunk_id -> the text extraction read, for one version's active chunks: the
        coreference-annotated copy where there was one, the chunk's text otherwise."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, coalesce(extraction_text, text) AS extraction_input
                FROM chunks
                WHERE doc_id = ? AND document_version_id = ? AND active = 1
                """,
                (doc_id, document_version_id),
            ).fetchall()
        return {str(row["chunk_id"]): str(row["extraction_input"] or "") for row in rows}

    def get_content_hashes(
        self,
        doc_id: str,
        document_version_id: str | None = None,
        *,
        active_only: bool = True,
    ) -> dict[str, str]:
        """Map content_hash -> chunk_id for one document (WP2-D).

        Position-free, so it still matches after text is inserted earlier in the file.
        This is what WP7 will use to reuse assertions instead of re-extracting.
        """
        clauses = ["doc_id = ?", "content_hash IS NOT NULL", "content_hash <> ''"]
        params: list[object] = [doc_id]
        if document_version_id is not None:
            clauses.append("document_version_id = ?")
            params.append(document_version_id)
        if active_only:
            clauses.append("active = 1")
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT content_hash, chunk_id FROM chunks WHERE {' AND '.join(clauses)}",
                params,
            ).fetchall()
        return {str(row["content_hash"]): str(row["chunk_id"]) for row in rows}

    def get_chunk_ids_by_hash(
        self,
        doc_id: str,
        document_version_id: str | None = None,
        *,
        active_only: bool = False,
    ) -> dict[str, str]:
        """Map existing chunk_hash -> chunk_id for one document."""
        clauses = ["doc_id = ?"]
        params: list[object] = [doc_id]
        if document_version_id is not None:
            clauses.append("document_version_id = ?")
            params.append(document_version_id)
        if active_only:
            clauses.append("active = 1")
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT chunk_hash, chunk_id FROM chunks WHERE {' AND '.join(clauses)}",
                params,
            ).fetchall()
        return {str(row["chunk_hash"]): str(row["chunk_id"]) for row in rows}

    def get_chunks_for_doc(
        self,
        doc_id: str,
        document_version_id: str | None = None,
        *,
        active_only: bool = False,
    ) -> list[sqlite3.Row]:
        """Fetch all chunk rows for one document."""
        clauses = ["doc_id = ?"]
        params: list[object] = [doc_id]
        if document_version_id is not None:
            clauses.append("document_version_id = ?")
            params.append(document_version_id)
        if active_only:
            clauses.append("active = 1")
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT chunk_id, doc_id, text, section, page, chunk_hash, embedding_version,
                       content_kind, document_version_id, active
                FROM chunks
                WHERE {' AND '.join(clauses)}
                ORDER BY position ASC
                """,
                params,
            ).fetchall()
        return rows

    def get_queryable_chunks_for_docs(self, doc_ids: Iterable[str]) -> list[sqlite3.Row]:
        """Active chunks of these documents, fenced to active versions, in reading order.

        For passages fetched by document rather than by search (ROADMAP C2), which must obey
        the same fence the vector retriever applies to its hits.
        """
        ids = list(dict.fromkeys(str(doc_id) for doc_id in doc_ids if doc_id))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._connection() as conn:
            return conn.execute(
                f"""
                SELECT c.chunk_id, c.doc_id, c.text, c.section, c.page, c.document_version_id
                FROM chunks c
                JOIN document_versions v
                  ON v.version_id = c.document_version_id AND v.status = 'active'
                WHERE c.active = 1 AND c.doc_id IN ({placeholders})
                ORDER BY c.doc_id, c.position ASC
                """,
                ids,
            ).fetchall()

    def get_chunk_by_id(
        self,
        chunk_id: str,
        *,
        document_version_id: str | None = None,
        active_only: bool = True,
    ) -> sqlite3.Row | None:
        """Resolve one exact evidence chunk, optionally constrained to its version."""
        clauses = ["chunk_id = ?"]
        params: list[object] = [chunk_id]
        if document_version_id is not None:
            clauses.append("document_version_id = ?")
            params.append(document_version_id)
        if active_only:
            clauses.append("active = 1")
        with self._connection() as conn:
            return conn.execute(
                f"SELECT * FROM chunks WHERE {' AND '.join(clauses)}",
                params,
            ).fetchone()

    def delete_chunks_by_hashes(self, doc_id: str, hashes: Iterable[str]) -> None:
        """Delete only removed chunk hashes for delta update flow."""
        hash_list = list(hashes)
        if not hash_list:
            return
        placeholders = ",".join(["?"] * len(hash_list))
        params = [doc_id, *hash_list]
        with self._connection() as conn:
            conn.execute(
                f"DELETE FROM chunks WHERE doc_id = ? AND chunk_hash IN ({placeholders})",
                params,
            )

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        document_version_id: str | None = None,
        *,
        active: bool = True,
    ) -> None:
        """Upsert chunk rows emitted by chunker."""
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, text, token_count, position, section, page,
                    chunk_hash, embedding_version, content_kind, content_hash,
                    document_version_id, extraction_text, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    text = excluded.text,
                    token_count = excluded.token_count,
                    position = excluded.position,
                    section = excluded.section,
                    page = excluded.page,
                    chunk_hash = excluded.chunk_hash,
                    content_kind = excluded.content_kind,
                    content_hash = excluded.content_hash,
                    document_version_id = excluded.document_version_id,
                    extraction_text = excluded.extraction_text,
                    active = excluded.active
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.text,
                        chunk.token_count,
                        chunk.position,
                        chunk.section_heading,
                        chunk.page_number,
                        chunk.chunk_hash,
                        None,
                        chunk.content_kind,
                        chunk.content_hash,
                        document_version_id,
                        chunk.extraction_text,
                        int(active),
                    )
                    for chunk in chunks
                ],
            )

    def set_chunk_embedding_version(self, chunk_ids: Sequence[str], embedding_version: str) -> None:
        """Set embedding_version for successfully indexed chunk IDs."""
        if not chunk_ids:
            return
        placeholders = ",".join(["?"] * len(chunk_ids))
        params = [embedding_version, *chunk_ids]
        with self._connection() as conn:
            conn.execute(
                f"UPDATE chunks SET embedding_version = ? WHERE chunk_id IN ({placeholders})",
                params,
            )

    def upsert_doc_entities(
        self,
        doc_id: str,
        entities: Sequence[dict[str, object]],
        document_version_id: str | None = None,
        *,
        active: bool = True,
    ) -> None:
        """Upsert canonical entity records for a document."""
        if document_version_id is not None:
            with self._connection() as conn:
                conn.executemany(
                    """
                    INSERT INTO version_doc_entities (
                        document_version_id, doc_id, entity_id, canonical_name, aliases, active
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_version_id, entity_id) DO UPDATE SET
                        canonical_name = excluded.canonical_name,
                        aliases = excluded.aliases,
                        active = excluded.active
                    """,
                    [
                        (
                            document_version_id,
                            doc_id,
                            str(entity["entity_id"]),
                            str(entity["canonical_name"]),
                            json.dumps(entity.get("aliases", [])),
                            int(active),
                        )
                        for entity in entities
                    ],
                )
            return
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT INTO doc_entities (
                    doc_id, entity_id, canonical_name, aliases
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(doc_id, entity_id) DO UPDATE SET
                    canonical_name = excluded.canonical_name,
                    aliases = excluded.aliases
                """,
                [
                    (
                        doc_id,
                        str(entity["entity_id"]),
                        str(entity["canonical_name"]),
                        json.dumps(entity.get("aliases", [])),
                    )
                    for entity in entities
                ],
            )

    def get_doc_entities(
        self,
        doc_id: str,
        document_version_id: str | None = None,
        *,
        active_only: bool = False,
    ) -> list[dict[str, object]]:
        """Fetch stored canonical entities for a document."""
        if document_version_id is not None or active_only:
            clauses = ["doc_id = ?"]
            params: list[object] = [doc_id]
            if document_version_id is not None:
                clauses.append("document_version_id = ?")
                params.append(document_version_id)
            if active_only:
                clauses.append("active = 1")
            with self._connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT entity_id, canonical_name, aliases
                    FROM version_doc_entities
                    WHERE {' AND '.join(clauses)}
                    """,
                    params,
                ).fetchall()
            return [
                {
                    "entity_id": str(row["entity_id"]),
                    "canonical_name": str(row["canonical_name"]),
                    "aliases": json.loads(row["aliases"]) if row["aliases"] else [],
                }
                for row in rows
            ]
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT entity_id, canonical_name, aliases FROM doc_entities WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
        results: list[dict[str, object]] = []
        for row in rows:
            aliases = json.loads(row["aliases"]) if row["aliases"] else []
            results.append(
                {
                    "entity_id": str(row["entity_id"]),
                    "canonical_name": str(row["canonical_name"]),
                    "aliases": aliases,
                }
            )
        return results

    def set_system_metadata(self, key: str, value: str) -> None:
        """Upsert system metadata entry."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO system_metadata (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )

    def get_system_metadata(self, key: str) -> str | None:
        """Fetch system metadata value."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT value FROM system_metadata WHERE key = ?",
                (key,),
            ).fetchone()
        return str(row["value"]) if row else None

    def log_ingestion_run(
        self,
        run_id: str,
        doc_id: str,
        started_at: str,
        finished_at: str,
        status: str,
        entities_created: int = 0,
        relations_created: int = 0,
        entities_merged: int = 0,
        document_version_id: str | None = None,
    ) -> None:
        """Write one ingestion_log row for a document run."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (
                    run_id, doc_id, started_at, finished_at,
                    entities_created, relations_created, entities_merged, status,
                    document_version_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    doc_id,
                    started_at,
                    finished_at,
                    entities_created,
                    relations_created,
                    entities_merged,
                    status,
                    document_version_id,
                ),
            )
