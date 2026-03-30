"""SQLite metadata store for ingestion pipeline state."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from models.ingestion import Chunk


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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    format TEXT,
                    checksum TEXT UNIQUE,
                    word_count INTEGER,
                    ingested_at TEXT,
                    status TEXT,
                    pipeline_stage TEXT,
                    last_error TEXT
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
                    embedding_version TEXT
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

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_hash
                    ON chunks (doc_id, chunk_hash);

                CREATE INDEX IF NOT EXISTS idx_entity_mentions_doc
                    ON entity_mentions (doc_id);

                CREATE INDEX IF NOT EXISTS idx_doc_entities_name
                    ON doc_entities (doc_id, canonical_name);
                """
            )

    def get_document_by_source(self, source_path: str) -> sqlite3.Row | None:
        """Fetch document row by source path."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
        return row

    def start_document(
        self,
        doc_id: str,
        source_path: str,
        doc_format: str,
        checksum: str,
    ) -> None:
        """Upsert document row at pipeline start."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, source_path, format, checksum, word_count,
                    ingested_at, status, pipeline_stage, last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    format = excluded.format,
                    checksum = excluded.checksum,
                    status = excluded.status,
                    pipeline_stage = excluded.pipeline_stage,
                    last_error = excluded.last_error
                """,
                (
                    doc_id,
                    source_path,
                    doc_format,
                    checksum,
                    0,
                    utc_now_iso(),
                    "processing",
                    "PARSING",
                    None,
                ),
            )

    def finish_document(self, doc_id: str, word_count: int, pipeline_stage: str) -> None:
        """Mark ingestion completion for the latest stage."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE documents
                SET word_count = ?,
                    ingested_at = ?,
                    status = ?,
                    pipeline_stage = ?,
                    last_error = NULL
                WHERE doc_id = ?
                """,
                (word_count, utc_now_iso(), "complete", pipeline_stage, doc_id),
            )

    def update_pipeline_stage(self, doc_id: str, stage: str) -> None:
        """Update document pipeline stage explicitly."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE documents SET pipeline_stage = ? WHERE doc_id = ?",
                (stage, doc_id),
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

    def get_chunk_ids_by_hash(self, doc_id: str) -> dict[str, str]:
        """Map existing chunk_hash -> chunk_id for one document."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT chunk_hash, chunk_id FROM chunks WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
        return {str(row["chunk_hash"]): str(row["chunk_id"]) for row in rows}

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

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        """Upsert chunk rows emitted by chunker."""
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, doc_id, text, token_count, position, section, page, chunk_hash, embedding_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    text = excluded.text,
                    token_count = excluded.token_count,
                    position = excluded.position,
                    section = excluded.section,
                    page = excluded.page,
                    chunk_hash = excluded.chunk_hash
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
                    )
                    for chunk in chunks
                ],
            )

    def upsert_doc_entities(self, doc_id: str, entities: Sequence[dict[str, object]]) -> None:
        """Upsert canonical entity records for a document."""
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

    def delete_doc_entities(self, doc_id: str, entity_ids: Sequence[str]) -> None:
        """Delete canonical entity records for a document."""
        if not entity_ids:
            return
        placeholders = ",".join(["?"] * len(entity_ids))
        params = [doc_id, *entity_ids]
        with self._connection() as conn:
            conn.execute(
                f"DELETE FROM doc_entities WHERE doc_id = ? AND entity_id IN ({placeholders})",
                params,
            )

    def get_doc_entities(self, doc_id: str) -> list[dict[str, object]]:
        """Fetch stored canonical entities for a document."""
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

    def upsert_entity_mentions(
        self,
        mentions: Sequence[tuple[str, str, str, float]],
    ) -> None:
        """Insert entity mention rows (entity_id, chunk_id, doc_id, confidence)."""
        if not mentions:
            return
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT INTO entity_mentions (entity_id, chunk_id, doc_id, confidence)
                VALUES (?, ?, ?, ?)
                """,
                mentions,
            )

    def delete_entity_mentions_for_chunks(self, doc_id: str, chunk_ids: Sequence[str]) -> None:
        """Delete entity mention rows for specific chunks."""
        if not chunk_ids:
            return
        placeholders = ",".join(["?"] * len(chunk_ids))
        params = [doc_id, *chunk_ids]
        with self._connection() as conn:
            conn.execute(
                f"DELETE FROM entity_mentions WHERE doc_id = ? AND chunk_id IN ({placeholders})",
                params,
            )

    def get_entity_ids_for_chunks(self, doc_id: str, chunk_ids: Sequence[str]) -> set[str]:
        """Get entity IDs mentioned in specific chunks."""
        if not chunk_ids:
            return set()
        placeholders = ",".join(["?"] * len(chunk_ids))
        params = [doc_id, *chunk_ids]
        with self._connection() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT entity_id FROM entity_mentions WHERE doc_id = ? AND chunk_id IN ({placeholders})",
                params,
            ).fetchall()
        return {str(row["entity_id"]) for row in rows}

    def get_entity_ids_with_mentions(self, doc_id: str) -> set[str]:
        """Get entity IDs still mentioned in any chunk for a document."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT DISTINCT entity_id FROM entity_mentions WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
        return {str(row["entity_id"]) for row in rows}

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
    ) -> None:
        """Write one ingestion_log row for a document run."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log (
                    run_id, doc_id, started_at, finished_at,
                    entities_created, relations_created, entities_merged, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
