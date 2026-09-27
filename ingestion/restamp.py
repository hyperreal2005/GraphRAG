"""Record a re-embed in the metadata store, after proving it happened.

`scripts/reembed_corpus.py` rebuilds a corpus's vectors without re-running ingestion, which is
the only affordable way to change embedding model: extraction is most of a build's cost and
does not depend on the embedding. But four records in SQLite describe the embedding, and until
2026-09-27 the re-embed left all four describing the old model (REFACTOR3 WP20):

- `system_metadata`, the store-level contract. `PhaseOnePipeline` refuses to start against a
  store whose contract differs from its settings, so every ingest into both re-embedded
  corpora was refused, telling the operator to rebuild vectors that had been rebuilt.
- each active version's embedding columns;
- each document's `active_pipeline_fingerprint`, so that with the contract fixed every
  document would read as changed and be re-ingested;
- each active chunk's `embedding_version`.

Nothing is stamped on trust. The served vectors must carry the embedding the stamp claims,
and each stored fingerprint must be *reproduced* from the settings the document was built
with: the current settings, the store's recorded embedding, and any difference the operator
declares in `built_with`. A fingerprint that cannot be reproduced means something else about
how the corpus was built differs from the current pipeline, and stamping it as current would
hide exactly that.

A declared difference is carried into the new stamp, because it describes the corpus as it
is: a corpus chunked into table blocks stays stamped with blocks. Only a difference named in
`equivalent` is stamped with the current value instead - one the operator has shown cannot
change what was written.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from config import Settings
from ingestion.pipeline import (
    FINGERPRINT_KEYS_ADDED_LATER,
    embedding_contract,
    fingerprint_of,
    pipeline_fingerprint,
    pipeline_fingerprint_payload,
)
from models.refactor import DocumentVersion

# Fingerprint keys that are not settings a build can differ in.
_NOT_DECLARABLE = {"pipeline_version", "chunk_fingerprint_version", "target_stage"}
HISTORY_KEY = "restamp_history"


@dataclass
class DocumentStamp:
    doc_id: str
    version_id: str
    old: str
    new: str


@dataclass
class RestampPlan:
    db: Path
    contract_before: dict[str, str | None]
    contract_after: dict[str, str | None]
    built_with: dict[str, Any]
    equivalent: tuple[str, ...]
    documents: list[DocumentStamp] = field(default_factory=list)
    already_current: list[str] = field(default_factory=list)
    refused: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    legacy: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems and not self.refused

    @property
    def stamps(self) -> set[str]:
        return {doc.new for doc in self.documents}


def declarable_keys(settings: Settings) -> set[str]:
    """Settings a corpus can have been built with differently. Embedding settings are not
    among them: those come from the store's own contract, not from the operator.
    `coref_enabled` is: it decides whether the fingerprint carries `coref_annotation`."""
    keys = {
        key for key in pipeline_fingerprint_payload(settings, "DONE") if hasattr(settings, key)
    } - _NOT_DECLARABLE
    return (keys - set(embedding_contract(settings))) | {"coref_enabled"}


def parse_declared(settings: Settings, pairs: Iterable[str]) -> dict[str, Any]:
    """`KEY=VALUE` pairs, typed like the setting they override."""
    allowed = declarable_keys(settings)
    declared: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"expected KEY=VALUE, got {pair!r}")
        if key not in allowed:
            raise ValueError(f"{key!r} is not a fingerprinted setting that can be declared; "
                             f"choose from {sorted(allowed)}")
        declared[key] = _typed(getattr(settings, key), raw.strip())
    return declared


def _typed(current: Any, raw: str) -> Any:
    if isinstance(current, bool):
        lowered = raw.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if current is None and raw.lower() == "none":
        return None
    return raw


def _with(settings: Settings, values: Mapping[str, Any]) -> Settings:
    # copy + setattr, not dataclasses.replace: Settings.__post_init__ re-reads the
    # environment and would silently overwrite the values being set.
    derived = copy.copy(settings)
    for key, value in values.items():
        setattr(derived, key, value)
    return derived


def _contract_as_settings(contract: Mapping[str, str | None]) -> dict[str, Any]:
    values: dict[str, Any] = dict(contract)
    if values.get("embedding_dim") is not None:
        values["embedding_dim"] = int(values["embedding_dim"])
    return values


def _legacy_fingerprint(settings: Settings, target_stage: str) -> str:
    payload = pipeline_fingerprint_payload(settings, target_stage)
    for key in FINGERPRINT_KEYS_ADDED_LATER:
        payload.pop(key, None)
    return fingerprint_of(payload)


def _table_rows_per_chunk(conn: sqlite3.Connection) -> tuple[int, int]:
    """Active chunks holding one rendered table row, and chunks holding several."""
    single = multi = 0
    for (text,) in conn.execute("SELECT text FROM chunks WHERE active = 1"):
        rows = sum(1 for line in (text or "").split("\n") if " | " in line and ": " in line)
        if rows >= 2:
            multi += 1
        elif rows == 1:
            single += 1
    return single, multi


def plan_restamp(
    db: Path,
    settings: Settings,
    *,
    built_with: Mapping[str, Any] | None = None,
    equivalent: Iterable[str] = (),
    vector_problems: Iterable[str] = (),
) -> RestampPlan:
    """Decide, without writing anything, whether and how the store can be re-stamped."""
    built_with = dict(built_with or {})
    equivalent = tuple(equivalent)
    unknown = set(built_with) - declarable_keys(settings)
    if unknown:
        raise ValueError(f"not declarable: {sorted(unknown)}")
    stray = set(equivalent) - set(built_with)
    if stray:
        raise ValueError(f"an equivalent setting must also be declared in built_with: {sorted(stray)}")

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        before = {
            key: _system_metadata(conn, key) for key in embedding_contract(settings)
        }
        after = embedding_contract(settings)
        plan = RestampPlan(
            db=db, contract_before=before, contract_after=after,
            built_with=built_with, equivalent=equivalent,
        )
        plan.problems.extend(vector_problems)
        if before.get("embedding_model_id") is None:
            plan.problems.append("the store has no embedding contract to reproduce a build from")
            return plan

        as_built = _with(settings, {**_contract_as_settings(before), **built_with})
        stamp_settings = _with(settings, {k: v for k, v in built_with.items() if k not in equivalent})

        single, multi = _table_rows_per_chunk(conn)
        blocks = stamp_settings.chunk_table_block_target_tokens
        if blocks <= 0 and multi:
            plan.problems.append(
                f"the stamp says one chunk per table row, but {multi} active chunks hold "
                "several rows - declare the block target the corpus was chunked with"
            )
        if blocks > 0 and single and not multi:
            plan.problems.append(
                f"the stamp says table blocks of {blocks} tokens, but no active chunk holds more "
                f"than one row ({single} hold exactly one)"
            )

        target_version = _version_columns(settings)
        rows = conn.execute(
            """
            SELECT d.doc_id, d.target_stage, d.active_pipeline_fingerprint AS stored,
                   v.version_id, v.embedding_model, v.embedding_schema_version
            FROM documents d
            JOIN document_versions v ON v.doc_id = d.doc_id AND v.status = 'active'
            WHERE d.status = 'complete'
            ORDER BY d.doc_id
            """
        ).fetchall()
        for row in rows:
            doc_id = str(row["doc_id"])
            stage = str(row["target_stage"] or "DONE")
            stored = str(row["stored"] or "")
            new = pipeline_fingerprint(stamp_settings, stage)
            if (
                stored == new
                and row["embedding_model"] == target_version["embedding_model"]
                and row["embedding_schema_version"] == target_version["embedding_schema_version"]
            ):
                plan.already_current.append(doc_id)
                continue
            if stored == pipeline_fingerprint(as_built, stage):
                pass
            elif stored == _legacy_fingerprint(as_built, stage):
                plan.legacy += 1
                missing = [key for key in FINGERPRINT_KEYS_ADDED_LATER if key not in built_with]
                if missing:
                    plan.refused[doc_id] = (
                        f"the stored fingerprint predates {missing}, so it cannot say how the "
                        "corpus was built there; declare the value in built_with"
                    )
                    continue
            else:
                plan.refused[doc_id] = (
                    "the stored fingerprint is not reproduced by the current settings with this "
                    "store's embedding and the declared differences - something else about how "
                    "it was built differs from the current pipeline"
                )
                continue
            plan.documents.append(
                DocumentStamp(doc_id=doc_id, version_id=str(row["version_id"]), old=stored, new=new)
            )
        return plan
    finally:
        conn.close()


def _system_metadata(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM system_metadata WHERE key = ?", (key,)).fetchone()
    return None if row is None else row[0]


def _version_columns(settings: Settings) -> dict[str, Any]:
    """The embedding columns ingestion would record for a version built with `settings`,
    normalised by the same model it uses, so a re-stamped row matches a freshly built one."""
    version = DocumentVersion(
        version_id="restamp",
        doc_id="restamp",
        checksum="0" * 64,
        pipeline_fingerprint="restamp",
        generative_model=settings.gemini_model,
        embedding_model=settings.embedding_model_id,
        embedding_dim=settings.embedding_dim,
        embedding_prompt_version=settings.embedding_prompt_version,
        embedding_schema_version=settings.embedding_version,
        embedding_normalization_policy=settings.embedding_normalization_policy,
        document_embedding_instruction=settings.embedding_document_prefix,
        query_embedding_instruction=settings.embedding_query_prefix,
        graph_schema_version=settings.graph_schema_version,
    )
    return {
        "embedding_model": version.embedding_model,
        "embedding_dim": version.embedding_dim,
        "embedding_prompt_version": version.embedding_prompt_version,
        "embedding_schema_version": version.embedding_schema_version,
        "embedding_normalization_policy": version.embedding_normalization_policy,
        "document_embedding_instruction": version.document_embedding_instruction,
        "query_embedding_instruction": version.query_embedding_instruction,
    }


def apply_restamp(plan: RestampPlan, settings: Settings, *, backup: str | None = None) -> None:
    """Write the plan in one transaction. Refuses a plan with any problem or refusal: the
    store-level contract can describe one embedding only, so a store is re-stamped whole."""
    if not plan.ok:
        raise RuntimeError("refusing to apply a restamp plan with problems or refused documents")
    if not plan.documents and plan.contract_before == plan.contract_after:
        return
    columns = _version_columns(settings)
    conn = sqlite3.connect(plan.db)
    try:
        with conn:
            for key, value in plan.contract_after.items():
                conn.execute(
                    "INSERT INTO system_metadata (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )
            for doc in plan.documents:
                moved = conn.execute(
                    """
                    UPDATE documents
                    SET active_pipeline_fingerprint = ?,
                        desired_pipeline_fingerprint = CASE
                            WHEN desired_pipeline_fingerprint IS NULL
                              OR desired_pipeline_fingerprint = active_pipeline_fingerprint
                            THEN ? ELSE desired_pipeline_fingerprint END
                    WHERE doc_id = ? AND active_pipeline_fingerprint = ?
                    """,
                    (doc.new, doc.new, doc.doc_id, doc.old),
                ).rowcount
                if moved != 1:
                    raise RuntimeError(f"document {doc.doc_id} changed since it was planned")
                moved = conn.execute(
                    """
                    UPDATE document_versions
                    SET embedding_model = ?, embedding_dim = ?, embedding_prompt_version = ?,
                        embedding_schema_version = ?, embedding_normalization_policy = ?,
                        document_embedding_instruction = ?, query_embedding_instruction = ?,
                        pipeline_fingerprint = ?
                    WHERE version_id = ? AND status = 'active'
                    """,
                    (
                        columns["embedding_model"], columns["embedding_dim"],
                        columns["embedding_prompt_version"], columns["embedding_schema_version"],
                        columns["embedding_normalization_policy"],
                        columns["document_embedding_instruction"],
                        columns["query_embedding_instruction"],
                        doc.new, doc.version_id,
                    ),
                ).rowcount
                if moved != 1:
                    raise RuntimeError(f"version {doc.version_id} is no longer active")
                conn.execute(
                    "UPDATE chunks SET embedding_version = ? "
                    "WHERE document_version_id = ? AND active = 1",
                    (settings.embedding_version, doc.version_id),
                )
            history = json.loads(_system_metadata(conn, HISTORY_KEY) or "[]")
            history.append(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "embedding_from": plan.contract_before,
                    "embedding_to": plan.contract_after,
                    "built_with": plan.built_with,
                    "equivalent": list(plan.equivalent),
                    "documents": len(plan.documents),
                    "fingerprints_from": sorted({doc.old for doc in plan.documents}),
                    "fingerprints_to": sorted(plan.stamps),
                    "backup": backup,
                }
            )
            conn.execute(
                "INSERT INTO system_metadata (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (HISTORY_KEY, json.dumps(history)),
            )
    finally:
        conn.close()


def backup_store(db: Path) -> tuple[Path, str]:
    """Copy the store through SQLite's backup API, beside the original. Returns the copy and
    its SHA-256."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = db.with_name(f"{db.stem}.pre-restamp-{stamp}{db.suffix}")
    source = sqlite3.connect(db)
    copy_conn = sqlite3.connect(target)
    try:
        source.backup(copy_conn)
    finally:
        copy_conn.close()
        source.close()
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return target, digest.hexdigest()


def verify_served_vectors(
    settings: Settings,
    *,
    chunks_collection: str,
    entities_collection: str,
    db: Path,
    client: Any | None = None,
) -> list[str]:
    """Every point serving this store's active versions must carry the embedding the stamp
    will claim, and every active chunk must have a point."""
    conn = sqlite3.connect(db)
    try:
        versions = {
            v for (v,) in conn.execute("SELECT version_id FROM document_versions WHERE status = 'active'")
        }
        chunk_ids = {c for (c,) in conn.execute("SELECT chunk_id FROM chunks WHERE active = 1")}
    finally:
        conn.close()
    if client is None:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key,
                              timeout=settings.qdrant_timeout_seconds)

    problems: list[str] = []
    for collection, is_chunks in ((chunks_collection, True), (entities_collection, False)):
        served = wrong = 0
        seen: set[str] = set()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection,
                limit=2000,
                offset=offset,
                with_payload=["document_version_id", "embedding_model_id",
                              "embedding_schema_version", "chunk_id"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                if str(payload.get("document_version_id")) not in versions:
                    continue
                served += 1
                if (
                    payload.get("embedding_model_id") != settings.embedding_model_id
                    or payload.get("embedding_schema_version") != settings.embedding_version
                ):
                    wrong += 1
                if is_chunks:
                    seen.add(str(payload.get("chunk_id") or point.id))
            if offset is None:
                break
        if not served:
            problems.append(f"{collection} serves no point of this store's active versions")
        if wrong:
            problems.append(
                f"{collection}: {wrong} of {served} points were not embedded with "
                f"{settings.embedding_model_id} / {settings.embedding_version}"
            )
        if is_chunks and chunk_ids - seen:
            problems.append(f"{collection}: {len(chunk_ids - seen)} active chunks have no point")
    return problems
