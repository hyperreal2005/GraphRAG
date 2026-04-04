"""Global settings for GraphRAG."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    """Configuration defaults aligned to architecture reference."""

    metadata_db_path: Path = Path("data/metadata.db")
    artifact_dir: Path = Path("artifacts/phase1")
    supported_formats: tuple[str, ...] = ("pdf", "txt", "md", "csv", "json")
    chunk_target_tokens: int = 700
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 100
    coref_profile: str = "balanced"
    strict_coref: bool = True
    enable_graph: bool = True
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_max_workers: int = 2
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "your_password"
    gliner_model: str = "urchade/gliner_base"
    gliner_labels: tuple[str, ...] = (
        "Person",
        "Organization",
        "Location",
        "Event",
        "Concept",
        "Process",
        "Method",
        "Tool",
        "Dataset",
        "Metric",
        "Date",
        "Time",
        "Role",
        "Component",
    )
    disamb_model: str = "all-MiniLM-L6-v2"
    disamb_threshold: float = 0.92
    embedding_model_id: str = "gemini-embedding-001"
    embedding_dim: int = 3072
    embedding_version: str = "v1"
    gemini_embedding_task_type: str = "RETRIEVAL_DOCUMENT"
    embedding_batch_size: int = 32
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_timeout_seconds: float = 30.0
    qdrant_chunks_collection: str = "chunks"
    qdrant_entities_collection: str = "entities"
    auto_backfill_vectors: bool = True

    def ensure_directories(self) -> None:
        """Ensure required directories exist before execution."""
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        """Allow env vars to override sensitive defaults."""
        self.gemini_api_key = self.gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        try:
            self.gemini_max_workers = int(os.getenv("GEMINI_MAX_WORKERS", self.gemini_max_workers))
        except ValueError:
            self.gemini_max_workers = self.gemini_max_workers
        try:
            self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", self.embedding_batch_size))
        except ValueError:
            self.embedding_batch_size = self.embedding_batch_size
        try:
            self.qdrant_timeout_seconds = float(os.getenv("QDRANT_TIMEOUT_SECONDS", self.qdrant_timeout_seconds))
        except ValueError:
            self.qdrant_timeout_seconds = self.qdrant_timeout_seconds
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", self.neo4j_password)
        self.neo4j_uri = os.getenv("NEO4J_URI", self.neo4j_uri)
        self.neo4j_user = os.getenv("NEO4J_USER", self.neo4j_user)
        self.qdrant_url = os.getenv("QDRANT_URL", self.qdrant_url)
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", self.qdrant_api_key)
        self.qdrant_chunks_collection = os.getenv("QDRANT_CHUNKS_COLLECTION", self.qdrant_chunks_collection)
        self.qdrant_entities_collection = os.getenv("QDRANT_ENTITIES_COLLECTION", self.qdrant_entities_collection)
        self.embedding_model_id = os.getenv("EMBEDDING_MODEL_ID", self.embedding_model_id)
        self.gemini_embedding_task_type = os.getenv(
            "GEMINI_EMBED_TASK_TYPE",
            self.gemini_embedding_task_type,
        )
        try:
            self.embedding_dim = int(os.getenv("EMBEDDING_DIM", self.embedding_dim))
        except ValueError:
            self.embedding_dim = self.embedding_dim
        auto_backfill_env = os.getenv("AUTO_BACKFILL_VECTORS")
        if auto_backfill_env is not None:
            self.auto_backfill_vectors = auto_backfill_env.strip().lower() not in {"0", "false", "no"}
