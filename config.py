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
    embedding_model_id: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    embedding_version: str = "v1"

    def ensure_directories(self) -> None:
        """Ensure required directories exist before execution."""
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        """Allow env vars to override sensitive defaults."""
        self.gemini_api_key = self.gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", self.neo4j_password)
        self.neo4j_uri = os.getenv("NEO4J_URI", self.neo4j_uri)
        self.neo4j_user = os.getenv("NEO4J_USER", self.neo4j_user)
