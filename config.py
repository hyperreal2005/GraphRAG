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
    qdrant_communities_collection: str = "communities"
    auto_backfill_vectors: bool = True
    query_embedding_task_type: str = "RETRIEVAL_QUERY"
    entity_link_resolve_threshold: float = 0.85
    entity_link_ambiguous_threshold: float = 0.70
    query_vector_top_k: int = 15
    query_graph_local_hops: int = 3
    query_graph_multihop_hops: int = 4
    query_graph_path_limit: int = 25
    query_graph_allow_ambiguous_fallback: bool = True
    query_graph_ambiguous_min_similarity: float = 0.75
    query_graph_ambiguous_score_penalty: float = 0.85
    query_context_max_items: int = 20
    query_context_max_chars: int = 12000
    query_filter_boilerplate_chunks: bool = True
    query_citation_max: int = 8
    query_temperature_local: float = 0.1
    query_temperature_multihop: float = 0.1
    query_temperature_comparative: float = 0.2
    query_temperature_global: float = 0.35
    query_temperature_hybrid: float = 0.2
    query_cache_enabled: bool = False
    query_cache_ttl_seconds: int = 3600

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
        try:
            self.query_vector_top_k = int(os.getenv("QUERY_VECTOR_TOP_K", self.query_vector_top_k))
        except ValueError:
            self.query_vector_top_k = self.query_vector_top_k
        try:
            self.query_graph_local_hops = int(os.getenv("QUERY_GRAPH_LOCAL_HOPS", self.query_graph_local_hops))
        except ValueError:
            self.query_graph_local_hops = self.query_graph_local_hops
        try:
            self.query_graph_multihop_hops = int(
                os.getenv("QUERY_GRAPH_MULTIHOP_HOPS", self.query_graph_multihop_hops)
            )
        except ValueError:
            self.query_graph_multihop_hops = self.query_graph_multihop_hops
        try:
            self.query_graph_path_limit = int(os.getenv("QUERY_GRAPH_PATH_LIMIT", self.query_graph_path_limit))
        except ValueError:
            self.query_graph_path_limit = self.query_graph_path_limit
        try:
            self.query_graph_ambiguous_min_similarity = float(
                os.getenv("QUERY_GRAPH_AMBIGUOUS_MIN_SIMILARITY", self.query_graph_ambiguous_min_similarity)
            )
        except ValueError:
            self.query_graph_ambiguous_min_similarity = self.query_graph_ambiguous_min_similarity
        try:
            self.query_graph_ambiguous_score_penalty = float(
                os.getenv("QUERY_GRAPH_AMBIGUOUS_SCORE_PENALTY", self.query_graph_ambiguous_score_penalty)
            )
        except ValueError:
            self.query_graph_ambiguous_score_penalty = self.query_graph_ambiguous_score_penalty
        try:
            self.query_context_max_items = int(os.getenv("QUERY_CONTEXT_MAX_ITEMS", self.query_context_max_items))
        except ValueError:
            self.query_context_max_items = self.query_context_max_items
        try:
            self.query_context_max_chars = int(os.getenv("QUERY_CONTEXT_MAX_CHARS", self.query_context_max_chars))
        except ValueError:
            self.query_context_max_chars = self.query_context_max_chars
        try:
            self.query_citation_max = int(os.getenv("QUERY_CITATION_MAX", self.query_citation_max))
        except ValueError:
            self.query_citation_max = self.query_citation_max
        try:
            self.entity_link_resolve_threshold = float(
                os.getenv("ENTITY_LINK_RESOLVE_THRESHOLD", self.entity_link_resolve_threshold)
            )
        except ValueError:
            self.entity_link_resolve_threshold = self.entity_link_resolve_threshold
        try:
            self.entity_link_ambiguous_threshold = float(
                os.getenv("ENTITY_LINK_AMBIGUOUS_THRESHOLD", self.entity_link_ambiguous_threshold)
            )
        except ValueError:
            self.entity_link_ambiguous_threshold = self.entity_link_ambiguous_threshold
        try:
            self.query_temperature_local = float(os.getenv("QUERY_TEMP_LOCAL", self.query_temperature_local))
        except ValueError:
            self.query_temperature_local = self.query_temperature_local
        try:
            self.query_temperature_multihop = float(os.getenv("QUERY_TEMP_MULTIHOP", self.query_temperature_multihop))
        except ValueError:
            self.query_temperature_multihop = self.query_temperature_multihop
        try:
            self.query_temperature_comparative = float(
                os.getenv("QUERY_TEMP_COMPARATIVE", self.query_temperature_comparative)
            )
        except ValueError:
            self.query_temperature_comparative = self.query_temperature_comparative
        try:
            self.query_temperature_global = float(os.getenv("QUERY_TEMP_GLOBAL", self.query_temperature_global))
        except ValueError:
            self.query_temperature_global = self.query_temperature_global
        try:
            self.query_temperature_hybrid = float(os.getenv("QUERY_TEMP_HYBRID", self.query_temperature_hybrid))
        except ValueError:
            self.query_temperature_hybrid = self.query_temperature_hybrid
        try:
            self.query_cache_ttl_seconds = int(os.getenv("QUERY_CACHE_TTL_SECONDS", self.query_cache_ttl_seconds))
        except ValueError:
            self.query_cache_ttl_seconds = self.query_cache_ttl_seconds
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", self.neo4j_password)
        self.neo4j_uri = os.getenv("NEO4J_URI", self.neo4j_uri)
        self.neo4j_user = os.getenv("NEO4J_USER", self.neo4j_user)
        self.qdrant_url = os.getenv("QDRANT_URL", self.qdrant_url)
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", self.qdrant_api_key)
        self.qdrant_chunks_collection = os.getenv("QDRANT_CHUNKS_COLLECTION", self.qdrant_chunks_collection)
        self.qdrant_entities_collection = os.getenv("QDRANT_ENTITIES_COLLECTION", self.qdrant_entities_collection)
        self.qdrant_communities_collection = os.getenv(
            "QDRANT_COMMUNITIES_COLLECTION",
            self.qdrant_communities_collection,
        )
        self.embedding_model_id = os.getenv("EMBEDDING_MODEL_ID", self.embedding_model_id)
        self.gemini_embedding_task_type = os.getenv(
            "GEMINI_EMBED_TASK_TYPE",
            self.gemini_embedding_task_type,
        )
        self.query_embedding_task_type = os.getenv("QUERY_EMBED_TASK_TYPE", self.query_embedding_task_type)
        try:
            self.embedding_dim = int(os.getenv("EMBEDDING_DIM", self.embedding_dim))
        except ValueError:
            self.embedding_dim = self.embedding_dim
        cache_enabled_env = os.getenv("QUERY_CACHE_ENABLED")
        if cache_enabled_env is not None:
            self.query_cache_enabled = cache_enabled_env.strip().lower() not in {"0", "false", "no"}
        ambiguous_graph_fallback_env = os.getenv("QUERY_GRAPH_ALLOW_AMBIGUOUS_FALLBACK")
        if ambiguous_graph_fallback_env is not None:
            self.query_graph_allow_ambiguous_fallback = ambiguous_graph_fallback_env.strip().lower() not in {
                "0",
                "false",
                "no",
            }
        filter_boilerplate_env = os.getenv("QUERY_FILTER_BOILERPLATE_CHUNKS")
        if filter_boilerplate_env is not None:
            self.query_filter_boilerplate_chunks = filter_boilerplate_env.strip().lower() not in {"0", "false", "no"}
        auto_backfill_env = os.getenv("AUTO_BACKFILL_VECTORS")
        if auto_backfill_env is not None:
            self.auto_backfill_vectors = auto_backfill_env.strip().lower() not in {"0", "false", "no"}
