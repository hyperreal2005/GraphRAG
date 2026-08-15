"""Global settings for GraphRAG."""

from __future__ import annotations

import os
from dataclasses import dataclass
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
    gemini_model: str = "gemini-3.5-flash-lite"
    gemini_max_workers: int = 2
    ingestion_provider_timeout_seconds: float = 120.0
    ingestion_progress_interval_chunks: int = 5
    graph_exclude_reference_sections: bool = True
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "your_password"
    neo4j_query_user: str | None = None
    neo4j_query_password: str | None = None
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
    cross_doc_auto_merge_threshold: float = 0.95
    cross_doc_llm_min_threshold: float = 0.80
    cross_doc_llm_max_threshold: float = 0.95
    cross_doc_candidate_k: int = 5
    cross_doc_auto_merge_margin: float = 0.05
    cross_doc_llm_confidence_threshold: float = 0.90
    cross_doc_same_type_required_for_auto: bool = True
    cross_doc_llm_enabled: bool = True
    embedding_model_id: str = "gemini-embedding-2"
    embedding_dim: int = 3072
    embedding_version: str = "embedding2-3072-r1"
    embedding_prompt_version: str = "embedding2-r1-v1"
    embedding_normalization_policy: str = "provider-default"
    embedding_document_prefix: str = "task: search result | title: {title} | text: "
    embedding_query_prefix: str = "task: search result | query: "
    gemini_embedding_task_type: str = "RETRIEVAL_DOCUMENT"
    embedding_batch_size: int = 32
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_timeout_seconds: float = 30.0
    qdrant_chunks_collection: str = "chunks"
    qdrant_entities_collection: str = "entities"
    qdrant_communities_collection: str = "communities"
    auto_backfill_vectors: bool = True
    extraction_prompt_version: str = "assertion-r1-v2"
    assertion_review_confidence_threshold: float = 0.50
    extraction_max_attempts: int = 3
    extraction_retry_base_seconds: float = 1.0
    graph_schema_version: int = 3
    outbox_max_attempts: int = 5
    outbox_retry_base_seconds: float = 1.0
    outbox_lease_seconds: float = 60.0
    query_embedding_task_type: str = "RETRIEVAL_QUERY"
    entity_link_resolve_threshold: float = 0.85
    entity_link_ambiguous_threshold: float = 0.70
    entity_link_resolve_margin: float = 0.05
    enforce_embedding_metadata: bool = True
    query_vector_top_k: int = 15
    query_mmr_enabled: bool = True
    query_mmr_lambda: float = 0.65
    query_mmr_candidate_multiplier: int = 3
    query_graph_local_hops: int = 3
    query_graph_multihop_hops: int = 4
    query_graph_path_limit: int = 25
    query_analysis_timeout_seconds: float = 15.0
    query_entity_link_timeout_seconds: float = 20.0
    query_graph_timeout_seconds: float = 15.0
    query_vector_timeout_seconds: float = 15.0
    query_rerank_timeout_seconds: float = 15.0
    query_generation_timeout_seconds: float = 45.0
    # Raw LLM-generated Cypher is intentionally opt-in. The default query path
    # uses application-owned, bounded Cypher templates.
    query_graph_dynamic_enabled: bool = False
    query_graph_dynamic_fallback_to_static: bool = True
    query_graph_schema_cache_ttl_seconds: int = 300
    query_graph_dynamic_model: str | None = None
    query_graph_dynamic_temperature: float = 0.0
    query_graph_dynamic_max_retries: int = 1
    query_graph_transaction_timeout_seconds: float = 5.0
    query_graph_allow_ambiguous_fallback: bool = True
    query_graph_ambiguous_min_similarity: float = 0.75
    query_graph_ambiguous_score_penalty: float = 0.85
    query_context_max_items: int = 20
    query_context_max_chars: int = 12000
    query_filter_boilerplate_chunks: bool = True
    query_include_reference_chunks: bool = False
    query_citation_max: int = 8
    query_temperature_local: float = 0.1
    query_temperature_multihop: float = 0.1
    query_temperature_comparative: float = 0.2
    query_temperature_global: float = 0.35
    query_temperature_hybrid: float = 0.2
    cohere_api_key: str | None = None
    query_rerank_enabled: bool = True
    query_rerank_model: str = "rerank-v3.5"
    query_rerank_top_n: int = 15
    query_cache_enabled: bool = False
    query_cache_ttl_seconds: int = 3600

    def ensure_directories(self) -> None:
        """Ensure required directories exist before execution."""
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        """Allow env vars to override sensitive defaults."""
        self.gemini_api_key = self.gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", self.gemini_model).strip() or self.gemini_model
        try:
            self.gemini_max_workers = int(os.getenv("GEMINI_MAX_WORKERS", self.gemini_max_workers))
        except ValueError:
            self.gemini_max_workers = self.gemini_max_workers
        try:
            self.ingestion_provider_timeout_seconds = float(
                os.getenv(
                    "INGESTION_PROVIDER_TIMEOUT_SECONDS",
                    self.ingestion_provider_timeout_seconds,
                )
            )
        except ValueError:
            pass
        try:
            self.ingestion_progress_interval_chunks = int(
                os.getenv(
                    "INGESTION_PROGRESS_INTERVAL_CHUNKS",
                    self.ingestion_progress_interval_chunks,
                )
            )
        except ValueError:
            pass
        try:
            self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", self.embedding_batch_size))
        except ValueError:
            self.embedding_batch_size = self.embedding_batch_size
        try:
            self.cross_doc_auto_merge_threshold = float(
                os.getenv("CROSS_DOC_AUTO_MERGE_THRESHOLD", self.cross_doc_auto_merge_threshold)
            )
        except ValueError:
            self.cross_doc_auto_merge_threshold = self.cross_doc_auto_merge_threshold
        try:
            self.cross_doc_llm_min_threshold = float(
                os.getenv("CROSS_DOC_LLM_MIN_THRESHOLD", self.cross_doc_llm_min_threshold)
            )
        except ValueError:
            self.cross_doc_llm_min_threshold = self.cross_doc_llm_min_threshold
        try:
            self.cross_doc_llm_max_threshold = float(
                os.getenv("CROSS_DOC_LLM_MAX_THRESHOLD", self.cross_doc_llm_max_threshold)
            )
        except ValueError:
            self.cross_doc_llm_max_threshold = self.cross_doc_llm_max_threshold
        try:
            self.cross_doc_candidate_k = int(os.getenv("CROSS_DOC_CANDIDATE_K", self.cross_doc_candidate_k))
        except ValueError:
            self.cross_doc_candidate_k = self.cross_doc_candidate_k
        try:
            self.cross_doc_auto_merge_margin = float(
                os.getenv("CROSS_DOC_AUTO_MERGE_MARGIN", self.cross_doc_auto_merge_margin)
            )
        except ValueError:
            self.cross_doc_auto_merge_margin = self.cross_doc_auto_merge_margin
        try:
            self.cross_doc_llm_confidence_threshold = float(
                os.getenv("CROSS_DOC_LLM_CONFIDENCE_THRESHOLD", self.cross_doc_llm_confidence_threshold)
            )
        except ValueError:
            self.cross_doc_llm_confidence_threshold = self.cross_doc_llm_confidence_threshold
        try:
            self.qdrant_timeout_seconds = float(os.getenv("QDRANT_TIMEOUT_SECONDS", self.qdrant_timeout_seconds))
        except ValueError:
            self.qdrant_timeout_seconds = self.qdrant_timeout_seconds
        try:
            self.query_vector_top_k = int(os.getenv("QUERY_VECTOR_TOP_K", self.query_vector_top_k))
        except ValueError:
            self.query_vector_top_k = self.query_vector_top_k
        try:
            self.query_mmr_lambda = float(os.getenv("QUERY_MMR_LAMBDA", self.query_mmr_lambda))
        except ValueError:
            self.query_mmr_lambda = self.query_mmr_lambda
        try:
            self.query_mmr_candidate_multiplier = int(
                os.getenv("QUERY_MMR_CANDIDATE_MULTIPLIER", self.query_mmr_candidate_multiplier)
            )
        except ValueError:
            self.query_mmr_candidate_multiplier = self.query_mmr_candidate_multiplier
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
            self.query_graph_schema_cache_ttl_seconds = int(
                os.getenv("QUERY_GRAPH_SCHEMA_CACHE_TTL_SECONDS", self.query_graph_schema_cache_ttl_seconds)
            )
        except ValueError:
            self.query_graph_schema_cache_ttl_seconds = self.query_graph_schema_cache_ttl_seconds
        try:
            self.query_graph_dynamic_temperature = float(
                os.getenv("QUERY_GRAPH_DYNAMIC_TEMPERATURE", self.query_graph_dynamic_temperature)
            )
        except ValueError:
            self.query_graph_dynamic_temperature = self.query_graph_dynamic_temperature
        try:
            self.query_graph_dynamic_max_retries = int(
                os.getenv("QUERY_GRAPH_DYNAMIC_MAX_RETRIES", self.query_graph_dynamic_max_retries)
            )
        except ValueError:
            self.query_graph_dynamic_max_retries = self.query_graph_dynamic_max_retries
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
            self.entity_link_resolve_margin = float(
                os.getenv("ENTITY_LINK_RESOLVE_MARGIN", self.entity_link_resolve_margin)
            )
        except ValueError:
            pass
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
        try:
            self.query_rerank_top_n = int(os.getenv("QUERY_RERANK_TOP_N", self.query_rerank_top_n))
        except ValueError:
            self.query_rerank_top_n = self.query_rerank_top_n
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", self.neo4j_password)
        self.neo4j_uri = os.getenv("NEO4J_URI", self.neo4j_uri)
        self.neo4j_user = os.getenv("NEO4J_USER", self.neo4j_user)
        self.neo4j_query_user = os.getenv("NEO4J_QUERY_USER", self.neo4j_query_user or self.neo4j_user)
        self.neo4j_query_password = os.getenv(
            "NEO4J_QUERY_PASSWORD",
            self.neo4j_query_password or self.neo4j_password,
        )
        self.cohere_api_key = self.cohere_api_key or os.getenv("COHERE_API_KEY")
        self.qdrant_url = os.getenv("QDRANT_URL", self.qdrant_url)
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", self.qdrant_api_key)
        self.qdrant_chunks_collection = os.getenv("QDRANT_CHUNKS_COLLECTION", self.qdrant_chunks_collection)
        self.qdrant_entities_collection = os.getenv("QDRANT_ENTITIES_COLLECTION", self.qdrant_entities_collection)
        self.qdrant_communities_collection = os.getenv(
            "QDRANT_COMMUNITIES_COLLECTION",
            self.qdrant_communities_collection,
        )
        self.embedding_model_id = os.getenv("EMBEDDING_MODEL_ID", self.embedding_model_id)
        self.embedding_version = os.getenv("EMBEDDING_VERSION", self.embedding_version)
        self.embedding_prompt_version = os.getenv(
            "EMBEDDING_PROMPT_VERSION",
            self.embedding_prompt_version,
        )
        self.embedding_normalization_policy = os.getenv(
            "EMBEDDING_NORMALIZATION_POLICY",
            self.embedding_normalization_policy,
        )
        self.embedding_document_prefix = os.getenv(
            "EMBEDDING_DOCUMENT_PREFIX",
            self.embedding_document_prefix,
        )
        self.embedding_query_prefix = os.getenv("EMBEDDING_QUERY_PREFIX", self.embedding_query_prefix)
        self.extraction_prompt_version = os.getenv(
            "EXTRACTION_PROMPT_VERSION",
            self.extraction_prompt_version,
        )
        try:
            self.assertion_review_confidence_threshold = float(
                os.getenv(
                    "ASSERTION_REVIEW_CONFIDENCE_THRESHOLD",
                    self.assertion_review_confidence_threshold,
                )
            )
        except ValueError:
            self.assertion_review_confidence_threshold = self.assertion_review_confidence_threshold
        try:
            self.extraction_max_attempts = int(
                os.getenv("EXTRACTION_MAX_ATTEMPTS", self.extraction_max_attempts)
            )
        except ValueError:
            self.extraction_max_attempts = self.extraction_max_attempts
        try:
            self.outbox_lease_seconds = float(
                os.getenv("OUTBOX_LEASE_SECONDS", self.outbox_lease_seconds)
            )
        except ValueError:
            self.outbox_lease_seconds = self.outbox_lease_seconds
        try:
            self.extraction_retry_base_seconds = float(
                os.getenv("EXTRACTION_RETRY_BASE_SECONDS", self.extraction_retry_base_seconds)
            )
        except ValueError:
            self.extraction_retry_base_seconds = self.extraction_retry_base_seconds
        self.query_rerank_model = os.getenv("QUERY_RERANK_MODEL", self.query_rerank_model)
        dynamic_model = os.getenv("QUERY_GRAPH_DYNAMIC_MODEL")
        if dynamic_model is not None and dynamic_model.strip():
            self.query_graph_dynamic_model = dynamic_model.strip()
        try:
            self.query_graph_transaction_timeout_seconds = float(
                os.getenv(
                    "QUERY_GRAPH_TRANSACTION_TIMEOUT_SECONDS",
                    self.query_graph_transaction_timeout_seconds,
                )
            )
        except ValueError:
            self.query_graph_transaction_timeout_seconds = 5.0
        timeout_fields = {
            "query_analysis_timeout_seconds": "QUERY_ANALYSIS_TIMEOUT_SECONDS",
            "query_entity_link_timeout_seconds": "QUERY_ENTITY_LINK_TIMEOUT_SECONDS",
            "query_graph_timeout_seconds": "QUERY_GRAPH_TIMEOUT_SECONDS",
            "query_vector_timeout_seconds": "QUERY_VECTOR_TIMEOUT_SECONDS",
            "query_rerank_timeout_seconds": "QUERY_RERANK_TIMEOUT_SECONDS",
            "query_generation_timeout_seconds": "QUERY_GENERATION_TIMEOUT_SECONDS",
        }
        for attribute, environment_name in timeout_fields.items():
            try:
                setattr(self, attribute, float(os.getenv(environment_name, getattr(self, attribute))))
            except ValueError:
                pass
        self.gemini_embedding_task_type = os.getenv(
            "GEMINI_EMBED_TASK_TYPE",
            self.gemini_embedding_task_type,
        )
        self.query_embedding_task_type = os.getenv("QUERY_EMBED_TASK_TYPE", self.query_embedding_task_type)
        try:
            self.embedding_dim = int(os.getenv("EMBEDDING_DIM", self.embedding_dim))
        except ValueError:
            self.embedding_dim = self.embedding_dim
        try:
            self.graph_schema_version = int(os.getenv("GRAPH_SCHEMA_VERSION", self.graph_schema_version))
        except ValueError:
            self.graph_schema_version = self.graph_schema_version
        try:
            self.outbox_max_attempts = int(os.getenv("OUTBOX_MAX_ATTEMPTS", self.outbox_max_attempts))
        except ValueError:
            self.outbox_max_attempts = self.outbox_max_attempts
        try:
            self.outbox_retry_base_seconds = float(
                os.getenv("OUTBOX_RETRY_BASE_SECONDS", self.outbox_retry_base_seconds)
            )
        except ValueError:
            self.outbox_retry_base_seconds = self.outbox_retry_base_seconds
        cache_enabled_env = os.getenv("QUERY_CACHE_ENABLED")
        if cache_enabled_env is not None:
            self.query_cache_enabled = cache_enabled_env.strip().lower() not in {"0", "false", "no"}
        mmr_enabled_env = os.getenv("QUERY_MMR_ENABLED")
        if mmr_enabled_env is not None:
            self.query_mmr_enabled = mmr_enabled_env.strip().lower() not in {"0", "false", "no"}
        rerank_enabled_env = os.getenv("QUERY_RERANK_ENABLED")
        if rerank_enabled_env is not None:
            self.query_rerank_enabled = rerank_enabled_env.strip().lower() not in {"0", "false", "no"}
        dynamic_graph_enabled_env = os.getenv("QUERY_GRAPH_DYNAMIC_ENABLED")
        if dynamic_graph_enabled_env is not None:
            self.query_graph_dynamic_enabled = dynamic_graph_enabled_env.strip().lower() not in {"0", "false", "no"}
        dynamic_graph_fallback_env = os.getenv("QUERY_GRAPH_DYNAMIC_FALLBACK_TO_STATIC")
        if dynamic_graph_fallback_env is not None:
            self.query_graph_dynamic_fallback_to_static = dynamic_graph_fallback_env.strip().lower() not in {
                "0",
                "false",
                "no",
            }
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
        include_references_env = os.getenv("QUERY_INCLUDE_REFERENCE_CHUNKS")
        if include_references_env is not None:
            self.query_include_reference_chunks = include_references_env.strip().lower() not in {
                "0",
                "false",
                "no",
            }
        exclude_reference_graph_env = os.getenv("GRAPH_EXCLUDE_REFERENCE_SECTIONS")
        if exclude_reference_graph_env is not None:
            self.graph_exclude_reference_sections = exclude_reference_graph_env.strip().lower() not in {
                "0",
                "false",
                "no",
            }
        embedding_metadata_env = os.getenv("ENFORCE_EMBEDDING_METADATA")
        if embedding_metadata_env is not None:
            self.enforce_embedding_metadata = embedding_metadata_env.strip().lower() not in {
                "0",
                "false",
                "no",
            }
        cross_doc_same_type_env = os.getenv("CROSS_DOC_SAME_TYPE_REQUIRED_FOR_AUTO")
        if cross_doc_same_type_env is not None:
            self.cross_doc_same_type_required_for_auto = cross_doc_same_type_env.strip().lower() not in {
                "0",
                "false",
                "no",
            }
        cross_doc_llm_env = os.getenv("CROSS_DOC_LLM_ENABLED")
        if cross_doc_llm_env is not None:
            self.cross_doc_llm_enabled = cross_doc_llm_env.strip().lower() not in {"0", "false", "no"}
        auto_backfill_env = os.getenv("AUTO_BACKFILL_VECTORS")
        if auto_backfill_env is not None:
            self.auto_backfill_vectors = auto_backfill_env.strip().lower() not in {"0", "false", "no"}
        if self.query_graph_dynamic_model is None:
            self.query_graph_dynamic_model = self.gemini_model
