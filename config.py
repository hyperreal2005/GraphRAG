"""Global settings for GraphRAG."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def default_metadata_db() -> str:
    """The store a CLI uses when none is passed: METADATA_DB_PATH, else data/metadata.db.

    Read by the CLIs after they load .env, never inside Settings, so an explicit
    `metadata_db_path` - a test's temporary store, a `--db` flag - is never overridden.
    """
    return os.getenv("METADATA_DB_PATH") or "data/metadata.db"


@dataclass
class Settings:
    """Configuration defaults aligned to architecture reference."""

    metadata_db_path: Path = Path("data/metadata.db")
    # Where document cards come from: the corpus builder's manifest (titles) and the news
    # index (publisher, date). Files neither knows fall back to their first heading.
    corpus_manifest_path: Path = Path("data/corpus/MANIFEST.json")
    corpus_news_index_path: Path = Path("data/corpus/structured/news_index.csv")
    artifact_dir: Path = Path("artifacts/phase1")
    supported_formats: tuple[str, ...] = ("pdf", "txt", "md", "csv", "json")
    chunk_target_tokens: int = 700
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 100
    # 0 = one chunk per table row (the corpus of record, wp24; wp2f too). A positive value groups
    # consecutive rows into blocks of up to that many tokens (wp2g was built with 700).
    # Measured on the PDF/table judged arm with Cohere reranking, 2026-09-26: rows answered
    # 47/47, blocks 44/47 - a row is a precise match for a single cell, a block dilutes it.
    # Blocks cost 3.3x fewer extraction calls; answers are the product. See REFACTOR3 WP19.
    # Part of the pipeline fingerprint: ingesting into a corpus built with another value
    # rebuilds it (env CHUNK_TABLE_BLOCK_TARGET_TOKENS).
    chunk_table_block_target_tokens: int = 0
    # Gate 18 (REFACTOR3 WP24): extract consecutive table-row chunks in one call, up to this
    # many tokens of rows, while the chunks stay one per row. Every entity and relation is
    # filed under its row and grounded in it, so assertions and citations stay row-level.
    # 0 = one call per row. Part of the fingerprint when on (env EXTRACTION_TABLE_BATCH_TOKENS).
    # 600 = the corpus of record's build: 19.4 provider calls per PDF against 57.5 unbatched.
    extraction_table_batch_tokens: int = 600
    extraction_table_batch_max_rows: int = 30
    # WP2-A: merge parser items into logical sections before coref/chunking (B1).
    section_group_enabled: bool = True
    section_group_target_tokens: int = 700
    section_group_max_tokens: int = 2000
    section_group_page_gap_tolerance: int = 1
    # Coreference annotates the text extraction reads - "it [Rusnano]" - and never the
    # stored text (ingestion.coref). Until 2026-09-27 the stage ran and changed nothing: it
    # read a resolved text fastcoref never returns (REFACTOR3 WP20).
    # OFF BY MEASUREMENT. Re-DocRED, 100 documents, two runs off and one on, compared per
    # document: overall entity recall -17 +/- 16 gold entities, connected pairs -11 +/- 15,
    # relation precision unchanged - no gain. ORG and TIME rose and MISC and LOC fell,
    # within noise except on list-heavy documents, where annotation systematically cost
    # entities: Ramey Idriss's songs 13/14 found in 3 of 3 plain extractions, 3-9 annotated.
    # Off means the model is not loaded. On, it joins the pipeline fingerprint, so enabling
    # it re-ingests a corpus built without it (env COREF_ENABLED).
    coref_enabled: bool = False
    coref_profile: str = "balanced"
    strict_coref: bool = True
    enable_graph: bool = True
    gemini_api_key: str | None = None
    # The query-time model: analysis, answer generation, LLM reranking (env GEMINI_MODEL).
    gemini_model: str = "gemini-3.5-flash-lite"
    # The build-time model: extraction, merge adjudication and community summaries (env
    # INGESTION_MODEL). Separate since 2026-09-27 so trying another answer model does not
    # change the graph's identity: it is part of the pipeline fingerprint, and extraction is
    # reused only across chunks the same model extracted, so changing it re-extracts the
    # corpus. Default = the model the corpus of record (wp24) was built with. Measured against
    # 3.5-flash-lite on Re-DocRED (gold entities z +3.6, connected pairs z +5.1, relation
    # precision 18.8% -> 21.1%) and on the papers' gold passages (answer captured 38% -> 76%
    # for table rows, 37% -> 60% for prose) - REFACTOR3 WP25.
    ingestion_model: str = "gemini-3.8-flash"
    gemini_max_workers: int = 2
    ingestion_provider_timeout_seconds: float = 120.0
    ingestion_progress_interval_chunks: int = 5
    # Hard ceiling on extractable chunks per document, checked before the first
    # provider call. Bounds what one bad input can cost. 0 disables the check.
    ingestion_max_chunks_per_document: int = 400
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
    # WP2-B: identity is name-first, unconditionally - the disambiguator matches on the
    # normalized name and its aliases, and type only blocks on a coarse-class conflict.
    # An `entity_identity_name_first` flag stood here until 2026-09-01 and was never read
    # by anything, so it advertised a switch that did not exist.
    entity_coarse_class_blocking: bool = True
    entity_same_type_bonus: float = 0.02
    cross_doc_auto_merge_threshold: float = 0.95
    cross_doc_llm_min_threshold: float = 0.80
    cross_doc_llm_max_threshold: float = 0.95
    cross_doc_candidate_k: int = 5
    cross_doc_auto_merge_margin: float = 0.05
    cross_doc_llm_confidence_threshold: float = 0.90
    cross_doc_same_type_required_for_auto: bool = True
    cross_doc_llm_enabled: bool = True
    # Switched from `gemini-embedding-2` on 2026-09-05. That model sheds load
    # server-side regardless of billing tier: measured interleaved against -001 on one key
    # in one minute, it failed 8 of 30 calls where -001 failed 0 of 30, and it failed even
    # at a paced 30 requests/minute - 0.6% of the project's 5,000 RPM limit, with the
    # dashboard showing peak 214/5K. The 429 carries no quota metric and no Retry-After,
    # and points at Vertex capacity docs. It cost five judged runs before it was diagnosed.
    #
    # Retrieval quality is a wash on this corpus (49 judged questions, same 1,233 chunks):
    # -2 recall@10 1.000 / MRR 0.708 / median rank 2; -001 recall@10 0.980 / MRR 0.740 /
    # median rank 1. Each wins one metric by one question, which is noise at this n.
    # Reliability is not a wash, so -001 wins.
    embedding_model_id: str = "gemini-embedding-001"
    embedding_dim: int = 3072
    embedding_version: str = "embedding001-3072-r4"
    embedding_prompt_version: str = "embedding001-tasktype-v1"
    embedding_normalization_policy: str = "provider-default"
    # WP2-E: aligned to the documented asymmetric form. The document side carries no
    # `task:` component; only the query side does.
    embedding_document_prefix: str = "title: {title} | text: "
    embedding_query_prefix: str = "task: search result | query: "
    gemini_embedding_task_type: str = "RETRIEVAL_DOCUMENT"
    # WP2-E: model behaviour is selected by explicit capability, never by matching a
    # model-name string. Setting the documented `gemini-embedding-2-preview` id used
    # to silently flip the pipeline back to legacy behaviour (REVIEW1 B10).
    # -001 takes task_type natively and needs no instruction prefix; -2 took the
    # opposite convention. WP2-E exists so this is configuration, not a code branch.
    embedding_model_supports_task_type: bool = True
    embedding_model_uses_instruction_prefix: bool = False
    embedding_probe_on_startup: bool = True
    embedding_batch_size: int = 32
    # Retry a *throttled* embedding call rather than letting it fail the whole vector
    # stage: the outbox's three attempts re-embed every chunk in the document to recover
    # one 429. A rebuild hit 26 of them in six documents and lost a document that way.
    embedding_rate_limit_attempts: int = 5
    embedding_rate_limit_base_seconds: float = 1.0
    # Total backoff ceiling, not just an attempt count. The query path gives the vector
    # stage 15s and entity linking 20s, so an unbounded 2+4+8+16s backoff converts a
    # recoverable 429 into a stage timeout - worse than one failed call. Slow retries
    # belong at the outbox, which has no deadline.
    embedding_rate_limit_max_total_seconds: float = 6.0
    # Process-wide minimum gap between embedding requests. Default 0 (off). Concurrency is
    # the wrong lever against a per-minute quota, and the query path has no worker count to
    # turn down - its three retrieval stages are dispatched by one asyncio.gather.
    embedding_min_interval_seconds: float = 0.0
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_timeout_seconds: float = 30.0
    # Points per upsert request. One request per document is unbounded: a 987-chunk
    # document built a 47.7 MB body against Qdrant 33.5 MB limit and could never
    # succeed, however many times it was retried.
    qdrant_upsert_batch_size: int = 64
    qdrant_chunks_collection: str = "chunks"
    qdrant_entities_collection: str = "entities"
    qdrant_communities_collection: str = "communities"
    # Phase 8 (built 2026-09-01). The entity graph is sparse - 9,141 active entities joined
    # by 3,819 distinct pairs - so Louvain returns 973 communities of which 296 hold one or
    # two members. A two-entity cluster carries no theme, so the floor is a correctness
    # control first and a cost control second: >=10 members keeps 218 communities covering
    # 4,596 entities, against 973 summary calls for all of them.
    community_min_size: int = 10
    community_seed: int = 17
    community_summary_max_entities: int = 40
    community_summary_max_relations: int = 60
    # Phase 9 healer. The graph is schema-free, so which predicates are single-valued is
    # not declared anywhere: it is inferred from the corpus. A predicate whose subjects
    # almost always carry exactly one object is treated as functional, and a subject with
    # several is then a contradiction. Without this, all 1,528 subject/predicate pairs with
    # multiple objects report as conflicts, most of them legitimately multi-valued.
    # Matches GraphWriter.find_value_conflicts, which the healer delegates to and whose
    # 0.9 / 5 defaults are already measured and documented (1,520 raw conflicts -> 75).
    heal_functional_threshold: float = 0.9
    heal_functional_min_support: int = 5
    # The Refactor 2 prompt, retained after Refactor 3 measured three variants against it
    # and rejected all three (docs/REFACTOR3.md WP2 and WP3):
    #
    #   r3-v1  entity categories stated unconditionally. Re-DocRED entity recall 52.7% ->
    #          60.6%, but on arXiv it traded the subject matter for scenery: added UK,
    #          NASA and bibliography surnames, dropped RAG, BM25, TF-IDF, MRR@10.
    #   r3-v2  the same categories made conditional. Protected arXiv, fell to 50.6% on
    #          Re-DocRED - below baseline. One global entity instruction cannot serve both.
    #   r3-v3  relation density. Recall 5.3% -> 5.5% (~7 relations in 3,655) bought with
    #          precision 20.3% -> 18.2%, and the confidence instruction that was meant to
    #          route uncertain relations to review produced 0 of 1,100 below 0.5.
    extraction_prompt_version: str = "assertion-r2-v1"
    # WP7: the extraction, embedding and judge prompts were all versioned; the generation
    # prompt was not, so two runs with different answer instructions produced reports that
    # claimed to be comparable. "attribution" marks the rule added after the judge found
    # the generator asserting which publication reported a fact the evidence only stated.
    # "document-cards" (WP23): evidence carries its document's title, publisher and date.
    # "attribution-check" (WP24): a claim naming an outlet must cite evidence from it; the
    # retry message says so. "prose-resynthesis" (WP26): re-synthesis has its own plain-text
    # instruction, and "TechCrunch reported..." is checked at a sentence start.
    # "polar-verdict" (ROADMAP C1): yes/no questions carry a verdict the answer opens with.
    generation_prompt_version: str = "answer-r7-polar-verdict-v1"
    assertion_review_confidence_threshold: float = 0.50
    # Refactor 3: an assertion whose predicate the corpus uses fewer than this many
    # times is flagged for review, unless a second document corroborates the claim.
    # Measured against gold, predicates used 1-4 times match a gold relation about 1%
    # of the time against 43.6% for those used 20+ times. Absolute count, so a small
    # corpus flags more - see ingestion/review_status.py.
    review_rare_predicate_below: int = 5
    extraction_max_attempts: int = 3
    # WP7: reuse a chunk's assertions when a re-ingest would extract it identically (A4):
    # the same text read (coreference annotations included) by the same extractor - the
    # same extraction prompt version and generative model, per assertion provenance. A
    # prompt or model change therefore re-extracts; every other fingerprint change reuses.
    extraction_reuse_enabled: bool = True
    # Opt-in: carry assertions across a prompt or model change anyway, to avoid the cost of
    # re-extraction (e.g. a cosmetic prompt edit). The reused assertions keep the prompt
    # version and model that actually produced them. Until 2026-09-27 this was the only
    # behaviour, so a new prompt never reached text that had not changed (REFACTOR3 WP22).
    extraction_reuse_across_prompts: bool = False
    # WP3-C: share of chunks that may be quarantined before the document fails.
    extraction_chunk_failure_ratio: float = 0.05
    extraction_retry_base_seconds: float = 1.0
    graph_schema_version: int = 5
    outbox_max_attempts: int = 5
    outbox_retry_base_seconds: float = 1.0
    outbox_lease_seconds: float = 60.0
    # WP3-B: age at which a mid-build version is treated as abandoned by the sweeper.
    outbox_orphan_sweep_hours: float = 24.0
    query_embedding_task_type: str = "RETRIEVAL_QUERY"
    # WP8 calibration, 2026-08-27 (REVIEW1 `B7`, second half). Swept together over 60
    # exact-name positives and 12 absent-name negatives on the WP2-F corpus; see
    # docs/REFACTOR2.md section 8.6. The margin was the damaging parameter, not the
    # threshold: at 0.05 only 44 of 60 exact names resolved, at 0.00 it is 57, and the
    # absent names are still refused (0 false links). After WP2-B a technical corpus is
    # full of legitimate near-neighbours ("Qwen3-4B-Instruct" beside "Qwen3-4B"), so a
    # top-2 gap requirement mostly punishes correct answers rather than catching ambiguity.
    # Recalibrated 2026-09-16 for gemini-embedding-001; the 0.75 above was measured on
    # gemini-embedding-2, whose cosine distribution sits higher. On -001 the 12 absent
    # names in the fixture top out at 0.711 and the 57 reachable positives bottom at
    # 0.724, so 0.72 links every reachable positive with zero false links. At 0.75 it lost
    # 12 of 57 - and that, not chunking, is why graph retrieval fell silent on wp2g
    # (33 -> 8 of 50 questions with graph evidence). This value is per embedding model.
    entity_link_resolve_threshold: float = 0.72
    entity_link_ambiguous_threshold: float = 0.70
    entity_link_resolve_margin: float = 0.00
    entity_link_candidate_k: int = 5
    entity_link_mention_weight: float = 0.60
    enforce_embedding_metadata: bool = True
    query_vector_top_k: int = 15
    # WP6: lexical channel fused with dense retrieval by reciprocal rank (A5).
    # DEFAULT OFF - measured and rejected 2026-08-27. On the judged set, holding corpus
    # and questions fixed and toggling only this flag, recall@10 fell 0.653 to 0.612 and
    # MRR 0.463 to 0.449. The code and its tests are kept so the decision can be revisited
    # with weighted fusion; see docs/REFACTOR2.md section 8.5.
    query_sparse_enabled: bool = False
    query_rrf_k: int = 60
    # Expected chunk length used for BM25 length normalization. The configured chunk
    # target, not a measured corpus statistic, so a chunk weight never changes because
    # some unrelated document was ingested.
    sparse_average_chunk_tokens: int = 700
    query_mmr_enabled: bool = True
    query_mmr_lambda: float = 0.65
    query_mmr_candidate_multiplier: int = 3
    query_graph_local_hops: int = 3
    query_graph_multihop_hops: int = 4
    query_graph_path_limit: int = 25
    # WP4-B: per-level frontier cap for level-wise BFS traversal.
    query_graph_level_fanout: int = 64
    query_analysis_timeout_seconds: float = 15.0
    query_entity_link_timeout_seconds: float = 20.0
    query_graph_timeout_seconds: float = 15.0
    query_vector_timeout_seconds: float = 15.0
    query_rerank_timeout_seconds: float = 15.0
    query_generation_timeout_seconds: float = 45.0
    # NOT raw generated Cypher, despite the name. Since WP4-B the traversal is a constant
    # level-wise BFS in retrieval/graph_traversal.py; enabling this asks the model for one
    # thing only - which relationship types to follow - validated against the live schema
    # and clamped by _compile_logical_plan. Nothing model-authored is ever executed, so
    # finding A7 (Neo4j Community has no RBAC) is not an argument against it.
    # It stays off because it costs an extra provider call on every query and has never
    # been measured to help, not because it is dangerous.
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
    # Refactor 3: score multiplier for an assertion flagged for review at ingest.
    # Admitted to retrieval but ranked below confident evidence, so that raising
    # extraction recall does not mean trusting uncertain facts equally.
    query_flagged_assertion_penalty: float = 0.60
    query_context_max_items: int = 20
    # The single biggest retrieval win in the project, and it was a config value.
    # At 12,000 the packed context held roughly three 700-word chunks, and
    # `_pack_context` stops at the first item that will not fit - so gold evidence
    # that retrieval had already found, at median rank 7, was discarded before the
    # model ever saw it. Measured on the judged set: recall@10 0.653 at 12,000,
    # 0.878 at 24,000, 0.918 at 40,000, against a 0.85 gate that had never been met.
    query_context_max_chars: int = 40000
    query_filter_boilerplate_chunks: bool = True
    query_include_reference_chunks: bool = False
    query_citation_max: int = 8
    # Refactor 3 WP5/WP6. When the model judges the retrieved evidence does not
    # address the question, route to the insufficient-evidence response instead of
    # answering. Measured on the 50-question judged set: claim faithfulness 0.941
    # -> 1.000 and abstention 0.000 -> 1.000, but questions answered fell 49 -> 30.
    # That trade is a product judgement - trustworthiness against answer rate - so
    # it is a setting rather than a fixed behaviour. False restores the previous
    # behaviour of always answering.
    generation_refuse_when_unanswerable: bool = True
    # ROADMAP C1. A polar question ("Was there consistency between the TechCrunch and
    # Fortune reports?") gets an explicit verdict: a `verdict` field, an answer that opens
    # with it, and a false premise answered "no" instead of refused. Adopted under the C1
    # decision rule: correct verdicts 15/41 -> 32/41 (news 2 -> 15 of 18, held out 13 -> 17
    # of 23), faithfulness 0.938 / 0.927, abstention controls 9 -> 10 of 10.
    generation_polar_verdicts: bool = True
    # ROADMAP C1, variant 2. A question asking what a named publication reported, when no
    # evidence comes from that publication, is refused rather than answered from other
    # outlets' coverage of the story (the B1 abstention miss). Measured and not adopted: it
    # refused that question (20/20) but also 11 news questions V1 answers, 6 of them with a
    # gold article at rank 1 - news verdicts 15 -> 9 of 18, entity answers 31 -> 27.
    generation_refuse_absent_named_source: bool = False
    # ROADMAP C2. For a GLOBAL or HYBRID question, each retrieved community adds this many of
    # its most central member passages, one per document, so the answer can cite real text and
    # draw on more of the theme's documents. 0 is off. Measured at 3 and not adopted: uncited
    # claims 0.101 -> 0.068, but theme recall 0.567 -> 0.614, inside run-to-run noise.
    query_community_grounding_passages: int = 0
    # ROADMAP C2. A bonus of at most this much cosine for the community spanning the most
    # documents among the candidates, log-scaled. 0 is off. Measured at 0.05 and not adopted:
    # theme recall 0.567 -> 0.654, but uncited claims rose to 0.112, and the overview question
    # it was meant for did not move.
    query_community_size_prior: float = 0.0
    query_temperature_local: float = 0.1
    query_temperature_multihop: float = 0.1
    query_temperature_comparative: float = 0.2
    query_temperature_global: float = 0.35
    query_temperature_hybrid: float = 0.2
    cohere_api_key: str | None = None
    query_rerank_enabled: bool = True
    # rerank-v4.0-pro on measurement (REFACTOR3 WP17, WP19). Swept offline on the judged
    # news questions, recall@10 / MRR: rank fusion 0.959 / 0.629, rerank-v3.5 0.918 / 0.755,
    # rerank-v4.0-pro 0.980 / 0.809. Live it read MRR 0.820 on the tuned set and 0.813 on 60
    # held-out questions on wp2f; on wp24, the corpus of record from 2026-10-01, 0.845 and
    # 0.903 (REFACTOR3 WP26). The gate is 0.75.
    query_rerank_model: str = "rerank-v4.0-pro"
    query_rerank_top_n: int = 15
    # Cohere rate limiting. Pacing is off by default; set it for a trial key (limited per
    # minute) so a run stays under the limit rather than degrading queries on 429. The
    # retry budget must stay well under query_rerank_timeout_seconds, which bounds the
    # whole fusion stage: overrunning it loses the fused context entirely.
    query_rerank_min_interval_seconds: float = 0.0
    query_rerank_rate_limit_max_total_seconds: float = 3.0
    # Per-request ceiling on a Cohere call, and one deadline for pacing + requests +
    # retries together. Both sit inside query_rerank_timeout_seconds, which bounds the
    # whole fusion stage: a stalled call used to outlive it and discard the fused context
    # (seven questions in one judged run). Now reranking degrades to fused order instead.
    # 4 s (was 5) so a stall can be retried once inside the 10 s budget: 4 + 0.5 + 4 < 10.
    # A normal rerank-v4.0-pro call takes under a second (WP23).
    query_rerank_request_timeout_seconds: float = 4.0
    query_rerank_total_budget_seconds: float = 10.0
    # The architecture's stated alternative to Cohere (5.5): a local cross-encoder, no key,
    # no per-query cost, ~0.65 s for 60 candidates on CPU. Phase 6's "fallback" returned the
    # input order unchanged, so until WP17 no reranker had ever run. Measured, every
    # cross-encoder made retrieval worse than no reranking (table below). It stays a
    # fallback tier behind Cohere, off by default.
    query_rerank_local_enabled: bool = False
    query_rerank_local_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    query_rerank_local_max_items: int = 60
    # LLM-as-reranker: the generation model orders the top candidates, one call per query.
    # Enabled 2026-09-25 on measurement. Every cross-encoder tried made retrieval worse on
    # this corpus's long compositional questions; the model that answers them orders them
    # well. Swept on wp2g, 49 judged questions, recall@10 / MRR / offered-p@8 / ms per query:
    #   rank fusion only              0.959  0.629  0.253      29
    #   ms-marco-MiniLM-L-6 (named)   0.918  0.543  0.218   2,783
    #   bge-reranker-base             0.816  0.427  0.237  15,911
    #   bge-reranker-v2-m3            0.898  0.659  0.254  76,337
    #   LLM (gemini-3.5-flash-lite)   0.980  0.682  0.272   1,315
    #
    # DEFAULT OFF. The offline sweep above ranks the tiers correctly but overstates the
    # LLM's gain: its baseline reads recall 0.959 / MRR 0.629 where the live pipeline reads
    # 0.980 / 0.662 for the same configuration, so its deltas do not transfer. Measured
    # live on the judged set, the LLM tier is net-neutral on the gates - MRR +0.018,
    # recall -0.020, citation precision -0.005, faithfulness -0.003, all inside the
    # ~0.02 that one question is worth at n=49 - while costing one generation call and
    # 1.3 s per query and raising degraded_rate 0.220 -> 0.260. It does lift graph path
    # utility 0.610 -> 0.780, which is the one real effect and the reason to revisit it if
    # graph evidence is ever the thing being optimised.
    query_rerank_llm_enabled: bool = False
    query_rerank_llm_top_n: int = 20
    query_rerank_llm_snippet_chars: int = 400
    query_cache_enabled: bool = False
    query_cache_ttl_seconds: int = 3600
    # Architecture 5.1: global and summary answers change least, so they live longer.
    query_cache_global_ttl_seconds: int = 21600
    # ROADMAP D2. The query cache and session chains live in Redis (architecture 4.3); when it
    # cannot be reached, an in-process cache stands in and no query fails over it.
    redis_url: str = "redis://127.0.0.1:6380/0"
    # Every key GraphRAG writes starts with this, so even a Redis shared with another application
    # could never mix their keys.
    redis_key_prefix: str = "graphrag:"
    # ROADMAP D2 (architecture 9.5). A follow-up question is read in the light of the session's
    # earlier turns: rewritten to stand alone, with their entities as extra graph seeds.
    # Measured and adopted: on 30 follow-ups ambiguous by construction, 5 answered alone, 21
    # with memory, 21 asked in full; topic switches unchanged. Off as a library default so
    # tests and evaluations never touch Redis; main_query.py (and the API) turn it on.
    session_memory_enabled: bool = False
    # The architecture's mechanism: the previous turn's linked entities seed graph traversal.
    # A setting so the rewrite can be measured with and without it.
    session_entity_seeds: bool = True
    session_max_turns: int = 10
    session_context_turns: int = 3
    session_ttl_seconds: int = 86400
    # ROADMAP D3. The API binds to localhost; `main_api.py` refuses any other address unless
    # API_TOKEN is set, and with a token every request must carry it. Ingest requests may only
    # name paths under these roots, relative to the working directory.
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_token: str | None = None
    api_ingest_roots: tuple[str, ...] = ("data",)
    # ROADMAP D4. Scheduled work, inside the API process (`main_api.py` turns it on; off as a
    # library default so nothing in tests or evaluations schedules anything).
    scheduler_enabled: bool = False
    scheduler_tick_seconds: int = 300
    # Architecture 6.3: rebuild the communities after this many documents change (added,
    # removed or re-versioned). 0 turns the automatic rebuild off.
    community_rebuild_every_n_documents: int = 50
    # Architecture 6.1, read-only here: a nightly healer report from this local hour.
    healer_report_enabled: bool = True
    healer_report_hour: int = 3
    report_dir: Path = Path("data/reports")
    # WP1 evaluation harness (docs/REFACTOR2.md section 3)
    eval_judge_model: str = "gemini-3.5-flash-lite"
    # Phase 10.1. Off by default so nothing needs a collector; enable to send every query
    # and ingestion stage to Phoenix at the endpoint below (`python -m phoenix.server.main
    # serve` runs one locally on :6006). Provider calls nest under their stage.
    tracing_enabled: bool = False
    tracing_endpoint: str = "http://localhost:6006/v1/traces"
    tracing_project: str = "graphrag"
    eval_judge_prompt_version: str = "judge-r2-v1"

    def ensure_directories(self) -> None:
        """Ensure required directories exist before execution."""
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def __post_init__(self) -> None:
        """Allow env vars to override sensitive defaults."""
        self.gemini_api_key = self.gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", self.gemini_model).strip() or self.gemini_model
        self.ingestion_model = (
            os.getenv("INGESTION_MODEL", self.ingestion_model).strip() or self.ingestion_model
        )
        try:
            self.section_group_target_tokens = max(
                1, int(os.getenv("SECTION_GROUP_TARGET_TOKENS", self.section_group_target_tokens))
            )
        except ValueError:
            pass
        try:
            self.section_group_max_tokens = max(
                1, int(os.getenv("SECTION_GROUP_MAX_TOKENS", self.section_group_max_tokens))
            )
        except ValueError:
            pass
        try:
            self.chunk_table_block_target_tokens = max(
                0,
                int(os.getenv("CHUNK_TABLE_BLOCK_TARGET_TOKENS", self.chunk_table_block_target_tokens)),
            )
        except ValueError:
            pass
        for attribute, environment_name in (
            ("extraction_table_batch_tokens", "EXTRACTION_TABLE_BATCH_TOKENS"),
            ("extraction_table_batch_max_rows", "EXTRACTION_TABLE_BATCH_MAX_ROWS"),
        ):
            try:
                setattr(self, attribute, max(0, int(os.getenv(environment_name, getattr(self, attribute)))))
            except ValueError:
                pass
        coref_raw = os.getenv("COREF_ENABLED")
        if coref_raw is not None:
            self.coref_enabled = coref_raw.strip().lower() not in {"0", "false", "no", "off", ""}
        try:
            self.section_group_page_gap_tolerance = max(
                0,
                int(os.getenv(
                    "SECTION_GROUP_PAGE_GAP_TOLERANCE",
                    self.section_group_page_gap_tolerance,
                )),
            )
        except ValueError:
            pass
        try:
            self.entity_same_type_bonus = max(
                0.0, min(0.2, float(os.getenv("ENTITY_SAME_TYPE_BONUS", self.entity_same_type_bonus)))
            )
        except ValueError:
            pass
        coarse_block_env = os.getenv("ENTITY_COARSE_CLASS_BLOCKING")
        if coarse_block_env is not None:
            self.entity_coarse_class_blocking = coarse_block_env.strip().lower() not in {"0", "false", "no"}
        section_group_env = os.getenv("SECTION_GROUP_ENABLED")
        if section_group_env is not None:
            self.section_group_enabled = section_group_env.strip().lower() not in {"0", "false", "no"}
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
            self.qdrant_upsert_batch_size = max(
                1, int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", self.qdrant_upsert_batch_size))
            )
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
            self.query_graph_level_fanout = max(
                1, int(os.getenv("QUERY_GRAPH_LEVEL_FANOUT", self.query_graph_level_fanout))
            )
        except ValueError:
            pass
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
            self.entity_link_candidate_k = max(
                2,
                min(20, int(os.getenv("ENTITY_LINK_CANDIDATE_K", self.entity_link_candidate_k))),
            )
        except ValueError:
            pass
        try:
            self.entity_link_mention_weight = max(
                0.0,
                min(
                    1.0,
                    float(os.getenv("ENTITY_LINK_MENTION_WEIGHT", self.entity_link_mention_weight)),
                ),
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
            self.query_cache_global_ttl_seconds = int(
                os.getenv("QUERY_CACHE_GLOBAL_TTL_SECONDS", self.query_cache_global_ttl_seconds)
            )
        except ValueError:
            pass
        self.redis_url = os.getenv("REDIS_URL", self.redis_url)
        self.redis_key_prefix = os.getenv("REDIS_KEY_PREFIX", self.redis_key_prefix)
        self.api_host = os.getenv("API_HOST", self.api_host)
        try:
            self.api_port = int(os.getenv("API_PORT", self.api_port))
        except ValueError:
            pass
        self.api_token = os.getenv("API_TOKEN") or self.api_token
        for name in ("scheduler_tick_seconds", "community_rebuild_every_n_documents", "healer_report_hour"):
            try:
                setattr(self, name, max(0, int(os.getenv(name.upper(), getattr(self, name)))))
            except ValueError:
                pass
        healer_env = os.getenv("HEALER_REPORT_ENABLED")
        if healer_env is not None:
            self.healer_report_enabled = healer_env.strip().lower() not in {"0", "false", "no"}
        self.report_dir = Path(os.getenv("REPORT_DIR", str(self.report_dir)))
        roots_env = os.getenv("API_INGEST_ROOTS")
        if roots_env:
            self.api_ingest_roots = tuple(r.strip() for r in roots_env.split(",") if r.strip())
        for name in ("session_max_turns", "session_context_turns", "session_ttl_seconds"):
            try:
                setattr(self, name, max(1, int(os.getenv(name.upper(), getattr(self, name)))))
            except ValueError:
                pass
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
        try:
            self.community_min_size = int(os.getenv("COMMUNITY_MIN_SIZE", self.community_min_size))
        except (TypeError, ValueError):
            self.community_min_size = self.community_min_size
        self.embedding_model_id = os.getenv("EMBEDDING_MODEL_ID", self.embedding_model_id)
        for attribute, environment_name in (
            ("embedding_model_supports_task_type", "EMBEDDING_MODEL_SUPPORTS_TASK_TYPE"),
            ("embedding_model_uses_instruction_prefix", "EMBEDDING_MODEL_USES_INSTRUCTION_PREFIX"),
            ("embedding_probe_on_startup", "EMBEDDING_PROBE_ON_STARTUP"),
        ):
            raw_value = os.getenv(environment_name)
            if raw_value is not None:
                setattr(self, attribute, raw_value.strip().lower() not in {"0", "false", "no"})
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
        self.generation_prompt_version = os.getenv(
            "GENERATION_PROMPT_VERSION",
            self.generation_prompt_version,
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
            self.ingestion_max_chunks_per_document = max(
                0,
                int(os.getenv(
                    "INGESTION_MAX_CHUNKS_PER_DOCUMENT",
                    self.ingestion_max_chunks_per_document,
                )),
            )
        except ValueError:
            pass
        refuse_env = os.getenv("GENERATION_REFUSE_WHEN_UNANSWERABLE")
        if refuse_env is not None:
            self.generation_refuse_when_unanswerable = (
                refuse_env.strip().lower() not in {"0", "false", "no"}
            )
        try:
            self.query_community_grounding_passages = max(
                0,
                int(os.getenv("QUERY_COMMUNITY_GROUNDING_PASSAGES", self.query_community_grounding_passages)),
            )
        except ValueError:
            pass
        try:
            self.query_community_size_prior = max(
                0.0, float(os.getenv("QUERY_COMMUNITY_SIZE_PRIOR", self.query_community_size_prior))
            )
        except ValueError:
            pass
        polar_env = os.getenv("GENERATION_POLAR_VERDICTS")
        if polar_env is not None:
            self.generation_polar_verdicts = polar_env.strip().lower() not in {"0", "false", "no"}
        absent_source_env = os.getenv("GENERATION_REFUSE_ABSENT_NAMED_SOURCE")
        if absent_source_env is not None:
            self.generation_refuse_absent_named_source = (
                absent_source_env.strip().lower() not in {"0", "false", "no"}
            )
        sparse_env = os.getenv("QUERY_SPARSE_ENABLED")
        if sparse_env is not None:
            self.query_sparse_enabled = sparse_env.strip().lower() not in {"0", "false", "no"}
        try:
            self.query_rrf_k = max(1, int(os.getenv("QUERY_RRF_K", self.query_rrf_k)))
        except ValueError:
            pass
        reuse_env = os.getenv("EXTRACTION_REUSE_ENABLED")
        if reuse_env is not None:
            self.extraction_reuse_enabled = reuse_env.strip().lower() not in {"0", "false", "no"}
        across_env = os.getenv("EXTRACTION_REUSE_ACROSS_PROMPTS")
        if across_env is not None:
            self.extraction_reuse_across_prompts = across_env.strip().lower() not in {"0", "false", "no", ""}
        try:
            self.extraction_chunk_failure_ratio = max(
                0.0,
                min(1.0, float(os.getenv(
                    "EXTRACTION_CHUNK_FAILURE_RATIO",
                    self.extraction_chunk_failure_ratio,
                ))),
            )
        except ValueError:
            pass
        try:
            self.outbox_lease_seconds = float(
                os.getenv("OUTBOX_LEASE_SECONDS", self.outbox_lease_seconds)
            )
        except ValueError:
            self.outbox_lease_seconds = self.outbox_lease_seconds
        try:
            self.outbox_orphan_sweep_hours = max(
                0.0,
                float(os.getenv("OUTBOX_ORPHAN_SWEEP_HOURS", self.outbox_orphan_sweep_hours)),
            )
        except ValueError:
            pass
        try:
            self.extraction_retry_base_seconds = float(
                os.getenv("EXTRACTION_RETRY_BASE_SECONDS", self.extraction_retry_base_seconds)
            )
        except ValueError:
            self.extraction_retry_base_seconds = self.extraction_retry_base_seconds
        self.query_rerank_model = os.getenv("QUERY_RERANK_MODEL", self.query_rerank_model)
        try:
            self.query_rerank_min_interval_seconds = float(
                os.getenv("QUERY_RERANK_MIN_INTERVAL_SECONDS", self.query_rerank_min_interval_seconds)
            )
        except (TypeError, ValueError):
            self.query_rerank_min_interval_seconds = self.query_rerank_min_interval_seconds
        self.query_rerank_local_model = os.getenv(
            "QUERY_RERANK_LOCAL_MODEL", self.query_rerank_local_model
        )
        tracing_env = os.getenv("TRACING_ENABLED")
        if tracing_env is not None:
            self.tracing_enabled = tracing_env.strip().lower() not in {"0", "false", "no"}
        self.tracing_endpoint = os.getenv("TRACING_ENDPOINT", self.tracing_endpoint)
        self.tracing_project = os.getenv("TRACING_PROJECT", self.tracing_project)
        try:
            self.embedding_min_interval_seconds = float(
                os.getenv("EMBEDDING_MIN_INTERVAL_SECONDS", self.embedding_min_interval_seconds)
            )
        except (TypeError, ValueError):
            self.embedding_min_interval_seconds = self.embedding_min_interval_seconds
        try:
            self.embedding_rate_limit_max_total_seconds = float(
                os.getenv(
                    "EMBEDDING_RATE_LIMIT_MAX_TOTAL_SECONDS",
                    self.embedding_rate_limit_max_total_seconds,
                )
            )
        except (TypeError, ValueError):
            self.embedding_rate_limit_max_total_seconds = self.embedding_rate_limit_max_total_seconds
        try:
            self.embedding_rate_limit_attempts = int(
                os.getenv("EMBEDDING_RATE_LIMIT_ATTEMPTS", self.embedding_rate_limit_attempts)
            )
        except (TypeError, ValueError):
            self.embedding_rate_limit_attempts = self.embedding_rate_limit_attempts
        local_rerank_env = os.getenv("QUERY_RERANK_LOCAL_ENABLED")
        if local_rerank_env is not None:
            self.query_rerank_local_enabled = local_rerank_env.strip().lower() not in {"0", "false", "no"}
        try:
            self.query_rerank_local_max_items = int(
                os.getenv("QUERY_RERANK_LOCAL_MAX_ITEMS", self.query_rerank_local_max_items)
            )
        except (TypeError, ValueError):
            self.query_rerank_local_max_items = self.query_rerank_local_max_items
        self.eval_judge_model = os.getenv("EVAL_JUDGE_MODEL", self.eval_judge_model)
        self.eval_judge_prompt_version = os.getenv(
            "EVAL_JUDGE_PROMPT_VERSION",
            self.eval_judge_prompt_version,
        )
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
        session_env = os.getenv("SESSION_MEMORY_ENABLED")
        if session_env is not None:
            self.session_memory_enabled = session_env.strip().lower() not in {"0", "false", "no"}
        seeds_env = os.getenv("SESSION_ENTITY_SEEDS")
        if seeds_env is not None:
            self.session_entity_seeds = seeds_env.strip().lower() not in {"0", "false", "no"}
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
        if self.query_graph_dynamic_model is None:
            self.query_graph_dynamic_model = self.gemini_model
