"""Deterministic local sparse term weighting for hybrid retrieval (WP6, finding `A5`).

Dense retrieval alone misses exact-token queries — an identifier, a metric name, a rare
proper noun — because those tokens carry little of the sentence embedding's mass. The fix
is a lexical channel, but it must not add a provider dependency: everything here is
computed locally from the chunk text and is a pure function of that text plus
`SPARSE_TOKENIZER_VERSION`.

Document-frequency weighting is deliberately *not* computed here. Qdrant applies IDF
server-side when the sparse vector is configured with `Modifier.IDF`, using the collection
it actually holds. Computing IDF locally would require a corpus pass that goes stale the
moment a document is added, and would silently disagree with the index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Bump whenever tokenization, weighting, or the stopword list changes: the stored vectors
# and the query vector must be produced by the same version or scores are meaningless.
SPARSE_TOKENIZER_VERSION = "bm25-local-r2-v1"

# BM25 term-frequency saturation and length normalization.
BM25_K1 = 1.2
BM25_B = 0.75

_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_\-\.]*", re.ASCII)

# Only closed-class words that carry no retrieval signal. Deliberately short: aggressive
# stopword removal is what makes lexical search fail on phrases like "part of speech".
_STOPWORDS = frozenset(
    """
    a an and are as at be been but by for from had has have he her his i in into is it
    its of on or she that the their them then there these they this to was were what
    when which who will with would you your
    """.split()
)

# Qdrant sparse indices are unsigned 32-bit. Hashing the token avoids maintaining a
# vocabulary file that would have to be versioned alongside every index.
_INDEX_SPACE = 2**32


@dataclass(frozen=True)
class SparseVector:
    """A Qdrant-shaped sparse vector: parallel index and value arrays."""

    indices: list[int]
    values: list[float]

    def __bool__(self) -> bool:
        return bool(self.indices)

    def as_dict(self) -> dict[str, list]:
        return {"indices": list(self.indices), "values": list(self.values)}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords removed, order preserved.

    Interior `.`, `-` and `_` are kept so identifiers survive whole (`gemini-embedding-2`,
    `v1.2`, `RAG_pipeline`), but trailing ones are stripped: sentence-final punctuation
    would otherwise make `v1.2.` a different term from `v1.2`.
    """
    tokens = []
    for raw in _TOKEN_PATTERN.findall(str(text).lower()):
        token = raw.rstrip("._-")
        if len(token) > 1 and token not in _STOPWORDS:
            tokens.append(token)
    return tokens


def token_index(token: str) -> int:
    """Stable index for a token. Python's `hash` is salted per process, so not that."""
    import hashlib

    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % _INDEX_SPACE


def encode_document(text: str, *, average_length: float) -> SparseVector:
    """BM25 term-frequency weights for one chunk, without the IDF factor.

    `average_length` is the expected chunk length in tokens — the configured chunk target,
    not a measured corpus statistic, so the weighting of a chunk never changes because
    some other document was ingested.
    """
    tokens = tokenize(text)
    if not tokens:
        return SparseVector([], [])
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    norm = 1.0 - BM25_B + BM25_B * (len(tokens) / max(1.0, float(average_length)))
    weights: dict[int, float] = {}
    for token, count in counts.items():
        weight = (count * (BM25_K1 + 1.0)) / (count + BM25_K1 * norm)
        index = token_index(token)
        # A hash collision is rare but must not silently drop a term.
        weights[index] = weights.get(index, 0.0) + weight
    ordered = sorted(weights.items())
    return SparseVector([index for index, _ in ordered], [value for _, value in ordered])


def encode_query(text: str) -> SparseVector:
    """Query-side weights. Each distinct term counts once; Qdrant supplies the IDF."""
    tokens = tokenize(text)
    if not tokens:
        return SparseVector([], [])
    indices = sorted({token_index(token) for token in tokens})
    return SparseVector(indices, [1.0] * len(indices))
