"""Reciprocal Rank Fusion (RRF) of the vector and BM25 rankings.

    rrf(d) = sum over retrievers r in which d appears of  1 / (k + rank_r(d))     (k = 60 by default)

This is a rank-fusion method, not a cross-encoder or LLM reranker: it only uses positions in the
two lists, never the text. A chunk found by one retriever only simply gets one term.

``score`` (contract: ``RetrievedChunk.score``, "normalized 0..1") is ``rrf / (2 / (k + 1))``, i.e.
relative to the best possible value (rank 1 in both lists). It expresses rank agreement between the
retrievers; it is NOT a probability and NOT comparable to cosine similarity.

Ties are broken deterministically: higher cosine similarity first (unknown last), then chunk_id.
"""

from __future__ import annotations

from typing import Sequence

from app.retrieval.models import Bm25Hit, FusedCandidate, VectorHit

_SYSTEMS = 2


def rrf_fuse(vector_hits: Sequence[VectorHit], bm25_hits: Sequence[Bm25Hit], k: int = 60) -> list[FusedCandidate]:
    """Fuse and deduplicate by ``chunk_id``; returns candidates best-first."""
    if k < 1:
        raise ValueError("RRF constant k must be >= 1")
    best = _SYSTEMS / (k + 1)
    by_id: dict[str, FusedCandidate] = {}

    def entry(chunk) -> FusedCandidate:
        cid = chunk.metadata.chunk_id
        if cid not in by_id:
            by_id[cid] = FusedCandidate(chunk=chunk, rrf=0.0, score=0.0)
        return by_id[cid]

    for h in vector_hits:
        e = entry(h.chunk)
        if e.vector_rank is None:  # first occurrence wins if a list ever repeats an ID
            e.vector_rank, e.vector_score = h.rank, h.similarity
            e.rrf += 1.0 / (k + h.rank)
    for h in bm25_hits:
        e = entry(h.chunk)
        if e.bm25_rank is None:
            e.bm25_rank, e.coverage = h.rank, h.coverage
            e.rrf += 1.0 / (k + h.rank)

    for e in by_id.values():
        e.score = min(1.0, e.rrf / best)
    return sorted(
        by_id.values(),
        key=lambda e: (-e.rrf, -(e.vector_score if e.vector_score is not None else float("-inf")),
                       e.chunk.metadata.chunk_id),
    )
