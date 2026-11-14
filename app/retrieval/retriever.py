"""Central retrieval service: vector + BM25 → RRF → evidence analysis → conflict detection.

No LLM is involved and nothing is generated. Depends only on the ``Embedder`` protocol, a store
satisfying ``RetrievalStore``, and the shared schemas; it does not import ``app.generation``.
"""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Optional

from app.embeddings.base import Embedder
from app.retrieval.bm25_retriever import Bm25Retriever
from app.retrieval.conflict_detector import detect_conflicts
from app.retrieval.evidence_analyzer import analyze_evidence
from app.retrieval.hybrid_ranker import rrf_fuse
from app.retrieval.models import InvalidQueryError, RetrievalSettings, RetrievalStore
from app.retrieval.text import tokenize
from app.retrieval.vector_retriever import VectorRetriever
from app.schemas import RetrievalResult, RetrievedChunk

logger = logging.getLogger(__name__)


def build_where(doc_ids: Optional[list[str]] = None, versions: Optional[list[str]] = None) -> Optional[dict]:
    """Chroma ``where`` filter from the QueryRequest's optional ``doc_ids`` / ``versions``."""
    clauses = []
    if doc_ids:
        clauses.append({"doc_id": {"$in": list(doc_ids)}})
    if versions is not None and len(versions) > 0:
        clauses.append({"version": {"$in": list(versions)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


class Retriever:
    def __init__(self, embedder: Embedder, store: RetrievalStore, settings: Optional[RetrievalSettings] = None):
        self.settings = settings or RetrievalSettings()
        self._store = store
        self._vector = VectorRetriever(embedder, store)
        self._bm25 = Bm25Retriever(store, self.settings.bm25_k1, self.settings.bm25_b)

    def retrieve(self, question: str, *, top_k: int = 5, where: Optional[dict] = None) -> RetrievalResult:
        """Return the contract's ``RetrievalResult`` for ``question``.

        Raises ``InvalidQueryError`` for a blank question or ``top_k < 1``; embedding failures
        (``EmbeddingError``) and vector-store failures (``VectorStoreError``) propagate unchanged.
        """
        if not isinstance(question, str) or not question.strip():
            raise InvalidQueryError("question must not be blank")
        if not re.search(r"\w", question):
            raise InvalidQueryError("question must contain letters or digits")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise InvalidQueryError("top_k must be a positive integer")
        s = self.settings
        pool = max(s.top_n, top_k)
        t0 = time.perf_counter()

        embedding = self._vector.embed(question)
        t1 = time.perf_counter()
        vector_hits = self._vector.search(embedding, pool, where)
        bm25_hits = self._bm25.search(question, pool, where)
        t2 = time.perf_counter()

        fused = rrf_fuse(vector_hits, bm25_hits, s.rrf_k)
        top = fused[:top_k]

        # Chunks found only by BM25 have no cosine yet: compute it exactly from stored embeddings.
        missing = [c.chunk.metadata.chunk_id for c in top if c.vector_score is None]
        if missing:
            stored = self._store.get_embeddings(missing)
            for c in top:
                if c.vector_score is None:
                    emb = stored.get(c.chunk.metadata.chunk_id)
                    c.vector_score = cosine(embedding, emb) if emb is not None else 0.0
                    if emb is None:
                        logger.warning("no stored embedding for %s; vector_score set to 0.0",
                                       c.chunk.metadata.chunk_id)

        chunks = [
            RetrievedChunk(chunk=c.chunk, score=c.score, vector_score=c.vector_score or 0.0, rank=i)
            for i, c in enumerate(top, 1)
        ]
        coverage = {c.chunk.metadata.chunk_id: c.coverage for c in top}
        assessment = analyze_evidence(chunks, coverage, s, len(set(tokenize(question))))
        conflicts = detect_conflicts(chunks, assessment.relevant_ids)

        logger.info(
            "retrieve: %d vector + %d bm25 -> %d fused, returned %d, evidence=%s, conflicts=%d (embed %.0f ms, search %.0f ms)",
            len(vector_hits), len(bm25_hits), len(fused), len(chunks), assessment.strength,
            len(conflicts.groups), (t1 - t0) * 1000, (t2 - t1) * 1000)
        return RetrievalResult(
            question=question,
            chunks=chunks,
            conflicts=conflicts,
            evidence_strength=assessment.strength,
            top_score=max((c.vector_score for c in chunks), default=0.0),
            candidates_considered=len(fused),
        )
