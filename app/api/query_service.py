"""Query orchestration (business logic kept out of the route handler):
question -> hybrid retrieval (embedding, vector, BM25, RRF, evidence, conflicts) -> grounded generation."""

from __future__ import annotations

import time

from app.generation import AnswerGenerator
from app.retrieval import Retriever, build_where
from app.schemas import QueryRequest, QueryResponse


class QueryService:
    def __init__(self, retriever: Retriever, generator: AnswerGenerator):
        self._retriever = retriever
        self._generator = generator

    def query(self, req: QueryRequest) -> QueryResponse:
        """Raises ``InvalidQueryError``, ``BedrockError``, ``VectorStoreError`` or ``GenerationError``."""
        t0 = time.perf_counter()
        retrieval = self._retriever.retrieve(req.question, top_k=req.top_k,
                                             where=build_where(req.doc_ids, req.versions))
        t1 = time.perf_counter()
        answer = self._generator.generate(retrieval)
        t2 = time.perf_counter()
        return QueryResponse(
            answer=answer,
            retrieved=retrieval.chunks if req.debug else None,
            # the question embedding is included in "retrieve"
            latency_ms={"retrieve": round((t1 - t0) * 1000), "generate": round((t2 - t1) * 1000),
                        "total": round((t2 - t0) * 1000)},
        )
