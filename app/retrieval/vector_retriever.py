"""Semantic retrieval: embed the question (Titan) and query ChromaDB."""

from __future__ import annotations

from typing import Optional

from app.embeddings.base import Embedder
from app.retrieval.models import RetrievalStore, VectorHit


class VectorRetriever:
    def __init__(self, embedder: Embedder, store: RetrievalStore):
        self._embedder = embedder
        self._store = store

    def embed(self, question: str) -> list[float]:
        """Embed with the same model and dimension used for the indexed documents."""
        return self._embedder.embed_query(question)

    def search(self, embedding: list[float], n: int, where: Optional[dict] = None) -> list[VectorHit]:
        """Top-n chunks by cosine similarity, best first, with 1-based ranks."""
        results = self._store.query(embedding, n, where)
        return [VectorHit(chunk, sim, rank) for rank, (chunk, sim) in enumerate(results, 1)]
