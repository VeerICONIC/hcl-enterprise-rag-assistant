"""Lexical retrieval with Okapi BM25 over indexed chunk text.

Own implementation (no dependency): IDF is ``ln(1 + (N - n + 0.5) / (n + 0.5))``, which is
never negative, unlike some library variants that misbehave on small corpora.

Caching: the index is built from ``store.get_all_chunks(where)`` and cached per metadata filter,
keyed by the store's ``revision`` counter, which every upsert/delete through the store bumps. So
additions and deletions invalidate it automatically and a query on an unchanged collection does
not rebuild anything. A store without a ``revision`` attribute is never cached (always rebuilt).
Limitation: the counter is per process; writes made by another process to the same Chroma
directory are not noticed until this process writes or restarts.
"""

from __future__ import annotations

import json
import math
import threading
from collections import Counter
from typing import Optional, Sequence

from app.retrieval.models import Bm25Hit, RetrievalStore
from app.retrieval.text import tokenize
from app.schemas import Chunk


class Bm25Index:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self._chunks = list(chunks)
        self._k1, self._b = k1, b
        self._tf = [Counter(tokenize(c.text)) for c in self._chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self._chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query_terms: Sequence[str], n: int) -> list[Bm25Hit]:
        terms = list(dict.fromkeys(query_terms))
        if not terms or not self._chunks:
            return []
        scored: list[tuple[float, float, Chunk]] = []
        for chunk, tf, dl in zip(self._chunks, self._tf, self._len):
            score = 0.0
            present = 0
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                present += 1
                norm = 1 - self._b + self._b * (dl / self._avg if self._avg else 1.0)
                score += self._idf[t] * f * (self._k1 + 1) / (f + self._k1 * norm)
            if score > 0:
                scored.append((score, present / len(terms), chunk))
        scored.sort(key=lambda r: (-r[0], r[2].metadata.chunk_id))
        return [Bm25Hit(c, s, rank, cov) for rank, (s, cov, c) in enumerate(scored[:n], 1)]


_MAX_CACHED_FILTERS = 8


class Bm25Retriever:
    def __init__(self, store: RetrievalStore, k1: float = 1.5, b: float = 0.75):
        self._store, self._k1, self._b = store, k1, b
        self._cache: dict[str, tuple[int, Bm25Index]] = {}
        self._lock = threading.Lock()
        self.builds = 0  # number of index builds (observable in tests)

    def _index(self, where: Optional[dict]) -> Bm25Index:
        revision = getattr(self._store, "revision", None)
        key = json.dumps(where, sort_keys=True, default=str) if where else ""
        with self._lock:
            hit = self._cache.get(key)
            if revision is not None and hit is not None and hit[0] == revision:
                return hit[1]
            index = Bm25Index(self._store.get_all_chunks(where), self._k1, self._b)
            self.builds += 1
            if revision is not None:
                if len(self._cache) >= _MAX_CACHED_FILTERS:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = (revision, index)
            return index

    def search(self, query: str, n: int, where: Optional[dict] = None) -> list[Bm25Hit]:
        """Top-n lexical matches. Empty/punctuation-only queries and empty corpora give []."""
        terms = tokenize(query)
        if not terms:
            return []
        return self._index(where).search(terms, n)
