"""Deterministic stand-in embedder for OFFLINE dry runs (no AWS). A hashed bag of words: texts sharing
words get similar vectors. It exercises the retrieval machinery (BM25, RRF, evidence, conflicts) but is
NOT Titan, so its numbers say nothing about real embedding quality."""

from __future__ import annotations

import hashlib
import math
import re

from app.retrieval.text import STOPWORDS


class HashedBagOfWordsEmbedder:
    model_id = "offline-hashed-bag-of-words"

    def __init__(self, dimension: int = 256):
        self.dimension = dimension

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dimension
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if tok in STOPWORDS:
                continue
            v[int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big") % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
