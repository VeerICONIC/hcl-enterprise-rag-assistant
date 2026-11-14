"""Retrieval-internal types. Public result types stay in ``app.schemas`` (the contract)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

from pydantic import BaseModel, Field

from app.core.config import Settings
from app.schemas import Chunk


class InvalidQueryError(ValueError):
    """The question or retrieval parameters are invalid."""


class RetrievalSettings(BaseModel):
    """Tunable retrieval parameters. ``from_settings`` maps the application Settings."""

    top_n: int = Field(20, ge=1, description="Candidates taken from each retriever before fusion.")
    min_relevance: float = Field(0.30, ge=0.0, le=1.0, description="Cosine similarity needed to count as relevant.")
    rrf_k: int = Field(60, ge=1)
    bm25_k1: float = Field(1.5, gt=0)
    bm25_b: float = Field(0.75, ge=0, le=1)
    # evidence heuristics (see evidence_analyzer.py)
    strong_margin: float = Field(0.15, ge=0, le=1)
    very_strong_margin: float = Field(0.30, ge=0, le=1)
    lexical_rescue_coverage: float = Field(0.80, ge=0, le=1)

    @classmethod
    def from_settings(cls, s: Settings) -> "RetrievalSettings":
        return cls(top_n=s.retrieval_top_n, min_relevance=s.min_relevance, rrf_k=s.rrf_k)


class RetrievalStore(Protocol):
    """What retrieval needs from a vector store (satisfied by ``ChromaStore``)."""

    def query(self, embedding: list[float], top_k: int,
              where: Optional[dict] = None) -> list[tuple[Chunk, float]]: ...

    def get_all_chunks(self, where: Optional[dict] = None) -> list[Chunk]: ...

    def get_embeddings(self, ids: list[str]) -> dict[str, list[float]]: ...


@dataclass(frozen=True)
class VectorHit:
    chunk: Chunk
    similarity: float   # cosine similarity
    rank: int           # 1-based


@dataclass(frozen=True)
class Bm25Hit:
    chunk: Chunk
    score: float        # raw BM25 score (only meaningful within one query)
    rank: int           # 1-based
    coverage: float     # fraction of distinct query terms present in the chunk


@dataclass
class FusedCandidate:
    chunk: Chunk
    rrf: float                       # raw reciprocal-rank-fusion sum
    score: float                     # rrf normalized to 0..1
    vector_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    vector_score: Optional[float] = None
    coverage: float = 0.0


@dataclass
class EvidenceAssessment:
    strength: Literal["strong", "weak", "none"]
    relevant_ids: set[str] = field(default_factory=set)
    relevant_count: int = 0
    distinct_documents: int = 0
    best_similarity: float = 0.0
    concentrated: bool = False       # all relevant evidence comes from a single document
    reasons: list[str] = field(default_factory=list)
