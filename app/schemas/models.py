"""Shared Pydantic v2 models.

Implements the contract in docs/api_contracts.md (sections 1, 3 and 7).
Field names and types follow the contract exactly; only validation
constraints and descriptions are added.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

DocType = Literal["pdf", "html", "md", "txt"]
ConflictKind = Literal["contradiction", "version_difference", "date_difference"]
EvidenceStrength = Literal["strong", "weak", "none"]
AnswerStatus = Literal["answered", "conflict", "needs_clarification", "not_found"]
Confidence = Literal["high", "medium", "low"]


def _check_page(v: int) -> int:
    if v != -1 and v < 1:
        raise ValueError("page must be 1-based, or -1 when not applicable")
    return v


class ChunkMetadata(BaseModel):
    chunk_id: str = Field(min_length=1, description='Deterministic: f"{doc_id}:v{version}:{ordinal:05d}"')
    doc_id: str = Field(min_length=1, description="Stable hash of normalized source path (or content).")
    filename: str = Field(min_length=1)
    title: str = Field(description='Document title ("" if unknown).')
    section: str = Field(description='Heading path, e.g. "Billing > Refunds"; "" if none.')
    page: int = Field(description="1-based page; -1 if not applicable (HTML/MD/TXT).")
    version: str = Field(description='"" if unknown; else e.g. "2.1" or "2024-03-01".')
    source: str = Field(description="Path or URL of origin.")
    doc_type: DocType
    ordinal: int = Field(ge=0, description="Chunk order within the document.")

    _page = field_validator("page")(_check_page)


class Chunk(BaseModel):
    text: str = Field(min_length=1)
    metadata: ChunkMetadata


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float = Field(ge=0.0, le=1.0, description="Final score, higher is better, normalized 0..1.")
    vector_score: float = Field(ge=-1.0, le=1.0, description="Cosine similarity.")
    rank: int = Field(ge=1, description="1-based rank after reranking.")


class ConflictGroup(BaseModel):
    topic: str
    chunk_ids: list[str]
    kind: ConflictKind
    description: str


class ConflictReport(BaseModel):
    has_conflict: bool
    groups: list[ConflictGroup] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    question: str
    chunks: list[RetrievedChunk] = Field(description="Ordered best-first.")
    conflicts: ConflictReport
    evidence_strength: EvidenceStrength
    top_score: float
    candidates_considered: int = Field(ge=0)


class Citation(BaseModel):
    chunk_id: str = Field(min_length=1)
    filename: str
    title: str
    section: str
    page: int
    version: str
    excerpt: str = Field(description="Verbatim substring of the chunk.")
    source: str = Field(description="Link or identifier.")

    _page = field_validator("page")(_check_page)


class Answer(BaseModel):
    status: AnswerStatus
    text: str
    citations: list[Citation]
    confidence: Confidence
    conflicts: ConflictReport
    clarifying_question: Optional[str] = None
    warnings: list[str] = Field(default_factory=list, description='e.g. "removed 1 invalid citation"')


class DocumentInfo(BaseModel):
    doc_id: str = Field(min_length=1)
    filename: str
    title: str
    version: str
    doc_type: DocType
    chunk_count: int = Field(ge=0)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1)
    doc_ids: Optional[list[str]] = None
    versions: Optional[list[str]] = None
    debug: bool = False

    @field_validator("question")
    @classmethod
    def _not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class QueryResponse(BaseModel):
    answer: Answer
    retrieved: Optional[list[RetrievedChunk]] = Field(
        None, description="Present only if the request set debug=True."
    )
    latency_ms: dict[str, int] = Field(description="embed, retrieve, generate, total")


class IngestResponse(BaseModel):
    documents: list[DocumentInfo]
    chunks_indexed: int = Field(ge=0)
    failures: list[dict] = Field(default_factory=list, description="{filename, reason}")
