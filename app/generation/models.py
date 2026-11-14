"""Generation-internal types. The public result is the contract's ``Answer`` (app.schemas)."""

from __future__ import annotations

from typing import Literal, Optional, Protocol

from pydantic import BaseModel, Field

from app.core.config import Settings


class GenerationError(RuntimeError):
    """The model returned nothing usable (empty, filtered or malformed). Message is user-safe."""


class LLMClient(Protocol):
    """api_contracts.md §6. ``messages`` are ``{"role": "user"|"assistant", "content": str}``."""

    def converse(self, system: str, messages: list[dict], *, max_tokens: int, temperature: float) -> str: ...


class GenerationSettings(BaseModel):
    max_tokens: int = Field(800, ge=64)
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    max_context_chunks: int = Field(6, ge=1)
    max_chunk_chars: int = Field(1800, ge=200, description="Evidence text per chunk placed in the prompt.")
    context_min_score: float = Field(0.30, ge=0.0, le=1.0, description="Cosine below which a chunk is left out of the prompt.")
    excerpt_chars: int = Field(300, ge=40, description="Maximum length of a citation excerpt.")

    @classmethod
    def from_settings(cls, s: Settings) -> "GenerationSettings":
        return cls(max_tokens=s.llm_max_tokens, temperature=s.llm_temperature,
                   max_context_chunks=s.max_context_chunks, context_min_score=s.min_relevance)


ModelStatus = Literal["answered", "insufficient_evidence", "needs_clarification", "conflict"]


class ModelClaim(BaseModel):
    text: str
    evidence: list[str] = Field(default_factory=list, description='Evidence aliases such as "E1".')


class ModelOutput(BaseModel):
    """What we ask Nova to return (as JSON). Aliases, not chunk IDs, so it cannot garble them."""

    status: ModelStatus
    answer: str = ""
    claims: list[ModelClaim] = Field(default_factory=list)
    clarifying_question: Optional[str] = None
