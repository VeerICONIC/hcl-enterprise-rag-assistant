"""Embedding interface (api_contracts.md §2). Retrieval depends only on this protocol."""

from __future__ import annotations

from typing import Protocol

from app.core.bedrock import BedrockError


class EmbeddingError(BedrockError):
    """Embedding failed. ``__cause__`` holds the underlying Bedrock error, if any."""


class Embedder(Protocol):
    dimension: int
    model_id: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
