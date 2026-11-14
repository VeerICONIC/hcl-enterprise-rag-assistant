"""Application configuration (pydantic-settings).

Values come from environment variables and an optional ``.env`` file.
AWS credentials are intentionally NOT settings: boto3's default credential
chain (SSO profile, environment, shared credentials file, IAM role) resolves
them. Only the region and an optional profile *name* are configured here.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Output dimensions supported by Titan Text Embeddings V2.
TITAN_V2_DIMENSIONS = (256, 512, 1024)

# File extensions the ingestion pipeline has parsers for.
KNOWN_EXTENSIONS = ("pdf", "html", "htm", "md", "markdown", "txt")


class ConfigurationError(RuntimeError):
    """Raised when settings are missing or invalid. Message is user-facing."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # unrelated .env entries (e.g. AWS_* keys) are never loaded
    )

    # --- AWS / Bedrock -----------------------------------------------------
    aws_region: str = Field(
        "us-east-1",
        description="AWS region for Bedrock Runtime. Verify model availability there.",
    )
    aws_profile: Optional[str] = Field(
        None, description="Optional named AWS profile (e.g. an SSO profile)."
    )
    bedrock_embed_model_id: str = Field(
        "amazon.titan-embed-text-v2:0", description="Titan embeddings model ID."
    )
    bedrock_embed_dim: int = Field(
        1024,
        description="Embedding dimension. Fixed once documents are indexed; "
        "changing it requires re-indexing.",
    )
    bedrock_llm_model_id: str = Field(
        "amazon.nova-lite-v1:0",
        description="Generation model ID (may need an inference profile such as "
        "'us.amazon.nova-lite-v1:0' in some regions/accounts).",
    )

    # --- Vector store ------------------------------------------------------
    chroma_path: str = Field("data/chroma", description="ChromaDB persistence directory.")
    chroma_collection: str = Field("enterprise_docs", description="Chroma collection name.")

    # --- Ingestion (M1) ----------------------------------------------------
    chunk_size: int = Field(
        1000, ge=100, description="Target chunk body size in characters (heading prefix excluded)."
    )
    chunk_overlap: int = Field(
        150, ge=0, description="Overlap between consecutive chunks of one section, in characters."
    )
    max_upload_size_mb: int = Field(25, ge=1, description="Maximum size of one uploaded file.")
    supported_file_types: str = Field(
        "pdf,html,htm,md,markdown,txt",
        description="Comma-separated file extensions accepted for ingestion.",
    )
    embedding_batch_size: int = Field(
        8, ge=1, le=64,
        description="Concurrent Titan requests per batch (Titan has no native batch API).",
    )

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_n: int = Field(
        20, ge=1, description="Candidates fetched from the vector store before reranking."
    )
    min_relevance: float = Field(
        0.30,
        ge=0.0,
        le=1.0,
        description="Minimum relevance for evidence to count as support. "
        "Initial guess; calibrate on the evaluation set.",
    )

    rrf_k: int = Field(
        60, ge=1, description="Reciprocal Rank Fusion constant k (score = sum 1/(k + rank))."
    )

    # --- Generation (M3) ---------------------------------------------------
    llm_max_tokens: int = Field(800, ge=64, le=4096, description="Max tokens Nova may generate per answer.")
    llm_temperature: float = Field(0.0, ge=0.0, le=1.0, description="Sampling temperature (0 = most deterministic).")
    max_context_chunks: int = Field(6, ge=1, le=20, description="Evidence chunks placed in the prompt.")

    reports_dir: str = Field("reports", description="Directory holding evaluation output (eval_results.json).")

    # --- Hosting -----------------------------------------------------------
    api_key: Optional[str] = Field(
        None, repr=False,
        description="Optional shared secret. When set, every route except /health requires the header X-API-Key.",
    )

    # --- Frontend ----------------------------------------------------------
    api_base_url: str = Field(
        "http://localhost:8000", description="Backend base URL used by the Streamlit frontend."
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def _blank_api_key_is_none(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v.strip() if isinstance(v, str) else v

    @field_validator("aws_profile", mode="before")
    @classmethod
    def _blank_profile_is_none(cls, v):
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("aws_region", "bedrock_embed_model_id", "bedrock_llm_model_id",
                     "chroma_path", "chroma_collection", "api_base_url")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("chroma_collection")
    @classmethod
    def _valid_collection_name(cls, v: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]", v):
            raise ValueError(
                "must be 3-512 characters from [a-zA-Z0-9._-], starting and ending with "
                "a letter or digit (ChromaDB rule)"
            )
        return v

    @field_validator("supported_file_types")
    @classmethod
    def _valid_file_types(cls, v: str) -> str:
        exts = [e.strip().lstrip(".").lower() for e in v.split(",") if e.strip()]
        if not exts:
            raise ValueError("must list at least one extension")
        unknown = sorted(set(exts) - set(KNOWN_EXTENSIONS))
        if unknown:
            raise ValueError(f"unsupported extension(s) {unknown}; parsers exist for {KNOWN_EXTENSIONS}")
        return ",".join(dict.fromkeys(exts))

    @field_validator("bedrock_embed_dim")
    @classmethod
    def _positive_dim(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @model_validator(mode="after")
    def _check_titan_dimension(self) -> "Settings":
        if ("titan-embed-text-v2" in self.bedrock_embed_model_id
                and self.bedrock_embed_dim not in TITAN_V2_DIMENSIONS):
            raise ValueError(
                f"BEDROCK_EMBED_DIM={self.bedrock_embed_dim} is not supported by Titan "
                f"Text Embeddings V2; use one of {TITAN_V2_DIMENSIONS}"
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(self.supported_file_types.split(","))

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["Invalid configuration (check environment variables / .env):"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "settings"
        lines.append(f"  - {loc.upper()}: {err['msg']}")
    return "\n".join(lines)


def load_settings(**overrides) -> Settings:
    """Build Settings, converting validation failures into ConfigurationError."""
    try:
        return Settings(**overrides)
    except ValidationError as exc:
        raise ConfigurationError(_format_validation_error(exc)) from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached process-wide settings. Call ``get_settings.cache_clear()`` in tests."""
    return load_settings()
