from pathlib import Path

import pytest

SETTING_VARS = [
    "AWS_REGION", "AWS_PROFILE", "BEDROCK_EMBED_MODEL_ID", "BEDROCK_EMBED_DIM",
    "BEDROCK_LLM_MODEL_ID", "CHROMA_PATH", "CHROMA_COLLECTION", "RETRIEVAL_TOP_N",
    "MIN_RELEVANCE", "API_BASE_URL", "CHUNK_SIZE", "CHUNK_OVERLAP", "MAX_UPLOAD_SIZE_MB",
    "SUPPORTED_FILE_TYPES", "EMBEDDING_BATCH_SIZE", "RRF_K", "LLM_MAX_TOKENS", "LLM_TEMPERATURE", "MAX_CONTEXT_CHUNKS", "REPORTS_DIR",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Isolate tests from the developer's environment and any local .env file."""
    for var in SETTING_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(Path(__file__).parent)  # no .env in tests/
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---- M1 fixtures -----------------------------------------------------------
from app.core.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(bedrock_embed_dim=256, chroma_path=str(tmp_path / "chroma"),
                    chroma_collection="test_docs", chunk_size=300, chunk_overlap=50,
                    reports_dir=str(tmp_path / "reports"))


@pytest.fixture
def embedder():
    from tests.helpers import FakeEmbedder
    return FakeEmbedder(256)


@pytest.fixture
def store(settings):
    from app.vectorstore.chroma_store import ChromaStore
    s = ChromaStore(settings)
    yield s
    s.close()


@pytest.fixture
def service(settings, embedder, store):
    from app.ingestion.ingestion_service import IngestionService
    return IngestionService(embedder, store, settings)


@pytest.fixture
def kw_embedder():
    from tests.helpers import KeywordEmbedder
    return KeywordEmbedder(256)


@pytest.fixture
def kw_service(settings, kw_embedder, store):
    from app.ingestion.ingestion_service import IngestionService
    return IngestionService(kw_embedder, store, settings)
