import pytest

from app.core.config import ConfigurationError, Settings, get_settings, load_settings


def test_defaults():
    s = load_settings()
    assert s.aws_region == "us-east-1"
    assert s.aws_profile is None
    assert s.bedrock_embed_model_id == "amazon.titan-embed-text-v2:0"
    assert s.bedrock_embed_dim == 1024
    assert s.bedrock_llm_model_id == "amazon.nova-lite-v1:0"
    assert s.chroma_path == "data/chroma"
    assert s.retrieval_top_n == 20
    assert 0 <= s.min_relevance <= 1
    assert s.api_base_url == "http://localhost:8000"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_PROFILE", "hack")
    monkeypatch.setenv("BEDROCK_EMBED_DIM", "512")
    monkeypatch.setenv("BEDROCK_LLM_MODEL_ID", "us.amazon.nova-lite-v1:0")
    monkeypatch.setenv("RETRIEVAL_TOP_N", "7")
    monkeypatch.setenv("MIN_RELEVANCE", "0.5")
    s = load_settings()
    assert (s.aws_region, s.aws_profile, s.bedrock_embed_dim) == ("eu-west-1", "hack", 512)
    assert s.bedrock_llm_model_id == "us.amazon.nova-lite-v1:0"
    assert s.retrieval_top_n == 7 and s.min_relevance == 0.5


def test_blank_profile_is_none(monkeypatch):
    monkeypatch.setenv("AWS_PROFILE", "  ")
    assert load_settings().aws_profile is None


def test_reads_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("AWS_REGION=ap-south-1\nAWS_SECRET_ACCESS_KEY=should-be-ignored\n")
    monkeypatch.chdir(tmp_path)
    s = load_settings()
    assert s.aws_region == "ap-south-1"
    assert not hasattr(s, "aws_secret_access_key")


@pytest.mark.parametrize("var,value,needle", [
    ("BEDROCK_EMBED_DIM", "999", "BEDROCK_EMBED_DIM"),
    ("BEDROCK_EMBED_DIM", "abc", "BEDROCK_EMBED_DIM"),
    ("BEDROCK_EMBED_DIM", "-5", "BEDROCK_EMBED_DIM"),
    ("RETRIEVAL_TOP_N", "0", "RETRIEVAL_TOP_N"),
    ("MIN_RELEVANCE", "1.5", "MIN_RELEVANCE"),
    ("AWS_REGION", "  ", "AWS_REGION"),
])
def test_invalid_values_give_useful_error(monkeypatch, var, value, needle):
    monkeypatch.setenv(var, value)
    with pytest.raises(ConfigurationError) as ei:
        load_settings()
    assert needle in str(ei.value)


def test_non_titan_model_allows_other_dims(monkeypatch):
    monkeypatch.setenv("BEDROCK_EMBED_MODEL_ID", "other.embed-model")
    monkeypatch.setenv("BEDROCK_EMBED_DIM", "768")
    assert load_settings().bedrock_embed_dim == 768


def test_get_settings_cached(monkeypatch):
    assert get_settings() is get_settings()
    assert isinstance(get_settings(), Settings)


# ---- M1 ingestion settings --------------------------------------------------
def test_ingestion_defaults():
    s = load_settings()
    assert (s.chunk_size, s.chunk_overlap, s.max_upload_size_mb, s.embedding_batch_size) == (1000, 150, 25, 8)
    assert s.supported_extensions == frozenset({"pdf", "html", "htm", "md", "markdown", "txt"})
    assert s.max_upload_bytes == 25 * 1024 * 1024


def test_supported_types_parsing(monkeypatch):
    monkeypatch.setenv("SUPPORTED_FILE_TYPES", " .PDF, txt ,txt")
    assert load_settings().supported_extensions == frozenset({"pdf", "txt"})


@pytest.mark.parametrize("var,value", [
    ("CHUNK_SIZE", "50"), ("CHUNK_OVERLAP", "-1"), ("MAX_UPLOAD_SIZE_MB", "0"),
    ("EMBEDDING_BATCH_SIZE", "0"), ("EMBEDDING_BATCH_SIZE", "500"),
    ("SUPPORTED_FILE_TYPES", "pdf,docx"), ("SUPPORTED_FILE_TYPES", " , "),
    ("CHROMA_COLLECTION", "ab"), ("CHROMA_COLLECTION", "bad name!"),
])
def test_invalid_ingestion_settings(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ConfigurationError) as ei:
        load_settings()
    assert var in str(ei.value)


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "200")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")
    with pytest.raises(ConfigurationError, match="CHUNK_OVERLAP"):
        load_settings()


def test_rrf_k_setting(monkeypatch):
    assert load_settings().rrf_k == 60
    monkeypatch.setenv("RRF_K", "10")
    assert load_settings().rrf_k == 10
    monkeypatch.setenv("RRF_K", "0")
    with pytest.raises(ConfigurationError, match="RRF_K"):
        load_settings()
