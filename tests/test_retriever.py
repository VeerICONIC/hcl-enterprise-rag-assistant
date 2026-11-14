import pytest

from app.core.bedrock import ThrottlingError
from app.embeddings.base import EmbeddingError
from app.retrieval import InvalidQueryError, RetrievalSettings, Retriever, build_where
from app.retrieval.retriever import cosine
from app.schemas import Chunk, ChunkMetadata, RetrievalResult
from app.vectorstore.chroma_store import VectorStoreError

HANDBOOK = (b"# Leave Policy\n\n## Annual Leave\n\nEmployees receive 20 days of annual leave per year.\n\n"
            b"## Password Reset\n\nTo reset your password open the account settings page and choose reset password.")


@pytest.fixture
def retriever(kw_embedder, store):
    return Retriever(kw_embedder, store, RetrievalSettings(top_n=10))


def doc_id(service, filename):
    return next(d.doc_id for d in service.list_documents() if d.filename == filename)


def test_complete_retrieval_is_contract_compliant(retriever, kw_service):
    kw_service.ingest_bytes("handbook.md", HANDBOOK)
    res = retriever.retrieve("How many days of annual leave do employees get?", top_k=3)
    assert isinstance(res, RetrievalResult)
    assert RetrievalResult.model_validate(res.model_dump()) == res
    assert res.question.startswith("How many days")
    top = res.chunks[0]
    assert "20 days of annual leave" in top.chunk.text and top.chunk.metadata.section == "Annual Leave"
    assert top.rank == 1 and [c.rank for c in res.chunks] == list(range(1, len(res.chunks) + 1))
    assert all(0 <= c.score <= 1 and -1 <= c.vector_score <= 1 for c in res.chunks)
    assert res.top_score == max(c.vector_score for c in res.chunks) and res.candidates_considered >= len(res.chunks)
    assert res.evidence_strength in {"strong", "weak"} and res.conflicts.has_conflict is False
    ids = [c.chunk.metadata.chunk_id for c in res.chunks]
    assert len(ids) == len(set(ids))


def test_top_k_limits_results_but_not_candidates(retriever, kw_service):
    kw_service.ingest_bytes("handbook.md", HANDBOOK)
    res = retriever.retrieve("annual leave", top_k=1)
    assert len(res.chunks) == 1 and res.candidates_considered >= 2


def test_paraphrase_still_finds_the_right_chunk_via_shared_terms(retriever, kw_service):
    kw_service.ingest_bytes("handbook.md", HANDBOOK)
    res = retriever.retrieve("how do I reset my account password", top_k=2)
    assert "reset" in res.chunks[0].chunk.text and res.chunks[0].chunk.metadata.section == "Password Reset"


def test_empty_index_returns_empty_none_result(retriever):
    res = retriever.retrieve("anything at all")
    assert res.chunks == [] and res.evidence_strength == "none"
    assert (res.top_score, res.candidates_considered, res.conflicts.has_conflict) == (0.0, 0, False)


def test_unrelated_question_has_no_evidence(retriever, kw_service):
    kw_service.ingest_bytes("handbook.md", HANDBOOK)
    res = retriever.retrieve("quantum chromodynamics lattice simulation", top_k=3)
    assert res.evidence_strength == "none"


def test_metadata_filtering_restricts_vector_and_bm25(retriever, kw_service):
    kw_service.ingest_bytes("one.txt", b"The router password is stored in the vault.")
    kw_service.ingest_bytes("two.txt", b"The printer password is on the sticker.")
    only_two = retriever.retrieve("password", top_k=5, where=build_where([doc_id(kw_service, "two.txt")]))
    assert [c.chunk.metadata.filename for c in only_two.chunks] == ["two.txt"]
    both = retriever.retrieve("password", top_k=5)
    assert {c.chunk.metadata.filename for c in both.chunks} == {"one.txt", "two.txt"}
    nothing = retriever.retrieve("password", where={"doc_id": "missing"})
    assert nothing.chunks == [] and nothing.evidence_strength == "none"


def test_conflict_report_for_two_versions_of_a_policy(retriever, kw_service):
    kw_service.ingest_bytes("leave_policy_v1.md", b"# Leave Policy\n\n## Annual Leave\n\nEmployees receive 20 days of annual leave per year.", version="1")
    kw_service.ingest_bytes("leave_policy_v2.md", b"# Leave Policy\n\n## Annual Leave\n\nEmployees receive 25 days of annual leave per year.", version="2")
    res = retriever.retrieve("How many days of annual leave do employees receive?", top_k=5)
    assert res.conflicts.has_conflict
    by_kind = {g.kind: g for g in res.conflicts.groups}
    assert set(by_kind) == {"version_difference", "contradiction"}
    assert by_kind["version_difference"].description.startswith("METADATA CONFLICT")
    assert by_kind["contradiction"].description.startswith("POSSIBLE")
    returned = {c.chunk.metadata.chunk_id for c in res.chunks}
    for g in res.conflicts.groups:
        assert len(g.chunk_ids) == 2 and set(g.chunk_ids) <= returned
    texts = " ".join(c.chunk.text for c in res.chunks)
    assert "20 days" in texts and "25 days" in texts  # both sides are preserved, neither dropped


def test_no_conflict_for_a_single_consistent_document(retriever, kw_service):
    kw_service.ingest_bytes("handbook.md", HANDBOOK)
    assert not retriever.retrieve("annual leave days", top_k=5).conflicts.has_conflict


def test_embedding_failure_propagates_as_bedrock_error(retriever, kw_embedder, kw_service):
    kw_service.ingest_bytes("handbook.md", HANDBOOK)
    kw_embedder.fail_with = EmbeddingError("model access not granted")
    with pytest.raises(EmbeddingError, match="model access"):
        retriever.retrieve("annual leave")
    kw_embedder.fail_with = ThrottlingError("slow down")
    with pytest.raises(ThrottlingError):
        retriever.retrieve("annual leave")


def test_chroma_failure_propagates(retriever, store, kw_service, monkeypatch):
    kw_service.ingest_bytes("handbook.md", HANDBOOK)

    def boom(*a, **k):
        raise VectorStoreError("database is locked")

    monkeypatch.setattr(store, "query", boom)
    with pytest.raises(VectorStoreError, match="locked"):
        retriever.retrieve("annual leave")


@pytest.mark.parametrize("q", ["", "   ", None, 5])
def test_invalid_question(retriever, q):
    with pytest.raises(InvalidQueryError):
        retriever.retrieve(q)


@pytest.mark.parametrize("k", [0, -1, True, 2.5])
def test_invalid_top_k(retriever, k):
    with pytest.raises(InvalidQueryError):
        retriever.retrieve("hello", top_k=k)


@pytest.mark.parametrize("q", ["?!?", "...", "--- ,,"])
def test_punctuation_only_question_is_invalid(retriever, kw_service, q):
    kw_service.ingest_bytes("handbook.md", HANDBOOK)
    with pytest.raises(InvalidQueryError, match="letters or digits"):
        retriever.retrieve(q)


# ---- BM25-only candidates get an exact cosine ---------------------------------------
class StubStore:
    def __init__(self, vector_chunks, all_chunks, embeddings):
        self.vector_chunks, self.all_chunks, self.embeddings = vector_chunks, all_chunks, embeddings

    def query(self, embedding, top_k, where=None):
        return [(c, 0.9) for c in self.vector_chunks][:top_k]

    def get_all_chunks(self, where=None):
        return self.all_chunks

    def get_embeddings(self, ids):
        return {i: self.embeddings[i] for i in ids if i in self.embeddings}


class FixedEmbedder:
    dimension, model_id = 2, "m"

    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def _chunk(cid, text):
    return Chunk(text=text, metadata=ChunkMetadata(chunk_id=cid, doc_id=cid.split(":")[0], filename="f.md", title="",
                                                   section="", page=-1, version="", source="s", doc_type="md", ordinal=0))


def test_bm25_only_hit_gets_cosine_from_stored_embedding():
    a, b = _chunk("a:0", "vector favourite text"), _chunk("b:0", "contains the rare keyword zebra")
    store = StubStore([a], [a, b], {"b:0": [0.6, 0.8]})
    res = Retriever(FixedEmbedder(), store, RetrievalSettings(top_n=5)).retrieve("zebra", top_k=5)
    by_id = {c.chunk.metadata.chunk_id: c for c in res.chunks}
    assert by_id["a:0"].vector_score == pytest.approx(0.9)       # from the vector search, unchanged
    assert by_id["b:0"].vector_score == pytest.approx(0.6)       # computed: cos([1,0],[.6,.8])
    assert 0 < by_id["b:0"].score <= 1


def test_bm25_only_hit_without_stored_embedding_falls_back_to_zero(caplog):
    a, b = _chunk("a:0", "vector favourite text"), _chunk("b:0", "contains the rare keyword zebra")
    store = StubStore([a], [a, b], {})
    with caplog.at_level("WARNING"):
        res = Retriever(FixedEmbedder(), store, RetrievalSettings()).retrieve("zebra")
    assert {c.chunk.metadata.chunk_id: c.vector_score for c in res.chunks}["b:0"] == 0.0
    assert "b:0" in caplog.text


# ---- helpers ---------------------------------------------------------------------------
def test_build_where():
    assert build_where() is None and build_where([], []) is None and build_where(None, []) is None
    assert build_where(["a"]) == {"doc_id": {"$in": ["a"]}}
    assert build_where(None, ["1", ""]) == {"version": {"$in": ["1", ""]}}
    assert build_where(["a"], ["1"]) == {"$and": [{"doc_id": {"$in": ["a"]}}, {"version": {"$in": ["1"]}}]}


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0) and cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0) and cosine([0, 0], [1, 0]) == 0.0


def test_retrieval_settings_from_app_settings(settings):
    rs = RetrievalSettings.from_settings(settings)
    assert (rs.top_n, rs.min_relevance, rs.rrf_k) == (settings.retrieval_top_n, settings.min_relevance, settings.rrf_k)
