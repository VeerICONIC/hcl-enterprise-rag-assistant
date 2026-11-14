import pytest
from pydantic import ValidationError

from app.core import schemas as core_schemas
from app.schemas import (
    Answer, Chunk, ChunkMetadata, Citation, ConflictGroup, ConflictReport, DocumentInfo,
    IngestResponse, QueryRequest, QueryResponse, RetrievalResult, RetrievedChunk,
)


def meta(**kw):
    base = dict(chunk_id="d1:v2.1:00000", doc_id="d1", filename="guide.pdf", title="Guide",
                section="Billing > Refunds", page=3, version="2.1", source="data/raw/guide.pdf",
                doc_type="pdf", ordinal=0)
    base.update(kw)
    return ChunkMetadata(**base)


def chunk(**kw):
    return Chunk(text="Refunds take 5 days.", metadata=meta(**kw))


def conflicts(has=False):
    return ConflictReport(has_conflict=has)


def answer(**kw):
    base = dict(status="answered", text="x", citations=[], confidence="high", conflicts=conflicts())
    base.update(kw)
    return Answer(**base)


def test_core_reexport_is_same_classes():
    assert core_schemas.Chunk is Chunk and core_schemas.QueryRequest is QueryRequest


def test_metadata_valid_and_sentinels():
    m = meta(page=-1, section="", title="", version="", doc_type="md")
    assert m.page == -1 and m.version == ""


@pytest.mark.parametrize("kw", [
    dict(page=0), dict(page=-2), dict(doc_type="docx"), dict(ordinal=-1),
    dict(chunk_id=""), dict(doc_id=""), dict(filename=""),
])
def test_metadata_rejects_invalid(kw):
    with pytest.raises(ValidationError):
        meta(**kw)


def test_metadata_requires_all_fields():
    with pytest.raises(ValidationError):
        ChunkMetadata(chunk_id="a", doc_id="a", filename="a")


def test_chunk_requires_text():
    with pytest.raises(ValidationError):
        Chunk(text="", metadata=meta())


def test_retrieved_chunk_bounds():
    RetrievedChunk(chunk=chunk(), score=0.9, vector_score=0.8, rank=1)
    RetrievedChunk(chunk=chunk(), score=0.0, vector_score=-0.1, rank=1)
    for kw in (dict(score=1.1), dict(score=-0.1), dict(vector_score=1.5), dict(rank=0)):
        base = dict(chunk=chunk(), score=0.5, vector_score=0.5, rank=1)
        base.update(kw)
        with pytest.raises(ValidationError):
            RetrievedChunk(**base)


def test_conflict_models():
    assert ConflictReport(has_conflict=False).groups == []
    g = ConflictGroup(topic="limit", chunk_ids=["a", "b"], kind="version_difference", description="d")
    assert ConflictReport(has_conflict=True, groups=[g]).groups[0].kind == "version_difference"
    with pytest.raises(ValidationError):
        ConflictGroup(topic="t", chunk_ids=[], kind="other", description="d")


def test_conflict_default_lists_not_shared():
    a, b = ConflictReport(has_conflict=False), ConflictReport(has_conflict=False)
    a.groups.append(ConflictGroup(topic="t", chunk_ids=[], kind="contradiction", description="d"))
    assert b.groups == []


def test_retrieval_result():
    r = RetrievalResult(question="q", chunks=[], conflicts=conflicts(), evidence_strength="none",
                        top_score=0.0, candidates_considered=0)
    assert r.evidence_strength == "none"
    with pytest.raises(ValidationError):
        RetrievalResult(question="q", chunks=[], conflicts=conflicts(), evidence_strength="medium",
                        top_score=0.0, candidates_considered=0)


def test_citation_and_answer():
    c = Citation(chunk_id="d1:v2.1:00000", filename="guide.pdf", title="Guide", section="s",
                 page=3, version="2.1", excerpt="Refunds take 5 days.", source="guide.pdf#s/p.3")
    a = answer(citations=[c])
    assert a.warnings == [] and a.clarifying_question is None
    for kw in (dict(status="maybe"), dict(confidence="certain")):
        with pytest.raises(ValidationError):
            answer(**kw)


def test_document_info():
    d = DocumentInfo(doc_id="d1", filename="a.md", title="", version="", doc_type="md", chunk_count=4)
    assert d.chunk_count == 4
    with pytest.raises(ValidationError):
        DocumentInfo(doc_id="d1", filename="a", title="", version="", doc_type="md", chunk_count=-1)


def test_query_request_defaults_and_validation():
    q = QueryRequest(question="How?")
    assert (q.top_k, q.doc_ids, q.versions, q.debug) == (5, None, None, False)
    for bad in ("", "   ", "x" * 2001):
        with pytest.raises(ValidationError):
            QueryRequest(question=bad)
    with pytest.raises(ValidationError):
        QueryRequest(question="q", top_k=0)


def test_query_response_and_ingest_response():
    qr = QueryResponse(answer=answer(), latency_ms={"embed": 1, "total": 2})
    assert qr.retrieved is None
    info = DocumentInfo(doc_id="d1", filename="a.md", title="", version="", doc_type="md", chunk_count=1)
    ir = IngestResponse(documents=[info], chunks_indexed=1)
    assert ir.failures == []
    with pytest.raises(ValidationError):
        IngestResponse(documents=[], chunks_indexed=-1)


def test_roundtrip_json():
    rc = RetrievedChunk(chunk=chunk(), score=0.7, vector_score=0.6, rank=2)
    assert RetrievedChunk.model_validate_json(rc.model_dump_json()) == rc
