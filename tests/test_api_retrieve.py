import pytest
from fastapi.testclient import TestClient

from app.api.container import Container
from app.api.main import create_app
from app.embeddings.base import EmbeddingError
from app.schemas import RetrievalResult
from app.vectorstore.chroma_store import VectorStoreError
from tests.test_retriever import HANDBOOK


@pytest.fixture
def client(settings, store, kw_embedder):
    app = create_app(Container(settings, store=store, embedder=kw_embedder))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def ingest(client, name, data, **form):
    r = client.post("/ingest", files=[("files", (name, data, "text/plain"))], data=form)
    assert r.status_code == 200, r.text
    return r.json()["documents"][0]


def test_successful_retrieval(client):
    ingest(client, "handbook.md", HANDBOOK)
    r = client.post("/retrieve", json={"question": "How many days of annual leave do employees get?", "top_k": 2})
    assert r.status_code == 200
    body = RetrievalResult.model_validate(r.json())
    assert len(body.chunks) == 2 and "20 days" in body.chunks[0].chunk.text
    assert body.chunks[0].rank == 1 and body.evidence_strength in {"strong", "weak"}
    assert set(body.chunks[0].chunk.metadata.model_dump()) >= {"chunk_id", "doc_id", "filename", "section", "page", "version"}
    assert "answer" not in r.json()  # M2 returns evidence only


def test_empty_index_gives_empty_result(client):
    r = client.post("/retrieve", json={"question": "anything"})
    assert r.status_code == 200
    assert r.json()["chunks"] == [] and r.json()["evidence_strength"] == "none"
    assert r.json()["conflicts"] == {"has_conflict": False, "groups": []}


def test_doc_ids_filter(client):
    one = ingest(client, "one.txt", b"The router password is stored in the vault.")
    ingest(client, "two.txt", b"The printer password is on the sticker.")
    r = client.post("/retrieve", json={"question": "password", "top_k": 5, "doc_ids": [one["doc_id"]]})
    assert [c["chunk"]["metadata"]["filename"] for c in r.json()["chunks"]] == ["one.txt"]


def test_versions_filter_and_conflicts(client):
    ingest(client, "leave_policy_v1.md", b"# Leave Policy\n\n## Annual Leave\n\nEmployees receive 20 days of annual leave per year.", version="1")
    ingest(client, "leave_policy_v2.md", b"# Leave Policy\n\n## Annual Leave\n\nEmployees receive 25 days of annual leave per year.", version="2")
    both = client.post("/retrieve", json={"question": "annual leave days", "top_k": 5}).json()
    assert both["conflicts"]["has_conflict"] and {g["kind"] for g in both["conflicts"]["groups"]} >= {"version_difference"}
    v2 = client.post("/retrieve", json={"question": "annual leave days", "versions": ["2"]}).json()
    assert [c["chunk"]["metadata"]["version"] for c in v2["chunks"]] == ["2"] and not v2["conflicts"]["has_conflict"]


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": "q", "top_k": 0},
                                     {"question": "x" * 2001}])
def test_invalid_requests_are_422_with_uniform_body(client, payload):
    r = client.post("/retrieve", json=payload)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_punctuation_only_question_is_400(client):
    r = client.post("/retrieve", json={"question": "?!?"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_query"


def test_embedding_failure_is_502_without_stack_trace(client, kw_embedder):
    kw_embedder.fail_with = EmbeddingError("model access not granted")
    r = client.post("/retrieve", json={"question": "hello"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "bedrock_error"
    assert "model access" in r.json()["error"]["message"] and "Traceback" not in r.text


def test_vector_store_failure_is_500_clean(client, store, monkeypatch):
    def boom(*a, **k):
        raise VectorStoreError("disk i/o error")

    monkeypatch.setattr(store, "query", boom)
    r = client.post("/retrieve", json={"question": "hello"})
    assert r.status_code == 500 and r.json()["error"]["code"] == "vector_store_error" and "Traceback" not in r.text


def test_unexpected_error_is_generic_500(client, kw_embedder):
    kw_embedder.fail_with = RuntimeError("secret internal detail")
    r = client.post("/retrieve", json={"question": "hello"})
    assert r.status_code == 500 and "secret" not in r.text


def test_existing_endpoints_still_work(client):
    doc = ingest(client, "n.txt", b"Some note text.")
    assert client.get("/health").json()["chroma_count"] == doc["chunk_count"]
    assert [d["doc_id"] for d in client.get("/documents").json()] == [doc["doc_id"]]
    assert client.delete(f"/documents/{doc['doc_id']}").json() == {"deleted_chunks": doc["chunk_count"]}
    assert client.post("/ingest", files=[("files", ("x.docx", b"x"))]).status_code == 400


def test_retrieve_builds_no_aws_client_until_used(monkeypatch):
    import boto3
    monkeypatch.setattr(boto3, "Session", lambda *a, **k: (_ for _ in ()).throw(AssertionError("AWS touched")))
    c = Container()
    create_app(c)
    assert c._retriever is None and c._store is None
