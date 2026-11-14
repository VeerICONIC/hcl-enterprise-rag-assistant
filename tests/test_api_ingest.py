import pytest
from fastapi.testclient import TestClient

from app.api.container import Container
from app.api.main import create_app
from app.core.bedrock import CredentialsError
from app.embeddings.base import EmbeddingError
from tests.helpers import SAMPLE_MD, make_pdf


@pytest.fixture
def client(settings, store, embedder):
    app = create_app(Container(settings, store=store, embedder=embedder))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def upload(client, files, **data):
    return client.post("/ingest", files=[("files", f) for f in files], data=data)


def test_health_needs_no_aws(client, settings):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "chroma_count": 0, "embed_model": settings.bedrock_embed_model_id,
                        "llm_model": settings.bedrock_llm_model_id}


def test_successful_upload_and_listing(client):
    r = upload(client, [("leave.md", SAMPLE_MD, "text/markdown"), ("n.txt", b"Some note text.", "text/plain")])
    assert r.status_code == 200
    body = r.json()
    assert body["failures"] == [] and len(body["documents"]) == 2
    assert body["chunks_indexed"] == sum(d["chunk_count"] for d in body["documents"])
    doc = next(d for d in body["documents"] if d["filename"] == "leave.md")
    assert set(doc) == {"doc_id", "filename", "title", "version", "doc_type", "chunk_count"}
    assert doc["title"] == "Employee Leave Policy" and doc["doc_type"] == "md"

    listed = client.get("/documents").json()
    assert {d["filename"] for d in listed} == {"leave.md", "n.txt"}
    assert client.get("/health").json()["chroma_count"] == body["chunks_indexed"]


def test_pdf_upload_and_version_field(client):
    pdf = make_pdf([[("Guide", 20, True), ("Body text of the guide.", 11, False)]])
    r = upload(client, [("g.pdf", pdf, "application/pdf")], version="3.0")
    assert r.status_code == 200 and r.json()["documents"][0]["version"] == "3.0"


def test_reupload_unchanged_reports_zero_new_chunks(client):
    upload(client, [("n.txt", b"Some note text.", "text/plain")])
    again = upload(client, [("n.txt", b"Some note text.", "text/plain")]).json()
    assert again["chunks_indexed"] == 0 and len(again["documents"]) == 1


def test_unsupported_type_is_400_with_useful_message(client):
    r = upload(client, [("report.docx", b"x", "application/octet-stream")])
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "unsupported_format" and "report.docx" in err["message"] and "supported" in err["message"]


def test_corrupt_and_empty_files_are_400(client):
    r = upload(client, [("bad.pdf", b"not a pdf", "application/pdf"), ("e.txt", b"", "text/plain")])
    assert r.status_code == 400
    assert "bad.pdf" in r.json()["error"]["message"] and "e.txt" in r.json()["error"]["message"]


def test_oversized_file_is_rejected(client, settings, monkeypatch):
    monkeypatch.setattr(type(settings), "max_upload_bytes", property(lambda self: 100))
    r = upload(client, [("big.txt", b"x" * 500, "text/plain")])
    assert r.status_code == 400 and r.json()["error"]["code"] == "file_too_large"
    assert "limit" in r.json()["error"]["message"]


def test_partial_success_returns_200_with_failures_listed(client):
    r = upload(client, [("ok.txt", b"fine text", "text/plain"), ("bad.docx", b"x", "application/octet-stream")])
    assert r.status_code == 200
    body = r.json()
    assert [d["filename"] for d in body["documents"]] == ["ok.txt"]
    assert body["failures"][0]["filename"] == "bad.docx" and "unsupported" in body["failures"][0]["reason"]


def test_embedding_failure_is_502_without_stack_trace(client, embedder):
    embedder.fail_with = EmbeddingError("model access not granted")
    r = upload(client, [("a.txt", b"some text", "text/plain")])
    assert r.status_code == 502
    err = r.json()["error"]
    assert err["code"] == "embedding_failed" and "model access" in err["message"]
    assert "Traceback" not in r.text
    assert client.get("/documents").json() == []


def test_unexpected_error_is_500_and_generic(client, embedder):
    embedder.fail_with = RuntimeError("internal secret")
    r = upload(client, [("a.txt", b"some text", "text/plain")])
    assert r.status_code == 500 and "internal secret" not in r.text
    assert r.json()["error"]["code"] == "internal_error"


def test_missing_files_field_is_422_uniform_body(client):
    r = client.post("/ingest")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_delete_document(client):
    doc = upload(client, [("n.txt", b"Some note text.", "text/plain")]).json()["documents"][0]
    r = client.delete(f"/documents/{doc['doc_id']}")
    assert r.status_code == 200 and r.json() == {"deleted_chunks": doc["chunk_count"]}
    assert client.get("/documents").json() == []
    assert client.delete(f"/documents/{doc['doc_id']}").json() == {"deleted_chunks": 0}


def test_unknown_route_uses_uniform_error(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "http_404"


def test_bedrock_error_outside_service_maps_to_502(settings, store, embedder):
    app = create_app(Container(settings, store=store, embedder=embedder))

    @app.get("/boom")
    def boom():
        raise CredentialsError("No AWS credentials found.")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom")
    assert r.status_code == 502 and r.json()["error"]["code"] == "bedrock_error"


def test_importing_app_and_building_container_touches_no_aws_or_chroma(monkeypatch, tmp_path):
    import boto3
    monkeypatch.setattr(boto3, "Session", lambda *a, **k: (_ for _ in ()).throw(AssertionError("AWS touched")))
    c = Container()
    create_app(c)
    assert c._store is None and c._embedder is None
