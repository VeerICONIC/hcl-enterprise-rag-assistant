import pytest

from app.core.bedrock import ThrottlingError
from app.embeddings.base import EmbeddingError
from app.ingestion import ids
from app.ingestion.errors import (
    CorruptDocumentError, EmptyDocumentError, FileTooLargeError, UnsupportedFormatError,
)
from app.vectorstore.chroma_store import VectorStoreError
from tests.helpers import SAMPLE_MD, make_pdf


def test_end_to_end_markdown(service, store, embedder):
    out = service.ingest_bytes("leave.md", SAMPLE_MD)
    assert out.status == "indexed" and out.chunks_indexed == out.document.chunk_count > 0
    assert store.count() == out.chunks_indexed
    chunks = store.get_chunks(out.doc_id)
    m = chunks[0].metadata
    assert (m.filename, m.doc_type, m.title, m.version, m.page) == (
        "leave.md", "md", "Employee Leave Policy", "", -1)
    assert m.doc_id == ids.doc_id_for("leave.md") and m.source == "upload:leave.md"
    assert m.chunk_id == f"{m.doc_id}:v:00000" and [c.metadata.ordinal for c in chunks] == list(range(len(chunks)))
    # heading context is embedded and stored
    annual = next(c for c in chunks if "20 days of annual leave" in c.text)
    assert annual.text.startswith("Document: Employee Leave Policy\nSection: Annual Leave\n\n")
    assert annual.metadata.section == "Annual Leave"
    assert any(c.metadata.section == "Annual Leave > Carry-over" for c in chunks)
    assert set(embedder.texts) == {c.text for c in chunks}


def test_end_to_end_pdf_keeps_pages(service, store):
    data = make_pdf([[("Billing", 20, True), ("Invoices are issued monthly.", 11, False)],
                     [("Limits", 20, True), ("The limit is 100 requests per minute.", 11, False)]])
    out = service.ingest_bytes("guide.pdf", data)
    by_page = {c.metadata.page: c for c in store.get_chunks(out.doc_id)}
    assert set(by_page) == {1, 2}
    assert "Invoices" in by_page[1].text and "Section: Billing" in by_page[1].text
    assert by_page[1].metadata.title == "guide"  # two H1s: no heading is trusted as the title
    assert "100 requests" in by_page[2].text and by_page[2].metadata.doc_type == "pdf"


def test_chroma_metadata_is_scalar_only(service, store):
    out = service.ingest_bytes("a.txt", b"Plain text document.")
    raw = store.collection.get(include=["metadatas"])["metadatas"][0]
    assert all(isinstance(v, (str, int, float, bool)) for v in raw.values())
    assert {"content_hash", "index_fingerprint", "total_chunks", "ingested_at"} <= set(raw)
    assert raw["total_chunks"] == out.document.chunk_count and raw["content_hash"] == ids.content_hash(b"Plain text document.")


def test_version_is_only_taken_from_caller(service, store):
    out = service.ingest_bytes("v.txt", b"Text mentioning version 9.9 inside.")
    assert out.document.version == ""
    out = service.ingest_bytes("v.txt", b"Text mentioning version 9.9 inside.", version=" 2.1 ")
    assert out.status == "updated" and out.document.version == "2.1"
    assert store.get_chunks(out.doc_id)[0].metadata.chunk_id.endswith(":v2.1:00000")


def test_ids_are_stable_and_distinct():
    assert ids.doc_id_for("A.md") == ids.doc_id_for("dir/a.md") == ids.doc_id_for("C:\\x\\a.md")
    assert ids.doc_id_for("a.md") != ids.doc_id_for("b.md")
    assert ids.chunk_id_for("doc_1", "2.1", 7) == "doc_1:v2.1:00007"


def test_reingest_unchanged_makes_no_embedding_calls_or_changes(service, store, embedder):
    first = service.ingest_bytes("leave.md", SAMPLE_MD)
    calls, count = embedder.calls, store.count()
    again = service.ingest_bytes("leave.md", SAMPLE_MD)
    assert again.status == "unchanged" and again.chunks_indexed == 0
    assert again.document.chunk_count == first.document.chunk_count
    assert embedder.calls == calls and store.count() == count


def test_reingest_changed_content_replaces_and_removes_stale(service, store):
    long_v1 = b"# Doc\n\n" + b"\n\n".join(f"Paragraph {i} " .encode() + b"alpha " * 40 for i in range(12))
    first = service.ingest_bytes("doc.md", long_v1)
    assert first.chunks_indexed > 3
    short_v2 = b"# Doc\n\nOnly one short paragraph now."
    second = service.ingest_bytes("doc.md", short_v2)
    assert second.status == "updated" and second.chunks_removed == first.chunks_indexed - second.chunks_indexed
    chunks = store.get_chunks(first.doc_id)
    assert len(chunks) == second.chunks_indexed == 1 and "short paragraph" in chunks[0].text
    assert store.count() == 1  # no stale or duplicate records


def test_reingest_leaves_other_documents_alone(service, store):
    other = service.ingest_bytes("other.txt", b"Totally unrelated content.")
    service.ingest_bytes("doc.md", b"# T\n\nversion one")
    service.ingest_bytes("doc.md", b"# T\n\nversion two")
    assert [c.text for c in store.get_chunks(other.doc_id)] == [store.get_chunks(other.doc_id)[0].text]
    assert store.count() == 2


def test_settings_change_triggers_reindex(settings, embedder, store):
    from app.core.config import Settings
    from app.ingestion.ingestion_service import IngestionService
    IngestionService(embedder, store, settings).ingest_bytes("d.txt", b"hello world " * 100)
    s2 = Settings(**{**settings.model_dump(), "chunk_size": 200, "chunk_overlap": 20})
    out = IngestionService(embedder, store, s2).ingest_bytes("d.txt", b"hello world " * 100)
    assert out.status == "updated"


def test_failed_parse_preserves_previous_version(service, store, embedder):
    first = service.ingest_bytes("doc.pdf", make_pdf([[("Body text of the original document.", 11, False)]]))
    before = [c.text for c in store.get_chunks(first.doc_id)]
    with pytest.raises(CorruptDocumentError):
        service.ingest_bytes("doc.pdf", b"%PDF-1.4 broken")
    assert [c.text for c in store.get_chunks(first.doc_id)] == before


def test_failed_embedding_preserves_previous_version(service, store, embedder):
    first = service.ingest_bytes("doc.md", b"# T\n\noriginal content")
    before = [c.text for c in store.get_chunks(first.doc_id)]
    embedder.fail_with = EmbeddingError("boom")
    with pytest.raises(EmbeddingError):
        service.ingest_bytes("doc.md", b"# T\n\nCHANGED content")
    assert [c.text for c in store.get_chunks(first.doc_id)] == before
    assert store.get_document_state(first.doc_id).content_hash == ids.content_hash(b"# T\n\noriginal content")


def test_failed_embedding_on_new_document_stores_nothing(service, store, embedder):
    embedder.fail_with = ThrottlingError("slow down")
    with pytest.raises(ThrottlingError):
        service.ingest_bytes("new.md", b"# T\n\ncontent")
    assert store.count() == 0


def test_failed_first_write_is_rolled_back(service, store, monkeypatch):
    real = store.upsert

    def partial(chunks, embeddings, extra=None):
        real(chunks[:1], embeddings[:1], extra)
        raise VectorStoreError("disk full")

    monkeypatch.setattr(store, "upsert", partial)
    with pytest.raises(VectorStoreError):
        service.ingest_bytes("big.txt", b"word " * 2000)
    assert store.count() == 0


def test_interrupted_update_is_detected_and_repaired(service, store, monkeypatch):
    data_v1 = b"# T\n\n" + b"alpha sentence here. " * 100
    first = service.ingest_bytes("d.md", data_v1)
    real = store.upsert

    def partial(chunks, embeddings, extra=None):
        real(chunks[:1], embeddings[:1], extra)
        raise VectorStoreError("crash")

    monkeypatch.setattr(store, "upsert", partial)
    with pytest.raises(VectorStoreError):
        service.ingest_bytes("d.md", data_v1 + b"\n\nNew trailing paragraph.")
    monkeypatch.undo()
    assert store.get_document_state(first.doc_id).index_fingerprint == ""  # inconsistent, not trusted
    repaired = service.ingest_bytes("d.md", data_v1 + b"\n\nNew trailing paragraph.")
    assert repaired.status == "updated"
    assert store.get_document_state(first.doc_id).chunk_count == repaired.document.chunk_count


def test_validation_errors(service, settings):
    with pytest.raises(UnsupportedFormatError, match="supported"):
        service.ingest_bytes("a.docx", b"x")
    with pytest.raises(EmptyDocumentError):
        service.ingest_bytes("a.txt", b"   \n ")
    with pytest.raises(EmptyDocumentError, match="no text"):
        service.ingest_bytes("a.md", b"# Only a heading")
    with pytest.raises(FileTooLargeError, match="limit"):
        service.ingest_bytes("a.txt", b"x" * (settings.max_upload_bytes + 1))


def test_batch_reports_every_failure_and_continues(service, store, embedder):
    result = service.ingest_files([
        ("good.md", b"# T\n\nfine"), ("bad.docx", b"x"), ("empty.txt", b""), ("broken.pdf", b"nope"),
    ])
    assert [o.filename for o in result.outcomes] == ["good.md"]
    assert {f.filename: f.code for f in result.failures} == {
        "bad.docx": "unsupported_format", "empty.txt": "empty_document", "broken.pdf": "corrupt_document"}
    resp = result.to_response()
    assert resp.chunks_indexed == 1 and {f["filename"] for f in resp.failures} == {"bad.docx", "empty.txt", "broken.pdf"}
    assert all(set(f) == {"filename", "reason"} for f in resp.failures)


def test_batch_classifies_infrastructure_failures(service, embedder):
    embedder.fail_with = EmbeddingError("access denied")
    result = service.ingest_files([("a.md", b"# T\n\nx")])
    f = result.failures[0]
    assert f.code == "embedding_failed" and not f.client_error and "access denied" in f.reason


def test_batch_unexpected_error_is_not_swallowed_silently(service, embedder):
    embedder.fail_with = RuntimeError("secret internal detail")
    f = service.ingest_files([("a.md", b"# T\n\nx")]).failures[0]
    assert f.code == "internal_error" and "secret internal detail" not in f.reason


def test_ingest_path_and_delete(service, store, tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("Some notes about limits.", encoding="utf-8")
    out = service.ingest_path(p)
    assert store.get_chunks(out.doc_id)[0].metadata.source == str(p.resolve())
    assert [d.doc_id for d in service.list_documents()] == [out.doc_id]
    assert service.delete_document(out.doc_id) == 1 and service.list_documents() == []


def test_missing_path_error(service, tmp_path):
    from app.ingestion.errors import IngestionError
    with pytest.raises(IngestionError, match="cannot read"):
        service.ingest_path(tmp_path / "missing.txt")
