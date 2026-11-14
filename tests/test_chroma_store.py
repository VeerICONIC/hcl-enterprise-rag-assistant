import pytest

from app.core.config import Settings
from app.schemas import Chunk, ChunkMetadata
from app.vectorstore.chroma_store import ChromaStore, VectorStoreError

DIM = 256


def vec(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


def chunk(doc="d1", ordinal=0, text="some text", **kw):
    meta = dict(chunk_id=f"{doc}:v:{ordinal:05d}", doc_id=doc, filename=f"{doc}.md", title="T",
                section="S", page=-1, version="", source=f"upload:{doc}.md", doc_type="md", ordinal=ordinal)
    meta.update(kw)
    return Chunk(text=text, metadata=ChunkMetadata(**meta))


EXTRA = {"content_hash": "h", "index_fingerprint": "f", "total_chunks": 2, "ingested_at": "2026-01-01T00:00:00+00:00"}


def test_add_and_read_back_with_contract_metadata(store):
    store.upsert([chunk(ordinal=0, text="alpha"), chunk(ordinal=1, text="beta", page=3, section="A > B")],
                 [vec(0), vec(1)], EXTRA)
    assert store.count() == 2
    got = store.get_chunks("d1")
    assert [c.text for c in got] == ["alpha", "beta"]
    assert got[1].metadata.page == 3 and got[1].metadata.section == "A > B"
    assert set(got[0].metadata.model_dump()) == set(ChunkMetadata.model_fields)  # no bookkeeping leaks out


def test_bookkeeping_state(store):
    store.upsert([chunk(ordinal=0), chunk(ordinal=1)], [vec(0), vec(1)], EXTRA)
    st = store.get_document_state("d1")
    assert st.chunk_count == 2 and st.content_hash == "h" and st.index_fingerprint == "f"
    assert st.total_chunks == 2 and st.ids == ["d1:v:00000", "d1:v:00001"]
    assert store.get_document_state("nope") is None


def test_upsert_same_ids_does_not_duplicate_and_updates_text(store):
    store.upsert([chunk(text="old")], [vec(0)], EXTRA)
    store.upsert([chunk(text="new")], [vec(1)], EXTRA)
    assert store.count() == 1 and store.get_chunks("d1")[0].text == "new"


def test_persists_across_instances(settings):
    s1 = ChromaStore(settings)
    s1.upsert([chunk()], [vec(0)], EXTRA)
    s1.close()
    s2 = ChromaStore(settings)
    try:
        assert s2.count() == 1 and s2.get_chunks("d1")[0].text == "some text"
    finally:
        s2.close()


def test_delete_document_leaves_others_untouched(store):
    store.upsert([chunk("d1", 0), chunk("d1", 1), chunk("d2", 0)], [vec(0), vec(1), vec(2)], EXTRA)
    assert store.delete_document("d1") == 2
    assert store.count() == 1 and store.get_document_state("d1") is None
    assert store.get_document_state("d2").chunk_count == 1
    assert store.delete_document("d1") == 0


def test_delete_ids_only_removes_those(store):
    store.upsert([chunk(ordinal=0), chunk(ordinal=1)], [vec(0), vec(1)], EXTRA)
    assert store.delete_ids(["d1:v:00001"]) == 1 and store.delete_ids([]) == 0
    assert store.get_document_state("d1").ids == ["d1:v:00000"]


def test_list_documents_aggregates(store):
    store.upsert([chunk("d1", 0), chunk("d1", 1), chunk("d2", 0, version="2")], [vec(0), vec(1), vec(2)], EXTRA)
    docs = {d.doc_id: d for d in store.list_documents()}
    assert docs["d1"].chunk_count == 2 and docs["d2"].version == "2" and docs["d2"].doc_type == "md"
    assert store.get_document_info("d1").chunk_count == 2 and store.get_document_info("zz") is None


def test_get_all_chunks(store):
    store.upsert([chunk("d1", 0), chunk("d2", 0)], [vec(0), vec(1)], EXTRA)
    assert len(store.get_all_chunks()) == 2


def test_inconsistent_fingerprints_are_not_trusted(store):
    store.upsert([chunk(ordinal=0)], [vec(0)], {**EXTRA, "index_fingerprint": "new"})
    store.upsert([chunk(ordinal=1)], [vec(1)], {**EXTRA, "index_fingerprint": "old"})
    assert store.get_document_state("d1").index_fingerprint == ""


def test_collection_uses_cosine_and_records_embedding_config(store):
    meta = store.collection.metadata
    assert meta["hnsw:space"] == "cosine" and meta["embed_dim"] == DIM
    assert meta["embed_model_id"] == "amazon.titan-embed-text-v2:0"


def test_rejects_wrong_dimension_and_mismatched_lengths(store):
    with pytest.raises(VectorStoreError, match="dimension"):
        store.upsert([chunk()], [[0.1] * 10], EXTRA)
    with pytest.raises(VectorStoreError):
        store.upsert([chunk()], [], EXTRA)
    with pytest.raises(VectorStoreError, match="duplicate"):
        store.upsert([chunk(), chunk()], [vec(0), vec(1)], EXTRA)


def test_rejects_invalid_extra_metadata(store):
    with pytest.raises(VectorStoreError):
        store.upsert([chunk()], [vec(0)], {"nested": {"a": 1}})
    with pytest.raises(VectorStoreError):
        store.upsert([chunk()], [vec(0)], {"filename": "override"})


def test_reopening_with_different_dimension_is_refused(settings, store):
    store.upsert([chunk()], [vec(0)], EXTRA)
    store.close()
    other = Settings(bedrock_embed_dim=512, chroma_path=settings.chroma_path,
                     chroma_collection=settings.chroma_collection)
    s2 = ChromaStore(other)
    try:
        with pytest.raises(VectorStoreError, match="Mixing vector spaces"):
            s2.count()
    finally:
        s2.close()


def test_reopening_with_different_model_is_refused(settings, store):
    store.count()
    store.close()
    other = Settings(bedrock_embed_dim=256, bedrock_embed_model_id="other.model", chroma_path=settings.chroma_path,
                     chroma_collection=settings.chroma_collection)
    s2 = ChromaStore(other)
    try:
        with pytest.raises(VectorStoreError, match="Mixing"):
            s2.health()
    finally:
        s2.close()


def test_health(store):
    h = store.health()
    assert h["chunks"] == 0 and h["collection"] == "test_docs"


def test_opening_does_not_create_duplicate_collections(settings):
    for _ in range(3):
        s = ChromaStore(settings)
        s.count()
        s.close()
    s = ChromaStore(settings)
    try:
        assert [c.name for c in s.collection._client.list_collections()] == ["test_docs"]
    finally:
        s.close()


def test_uninitialised_path_is_created(tmp_path):
    s = Settings(bedrock_embed_dim=256, chroma_path=str(tmp_path / "a" / "b"), chroma_collection="abc_docs")
    st = ChromaStore(s)
    try:
        assert st.count() == 0 and (tmp_path / "a" / "b").exists()
    finally:
        st.close()
