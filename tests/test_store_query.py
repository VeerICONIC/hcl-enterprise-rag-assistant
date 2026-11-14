import math

import pytest

from app.schemas import Chunk, ChunkMetadata
from app.vectorstore.chroma_store import VectorStoreError

DIM = 256
EXTRA = {"content_hash": "h", "index_fingerprint": "f", "total_chunks": 1, "ingested_at": "t"}


def chunk(cid, doc="d1", text=None, version="", **kw):
    meta = dict(chunk_id=cid, doc_id=doc, filename=f"{doc}.md", title="T", section="S", page=-1,
                version=version, source=f"upload:{doc}.md", doc_type="md", ordinal=int(cid[-1]))
    meta.update(kw)
    return Chunk(text=text or f"text {cid}", metadata=ChunkMetadata(**meta))


def unit(*pairs):
    v = [0.0] * DIM
    for i, x in pairs:
        v[i] = x
    return v


def load(store, items):
    """items: [(chunk, embedding)]"""
    store.upsert([c for c, _ in items], [e for _, e in items], EXTRA)


def test_similarity_is_one_minus_distance_and_sorted(store):
    load(store, [
        (chunk("a:0"), unit((0, 1.0))),                    # identical to query     -> 1.0
        (chunk("b:1"), unit((0, 0.6), (1, 0.8))),          # cos 0.6
        (chunk("c:2"), unit((1, 1.0))),                    # orthogonal             -> 0.0
        (chunk("d:3"), unit((0, -1.0))),                   # opposite               -> -1.0
    ])
    res = store.query(unit((0, 1.0)), top_k=10)
    assert [c.metadata.chunk_id for c, _ in res] == ["a:0", "b:1", "c:2", "d:3"]
    sims = [s for _, s in res]
    assert sims == pytest.approx([1.0, 0.6, 0.0, -1.0], abs=1e-5)
    assert sims == sorted(sims, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in sims)


def test_raw_distance_is_not_returned_as_similarity(store):
    load(store, [(chunk("a:0"), unit((0, 0.6), (1, 0.8)))])
    raw = store.collection.query(query_embeddings=[unit((0, 1.0))], n_results=1)["distances"][0][0]
    sim = store.query(unit((0, 1.0)), 1)[0][1]
    assert raw == pytest.approx(0.4, abs=1e-5) and sim == pytest.approx(0.6, abs=1e-5)


def test_returns_chunks_with_text_and_metadata(store):
    load(store, [(chunk("a:0", text="Document: X\n\nbody", version="2.1", page=3, section="A > B"), unit((0, 1.0)))])
    (c, sim), = store.query(unit((0, 1.0)), 1)
    assert isinstance(c, Chunk) and c.text == "Document: X\n\nbody"
    assert (c.metadata.version, c.metadata.page, c.metadata.section) == ("2.1", 3, "A > B")
    assert set(c.metadata.model_dump()) == set(ChunkMetadata.model_fields)


def test_top_k_limits_and_oversized_top_k_is_fine(store):
    load(store, [(chunk(f"x:{i}"), unit((i, 1.0))) for i in range(5)])
    assert len(store.query(unit((0, 1.0)), 2)) == 2
    assert len(store.query(unit((0, 1.0)), 50)) == 5


def test_empty_collection_returns_empty_list(store):
    assert store.query(unit((0, 1.0)), 5) == []


def test_identical_scores_are_ordered_by_chunk_id(store):
    load(store, [(chunk("z:0"), unit((0, 1.0))), (chunk("a:1"), unit((0, 1.0))), (chunk("m:2"), unit((0, 1.0)))])
    assert [c.metadata.chunk_id for c, _ in store.query(unit((0, 1.0)), 3)] == ["a:1", "m:2", "z:0"]


def test_metadata_filters(store):
    load(store, [
        (chunk("a:0", doc="d1", version="1"), unit((0, 1.0))),
        (chunk("b:1", doc="d2", version="2"), unit((0, 0.9), (1, 0.1))),
        (chunk("c:2", doc="d3", version=""), unit((0, 0.8), (1, 0.2))),
    ])
    q = unit((0, 1.0))
    ids = lambda r: [c.metadata.chunk_id for c, _ in r]  # noqa: E731
    assert ids(store.query(q, 5, where={"doc_id": "d2"})) == ["b:1"]
    assert ids(store.query(q, 5, where={"doc_id": {"$in": ["d1", "d3"]}})) == ["a:0", "c:2"]
    assert ids(store.query(q, 5, where={"$and": [{"doc_id": {"$in": ["d1", "d2"]}}, {"version": "2"}]})) == ["b:1"]
    assert ids(store.query(q, 5, where={"version": ""})) == ["c:2"]
    assert store.query(q, 5, where={"doc_id": "nope"}) == []
    assert len(store.query(q, 5, where={})) == 3  # empty filter == no filter


def test_invalid_where_is_a_clean_error(store):
    load(store, [(chunk("a:0"), unit((0, 1.0)))])
    with pytest.raises(VectorStoreError):
        store.query(unit((0, 1.0)), 1, where={"doc_id": {"$bogus": 1}})


@pytest.mark.parametrize("emb,match", [
    ([], "empty"), (None, "empty"), ([0.1] * 10, "dimension"), ([0.0] * DIM, "zero vector"),
    ([float("nan")] + [0.1] * (DIM - 1), "non-finite"), (["a"] * DIM, "non-numeric"),
])
def test_invalid_query_embeddings(store, emb, match):
    with pytest.raises(VectorStoreError, match=match):
        store.query(emb, 3)


@pytest.mark.parametrize("k", [0, -1, True, 2.5, "3"])
def test_invalid_top_k(store, k):
    with pytest.raises(VectorStoreError, match="top_k"):
        store.query(unit((0, 1.0)), k)


def test_malformed_stored_record_is_skipped_not_fatal(store, caplog):
    load(store, [(chunk("good:0"), unit((0, 1.0)))])
    store.collection.upsert(ids=["bad:1"], embeddings=[unit((0, 0.9), (1, 0.1))], documents=["x"],
                            metadatas=[{"doc_id": "d9"}])  # required fields missing
    with caplog.at_level("WARNING"):
        res = store.query(unit((0, 1.0)), 5)
    assert [c.metadata.chunk_id for c, _ in res] == ["good:0"]
    assert "bad:1" in caplog.text


def test_similarity_clamped_for_float_noise(store):
    load(store, [(chunk("a:0"), unit((0, 1.0)))])
    (_, sim), = store.query(unit((0, 1.0)), 1)
    assert 0.9999 <= sim <= 1.0 and not math.isnan(sim)


def test_get_embeddings_and_filtered_get_all_chunks(store):
    load(store, [(chunk("a:0", doc="d1"), unit((0, 1.0))), (chunk("b:1", doc="d2"), unit((1, 1.0)))])
    embs = store.get_embeddings(["a:0", "missing"])
    assert list(embs) == ["a:0"] and embs["a:0"][0] == pytest.approx(1.0) and len(embs["a:0"]) == DIM
    assert store.get_embeddings([]) == {}
    assert [c.metadata.chunk_id for c in store.get_all_chunks({"doc_id": "d2"})] == ["b:1"]
    assert [c.metadata.chunk_id for c in store.get_all_chunks()] == ["a:0", "b:1"]


def test_query_survives_restart(settings, store):
    from app.vectorstore.chroma_store import ChromaStore
    load(store, [(chunk("a:0"), unit((0, 1.0)))])
    store.close()
    s2 = ChromaStore(settings)
    try:
        assert s2.query(unit((0, 1.0)), 1)[0][0].metadata.chunk_id == "a:0"
    finally:
        s2.close()
