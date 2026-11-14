import pytest

from app.retrieval.bm25_retriever import Bm25Index, Bm25Retriever
from app.retrieval.conflict_detector import (
    detect_conflicts, extract_dates, extract_facts, family_key, filename_versions,
)
from app.retrieval.evidence_analyzer import analyze_evidence
from app.retrieval.hybrid_ranker import rrf_fuse
from app.retrieval.models import Bm25Hit, RetrievalSettings, VectorHit
from app.retrieval.text import strip_context_prefix, tokenize
from app.schemas import Chunk, ChunkMetadata, RetrievedChunk


def mk(cid, text="some text", doc=None, filename=None, version="", title="", section="", page=-1):
    doc = doc or cid.split(":")[0]
    return Chunk(text=text, metadata=ChunkMetadata(
        chunk_id=cid, doc_id=doc, filename=filename or f"{doc}.md", title=title, section=section, page=page,
        version=version, source=f"upload:{doc}.md", doc_type="md", ordinal=0))


def rc(cid, text="some text", vs=0.6, rank=1, score=0.5, **kw):
    return RetrievedChunk(chunk=mk(cid, text, **kw), score=score, vector_score=vs, rank=rank)


class StubStore:
    def __init__(self, chunks):
        self.chunks, self.where_seen = chunks, []

    def get_all_chunks(self, where=None):
        self.where_seen.append(where)
        return list(self.chunks)


# ---- tokenizer -------------------------------------------------------------------
def test_tokenize_keeps_identifiers_and_drops_noise():
    assert tokenize("Error E-4021 in v2.1: the API?") == ["error", "e-4021", "4021", "v2.1", "v2", "1", "api"]
    assert tokenize("?!... --- ,,,") == [] and tokenize("the of and") == [] and tokenize("") == []


def test_strip_context_prefix():
    assert strip_context_prefix("Document: T\nSection: A > B\n\nBody here") == "Body here"
    assert strip_context_prefix("Document: T\n\nBody") == "Body"
    assert strip_context_prefix("Body only") == "Body only"


# ---- BM25 ------------------------------------------------------------------------
CORPUS = [
    mk("a:0", "Error E-4021 means the license server is unreachable."),
    mk("b:1", "Reset your password from the account settings page."),
    mk("c:2", "The license covers one server per account. The account owner may add servers."),
]


def test_bm25_exact_keyword_and_identifier():
    hits = Bm25Retriever(StubStore(CORPUS)).search("E-4021", 5)
    assert [h.chunk.metadata.chunk_id for h in hits] == ["a:0"]
    assert hits[0].rank == 1 and hits[0].score > 0 and hits[0].coverage == 1.0


def test_bm25_rarer_terms_rank_higher_and_ranks_are_sequential():
    hits = Bm25Retriever(StubStore(CORPUS)).search("license password", 5)
    assert hits[0].chunk.metadata.chunk_id == "b:1"  # "password" is rarer than "license"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(h.score > 0 for h in hits)


def test_bm25_coverage_is_fraction_of_distinct_query_terms():
    hits = {h.chunk.metadata.chunk_id: h for h in Bm25Retriever(StubStore(CORPUS)).search("license server password", 5)}
    assert hits["a:0"].coverage == pytest.approx(2 / 3) and hits["b:1"].coverage == pytest.approx(1 / 3)


@pytest.mark.parametrize("q", ["", "   ", "?!...", "the of and", "---"])
def test_bm25_empty_or_punctuation_query_returns_nothing_and_skips_store(q):
    store = StubStore(CORPUS)
    assert Bm25Retriever(store).search(q, 5) == [] and store.where_seen == []


def test_bm25_empty_corpus_and_no_match():
    assert Bm25Retriever(StubStore([])).search("anything", 5) == []
    assert Bm25Retriever(StubStore(CORPUS)).search("zebra", 5) == []


def test_bm25_limit_and_filter_is_passed_to_store():
    store = StubStore(CORPUS)
    assert len(Bm25Retriever(store).search("server account license", 1, where={"doc_id": "a"})) == 1
    assert store.where_seen == [{"doc_id": "a"}]


def test_bm25_ties_are_broken_by_chunk_id_and_idf_is_never_negative():
    twins = [mk("z:0", "alpha beta"), mk("a:1", "alpha beta")]
    hits = Bm25Index(twins).search(["alpha"], 5)  # term in every doc: classic negative-IDF case
    assert [h.chunk.metadata.chunk_id for h in hits] == ["a:1", "z:0"] and hits[0].score > 0


def test_bm25_with_real_chroma_filter(kw_service, store):
    kw_service.ingest_bytes("one.txt", b"The router password is stored in the vault.")
    kw_service.ingest_bytes("two.txt", b"The printer password is on the sticker.")
    doc_two = [d.doc_id for d in kw_service.list_documents() if d.filename == "two.txt"][0]
    hits = Bm25Retriever(store).search("password", 5, where={"doc_id": doc_two})
    assert [h.chunk.metadata.filename for h in hits] == ["two.txt"]
    assert len(Bm25Retriever(store).search("password", 5)) == 2


# ---- RRF -------------------------------------------------------------------------
def vh(cid, rank, sim=0.5):
    return VectorHit(mk(cid), sim, rank)


def bh(cid, rank, cov=1.0):
    return Bm25Hit(mk(cid), 1.0, rank, cov)


def test_rrf_scores_order_and_normalization():
    out = rrf_fuse([vh("A:0", 1, .9), vh("B:0", 2, .8)], [bh("B:0", 1), bh("C:0", 2)], k=60)
    assert [e.chunk.metadata.chunk_id for e in out] == ["B:0", "A:0", "C:0"]
    by = {e.chunk.metadata.chunk_id: e for e in out}
    assert by["B:0"].rrf == pytest.approx(1 / 62 + 1 / 61)
    assert by["A:0"].rrf == pytest.approx(1 / 61) and by["C:0"].rrf == pytest.approx(1 / 62)
    assert by["B:0"].score == pytest.approx((1 / 62 + 1 / 61) / (2 / 61))
    assert (by["B:0"].vector_rank, by["B:0"].bm25_rank) == (2, 1)
    assert by["A:0"].bm25_rank is None and by["C:0"].vector_rank is None


def test_rrf_perfect_agreement_scores_one_and_scores_are_in_range():
    out = rrf_fuse([vh("A:0", 1), vh("B:0", 2)], [bh("A:0", 1), bh("B:0", 2)])
    assert out[0].score == pytest.approx(1.0) and all(0 < e.score <= 1 for e in out)


def test_rrf_deduplicates_and_preserves_vector_score_and_metadata():
    out = rrf_fuse([vh("A:0", 1, 0.77)], [bh("A:0", 3, 0.5)])
    assert len(out) == 1 and out[0].vector_score == 0.77 and out[0].coverage == 0.5
    assert out[0].chunk.metadata.filename == "A.md"
    dup = rrf_fuse([vh("A:0", 1, 0.9), vh("A:0", 2, 0.1)], [])
    assert len(dup) == 1 and dup[0].vector_score == 0.9


def test_rrf_chunk_in_only_one_list_and_empty_inputs():
    assert [e.chunk.metadata.chunk_id for e in rrf_fuse([vh("A:0", 1)], [])] == ["A:0"]
    assert [e.vector_score for e in rrf_fuse([], [bh("A:0", 1)])] == [None]
    assert rrf_fuse([], []) == []


def test_rrf_ties_are_deterministic():
    # equal rrf: the one with a known cosine goes first
    out = rrf_fuse([vh("b:0", 1, 0.5)], [bh("a:0", 1)])
    assert [e.chunk.metadata.chunk_id for e in out] == ["b:0", "a:0"]
    # equal rrf and equal cosine: chunk_id
    out = rrf_fuse([vh("z:0", 1, 0.5), vh("a:0", 1, 0.5)], [])
    assert [e.chunk.metadata.chunk_id for e in out] == ["a:0", "z:0"]
    again = rrf_fuse([vh("z:0", 1, 0.5), vh("a:0", 1, 0.5)], [])
    assert [e.chunk.metadata.chunk_id for e in again] == ["a:0", "z:0"]


def test_rrf_constant_changes_scores_and_must_be_valid():
    small = rrf_fuse([vh("A:0", 1)], [], k=1)[0].rrf
    large = rrf_fuse([vh("A:0", 1)], [], k=100)[0].rrf
    assert small == pytest.approx(0.5) and large == pytest.approx(1 / 101)
    with pytest.raises(ValueError):
        rrf_fuse([], [], k=0)


# ---- evidence --------------------------------------------------------------------
S = RetrievalSettings(min_relevance=0.30)


def analyze(chunks, coverage=None, terms=4):
    return analyze_evidence(chunks, coverage or {}, S, terms)


def test_no_results_is_none():
    a = analyze([])
    assert a.strength == "none" and a.relevant_count == 0 and a.reasons


def test_nothing_above_threshold_is_none():
    assert analyze([rc("a:0", vs=0.29), rc("b:0", vs=0.1)]).strength == "none"


def test_strong_needs_margin_and_support():
    assert analyze([rc("a:0", vs=0.70), rc("b:0", vs=0.50)]).strength == "strong"      # 2 chunks, best >= .45
    assert analyze([rc("a:0", vs=0.65)]).strength == "strong"                          # single, >= .60
    assert analyze([rc("a:0", vs=0.50)]).strength == "weak"                            # single, only > .45
    assert analyze([rc("a:0", vs=0.40), rc("b:0", vs=0.35)]).strength == "weak"        # two, but best < .45


def test_weak_when_barely_relevant():
    a = analyze([rc("a:0", vs=0.31), rc("b:0", vs=0.05)])
    assert a.strength == "weak" and a.relevant_count == 1


def test_distinct_documents_and_concentration():
    one_doc = analyze([rc("d1:0", vs=0.7, doc="d1"), rc("d1:1", vs=0.6, doc="d1")])
    assert one_doc.distinct_documents == 1 and one_doc.concentrated and one_doc.strength == "strong"
    two_docs = analyze([rc("d1:0", vs=0.7, doc="d1"), rc("d2:0", vs=0.6, doc="d2")])
    assert two_docs.distinct_documents == 2 and not two_docs.concentrated
    assert two_docs.relevant_ids == {"d1:0", "d2:0"}


def test_lexical_rescue_marks_exact_matches_relevant_but_never_strong():
    c = rc("a:0", vs=0.05)
    a = analyze([c], coverage={"a:0": 1.0}, terms=3)
    assert a.relevant_count == 1 and a.strength == "weak"
    assert analyze([c], coverage={"a:0": 1.0}, terms=1).strength == "none"   # single-term queries never rescue
    assert analyze([c], coverage={"a:0": 0.5}, terms=3).strength == "none"


def test_thresholds_follow_settings():
    strict = RetrievalSettings(min_relevance=0.6)
    assert analyze_evidence([rc("a:0", vs=0.55)], {}, strict, 3).strength == "none"


# ---- facts -----------------------------------------------------------------------
def test_extract_facts_units_currency_percent_and_skips_identifiers():
    facts = extract_facts("Employees get 20 days. Fee is $1,500 or 5% plus 3 requests. Error E-4021 in v2.1 (see 7).")
    got = {(f.unit, f.value) for f in facts}
    assert got == {("day", 20.0), ("$", 1500.0), ("%", 5.0), ("request", 3.0)}


def test_extract_dates_formats_and_blanking():
    dates, rest = extract_dates("Effective 2024-03-01, expires 31 March 2025 or March 5, and May 7th.")
    assert {d.key for d in dates} == {(None, 3, 5), (None, 5, 7), (2024, 3, 1), (2025, 3, 31)}
    assert "2024" not in rest and "31" not in rest
    assert extract_dates("You may 5 times try")[0] == []   # lowercase "may" is a verb, not a month


def test_family_key_and_filename_versions():
    assert family_key("leave_policy_v2.md", "") == family_key("Leave-Policy-FINAL.md", "") == "leave policy"
    assert family_key("x.md", "Leave Policy 2024") == "leave policy"
    assert filename_versions("leave_policy_v2.1.md") == {"v2.1"} and filename_versions("policy.md") == frozenset()


# ---- conflicts -------------------------------------------------------------------
def kinds(report):
    return [g.kind for g in report.groups]


def test_no_chunks_or_single_chunk_no_conflict():
    assert not detect_conflicts([]).has_conflict
    assert not detect_conflicts([rc("a:0")]).has_conflict


def test_same_document_is_never_a_conflict():
    r = detect_conflicts([
        rc("d1:0", "Annual leave is 20 days per year.", doc="d1", version="1", title="Leave Policy", section="Leave"),
        rc("d1:1", "Annual leave is 25 days per year.", doc="d1", version="1", title="Leave Policy", section="Leave", rank=2),
    ])
    assert not r.has_conflict and r.groups == []


def test_different_versions_in_metadata_is_a_metadata_conflict():
    r = detect_conflicts([
        rc("d1:0", "Employees receive annual leave.", doc="d1", filename="leave_policy_v1.md", version="1",
           title="Leave Policy", section="Annual Leave"),
        rc("d2:0", "Staff are entitled to annual leave.", doc="d2", filename="leave_policy_v2.md", version="2",
           title="Leave Policy", section="Annual Leave", rank=2),
    ])
    assert kinds(r) == ["version_difference"]
    g = r.groups[0]
    assert g.chunk_ids == ["d1:0", "d2:0"] and g.topic == "leave policy"
    assert g.description.startswith("METADATA CONFLICT") and "version 1" in g.description and "version 2" in g.description
    assert "NOT determined" in g.description


def test_filename_only_versions_are_labelled_possible_and_uncertainty_is_explicit():
    r = detect_conflicts([
        rc("d1:0", "Annual leave rules apply to everyone.", doc="d1", filename="leave_policy_v1.md", title="", section="Annual Leave"),
        rc("d2:0", "Annual leave rules were revised recently.", doc="d2", filename="leave_policy_v2.md", title="", section="Annual Leave", rank=2),
    ])
    g = r.groups[0]
    assert g.kind == "version_difference" and g.description.startswith("POSSIBLE")
    assert "filenames suggest different versions" in g.description and "No version metadata for" in g.description
    assert "leave_policy_v1.md" in g.description and "NOT determined" in g.description


def test_same_family_but_identical_recorded_version_is_not_a_version_conflict():
    r = detect_conflicts([
        rc("d1:0", "Annual leave rules.", doc="d1", filename="leave_a.md", version="3", title="Leave Policy", section="Annual Leave"),
        rc("d2:0", "Annual leave rules.", doc="d2", filename="leave_b.md", version="3", title="Leave Policy", section="Annual Leave", rank=2),
    ])
    assert "version_difference" not in kinds(r)


def test_different_documents_with_competing_values_are_flagged_as_possible_contradiction():
    r = detect_conflicts([
        rc("d1:0", "Annual leave is 20 days per year.", doc="d1", filename="handbook.md", title="Handbook", section="Leave"),
        rc("d2:0", "Annual leave is 25 days per year.", doc="d2", filename="faq.md", title="FAQ", section="Leave", rank=2),
    ])
    assert kinds(r) == ["contradiction"]
    g = r.groups[0]
    assert g.chunk_ids == ["d1:0", "d2:0"]
    assert g.description.startswith("POSSIBLE contradiction") and "20 days" in g.description and "25 days" in g.description
    assert "heuristic" in g.description and "handbook.md" in g.description and "faq.md" in g.description


def test_competing_dates_flagged_as_date_difference():
    r = detect_conflicts([
        rc("d1:0", "Unused annual leave expires on 31 March.", doc="d1", title="Handbook", section="Leave"),
        rc("d2:0", "Unused annual leave expires on 30 April.", doc="d2", title="FAQ", section="Leave", rank=2),
    ])
    assert kinds(r) == ["date_difference"] and r.groups[0].description.startswith("POSSIBLE date difference")


def test_unrelated_content_and_agreeing_values_are_not_flagged():
    r = detect_conflicts([
        rc("d1:0", "Passwords must contain 12 characters.", doc="d1", title="Security", section="Passwords"),
        rc("d2:0", "Refunds are processed within 5 days.", doc="d2", title="Billing", section="Refunds", rank=2),
    ])
    assert not r.has_conflict
    same = detect_conflicts([
        rc("d1:0", "Annual leave is 20 days per year.", doc="d1", title="A", section="Leave"),
        rc("d2:0", "Annual leave is 20 days per year.", doc="d2", title="B", section="Leave", rank=2),
    ])
    assert not same.has_conflict


def test_wording_differences_without_numbers_are_not_conflicts():
    r = detect_conflicts([
        rc("d1:0", "Employees may carry over unused leave with manager approval.", doc="d1", title="A", section="Leave"),
        rc("d2:0", "Unused leave can be carried over if a manager agrees.", doc="d2", title="B", section="Leave", rank=2),
    ])
    assert not r.has_conflict


def test_only_relevant_chunks_participate():
    chunks = [
        rc("d1:0", "Annual leave is 20 days per year.", doc="d1", title="A", section="Leave"),
        rc("d2:0", "Annual leave is 25 days per year.", doc="d2", title="B", section="Leave", rank=2),
    ]
    assert not detect_conflicts(chunks, relevant_ids={"d1:0"}).has_conflict
    assert detect_conflicts(chunks, relevant_ids={"d1:0", "d2:0"}).has_conflict


def test_evidence_text_is_preserved_in_inputs_not_modified():
    chunks = [rc("d1:0", "Annual leave is 20 days per year.", doc="d1", title="A", section="Leave"),
              rc("d2:0", "Annual leave is 25 days per year.", doc="d2", title="B", section="Leave", rank=2)]
    before = [c.model_dump() for c in chunks]
    detect_conflicts(chunks)
    assert [c.model_dump() for c in chunks] == before


def test_three_way_conflict_forms_one_group_with_all_ids():
    r = detect_conflicts([
        rc("d1:0", "Annual leave is 20 days per year.", doc="d1", title="A", section="Leave"),
        rc("d2:0", "Annual leave is 25 days per year.", doc="d2", title="B", section="Leave", rank=2),
        rc("d3:0", "Annual leave is 30 days per year.", doc="d3", title="C", section="Leave", rank=3),
    ])
    assert len(r.groups) == 1 and r.groups[0].chunk_ids == ["d1:0", "d2:0", "d3:0"]


# ---- BM25 cache / index consistency ----------------------------------------------------
def test_bm25_index_is_cached_and_invalidated_by_store_changes(kw_service, store):
    r = Bm25Retriever(store)
    kw_service.ingest_bytes("one.txt", b"The router password is stored in the vault.")
    assert [h.chunk.metadata.filename for h in r.search("password", 5)] == ["one.txt"]
    r.search("password", 5); r.search("vault", 5)
    assert r.builds == 1                                   # unchanged collection: no rebuild
    out = kw_service.ingest_bytes("two.txt", b"The printer password is on the sticker.")
    assert {h.chunk.metadata.filename for h in r.search("password", 5)} == {"one.txt", "two.txt"}
    assert r.builds == 2                                   # addition invalidated the cache
    kw_service.delete_document(out.doc_id)
    assert [h.chunk.metadata.filename for h in r.search("password", 5)] == ["one.txt"]
    assert r.builds == 3                                   # deletion invalidated it too
    kw_service.ingest_bytes("one.txt", b"The router password is stored in the safe.")
    assert r.search("vault", 5) == [] and r.builds == 4    # re-ingest (update) invalidated it


def test_bm25_cache_is_per_filter_and_skipped_without_revision():
    store = StubStore(CORPUS)                              # no `revision` attribute -> never cached
    r = Bm25Retriever(store)
    r.search("license", 5); r.search("license", 5)
    assert r.builds == 2


def test_identical_value_sets_agree_even_if_other_values_differ():
    a = "Employees receive 10 days of sick leave per year. A certificate is required for absences longer than 3 days."
    r = detect_conflicts([rc("d1:0", a, doc="d1", title="A", section="Sick Leave"),
                          rc("d2:0", a, doc="d2", title="B", section="Sick Leave", rank=2)])
    assert not r.has_conflict


def test_disjoint_value_sets_conflict_and_partial_overlap_does_not():
    a = "Annual leave is 20 days per year with up to 5 days carried over."
    b = "Annual leave is 25 days per year with up to 10 days carried over."
    assert detect_conflicts([rc("d1:0", a, doc="d1", title="A", section="Leave"),
                             rc("d2:0", b, doc="d2", title="B", section="Leave", rank=2)]).has_conflict
    c = "Annual leave is 20 days per year with up to 10 days carried over."
    assert not detect_conflicts([rc("d1:0", c, doc="d1", title="A", section="Leave"),
                                 rc("d2:0", b, doc="d2", title="B", section="Leave", rank=2)]).has_conflict


def test_different_named_sections_are_not_the_same_subject():
    r = detect_conflicts([
        rc("d1:0", "Employees receive 20 days of annual leave per year.", doc="d1", title="A", section="Annual Leave"),
        rc("d2:0", "Employees receive 10 days of sick leave per year.", doc="d2", title="B", section="Sick Leave", rank=2)])
    assert not r.has_conflict


def test_identical_passages_in_two_versions_are_not_a_version_conflict():
    same = "Employees receive 10 days of sick leave per year."
    r = detect_conflicts([
        rc("d1:0", same, doc="d1", filename="handbook_v1.md", version="1", title="Handbook", section="Sick Leave"),
        rc("d2:0", same, doc="d2", filename="handbook_v2.md", version="2", title="Handbook", section="Sick Leave", rank=2)])
    assert not r.has_conflict


def test_personal_pronouns_are_not_query_terms():
    assert tokenize("How do I reset my password?") == ["reset", "password"]
    assert tokenize("my leave and his leave") == ["leave", "leave"]
