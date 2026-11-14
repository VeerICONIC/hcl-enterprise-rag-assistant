import json
import math
import re
import sys
from pathlib import Path

import pytest

from app.core.config import Settings
from app.evaluation import answer_metrics, citation_metrics, evaluator, retrieval_metrics
from app.evaluation.dataset import (
    DatasetError, EvalQuestion, load_manifest, load_questions, normalize, resolve_relevant,
)
from app.evaluation.report_generator import render_markdown, write_reports
from app.schemas import Answer, Chunk, ChunkMetadata, Citation, ConflictReport
from tests.helpers import KeywordEmbedder

DATA = Path(__file__).resolve().parents[1] / "data" / "eval"


# ---- retrieval metrics ----------------------------------------------------------------
def test_recall_hit_mrr_ndcg_known_values():
    ranked, rel = ["a", "b", "c", "d"], {"c", "z"}
    assert retrieval_metrics.recall_at_k(ranked, rel, 3) == 0.5
    assert retrieval_metrics.recall_at_k(ranked, rel, 2) == 0.0
    assert retrieval_metrics.hit_rate_at_k(ranked, rel, 3) == 1.0 and retrieval_metrics.hit_rate_at_k(ranked, rel, 2) == 0.0
    assert retrieval_metrics.reciprocal_rank(ranked, rel) == pytest.approx(1 / 3)
    assert retrieval_metrics.reciprocal_rank(["x"], rel) == 0.0
    # DCG = 1/log2(4); ideal (2 relevant, k=4) = 1/log2(2) + 1/log2(3)
    assert retrieval_metrics.ndcg_at_k(ranked, rel, 4) == pytest.approx((1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert retrieval_metrics.ndcg_at_k(["a"], {"a"}, 5) == pytest.approx(1.0)
    assert retrieval_metrics.ndcg_at_k(["a"], set(), 5) == 0.0 and retrieval_metrics.recall_at_k(["a"], set(), 5) == 0.0


def test_mean_aggregation():
    rows = [retrieval_metrics.per_question(["a"], {"a"}, 5), retrieval_metrics.per_question(["b"], {"a"}, 5)]
    m = retrieval_metrics.mean(rows)
    assert m["recall@5"] == 0.5 and m["mrr"] == 0.5 and m["n"] == 2 and retrieval_metrics.mean([]) == {}


# ---- dataset ----------------------------------------------------------------------------
def test_shipped_dataset_is_valid_and_covers_all_required_categories():
    qs = load_questions(DATA / "questions.jsonl")
    assert len(qs) >= 25
    assert {q.category for q in qs} == {"direct", "paraphrase", "multi_document", "conflict", "ambiguous", "unanswerable", "similar_keywords"}
    assert {e.filename for e in load_manifest(DATA / "corpus.json")} == {p.name for p in (DATA / "corpus").iterdir()}
    versions = {e.filename: e.version for e in load_manifest(DATA / "corpus.json")}
    assert versions["employee_handbook_v1.md"] == "1.0" and versions["employee_handbook_v2.md"] == "2.0"


def test_every_shipped_label_resolves_to_indexed_chunks():
    qs = load_questions(DATA / "questions.jsonl")
    with evaluator.build_pipeline("offline", DATA / "corpus", load_manifest(DATA / "corpus.json")) as p:
        for q in qs:
            ids = resolve_relevant(q, p.chunks)       # raises DatasetError if a label matches nothing
            assert bool(ids) == bool(q.relevant)


def test_dataset_validation_errors(tmp_path):
    def write(*rows):
        f = tmp_path / "q.jsonl"
        f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        return f

    base = {"id": "a", "category": "direct", "question": "q?", "expected_status": "answered"}
    with pytest.raises(DatasetError, match="needs relevance labels"):
        load_questions(write(base))
    with pytest.raises(DatasetError, match="must not have"):
        load_questions(write({**base, "category": "unanswerable", "expected_status": "not_found", "relevant": [{"filename": "x", "contains": "y"}]}))
    ok = {**base, "relevant": [{"filename": "x", "contains": "y"}]}
    with pytest.raises(DatasetError, match="duplicate"):
        load_questions(write(ok, ok))
    with pytest.raises(DatasetError):
        load_questions(write({**ok, "category": "nonsense"}))


def test_unresolvable_label_is_a_dataset_error():
    q = EvalQuestion(id="x", category="direct", question="q", expected_status="answered",
                     relevant=[{"filename": "employee_handbook_v1.md", "contains": "text that is not there"}])
    with evaluator.build_pipeline("offline", DATA / "corpus", load_manifest(DATA / "corpus.json")) as p:
        with pytest.raises(DatasetError, match="matches no indexed chunk"):
            resolve_relevant(q, p.chunks)


def test_normalize():
    assert normalize("  A   b\n C ") == "a b c"


# ---- answer / citation metrics ----------------------------------------------------------
def answer(text, status="answered", cites=()):
    return Answer(status=status, text=text, citations=list(cites), confidence="high", conflicts=ConflictReport(has_conflict=False))


def chunk(cid="d:0", filename="a.md", text="Employees receive 20 days of annual leave per year.", section="Leave", page=-1):
    return Chunk(text=text, metadata=ChunkMetadata(chunk_id=cid, doc_id="d", filename=filename, title="T", section=section,
                                                   page=page, version="", source="s", doc_type="md", ordinal=0))


def cite(c, **kw):
    m = c.metadata
    base = dict(chunk_id=m.chunk_id, filename=m.filename, title=m.title, section=m.section, page=m.page, version=m.version,
                excerpt=c.text, source=m.filename)
    return Citation(**{**base, **kw})


def test_key_fact_recall():
    assert answer_metrics.key_fact_recall("It is 20 days and $50 per day", ["20 days", "$50", "missing"]) == pytest.approx(2 / 3)
    assert answer_metrics.key_fact_recall("anything", []) is None


def test_groundedness_proxy():
    c = chunk()
    good = answer("Employees receive 20 days of annual leave per year [1].", cites=[cite(c)])
    bad = answer("The company provides free lunch every Friday for everybody [1].", cites=[cite(c)])
    assert answer_metrics.groundedness(good, c.text) == 1.0
    assert answer_metrics.groundedness(bad, c.text) == 0.0
    assert answer_metrics.groundedness(answer("nothing", status="not_found"), c.text) is None


def test_status_and_conflict_metrics():
    q = EvalQuestion(id="q", category="conflict", question="q", expected_status="conflict", relevant=[{"filename": "a", "contains": "b"}])
    a1, a2 = chunk("d:0", "v1.md"), chunk("e:0", "v2.md")
    both = answer("v1 20 [1]; v2 25 [2]", "conflict", [cite(a1), cite(a2)])
    assert answer_metrics.status_ok(both, q) and answer_metrics.conflict_surfaced(both, q) is True
    one = answer("only v1 [1]", "conflict", [cite(a1)])
    assert answer_metrics.conflict_surfaced(one, q) is False
    assert not answer_metrics.status_ok(answer("x", "answered"), q)
    other = EvalQuestion(id="o", category="direct", question="q", expected_status="answered", relevant=[{"filename": "a", "contains": "b"}])
    assert answer_metrics.conflict_surfaced(both, other) is None


def test_citation_metrics_validity_precision_and_tamper_detection():
    c, other = chunk("d:0"), chunk("e:0", text="Unrelated passage text here.")
    retrieved = {"d:0": c, "e:0": other}
    ok = citation_metrics.check(answer("x [1][2]", cites=[cite(c), cite(other)]), retrieved, {"d:0"})
    assert ok == {"n": 2, "valid": 2, "location_correct": 2, "on_target": 1, "hit": True}
    bad = citation_metrics.check(answer("x", cites=[cite(c, excerpt="invented"), cite(c, page=9), cite(chunk("zz:0"))]), retrieved, {"d:0"})
    assert bad["n"] == 3 and bad["valid"] == 0 and bad["location_correct"] == 1 and bad["on_target"] == 2
    assert citation_metrics.check(answer("x", status="not_found"), retrieved, {"d:0"})["hit"] is False


# ---- the runner -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def offline_result():
    qs = load_questions(DATA / "questions.jsonl")
    with evaluator.build_pipeline("offline", DATA / "corpus", load_manifest(DATA / "corpus.json")) as p:
        return evaluator.run(p, qs, top_k=5)


def test_offline_run_structure_and_honest_labelling(offline_result):
    r = offline_result
    assert r["mode"] == "offline" and r["answers"] is None and r["citations"] is None
    assert r["n_questions"] == 28 and r["n_documents"] == 8 and r["reliability"]["failed_queries"] == 0
    assert "none" in r["llm_model"] and "offline" in r["embed_model"]
    assert r["retrieval"]["overall"]["recall@5"] >= 0.8 and r["retrieval"]["overall"]["n"] == 21
    assert set(r["retrieval"]["by_category"]) == {"direct", "paraphrase", "multi_document", "conflict", "similar_keywords"}


def test_offline_run_conflict_behaviour_on_the_synthetic_versions(offline_result):
    by_id = {q["id"]: q for q in offline_result["questions"]}
    assert by_id["q15"]["retrieval"]["conflicts"]["has_conflict"]            # 20 vs 25 days
    # q01 (sick leave): the two versions agree, so the identical sick-leave chunks must not be flagged. The
    # annual-leave chunks (20 vs 25 days) are also retrieved and DO differ; that residual is a known limit.
    sick_ids = {t["chunk_id"] for t in by_id["q01"]["retrieval"]["top"] if t["section"].endswith("Sick Leave")}
    assert sick_ids and not any(set(g["chunk_ids"]) & sick_ids for g in by_id["q01"]["retrieval"]["conflicts"]["groups"])
    kinds = {g["kind"] for g in by_id["q15"]["retrieval"]["conflicts"]["groups"]}
    assert "version_difference" in kinds


def test_evaluation_never_touches_the_users_collection(settings, tmp_path, monkeypatch):
    user_dir = tmp_path / "user_chroma"
    from app.vectorstore.chroma_store import ChromaStore
    s = Settings(bedrock_embed_dim=256, chroma_path=str(user_dir), chroma_collection="enterprise_docs")
    store = ChromaStore(s)
    store.count()
    before = {p.name for p in user_dir.iterdir()}
    with evaluator.build_pipeline("offline", DATA / "corpus", load_manifest(DATA / "corpus.json"), base=s):
        pass
    assert store.count() == 0 and {p.name for p in user_dir.iterdir()} == before
    store.close()


class EchoLLM:
    """Fake Nova: answers from the first evidence passage and cites it."""

    def converse(self, system, messages, *, max_tokens, temperature):
        prompt = messages[0]["content"]
        m = re.search(r"\[E1\][^\n]*\n([^\n]+)", prompt)
        passage = m.group(1).strip() if m else "unknown"
        first = re.split(r"(?<=[.!?])\s", passage)[0]
        return json.dumps({"status": "answered", "answer": f"{first} [E1]", "claims": [{"text": first, "evidence": ["E1"]}]})


def test_live_style_run_with_injected_embedder_and_llm_computes_answer_and_citation_metrics():
    qs = load_questions(DATA / "questions.jsonl")
    with evaluator.build_pipeline("live", DATA / "corpus", load_manifest(DATA / "corpus.json"),
                                  base=Settings(bedrock_embed_dim=256), embedder=KeywordEmbedder(256), llm=EchoLLM()) as p:
        r = evaluator.run(p, qs, top_k=5)
    assert r["mode"] == "live" and r["reliability"]["failed_queries"] == 0
    a, c = r["answers"], r["citations"]
    assert a["n"] == 28 and 0 <= a["status_accuracy"] <= 1 and a["groundedness"] is not None
    assert c["validity"] == 1.0 and c["location_correct"] == 1.0 and c["n_citations"] > 0
    assert any(f["reason"] == "wrong_status" for f in r["failure_cases"])    # the echo LLM never abstains/conflicts


def test_generation_or_retrieval_errors_are_counted_not_raised():
    class Boom:
        def converse(self, *a, **k):
            raise RuntimeError("model exploded")

    qs = load_questions(DATA / "questions.jsonl")[:3]
    with evaluator.build_pipeline("live", DATA / "corpus", load_manifest(DATA / "corpus.json"),
                                  base=Settings(bedrock_embed_dim=256), embedder=KeywordEmbedder(256), llm=Boom()) as p:
        r = evaluator.run(p, qs, 5)
    # the LLM is only called when retrieval found evidence, so count exactly those questions
    called = [q for q in r["questions"] if q["retrieval"]["evidence_strength"] != "none"]
    assert called and r["reliability"]["failed_queries"] == len(called)
    assert r["reliability"]["error_rate"] == pytest.approx(len(called) / 3)
    assert all("model exploded" in f["detail"] for f in r["failure_cases"] if f["reason"] == "error")


def test_setup_failure_is_reported():
    from app.evaluation.dataset import CorpusEntry
    with pytest.raises(evaluator.EvaluationSetupError, match="missing.md"):
        with evaluator.build_pipeline("offline", DATA / "corpus", [CorpusEntry(filename="missing.md")]):
            pass


def test_invalid_mode():
    with pytest.raises(ValueError):
        with evaluator.build_pipeline("bogus", DATA / "corpus", []):
            pass


# ---- reports --------------------------------------------------------------------------
def test_offline_report_is_clearly_labelled_and_has_five_plus_examples(offline_result, tmp_path):
    md_path, js_path = write_reports(json.loads(json.dumps(offline_result)), tmp_path / "out")
    md = md_path.read_text(encoding="utf-8")
    assert "OFFLINE DRY RUN" in md and "NOT representative" in md and "Mode: LIVE" not in md
    assert len(re.findall(r"^### Example \d+", md, re.M)) >= 5
    for section in ("## 1. Dataset", "## 2. Methodology", "## 3. Results", "## 4. Evidence-backed examples",
                    "## 5. Failure cases", "## 6. Human review (not automatically scored)", "## 7. Limitations", "## 8. Recommendations"):
        assert section in md
    assert "Not measured" in md and "automatic proxies" in md
    saved = json.loads(js_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "offline" and saved["report_path"].endswith("evaluation_report.md") and len(saved["examples"]) >= 5


def test_live_report_has_answer_tables(tmp_path):
    qs = load_questions(DATA / "questions.jsonl")
    with evaluator.build_pipeline("live", DATA / "corpus", load_manifest(DATA / "corpus.json"),
                                  base=Settings(bedrock_embed_dim=256), embedder=KeywordEmbedder(256), llm=EchoLLM()) as p:
        res = evaluator.run(p, qs, 5)
    md = render_markdown(res)
    assert "Mode: LIVE" in md and "Groundedness proxy" in md and "Citations: validity" in md and "OFFLINE DRY RUN" not in md


def test_run_evaluation_script(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_evaluation
    assert run_evaluation.main(["--out", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r" / "eval_results.json").exists() and (tmp_path / "r" / "evaluation_report.md").exists()

    def boom(*a, **k):
        raise evaluator.EvaluationSetupError("could not index x: No AWS credentials found")

    monkeypatch.setattr(run_evaluation.evaluator, "build_pipeline", boom)
    assert run_evaluation.main(["--live", "--out", str(tmp_path / "r2")]) == 1
    assert not (tmp_path / "r2").exists()          # no report is written when the run could not happen


# ---- methodology v2: groundedness proxy and failure severity --------------------------------------
def test_groundedness_v2_ignores_question_echo_but_checks_numbers():
    c = chunk(text="A lost or stolen laptop must be reported to the IT help desk within 24 hours.")
    q = "How long can I wait before telling IT that my notebook computer is missing?"
    good = answer("You can wait up to 24 hours before telling IT that your notebook computer is missing [1].", cites=[cite(c)])
    assert answer_metrics.groundedness(good, c.text) == 0.0                     # run-1 proxy: false alarm
    assert answer_metrics.groundedness_v2(good, c.text, q) == 1.0              # v2: only the number is a claim, and it is cited
    wrong = answer("You can wait up to 48 hours before telling IT that your notebook computer is missing [1].", cites=[cite(c)])
    assert answer_metrics.groundedness_v2(wrong, c.text, q) == 0.0             # an unsupported number is caught
    assert answer_metrics.groundedness_v2(answer("x", "not_found"), c.text, q) is None


def test_groundedness_v2_short_remainder_relies_on_numbers_not_connective_words():
    c = chunk(text="Unused annual leave may be carried over for up to 10 days.")
    ans = answer("The number of days differs between versions: 10 days in version 2.0 [1].", "conflict", [cite(c)])
    assert answer_metrics.groundedness_v2(ans, c.text + " version 2.0", "How many days of unused leave can be carried over?") == 1.0
    bad = answer("The number of days differs between versions: 12 days in version 2.0 [1].", "conflict", [cite(c)])
    assert answer_metrics.groundedness_v2(bad, c.text + " version 2.0", "How many days of unused leave can be carried over?") == 0.0


def test_groundedness_v2_conflict_framing_is_not_penalised():
    a = chunk("a:0", text="Full-time employees receive 20 days of annual leave per year.", filename="v1.md")
    b = chunk("b:0", text="Full-time employees receive 25 days of annual leave per year.", filename="v2.md")
    text = ("The number of days differs between versions of the Employee Handbook. According to version 1.0 [1], full-time "
            "employees receive 20 days. Version 2.0 [2] states 25 days.")
    ans = answer(text, "conflict", [cite(a), cite(b)])
    support = a.text + " " + b.text + " version 1.0 version 2.0"      # the model is shown the version labels too
    assert answer_metrics.groundedness_v2(ans, support, "How many days of annual leave do employees get?") == 1.0
    uncited_number = answer("Full-time employees receive 30 days [1].", "answered", [cite(a)])
    assert answer_metrics.groundedness_v2(uncited_number, a.text, "q") == 0.0


def test_failure_severity_and_counts_distinguish_defects_from_notes(offline_result):
    fc, counts = offline_result["failure_cases"], offline_result["failure_counts"]
    assert all(f["severity"] in ("failure", "review", "note") for f in fc)
    assert counts["entries"] == len(fc) and counts["questions_flagged"] <= counts["entries"]
    assert sum(counts["by_severity"].values()) == len(fc) and counts["n_questions"] == 28
    assert all(f["severity"] == "note" for f in fc if f["reason"] == "no_retrieval_abstention")
