from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from app.api.container import Container
from app.api.main import create_app
from app.generation import AnswerGenerator, GenerationSettings
from frontend import api_client
from frontend.api_client import ApiClient, ApiConnectionError, ApiError
from frontend.components.document_manager import validate_upload
from tests.test_api_query import GOOD
from tests.test_generation import ScriptedLLM
from tests.test_retriever import HANDBOOK

ROOT = Path(__file__).resolve().parents[1]


# ---- ApiClient against the real FastAPI app ----------------------------------------------
@pytest.fixture
def live(settings, store, kw_embedder):
    gen = AnswerGenerator(ScriptedLLM(GOOD), GenerationSettings(context_min_score=0.2))
    app = create_app(Container(settings, store=store, embedder=kw_embedder, generator=gen))
    http = TestClient(app, raise_server_exceptions=False)
    yield ApiClient("http://testserver", http), settings
    http.close()


def test_client_full_workflow_through_backend(live):
    client, _ = live
    assert client.health()["status"] == "ok" and client.documents() == []
    res = client.ingest([("handbook.md", HANDBOOK, "text/markdown")], version="2")
    assert res["documents"][0]["version"] == "2" and res["failures"] == []
    assert client.health()["chroma_count"] == res["chunks_indexed"]
    resp = client.query("How many days of annual leave do employees get?", top_k=3)
    assert resp["answer"]["status"] == "answered" and resp["retrieved"]
    doc = client.documents()[0]
    assert client.query("annual leave", doc_ids=[doc["doc_id"]])["answer"]["status"] in {"answered", "not_found"}
    assert client.delete_document(doc["doc_id"])["deleted_chunks"] == doc["chunk_count"]


def test_client_surfaces_backend_errors_readably(live):
    client, _ = live
    with pytest.raises(ApiError) as ei:
        client.ingest([("bad.docx", b"x", "application/octet-stream")])
    assert ei.value.status == 400 and ei.value.code == "unsupported_format" and "bad.docx" in ei.value.message
    with pytest.raises(ApiError) as ei:
        client.query("?!")
    assert ei.value.code == "invalid_query"


def test_client_evaluation_endpoint(live):
    client, settings = live
    assert client.latest_evaluation() is None
    out = Path(settings.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_results.json").write_text('{"mode": "live", "n_questions": 3}', encoding="utf-8")
    assert client.latest_evaluation() == {"mode": "live", "n_questions": 3}
    (out / "eval_results.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ApiError) as ei:
        client.latest_evaluation()
    assert ei.value.code == "evaluation_unreadable"


def test_client_evaluation_report_endpoint(live):
    client, settings = live
    assert client.latest_report() is None
    out = Path(settings.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation_report.md").write_text("# Evaluation Report\nbody", encoding="utf-8")
    assert client.latest_report() == "# Evaluation Report\nbody"


def test_client_connection_failures_are_friendly():
    def refuse(request):
        raise httpx.ConnectError("refused")

    c = ApiClient("http://localhost:9", httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(ApiConnectionError, match="Is FastAPI running"):
        c.health()

    def slow(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ApiConnectionError, match="in time"):
        ApiClient("http://x", httpx.Client(transport=httpx.MockTransport(slow))).health()


def test_client_handles_non_json_errors():
    c = ApiClient("http://x", httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500, text="boom"))))
    with pytest.raises(ApiError) as ei:
        c.health()
    assert ei.value.status == 500 and ei.value.message == "boom"


def test_client_does_not_touch_aws_or_chroma():
    import frontend.api_client as m
    src = Path(m.__file__).read_text(encoding="utf-8")
    assert "boto3" not in src and "chromadb" not in src


@pytest.mark.parametrize("name,size,ok", [("a.pdf", 10, True), ("A.MD", 10, True), ("a.exe", 10, False),
                                          ("noext", 10, False), ("a.txt", 0, False), ("a.txt", 26 * 1024 * 1024, False)])
def test_validate_upload(name, size, ok):
    assert (validate_upload(name, size) is None) is ok


# ---- Streamlit pages with a stub backend --------------------------------------------------
DOC = {"doc_id": "doc_1", "filename": "handbook.md", "title": "Leave Policy", "version": "2", "doc_type": "md", "chunk_count": 4}
CIT = {"chunk_id": "doc_1:v2:00000", "filename": "handbook.md", "title": "Leave Policy", "section": "Annual Leave", "page": 3,
       "version": "2", "excerpt": "Employees receive 20 days of annual leave per year.", "source": "handbook.md#Annual Leave/p.3"}
NO_CONFLICT = {"has_conflict": False, "groups": []}


def answer(status="answered", text="Employees receive 20 days [1].", citations=(CIT,), conflicts=NO_CONFLICT, **kw):
    return {"answer": {"status": status, "text": text, "citations": list(citations), "confidence": "high",
                       "conflicts": conflicts, "clarifying_question": kw.get("clarifying_question"), "warnings": kw.get("warnings", [])},
            "retrieved": [{"rank": 1, "score": 0.9, "vector_score": 0.77, "chunk": {"text": "t", "metadata": {
                "filename": "handbook.md", "section": "Annual Leave", "page": 3, "version": "2"}}}],
            "latency_ms": {"retrieve": 100, "generate": 900, "total": 1000}}


class FakeClient:
    base_url = "http://fake:8000"

    def __init__(self, **kw):
        self.docs, self.reply, self.evaluation = kw.get("docs", [DOC]), kw.get("reply", answer()), kw.get("evaluation")
        self.fail = kw.get("fail")
        self.report = kw.get("report")
        self.calls = []

    def _maybe_fail(self):
        if self.fail:
            raise self.fail

    def health(self):
        self._maybe_fail()
        return {"status": "ok", "chroma_count": 4, "embed_model": "amazon.titan-embed-text-v2:0", "llm_model": "amazon.nova-lite-v1:0"}

    def documents(self):
        self._maybe_fail()
        return list(self.docs)

    def query(self, question, **kw):
        self._maybe_fail()
        self.calls.append((question, kw))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def delete_document(self, doc_id):
        self.calls.append(("delete", doc_id))
        self.docs = [d for d in self.docs if d["doc_id"] != doc_id]
        return {"deleted_chunks": 4}

    def latest_evaluation(self):
        self._maybe_fail()
        return self.evaluation

    def latest_report(self):
        self._maybe_fail()
        return self.report


def page(path, fake):
    api_client._client = fake
    at = AppTest.from_file(str(ROOT / path), default_timeout=30)
    return at.run()


@pytest.fixture(autouse=True)
def reset_client():
    yield
    api_client._client = None


def texts(elements):
    return " ".join(str(getattr(e, "value", "")) for e in elements)


def test_overview_shows_real_backend_numbers():
    at = page("frontend/app.py", FakeClient())
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Indexed documents"] == "1" and metrics["Indexed chunks"] == "4" and metrics["Backend"] == "OK"
    assert "amazon.titan-embed-text-v2:0" in texts(at.markdown) and "amazon.nova-lite-v1:0" in texts(at.markdown)


def test_overview_when_backend_is_down_degrades_gracefully():
    at = page("frontend/app.py", FakeClient(fail=ApiConnectionError("Cannot reach the backend at http://fake:8000.")))
    assert not at.exception and any("Cannot reach the backend" in e.value for e in at.error)


def test_ask_page_sends_question_and_renders_answer_citations_and_ranking():
    fake = FakeClient()
    at = page("frontend/pages/2_Ask_Documents.py", fake)
    at.chat_input[0].set_value("How many days of leave?").run()
    assert not at.exception
    q, kw = fake.calls[0]
    assert q == "How many days of leave?" and kw["top_k"] == 5 and kw["debug"] is True
    body = texts(at.markdown)
    assert "Employees receive 20 days [1]." in body and "Confidence: high" in body
    labels = [e.label for e in at.expander]
    assert any(l.startswith("[1] handbook.md — Annual Leave · p.3 · v2") for l in labels)
    assert any(l.startswith("Retrieved evidence") for l in labels)
    assert "Employees receive 20 days of annual leave per year." in body            # excerpt
    # history survives the next interaction
    at.chat_input[0].set_value("Second question").run()
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]


def test_ask_page_not_found_and_clarification_and_conflict():
    nf = page("frontend/pages/2_Ask_Documents.py", FakeClient(reply=answer("not_found", "I could not find enough supporting evidence.", [])))
    nf.chat_input[0].set_value("unknown?").run()
    assert any("Not enough evidence" in w.value for w in nf.warning)

    cl = page("frontend/pages/2_Ask_Documents.py", FakeClient(reply=answer("needs_clarification", "", [], clarifying_question="Which version?")))
    cl.chat_input[0].set_value("leave?").run()
    assert any("Which version?" in i.value for i in cl.info)

    conflict = {"has_conflict": True, "groups": [{"topic": "leave policy", "kind": "version_difference", "chunk_ids": [CIT["chunk_id"]],
                                                  "description": "METADATA CONFLICT: versions 1 and 2."}]}
    cf = page("frontend/pages/2_Ask_Documents.py", FakeClient(reply=answer("conflict", "v1 says 20, v2 says 25 [1].", conflicts=conflict)))
    cf.chat_input[0].set_value("leave days?").run()
    assert any("sources may disagree" in w.value for w in cf.warning)
    assert any("Version difference: leave policy" in e.label for e in cf.expander)


def test_ask_page_shows_backend_errors_without_crashing():
    err = ApiError(502, "bedrock_error", "AWS is still verifying this account")
    at = page("frontend/pages/2_Ask_Documents.py", FakeClient(reply=err))
    at.chat_input[0].set_value("anything").run()
    assert not at.exception and any("still verifying" in e.value for e in at.error)
    down = page("frontend/pages/2_Ask_Documents.py", FakeClient(reply=ApiConnectionError("Cannot reach the backend")))
    down.chat_input[0].set_value("anything").run()
    assert not down.exception and any("Cannot reach" in e.value for e in down.error)


def test_ask_page_with_no_documents_shows_hint():
    at = page("frontend/pages/2_Ask_Documents.py", FakeClient(docs=[]))
    assert any("No documents are indexed yet" in i.value for i in at.info)


def test_documents_page_lists_and_deletes():
    fake = FakeClient()
    at = page("frontend/pages/1_Documents.py", fake)
    assert not at.exception and "handbook.md" in texts(at.markdown)
    at.button(key="del_doc_1").click().run()
    assert ("delete", "doc_1") in fake.calls and any("Deleted handbook.md" in s.value for s in at.success)


def test_evaluation_page_empty_and_with_results():
    empty = page("frontend/pages/3_Evaluation.py", FakeClient())
    assert any("No evaluation results" in i.value for i in empty.info)
    result = {"mode": "offline", "n_questions": 2, "n_documents": 3,
              "retrieval": {"overall": {"recall@5": 0.5, "mrr": 0.25, "ndcg@5": 0.3, "hit_rate@5": 0.5, "n": 2}, "by_category": {}},
              "answers": None, "citations": None, "reliability": {"failed_queries": 0, "empty_retrieval_rate": 0.0, "error_rate": 0.0},
              "failure_cases": [{"id": "q1", "category": "direct", "question": "Q?", "reason": "miss", "detail": "", "severity": "failure"},
                               {"id": "q2", "category": "unanswerable", "question": "Z?", "reason": "no_retrieval_abstention", "detail": "", "severity": "note"}],
              "failure_counts": {"entries": 2, "questions_flagged": 2, "questions_with_failure": 1, "n_questions": 2,
                                 "by_severity": {"failure": 1, "review": 0, "note": 1}, "by_reason": {}}}
    at = page("frontend/pages/3_Evaluation.py", FakeClient(evaluation=result, report="# Evaluation Report\n\nHello"))
    assert not at.exception
    assert any("Offline dry run" in w.value for w in at.warning)
    m = {x.label: x.value for x in at.metric}
    assert m["Recall@5"] == "0.500" and m["MRR"] == "0.250"
    assert any("not measured" in i.value for i in at.info)
    assert any("2 entries on 2 of 2 questions" in i.value and "0 queries actually failed" in i.value for i in at.info)
    labels = [e.label for e in at.expander]
    assert any(l.startswith("🔴 failure · q1") for l in labels) and any(l.startswith("⚪ note · q2") for l in labels)
    assert any("Full evaluation report" in l for l in labels) and "Hello" in texts(at.markdown)


def test_status_page():
    at = page("frontend/pages/4_System_Status.py", FakeClient())
    assert not at.exception
    m = {x.label: x.value for x in at.metric}
    assert m["FastAPI"] == "OK" and m["ChromaDB chunks"] == "4"
    assert not any("secret" in str(e.value).lower() for e in at.markdown)
    down = page("frontend/pages/4_System_Status.py", FakeClient(fail=ApiConnectionError("Cannot reach the backend")))
    assert not down.exception and any("Cannot reach" in e.value for e in down.error)
