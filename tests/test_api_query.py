import json

import pytest
from fastapi.testclient import TestClient

from app.api.container import Container
from app.api.main import create_app
from app.core.bedrock import ThrottlingError
from app.generation import AnswerGenerator, GenerationSettings
from app.schemas import QueryResponse
from tests.test_generation import ScriptedLLM
from tests.test_retriever import HANDBOOK

GOOD = {"status": "answered", "answer": "Employees receive 20 days of annual leave per year [E1].",
        "claims": [{"text": "Employees receive 20 days of annual leave per year.", "evidence": ["E1"]}]}


def make_client(settings, store, kw_embedder, llm):
    gen = AnswerGenerator(llm, GenerationSettings(context_min_score=0.2))
    app = create_app(Container(settings, store=store, embedder=kw_embedder, generator=gen))
    return TestClient(app, raise_server_exceptions=False)


def ingest(c, name, data, **form):
    r = c.post("/ingest", files=[("files", (name, data, "text/plain"))], data=form)
    assert r.status_code == 200, r.text


def test_query_returns_grounded_answer_with_validated_citations(settings, store, kw_embedder):
    llm = ScriptedLLM(GOOD)
    with make_client(settings, store, kw_embedder, llm) as c:
        ingest(c, "handbook.md", HANDBOOK)
        r = c.post("/query", json={"question": "How many days of annual leave do employees get?"})
    assert r.status_code == 200
    body = QueryResponse.model_validate(r.json())
    ans = body.answer
    assert ans.status == "answered" and ans.text.endswith("[1].") and len(ans.citations) == 1
    cit = ans.citations[0]
    assert cit.filename == "handbook.md" and cit.section == "Annual Leave" and "20 days" in cit.excerpt
    assert set(body.latency_ms) == {"retrieve", "generate", "total"} and body.retrieved is None
    prompt = llm.calls[0]["messages"][0]["content"]
    assert "document: handbook.md" in prompt and "How many days of annual leave" in prompt


def test_debug_returns_ranked_retrieved_chunks(settings, store, kw_embedder):
    with make_client(settings, store, kw_embedder, ScriptedLLM(GOOD)) as c:
        ingest(c, "handbook.md", HANDBOOK)
        r = c.post("/query", json={"question": "annual leave days", "debug": True, "top_k": 2})
    got = r.json()["retrieved"]
    assert len(got) == 2 and got[0]["rank"] == 1 and "vector_score" in got[0]


def test_unsupported_question_gets_not_found_without_calling_the_llm(settings, store, kw_embedder):
    llm = ScriptedLLM(GOOD)
    with make_client(settings, store, kw_embedder, llm) as c:
        ingest(c, "handbook.md", HANDBOOK)
        r = c.post("/query", json={"question": "quantum chromodynamics lattice simulation"})
    assert r.json()["answer"]["status"] == "not_found" and r.json()["answer"]["citations"] == [] and llm.calls == []


def test_empty_index_is_not_found(settings, store, kw_embedder):
    with make_client(settings, store, kw_embedder, ScriptedLLM(GOOD)) as c:
        r = c.post("/query", json={"question": "anything"})
    assert r.status_code == 200 and r.json()["answer"]["status"] == "not_found"


def test_conflicting_versions_are_surfaced(settings, store, kw_embedder):
    reply = {"status": "conflict", "answer": "v1 says 20 days [E1]; v2 says 25 days [E2].",
             "claims": [{"text": "v1 says 20 days.", "evidence": ["E1"]}, {"text": "v2 says 25 days.", "evidence": ["E2"]}]}
    with make_client(settings, store, kw_embedder, ScriptedLLM(reply)) as c:
        ingest(c, "leave_policy_v1.md", b"# Leave Policy\n\n## Annual Leave\n\nEmployees receive 20 days of annual leave per year.", version="1")
        ingest(c, "leave_policy_v2.md", b"# Leave Policy\n\n## Annual Leave\n\nEmployees receive 25 days of annual leave per year.", version="2")
        r = c.post("/query", json={"question": "How many days of annual leave do employees receive?"})
    ans = r.json()["answer"]
    assert ans["status"] == "conflict" and ans["conflicts"]["has_conflict"]
    assert {x["filename"] for x in ans["citations"]} == {"leave_policy_v1.md", "leave_policy_v2.md"}
    assert {x["version"] for x in ans["citations"]} == {"1", "2"}


def test_invented_citation_is_rejected_end_to_end(settings, store, kw_embedder):
    bad = {"status": "answered", "answer": "It is 20 days [E9].", "claims": [{"text": "It is 20 days.", "evidence": ["E9"]}]}
    with make_client(settings, store, kw_embedder, ScriptedLLM(bad)) as c:
        ingest(c, "handbook.md", HANDBOOK)
        r = c.post("/query", json={"question": "How many days of annual leave do employees get?"})
    assert r.json()["answer"]["status"] == "not_found" and r.json()["answer"]["citations"] == []


def test_bedrock_failure_is_502(settings, store, kw_embedder):
    with make_client(settings, store, kw_embedder, ScriptedLLM(ThrottlingError("Bedrock throttled the request"))) as c:
        ingest(c, "handbook.md", HANDBOOK)
        r = c.post("/query", json={"question": "How many days of annual leave do employees get?"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "bedrock_error" and "throttled" in r.json()["error"]["message"]


def test_unusable_model_output_is_502_generation_failed(settings, store, kw_embedder):
    with make_client(settings, store, kw_embedder, ScriptedLLM("garbage", "more garbage")) as c:
        ingest(c, "handbook.md", HANDBOOK)
        r = c.post("/query", json={"question": "How many days of annual leave do employees get?"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "generation_failed" and "Traceback" not in r.text


@pytest.mark.parametrize("payload,status", [({}, 422), ({"question": ""}, 422), ({"question": "q", "top_k": 0}, 422), ({"question": "?!"}, 400)])
def test_invalid_query_requests(settings, store, kw_embedder, payload, status):
    with make_client(settings, store, kw_embedder, ScriptedLLM(GOOD)) as c:
        assert c.post("/query", json=payload).status_code == status


def test_existing_routes_unaffected_by_query_wiring(settings, store, kw_embedder):
    with make_client(settings, store, kw_embedder, ScriptedLLM(GOOD)) as c:
        ingest(c, "n.txt", b"Some note text.")
        assert c.get("/health").status_code == 200 and len(c.get("/documents").json()) == 1
        assert c.post("/retrieve", json={"question": "note"}).status_code == 200


def test_query_builds_no_aws_client_until_used(monkeypatch):
    import boto3
    monkeypatch.setattr(boto3, "Session", lambda *a, **k: (_ for _ in ()).throw(AssertionError("AWS touched")))
    c = Container()
    create_app(c)
    assert c._generator is None and c._query_service is None
