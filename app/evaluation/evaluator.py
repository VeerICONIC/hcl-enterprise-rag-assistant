"""Evaluation runner: index the synthetic corpus into a TEMPORARY Chroma store (the user's own collection
is never touched), run every labelled question through the real Retriever (and AnswerGenerator in live
mode), and compute retrieval, answer, citation and reliability metrics.

Modes
-----
``live``    real Titan embeddings + real Nova Lite answers (needs working Bedrock access; billable).
``offline`` a hashed bag-of-words stand-in for the embedder and NO generation. Only the retrieval
            machinery is exercised; results must never be presented as real quality numbers.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from app.core.config import Settings, get_settings
from app.evaluation import answer_metrics, citation_metrics, retrieval_metrics
from app.evaluation.dataset import ANSWERABLE, CorpusEntry, EvalQuestion, resolve_relevant
from app.evaluation.offline import HashedBagOfWordsEmbedder
from app.generation import AnswerGenerator, BedrockNovaClient, GenerationSettings
from app.ingestion.ingestion_service import IngestionService
from app.retrieval import RetrievalSettings, Retriever
from app.schemas import Chunk
from app.vectorstore.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

EVAL_CHUNK_SIZE, EVAL_CHUNK_OVERLAP = 400, 60  # small, so the short synthetic documents yield several chunks


class EvaluationSetupError(RuntimeError):
    pass


@dataclass
class EvalPipeline:
    mode: str
    retriever: Retriever
    generator: Optional[AnswerGenerator]
    chunks: list[Chunk]
    embed_model: str
    llm_model: str
    n_documents: int
    settings: Settings = field(repr=False)


@contextlib.contextmanager
def build_pipeline(mode: str, corpus_dir: Path, manifest: list[CorpusEntry],
                   base: Optional[Settings] = None, embedder=None, llm=None,
                   chunk_size: int = EVAL_CHUNK_SIZE, chunk_overlap: int = EVAL_CHUNK_OVERLAP) -> Iterator[EvalPipeline]:
    """Index the corpus and yield a ready pipeline. ``embedder``/``llm`` may be injected (tests)."""
    if mode not in ("live", "offline"):
        raise ValueError("mode must be 'live' or 'offline'")
    base = base or get_settings()
    with tempfile.TemporaryDirectory(prefix="eval_chroma_", ignore_cleanup_errors=True) as tmp:
        overrides = dict(chroma_path=str(Path(tmp) / "chroma"), chroma_collection="eval_corpus",
                         chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if mode == "offline":
            overrides.update(bedrock_embed_dim=256, bedrock_embed_model_id="amazon.titan-embed-text-v2:0")
        settings = Settings(**{**base.model_dump(), **overrides})
        if embedder is None:
            if mode == "offline":
                embedder = HashedBagOfWordsEmbedder(settings.bedrock_embed_dim)
            else:
                from app.embeddings.titan import TitanEmbedder
                embedder = TitanEmbedder(settings=settings)
        store = ChromaStore(settings)
        try:
            service = IngestionService(embedder, store, settings)
            for entry in manifest:
                try:
                    service.ingest_path(Path(corpus_dir) / entry.filename, version=entry.version or None)
                except Exception as exc:
                    raise EvaluationSetupError(f"could not index {entry.filename}: {exc}") from exc
            generator = None
            if mode == "live" or llm is not None:
                generator = AnswerGenerator(llm or BedrockNovaClient(settings=settings),
                                            GenerationSettings.from_settings(settings))
            yield EvalPipeline(
                mode=mode,
                retriever=Retriever(embedder, store, RetrievalSettings.from_settings(settings)),
                generator=generator, chunks=store.get_all_chunks(),
                embed_model=getattr(embedder, "model_id", settings.bedrock_embed_model_id),
                llm_model=settings.bedrock_llm_model_id if generator else "none (generation not run)",
                n_documents=len(manifest), settings=settings)
        finally:
            store.close()


def _snippet(text: str, n: int = 160) -> str:
    body = text.split("\n\n", 1)[-1] if text.startswith("Document: ") else text
    body = " ".join(body.split())
    return body if len(body) <= n else body[:n].rstrip() + "…"


def _mean(values: list) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def run_question(p: EvalPipeline, q: EvalQuestion, top_k: int) -> dict:
    relevant = resolve_relevant(q, p.chunks) if q.relevant else set()
    rec: dict = {"id": q.id, "category": q.category, "question": q.question, "expected_status": q.expected_status,
                 "accepted_status": sorted(q.accepted), "reference_answer": q.reference_answer,
                 "relevant_ids": sorted(relevant), "error": None, "retrieval": None, "answer": None,
                 "answer_metrics": None, "citation": None}
    t0 = time.perf_counter()
    try:
        retrieval = p.retriever.retrieve(q.question, top_k=top_k)
    except Exception as exc:
        rec["error"] = f"retrieval failed: {type(exc).__name__}: {exc}"
        return rec
    ranked = [c.chunk.metadata.chunk_id for c in retrieval.chunks]
    rec["retrieval"] = {
        "top_ids": ranked,
        "top": [{"rank": c.rank, "chunk_id": c.chunk.metadata.chunk_id, "filename": c.chunk.metadata.filename,
                 "section": c.chunk.metadata.section, "page": c.chunk.metadata.page,
                 "version": c.chunk.metadata.version, "cosine": round(c.vector_score, 4),
                 "fusion_score": round(c.score, 4), "snippet": _snippet(c.chunk.text),
                 "relevant": c.chunk.metadata.chunk_id in relevant} for c in retrieval.chunks],
        "evidence_strength": retrieval.evidence_strength, "top_score": round(retrieval.top_score, 4),
        "conflicts": {"has_conflict": retrieval.conflicts.has_conflict,
                      "groups": [{"kind": g.kind, "topic": g.topic, "chunk_ids": g.chunk_ids,
                                  "description": g.description} for g in retrieval.conflicts.groups]},
        "metrics": retrieval_metrics.per_question(ranked, relevant, top_k) if relevant else None,
    }
    if p.generator is not None:
        try:
            ans = p.generator.generate(retrieval)
        except Exception as exc:
            rec["error"] = f"generation failed: {type(exc).__name__}: {exc}"
        else:
            by_id = {c.chunk.metadata.chunk_id: c.chunk for c in retrieval.chunks}
            cited_text = " ".join(
                f"{by_id[c.chunk_id].text} version {by_id[c.chunk_id].metadata.version} {by_id[c.chunk_id].metadata.filename} "
                f"{by_id[c.chunk_id].metadata.section} {by_id[c.chunk_id].metadata.title}"
                for c in ans.citations if c.chunk_id in by_id)
            rec["answer"] = ans.model_dump()
            rec["answer_metrics"] = {
                "key_fact_recall": answer_metrics.key_fact_recall(ans.text, q.key_facts) if ans.status in ("answered", "conflict") else None,
                "groundedness": answer_metrics.groundedness_v2(ans, cited_text, q.question),
                "groundedness_v1": answer_metrics.groundedness(ans, cited_text),
                "status_ok": answer_metrics.status_ok(ans, q),
                "conflict_surfaced": answer_metrics.conflict_surfaced(ans, q)}
            rec["citation"] = citation_metrics.check(ans, by_id, relevant)
    rec["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return rec


def run(p: EvalPipeline, questions: list[EvalQuestion], top_k: int = 5) -> dict:
    """Run all questions and return the full results structure (also what the dashboard displays)."""
    results = [run_question(p, q, top_k) for q in questions]
    return summarize(p, results, top_k)


def summarize(p: EvalPipeline, results: list[dict], top_k: int) -> dict:
    ok = [r for r in results if r["retrieval"] is not None]
    answerable = [r for r in ok if r["category"] in ANSWERABLE and r["retrieval"]["metrics"]]
    by_cat: dict[str, dict] = {}
    for cat in dict.fromkeys(r["category"] for r in answerable):
        by_cat[cat] = retrieval_metrics.mean(r["retrieval"]["metrics"] for r in answerable if r["category"] == cat)
    conflict_qs = [r for r in ok if r["category"] == "conflict"]
    other_answerable = [r for r in ok if r["category"] in ANSWERABLE and r["category"] != "conflict"]
    abstain_qs = [r for r in ok if r["category"] in ("unanswerable", "ambiguous")]
    retrieval = {
        "k": top_k,
        "overall": retrieval_metrics.mean(r["retrieval"]["metrics"] for r in answerable),
        "by_category": by_cat,
        "conflict_detection": {
            "recall_on_conflict_questions": _mean([r["retrieval"]["conflicts"]["has_conflict"] for r in conflict_qs]),
            "false_positive_rate_on_other_answerable": _mean([r["retrieval"]["conflicts"]["has_conflict"] for r in other_answerable]),
            "n_conflict_questions": len(conflict_qs), "n_other_answerable": len(other_answerable)},
        "abstention": {
            "evidence_none_rate_on_unanswerable_or_ambiguous": _mean([r["retrieval"]["evidence_strength"] == "none" for r in abstain_qs]),
            "n": len(abstain_qs)},
    }

    answers = citations = None
    gen = [r for r in results if r["answer"] is not None]
    if p.generator is not None and gen:
        am = [r["answer_metrics"] for r in gen]
        by_status = {}
        for cat in dict.fromkeys(r["category"] for r in gen):
            by_status[cat] = _mean([r["answer_metrics"]["status_ok"] for r in gen if r["category"] == cat])
        answers = {"n": len(gen), "key_fact_recall": _mean([a["key_fact_recall"] for a in am]),
                   "groundedness": _mean([a["groundedness"] for a in am]),
                   "groundedness_v1": _mean([a["groundedness_v1"] for a in am]),
                   "status_accuracy": _mean([a["status_ok"] for a in am]),
                   "status_accuracy_by_category": by_status,
                   "conflict_surfaced_rate": _mean([a["conflict_surfaced"] for a in am])}
        total = sum(r["citation"]["n"] for r in gen)
        cited = [r for r in gen if r["answer"]["status"] in ("answered", "conflict") and r["relevant_ids"]]
        citations = {"n_citations": total,
                     "validity": (sum(r["citation"]["valid"] for r in gen) / total) if total else None,
                     "precision": (sum(r["citation"]["on_target"] for r in gen) / total) if total else None,
                     "location_correct": (sum(r["citation"]["location_correct"] for r in gen) / total) if total else None,
                     "recall": _mean([bool(r["citation"]["hit"]) for r in cited])}

    failed = [r for r in results if r["error"]]
    reliability = {"n_questions": len(results), "failed_queries": len(failed),
                   "error_rate": len(failed) / len(results) if results else None,
                   "empty_retrieval_rate": _mean([not r["retrieval"]["top_ids"] for r in ok])}
    return {
        "mode": p.mode, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embed_model": p.embed_model, "llm_model": p.llm_model, "n_questions": len(results),
        "n_documents": p.n_documents, "n_chunks": len(p.chunks), "top_k": top_k,
        "chunking": {"size": p.settings.chunk_size, "overlap": p.settings.chunk_overlap},
        "retrieval": retrieval, "answers": answers, "citations": citations, "reliability": reliability,
        "failure_cases": (fc := failure_cases(results)), "failure_counts": failure_counts(fc, len(results)),
        "questions": results,
    }


# severity: "failure" = a defect by definition; "review" = a proxy flag that needs a human look;
# "note" = informational (for example retrieval did not abstain but the answer stage correctly declined)
SEVERITY = {"error": "failure", "retrieval_miss": "failure", "conflict_not_detected": "failure", "false_conflict": "failure", "retrieval_conflict_not_shown": "note",
            "wrong_status": "failure", "missing_key_facts": "failure", "invalid_citation": "failure",
            "low_groundedness_proxy": "review", "no_retrieval_abstention": "note"}


def failure_counts(cases: list[dict], n_questions: int) -> dict:
    """Entries are not failed queries: one question can raise several entries, and many are notes."""
    sev = {k: sum(c["severity"] == k for c in cases) for k in ("failure", "review", "note")}
    return {"entries": len(cases), "questions_flagged": len({c["id"] for c in cases}),
            "questions_with_failure": len({c["id"] for c in cases if c["severity"] == "failure"}),
            "n_questions": n_questions, "by_severity": sev,
            "by_reason": {r: sum(c["reason"] == r for c in cases) for r in dict.fromkeys(c["reason"] for c in cases)}}


def failure_cases(results: list[dict]) -> list[dict]:
    out: list[dict] = []

    def add(r, reason, detail=""):
        out.append({"id": r["id"], "category": r["category"], "question": r["question"], "reason": reason,
                    "detail": detail, "severity": SEVERITY[reason]})

    for r in results:
        if r["error"]:
            add(r, "error", r["error"])
            continue
        ret, cat = r["retrieval"], r["category"]
        if ret["metrics"] and ret["metrics"][f"hit_rate@{len(ret['top_ids']) or 1}"] == 0.0:
            add(r, "retrieval_miss", "no labelled-relevant chunk in the top results")
        if cat == "conflict" and not ret["conflicts"]["has_conflict"]:
            add(r, "conflict_not_detected", "retrieval reported no conflict")
        if cat in ANSWERABLE and cat != "conflict" and ret["conflicts"]["has_conflict"]:
            topics = "; ".join(g["topic"] for g in ret["conflicts"]["groups"])
            shown = r["answer"]["conflicts"]["has_conflict"] if r["answer"] else True   # offline: no answer, retrieval-level only
            if shown:
                add(r, "false_conflict", topics)
            else:
                add(r, "retrieval_conflict_not_shown", f"retrieval flagged ({topics}) among passages the answer does not cite; "
                    "the answer shown to the user has no conflict")
        if cat in ("unanswerable", "ambiguous") and ret["evidence_strength"] != "none":
            add(r, "no_retrieval_abstention", f"evidence_strength={ret['evidence_strength']} (the answer stage may still decline)")
        am = r["answer_metrics"]
        if am:
            if not am["status_ok"]:
                add(r, "wrong_status", f"got {r['answer']['status']}, accepted {r['accepted_status']}")
            elif am["key_fact_recall"] is not None and am["key_fact_recall"] < 1.0:
                add(r, "missing_key_facts", f"key-fact recall {am['key_fact_recall']:.2f}")
            if am["groundedness"] is not None and am["groundedness"] < 1.0:
                add(r, "low_groundedness_proxy", f"{am['groundedness']:.2f}")
        cit = r["citation"]
        if cit and cit["valid"] < cit["n"]:
            add(r, "invalid_citation", f"{cit['n'] - cit['valid']} of {cit['n']}")
    return out
