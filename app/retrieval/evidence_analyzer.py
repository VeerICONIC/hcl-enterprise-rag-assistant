"""Deterministic evidence-strength assessment. No LLM; cosine similarity is NOT a probability.

A returned chunk is **relevant** when either
  * its cosine similarity >= ``min_relevance`` (MIN_RELEVANCE), or
  * lexical rescue: it contains >= ``lexical_rescue_coverage`` (default 80%) of the query's
    distinct content terms AND the query has >= 2 such terms (catches exact identifiers such as
    error codes that embeddings can miss; never fires on one-word queries).

Strength:
  * ``none``   no relevant chunk (including: no results at all).
  * ``strong`` best cosine >= min_relevance + ``strong_margin`` (0.15) AND
               (>= 2 relevant chunks OR best cosine >= min_relevance + ``very_strong_margin`` (0.30)).
  * ``weak``   otherwise (at least one relevant chunk, but thin or only marginally similar).

Also reported (for logging and M3, not part of the contract): number of relevant chunks, distinct
source documents, and whether all relevant evidence comes from one document ("concentrated").
Concentration does not change the label: one authoritative document can be sufficient.

The thresholds are heuristics chosen before any live Titan data exists. They MUST be calibrated on
the M5 evaluation set; until then treat them as provisional.
"""

from __future__ import annotations

from typing import Sequence

from app.retrieval.models import EvidenceAssessment, RetrievalSettings
from app.schemas import RetrievedChunk


def analyze_evidence(chunks: Sequence[RetrievedChunk], coverage: dict[str, float],
                     settings: RetrievalSettings, query_term_count: int) -> EvidenceAssessment:
    if not chunks:
        return EvidenceAssessment("none", reasons=["no chunks retrieved"])

    relevant: list[RetrievedChunk] = []
    for c in chunks:
        by_vector = c.vector_score >= settings.min_relevance
        by_lexical = (query_term_count >= 2
                      and coverage.get(c.chunk.metadata.chunk_id, 0.0) >= settings.lexical_rescue_coverage)
        if by_vector or by_lexical:
            relevant.append(c)

    best = max(c.vector_score for c in chunks)
    docs = {c.chunk.metadata.doc_id for c in relevant}
    a = EvidenceAssessment(
        strength="none",
        relevant_ids={c.chunk.metadata.chunk_id for c in relevant},
        relevant_count=len(relevant),
        distinct_documents=len(docs),
        best_similarity=best,
        concentrated=len(docs) == 1 and len(relevant) > 1,
    )
    if not relevant:
        a.reasons.append(f"no chunk reached cosine {settings.min_relevance:.2f} or lexical rescue (best cosine {best:.2f})")
        return a

    strong_cut = settings.min_relevance + settings.strong_margin
    very_cut = settings.min_relevance + settings.very_strong_margin
    best_relevant = max(c.vector_score for c in relevant)
    if best_relevant >= strong_cut and (len(relevant) >= 2 or best_relevant >= very_cut):
        a.strength = "strong"
        a.reasons.append(f"{len(relevant)} relevant chunk(s), best cosine {best_relevant:.2f} >= {strong_cut:.2f}")
    else:
        a.strength = "weak"
        a.reasons.append(f"{len(relevant)} relevant chunk(s), best relevant cosine {best_relevant:.2f} "
                         f"(strong needs >= {strong_cut:.2f} with 2+ chunks, or >= {very_cut:.2f})")
    if a.concentrated:
        a.reasons.append("all relevant evidence comes from a single document")
    return a
