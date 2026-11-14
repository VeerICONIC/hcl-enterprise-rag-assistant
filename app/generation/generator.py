"""Grounded answer generation.

Flow: no evidence -> deterministic ``not_found`` (the LLM is never called). Otherwise build a prompt
from the retrieved chunks only, call the LLM, parse its JSON, then rebuild everything that matters
from the real retrieved chunks:

* citations come from chunk metadata, never from model text (see ``citation_builder``);
* inline ``[E#]`` markers are validated against the evidence list and renumbered ``[1]``, ``[2]``;
* claims without a valid citation are dropped (and the answer rebuilt from the supported claims);
* an answer left with no valid citation is withheld and reported as ``not_found``;
* conflicts detected by retrieval always travel with the answer (``Answer.conflicts``) and force
  status ``conflict``; the answer never silently picks a side.

Depends only on the shared schemas and its own modules (not on ``app.retrieval``).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.generation import prompts
from app.generation.citation_builder import (
    build_citation, marker_aliases, renumber_markers, strip_aliases,
)
from app.generation.models import GenerationError, GenerationSettings, LLMClient, ModelOutput
from app.generation.response_validator import parse_model_output
from app.schemas import Answer, Citation, ConflictReport, RetrievalResult

logger = logging.getLogger(__name__)

_ALIAS_NUM = re.compile(r"^\W*[Ee]?\s*(\d+)\W*$")
_RECENCY = re.compile(r"\b(latest|newest|newer|older|previous|previously|outdated|superseded|obsolete|current version)\b", re.I)


def _norm_alias(alias: str) -> str:
    """Tolerate "E1", "e1", "[E1]", "1", "E1," in the model's evidence lists."""
    m = _ALIAS_NUM.match(alias)
    return f"E{int(m.group(1))}" if m else alias.strip().upper()


NOT_FOUND_TEXT = ("I could not find enough supporting evidence in the indexed documents to answer this "
                  "question. Try rephrasing it, or upload the relevant documents.")


class AnswerGenerator:
    def __init__(self, llm: LLMClient, settings: Optional[GenerationSettings] = None):
        self._llm = llm
        self.settings = settings or GenerationSettings()

    # ---- public ---------------------------------------------------------------------
    def generate(self, retrieval: RetrievalResult) -> Answer:
        """Return the contract's ``Answer``. LLM/AWS failures propagate (``BedrockError``,
        ``GenerationError``); they are never turned into a fabricated answer."""
        if retrieval.evidence_strength == "none" or not retrieval.chunks:
            return self._not_found(retrieval.conflicts, [])
        evidence = self._select(retrieval)
        if not evidence:
            return self._not_found(retrieval.conflicts, ["no retrieved passage was relevant enough to use"])
        user = prompts.build_user_prompt(
            retrieval.question, evidence, retrieval.conflicts,
            weak_evidence=retrieval.evidence_strength == "weak", max_chunk_chars=self.settings.max_chunk_chars)
        output = self._call(user)
        return self._finalize(retrieval, evidence, output)

    # ---- helpers --------------------------------------------------------------------
    def _select(self, retrieval: RetrievalResult) -> list[prompts.EvidenceItem]:
        s = self.settings
        in_conflict = {cid for g in retrieval.conflicts.groups for cid in g.chunk_ids}
        picked = [rc for rc in retrieval.chunks
                  if rc.vector_score >= s.context_min_score or rc.chunk.metadata.chunk_id in in_conflict]
        if retrieval.evidence_strength == "weak" or not picked:
            # Weak evidence may rest on a keyword match whose cosine is low (for example a multi-topic chunk), so
            # the relevant passage can rank below others; show every returned passage and let the prompt's
            # caution (and the model's insufficient_evidence option) do the filtering.
            picked = list(retrieval.chunks)
        picked = picked[:s.max_context_chunks]
        return [prompts.EvidenceItem(f"E{i}", rc.chunk) for i, rc in enumerate(picked, 1)]

    def _call(self, user_prompt: str) -> ModelOutput:
        s = self.settings
        messages = [{"role": "user", "content": user_prompt}]
        raw = self._llm.converse(prompts.SYSTEM_PROMPT, messages, max_tokens=s.max_tokens, temperature=s.temperature)
        try:
            return parse_model_output(raw)
        except GenerationError as first:
            logger.warning("model output unusable (%s); retrying once", first)
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": prompts.REPAIR_PROMPT}]
        raw2 = self._llm.converse(prompts.SYSTEM_PROMPT, messages, max_tokens=s.max_tokens, temperature=s.temperature)
        return parse_model_output(raw2)  # raises GenerationError if still unusable

    @staticmethod
    def _not_found(conflicts: ConflictReport, warnings: list[str], text: str = "") -> Answer:
        return Answer(status="not_found", text=text.strip() or NOT_FOUND_TEXT, citations=[], confidence="low",
                      conflicts=conflicts, warnings=warnings)

    def _finalize(self, retrieval: RetrievalResult, evidence: list[prompts.EvidenceItem],
                  out: ModelOutput) -> Answer:
        conflicts = retrieval.conflicts
        warnings: list[str] = []

        if out.status == "insufficient_evidence":
            # inline [E#] markers are internal aliases, never shown to users on an answer with no citations
            return self._not_found(conflicts, warnings, strip_aliases(out.answer))
        if out.status == "needs_clarification":
            question = strip_aliases((out.clarifying_question or "").strip() or out.answer.strip())
            if not question:
                return self._not_found(conflicts, ["model asked for clarification without a question"])
            return Answer(status="needs_clarification", text=strip_aliases(out.answer).strip() or question, citations=[],
                          confidence="low", conflicts=conflicts, clarifying_question=question,
                          warnings=warnings)

        # answered / conflict: rebuild from validated pieces
        by_alias = {e.alias: e.chunk for e in evidence}
        text = out.answer.strip()
        kept_claims, dropped = [], 0
        for claim in out.claims:
            valid = [_norm_alias(a) for a in claim.evidence if _norm_alias(a) in by_alias]
            if len(valid) != len(claim.evidence):
                warnings.append("removed a citation that does not match any retrieved passage")
            if valid:
                kept_claims.append((claim.text.strip(), valid))
            elif claim.text.strip():
                dropped += 1
        if dropped and not kept_claims and [a for a in marker_aliases(text) if a in by_alias]:
            # the claim list was unusable, but the answer text carries valid inline citations: rely on those
            warnings.append("the model's claim list had no valid evidence; relied on the inline citations")
            dropped = 0
        if dropped:
            warnings.append(f"removed {dropped} statement(s) that had no valid citation")
            text = " ".join(f"{t} " + "".join(f"[{a}]" for a in v) for t, v in kept_claims)

        unknown = [a for a in marker_aliases(text) if a not in by_alias]
        if unknown:
            warnings.append("removed an inline citation that does not match any retrieved passage")

        order: list[str] = [a for a in marker_aliases(text) if a in by_alias]
        for _, valid in kept_claims:
            order += [a for a in valid if a not in order]
        if not order or not text:
            return self._not_found(conflicts, warnings + ["answer withheld: it had no valid citation to retrieved evidence"])

        claim_text = {a: " ".join(t for t, v in kept_claims if a in v) or text for a in order}
        citations: list[Citation] = [
            build_citation(by_alias[a], claim_text[a], self.settings.excerpt_chars) for a in order]
        final_text = renumber_markers(text, {a: i for i, a in enumerate(order, 1)})
        if not any(f"[{i}]" in final_text for i in range(1, len(order) + 1)):
            final_text += " " + "".join(f"[{i}]" for i in range(1, len(order) + 1))

        # Only conflicts that touch the evidence this answer actually cites matter to it. A conflict between
        # other retrieved passages (for example annual-leave versions retrieved for a sick-leave question)
        # must not turn an unrelated answer into a "conflict".
        cited_ids = {c.chunk_id for c in citations}
        groups = [g for g in conflicts.groups if set(g.chunk_ids) & cited_ids]
        relevant_conflicts = ConflictReport(has_conflict=bool(groups), groups=groups)
        for g in groups:
            uncited = [i for i in g.chunk_ids if i not in cited_ids]
            if uncited:
                warnings.append("a possibly conflicting passage was retrieved but is not cited in the answer")
        distinct_cited = {" ".join(by_alias[a].text.split("\n\n", 1)[-1].lower().split()) for a in order}
        model_conflict = (out.status == "conflict" and len({c.filename for c in citations}) >= 2
                          and len(distinct_cited) >= 2)       # identical passages in two files do not disagree
        status = "conflict" if (relevant_conflicts.has_conflict or model_conflict) else "answered"
        if status == "conflict" and len({c.filename for c in citations}) < 2:
            warnings.append("possible conflict between sources, but the answer cites only one document")
        evidence_text = " ".join(by_alias[a].text for a in order)
        recency = {m.group(0).lower() for m in _RECENCY.finditer(final_text)} - \
            {m.group(0).lower() for m in _RECENCY.finditer(evidence_text)}
        if recency:
            warnings.append("the answer calls a version " + "/".join(sorted(recency)) +
                            ", which the cited sources do not state; check which version applies")
        confidence = self._confidence(retrieval, status, warnings)
        return Answer(status=status, text=final_text, citations=citations, confidence=confidence,
                      conflicts=relevant_conflicts, warnings=list(dict.fromkeys(warnings)))

    @staticmethod
    def _confidence(retrieval: RetrievalResult, status: str, warnings: list[str]) -> str:
        """Heuristic label, not a calibrated probability."""
        if retrieval.evidence_strength == "weak":
            return "low"
        if status == "conflict" or warnings:
            return "medium"
        return "high"
