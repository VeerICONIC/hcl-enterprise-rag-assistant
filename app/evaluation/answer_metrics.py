"""Automatic answer-quality PROXIES. They are defined, deterministic and cheap, and they are NOT
human judgements of correctness:

* key-fact recall   - share of the question's expected key facts found (normalized substring) in the answer.
* groundedness (v2) - share of checkable answer sentences that are supported by the cited chunks' text:
                      content words (>=60% covered) AND every number in the sentence appears in the cited text.
                      Words copied from the question and conflict-framing words (version, sources, states...) are
                      ignored, because they restate the question rather than make a claim. A sentence is checkable
                      if it carries a citation marker or a number (the prompt requires every factual statement to
                      be cited). Word coverage is only judged when >=3 content words remain after exclusions (a shorter remainder is
                      mostly connectives such as "number" or "between"); numbers are always judged. Lexical only: it cannot
                      detect a fluent paraphrase of an unsupported claim, and an uncited qualitative sentence with no number
                      escapes the check.
* groundedness_v1   - the first, simpler proxy (run 1): every content word counts, numbers are not checked, question
                      words are not excluded. Kept so the effect of the change is visible on the same answers.
* status accuracy   - the answer status is one of the question's accepted statuses.
* conflict surfaced - for conflict questions: status "conflict" and citations from at least two documents.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from app.evaluation.dataset import EvalQuestion, normalize
from app.schemas import Answer

_WORD = re.compile(r"[a-z0-9$%.,]+")
_STOP = frozenset("the and for are was were with that this from have has had not but you your our can may will "
                  "per all any each its their they them than then into onto over also".split())
_MARKER = re.compile(r"\[\d+\]")


def key_fact_recall(text: str, key_facts: Sequence[str]) -> Optional[float]:
    if not key_facts:
        return None
    body = normalize(text)
    return sum(normalize(f) in body for f in key_facts) / len(key_facts)


def _terms(text: str) -> set[str]:
    out = set()
    for w in _WORD.findall(text.lower()):
        w = w.strip(".,")
        if len(w) > 2 and w not in _STOP:
            out.add(w)
    return out


_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")
_FRAMING = frozenset("version versions differs differ differs disagree disagrees sources source according states "
                     "stated state document documents handbook say says".split())


def _nums(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def groundedness_v2(answer: Answer, cited_text: str, question: str) -> Optional[float]:
    """See the module docstring. None when the answer makes no claims or has nothing checkable."""
    if answer.status not in ("answered", "conflict"):
        return None
    support, support_nums = _terms(cited_text), _nums(cited_text)
    skip, question_nums = _terms(question) | _FRAMING, _nums(question)
    results = []
    for raw in re.split(r"(?<=[.!?])\s+", answer.text):
        s = _MARKER.sub("", raw)
        terms = {w for w in _terms(s) - skip if not re.fullmatch(r"[\d.,$%]+", w)}   # numbers are judged via `nums`
        nums = _nums(s) - question_nums
        if not (_MARKER.search(raw) or nums) or not (len(terms) >= 3 or nums):
            continue
        words_ok = len(terms) < 3 or len(terms & support) / len(terms) >= 0.6
        results.append(words_ok and nums <= support_nums)
    return sum(results) / len(results) if results else None


def groundedness(answer: Answer, cited_text: str) -> Optional[float]:
    """Run-1 proxy (kept as ``groundedness_v1`` in results). None when the answer makes no claims or has no checkable sentence."""
    if answer.status not in ("answered", "conflict"):
        return None
    support = _terms(cited_text)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", _MARKER.sub("", answer.text)) if len(_terms(s)) >= 3]
    if not sentences:
        return None
    return sum(len(_terms(s) & support) / len(_terms(s)) >= 0.6 for s in sentences) / len(sentences)


def status_ok(answer: Answer, q: EvalQuestion) -> bool:
    return answer.status in q.accepted


def conflict_surfaced(answer: Answer, q: EvalQuestion) -> Optional[bool]:
    if q.category != "conflict":
        return None
    return answer.status == "conflict" and len({c.filename for c in answer.citations}) >= 2
