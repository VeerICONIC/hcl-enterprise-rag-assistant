"""Evaluation dataset: questions, corpus manifest, and label resolution.

Relevance labels are *anchored on text*, not chunk IDs (those change whenever chunking changes):
each label is ``{filename, contains}`` and is resolved at run time to the IDs of the indexed chunks of
that file whose text contains the phrase. A label that matches no chunk is a dataset bug and raises.
This intentionally differs from the contract's illustrative ``relevant_chunk_ids`` field (section 8):
the resolved IDs are recorded in the results so they remain traceable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, Field

from app.schemas import Chunk

Category = Literal["direct", "paraphrase", "multi_document", "conflict", "ambiguous", "unanswerable", "similar_keywords"]
ANSWERABLE = {"direct", "paraphrase", "multi_document", "conflict", "similar_keywords"}


class DatasetError(ValueError):
    pass


class RelevantSpan(BaseModel):
    filename: str
    contains: str


class EvalQuestion(BaseModel):
    id: str
    category: Category
    question: str
    relevant: list[RelevantSpan] = Field(default_factory=list)
    expected_status: Literal["answered", "conflict", "needs_clarification", "not_found"]
    acceptable_status: list[str] = Field(default_factory=list)
    reference_answer: str = ""
    key_facts: list[str] = Field(default_factory=list)
    notes: str = ""

    @property
    def accepted(self) -> set[str]:
        return set(self.acceptable_status) | {self.expected_status}


class CorpusEntry(BaseModel):
    filename: str
    version: str = ""


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_questions(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                questions.append(EvalQuestion.model_validate(json.loads(line)))
            except ValueError as exc:
                raise DatasetError(f"{path}:{n}: {exc}") from None
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise DatasetError("duplicate question ids")
    for q in questions:
        if q.category in ANSWERABLE and not q.relevant:
            raise DatasetError(f"{q.id}: answerable category '{q.category}' needs relevance labels")
        if q.category not in ANSWERABLE and q.relevant:
            raise DatasetError(f"{q.id}: '{q.category}' questions must not have relevance labels")
    return questions


def load_manifest(path: Path) -> list[CorpusEntry]:
    return [CorpusEntry.model_validate(e) for e in json.loads(Path(path).read_text(encoding="utf-8"))]


def resolve_relevant(q: EvalQuestion, chunks: Sequence[Chunk]) -> set[str]:
    """IDs of indexed chunks matching the question's labels."""
    found: set[str] = set()
    for span in q.relevant:
        needle = normalize(span.contains)
        hits = {c.metadata.chunk_id for c in chunks
                if c.metadata.filename == span.filename and needle in normalize(c.text)}
        if not hits:
            raise DatasetError(f"{q.id}: label {span.filename!r} / {span.contains!r} matches no indexed chunk")
        found |= hits
    return found
