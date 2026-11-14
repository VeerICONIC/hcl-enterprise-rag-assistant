"""Programmatic citations. Every field of a ``Citation`` is copied from a real retrieved chunk;
nothing the model writes (IDs, filenames, pages, sections) is ever used. The excerpt is a verbatim
substring of the chunk text, chosen as the sentence(s) that best overlap the cited claim."""

from __future__ import annotations

import re
from typing import Callable

from app.schemas import Chunk, ChunkMetadata, Citation

_WORD = re.compile(r"[a-z0-9]+")
_SPAN = re.compile(r"[^\n]*?(?:[.!?](?=\s|$)|\n|$)")
_MARKER = re.compile(r"\[\s*(E\d+(?:\s*[,;]\s*E\d+)*)\s*\]", re.I)
_ALIAS = re.compile(r"E\d+", re.I)


def _terms(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 or w.isdigit()}


def source_identifier(meta: ChunkMetadata) -> str:
    """``filename#section/p.N`` (parts omitted when unavailable)."""
    frag = [meta.section] if meta.section else []
    if meta.page != -1:
        frag.append(f"p.{meta.page}")
    return meta.filename + ("#" + "/".join(frag) if frag else "")


def body_offset(text: str) -> int:
    """Index where the chunk body starts (after the ``Document:``/``Section:`` header)."""
    if text.startswith("Document: ") or text.startswith("Section: "):
        idx = text.find("\n\n")
        if idx != -1:
            return idx + 2
    return 0


def best_excerpt(chunk_text: str, claim: str, max_chars: int) -> str:
    """Verbatim excerpt of ``chunk_text`` (body only) that best supports ``claim``."""
    off = body_offset(chunk_text)
    body = chunk_text[off:]
    spans = [(m.start(), m.end()) for m in _SPAN.finditer(body) if body[m.start():m.end()].strip()]
    if not spans:
        return chunk_text.strip()[:max_chars].strip() or chunk_text[:max_chars]
    claim_terms = _terms(claim)

    def overlap(i: int) -> int:
        return len(claim_terms & _terms(body[spans[i][0]:spans[i][1]]))

    scores = [overlap(i) for i in range(len(spans))]
    best = max(range(len(spans)), key=lambda i: (scores[i], -(spans[i][1] - spans[i][0])))
    start, end = spans[best]
    j = best + 1
    while j < len(spans) and scores[j] >= 2 and spans[j][1] - start <= max_chars:
        end = spans[j][1]
        j += 1
    excerpt = body[start:end].strip()
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars].rstrip()
    return excerpt


def build_citation(chunk: Chunk, claim_text: str, max_chars: int) -> Citation:
    m = chunk.metadata
    return Citation(
        chunk_id=m.chunk_id, filename=m.filename, title=m.title, section=m.section, page=m.page,
        version=m.version, excerpt=best_excerpt(chunk.text, claim_text, max_chars),
        source=source_identifier(m),
    )


def is_valid_citation(citation: Citation, chunk: Chunk) -> bool:
    """True iff the citation matches its chunk's metadata and its excerpt is a substring of the chunk."""
    m = chunk.metadata
    return (citation.chunk_id == m.chunk_id and citation.filename == m.filename and citation.page == m.page
            and citation.section == m.section and bool(citation.excerpt) and citation.excerpt in chunk.text)


def marker_aliases(text: str) -> list[str]:
    """Aliases cited inline in ``text`` in order of first appearance (upper-cased, deduplicated)."""
    seen: dict[str, None] = {}
    for m in _MARKER.finditer(text):
        for a in _ALIAS.findall(m.group(1)):
            seen.setdefault(a.upper(), None)
    return list(seen)


def renumber_markers(text: str, alias_to_number: dict[str, int]) -> str:
    """Replace ``[E1]``/``[E1, E2]`` with ``[1]``/``[1][2]``; markers naming unknown aliases are removed."""

    def repl(m: re.Match) -> str:
        nums = [alias_to_number[a.upper()] for a in _ALIAS.findall(m.group(1)) if a.upper() in alias_to_number]
        return "".join(f"[{n}]" for n in sorted(set(nums)))

    out = _MARKER.sub(repl, text)
    return re.sub(r"[ \t]{2,}", " ", re.sub(r"\s+([.,;:])", r"\1", out)).strip()


def strip_aliases(text: str) -> str:
    """Remove internal [E#] aliases from text that has no citations (refusals, clarifications).
    An alias that starts a sentence ("[E3] mentions ...") becomes "A retrieved passage" so the sentence still reads."""
    text = re.sub(r"(^|[.!?]\s+)" + _MARKER.pattern + r"(?=\s)", lambda m: f"{m.group(1)}A retrieved passage", text, flags=re.I)
    return renumber_markers(text, {})
