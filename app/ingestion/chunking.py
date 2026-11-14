"""Structure-aware chunking.

Rules
-----
* Chunks never cross a section (heading path) or a page boundary.
* Paragraphs and sentences are kept whole where possible; sentences are the unit of
  splitting and of overlap. Overlap never copies tables, lists or code.
* Tables, lists and code are atomic when they fit; oversized ones are split on line
  boundaries (table pieces repeat the header row).
* ``chunk_size`` counts characters of the chunk *body*; the ``Document:``/``Section:``
  context prefix is added afterwards by :func:`with_context`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.models import Block, ChunkDraft, ParsedDocument

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TABLE_SEP = re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$")
_PARA_JOIN = "\n\n"


@dataclass
class _Unit:
    text: str
    joiner: str          # separator placed before this unit when it is not first in a chunk
    overlappable: bool   # only prose may be repeated as overlap


def _split_words(text: str, size: int) -> list[str]:
    """Pack whitespace-delimited tokens into pieces <= size; hard-cut tokens longer than size."""
    pieces: list[str] = []
    cur = ""
    for tok in re.findall(r"\S+\s*", text):
        while len(tok) > size:
            if cur:
                pieces.append(cur.rstrip())
                cur = ""
            pieces.append(tok[:size])
            tok = tok[size:]
        if cur and len(cur) + len(tok.rstrip()) > size:
            pieces.append(cur.rstrip())
            cur = ""
        cur += tok
    if cur.strip():
        pieces.append(cur.rstrip())
    return pieces


def _sentences(text: str, size: int) -> list[str]:
    out: list[str] = []
    for part in _SENTENCE_SPLIT.split(text):
        part = part.strip()
        if not part:
            continue
        out.extend(_split_words(part, size) if len(part) > size else [part])
    return out


def _pack_lines(lines: list[str], size: int, header: list[str]) -> list[str]:
    """Pack lines into pieces <= size; each piece starts with ``header`` lines (tables)."""
    head = "\n".join(header)
    pieces: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for line in lines:
        if len(line) > size - (len(head) + 1 if head else 0):
            segs = _split_words(line, max(1, size - (len(head) + 1 if head else 0)))
        else:
            segs = [line]
        for seg in segs:
            add = len(seg) + (1 if cur else 0)
            if cur and cur_len + add > size:
                pieces.append("\n".join(cur))
                cur, cur_len = [], 0
            if not cur and head:
                cur, cur_len = list(header), len(head)
                add = len(seg) + 1
            cur.append(seg)
            cur_len += add
    if cur and (len(cur) > len(header) or not header):
        pieces.append("\n".join(cur))
    return pieces


def _block_units(block: Block, size: int) -> list[_Unit]:
    text = block.text
    if block.kind == "paragraph":
        if len(text) <= size:
            return [_Unit(text, _PARA_JOIN, True)]
        sents = _sentences(text, size)
        return [_Unit(s, _PARA_JOIN if i == 0 else " ", True) for i, s in enumerate(sents)]
    # table / list / code: atomic when they fit
    if len(text) <= size:
        return [_Unit(text, _PARA_JOIN, False)]
    lines = text.split("\n")
    header: list[str] = []
    if block.kind == "table" and len(lines) >= 2 and _TABLE_SEP.match(lines[1]):
        header, lines = lines[:2], lines[2:]
        if len(("\n".join(header))) > size // 2:  # header too big to repeat
            lines, header = header + lines, []
    pieces = _pack_lines(lines, size, header)
    return [_Unit(p, _PARA_JOIN if i == 0 else "\n", False) for i, p in enumerate(pieces)]


def _pack_units(units: list[_Unit], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur: list[_Unit] = []
    cur_len = 0
    new_units = 0

    def length(us: list[_Unit]) -> int:
        return sum(len(u.text) for u in us) + sum(len(u.joiner) for u in us[1:])

    def render(us: list[_Unit]) -> str:
        return "".join((u.joiner if i else "") + u.text for i, u in enumerate(us))

    for u in units:
        add = len(u.text) + (len(u.joiner) if cur else 0)
        if cur and cur_len + add > size:
            if new_units:
                chunks.append(render(cur))
            carry: list[_Unit] = []
            if overlap > 0:
                for prev in reversed(cur):
                    if not prev.overlappable or length([prev] + carry) > overlap:
                        break
                    carry.insert(0, prev)
                if carry and length(carry) + len(u.joiner) + len(u.text) > size:
                    carry = []
            cur, cur_len, new_units = carry, length(carry), 0
            add = len(u.text) + (len(u.joiner) if cur else 0)
        cur.append(u)
        cur_len += add
        new_units += 1
    if cur and new_units:
        chunks.append(render(cur))
    return [c.strip() for c in chunks if c.strip()]


def _section_path(stack: dict[int, str], title: str) -> str:
    path = [stack[k] for k in sorted(stack)]
    if path and title and path[0] == title:
        path = path[1:]  # the title is already in the chunk's Document line
    return " > ".join(path)


def chunk_document(doc: ParsedDocument, chunk_size: int, chunk_overlap: int) -> list[ChunkDraft]:
    """Split a preprocessed document into section/page-bounded chunk bodies."""
    if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("require chunk_size >= 1 and 0 <= chunk_overlap < chunk_size")

    stack: dict[int, str] = {}
    groups: list[tuple[str, int, list[Block]]] = []
    for block in doc.blocks:
        if block.kind == "heading":
            for lvl in [k for k in stack if k >= block.level]:
                del stack[lvl]
            stack[block.level] = block.text
            continue
        if not block.text.strip():
            continue
        section = _section_path(stack, doc.title)
        if groups and groups[-1][0] == section and groups[-1][1] == block.page:
            groups[-1][2].append(block)
        else:
            groups.append((section, block.page, [block]))

    drafts: list[ChunkDraft] = []
    for section, page, blocks in groups:
        units = [u for b in blocks for u in _block_units(b, chunk_size)]
        for body in _pack_units(units, chunk_size, chunk_overlap):
            drafts.append(ChunkDraft(section=section, page=page, body=body))
    return drafts


def with_context(title: str, section: str, body: str) -> str:
    """Prefix a chunk body with its document title and section path."""
    header = []
    if title:
        header.append(f"Document: {title}")
    if section:
        header.append(f"Section: {section}")
    return ("\n".join(header) + "\n\n" + body) if header else body
