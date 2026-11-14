"""Conservative text preprocessing.

Only layout noise is removed (line endings, invisible characters, trailing and
repeated whitespace, repeated blank lines). Words, numbers, identifiers,
punctuation, table rows and code are never altered.
"""

from __future__ import annotations

import re
import unicodedata

from app.ingestion.models import Block, ParsedDocument

# Zero-width space, BOM, soft hyphen, word joiner: invisible, never meaningful in extracted text.
_INVISIBLE = dict.fromkeys(map(ord, "​﻿­⁠"), None)
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_WS = re.compile(r"[ \t\f\v]+")
_BLANK_RUNS = re.compile(r"\n{3,}")

# Blocks whose internal spacing carries meaning.
_LAYOUT_KINDS = {"list", "table", "code"}


def normalize_text(text: str, *, preserve_layout: bool = False) -> str:
    """Normalize one text value. ``preserve_layout`` keeps indentation and inner spacing."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", "\n").replace(" ", "\n\n")
    text = unicodedata.normalize("NFC", text).translate(_INVISIBLE).replace("\xa0", " ")
    text = _CONTROL.sub("", text)
    lines = []
    for line in text.split("\n"):
        line = line.rstrip()
        if not preserve_layout:
            line = _INLINE_WS.sub(" ", line).strip()
        lines.append(line)
    return _BLANK_RUNS.sub("\n\n", "\n".join(lines)).strip("\n")


def preprocess_block(block: Block) -> Block | None:
    """Return the cleaned block, or None if nothing remains."""
    if block.kind == "heading":
        text = " ".join(normalize_text(block.text).split())
    else:
        text = normalize_text(block.text, preserve_layout=block.kind in _LAYOUT_KINDS)
    return block.with_text(text) if text.strip() else None


def preprocess_document(doc: ParsedDocument) -> ParsedDocument:
    """Clean every block, drop empty ones and keep order, levels and pages intact."""
    blocks = [b for b in (preprocess_block(b) for b in doc.blocks) if b is not None]
    return ParsedDocument(
        filename=doc.filename,
        doc_type=doc.doc_type,
        title=" ".join(normalize_text(doc.title).split()),
        blocks=blocks,
        warnings=list(doc.warnings),
        page_count=doc.page_count,
    )
