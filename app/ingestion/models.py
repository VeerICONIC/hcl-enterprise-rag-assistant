"""Internal, format-independent document representation produced by every parser."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Optional

BlockKind = Literal["heading", "paragraph", "list", "table", "code"]


@dataclass(frozen=True)
class Block:
    """A structural unit of a document.

    ``level`` is only meaningful for headings (1 = top). ``page`` is 1-based, or
    -1 when the format has no pages.
    """

    kind: BlockKind
    text: str
    level: int = 0
    page: int = -1

    def with_text(self, text: str) -> "Block":
        return replace(self, text=text)


@dataclass
class ParsedDocument:
    filename: str
    doc_type: Literal["pdf", "html", "md", "txt"]
    title: str = ""
    blocks: list[Block] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    page_count: Optional[int] = None


@dataclass
class ChunkDraft:
    """A chunk before IDs/metadata are attached."""

    section: str          # heading path, "A > B"; "" if none
    page: int             # 1-based, or -1
    body: str             # chunk text without the contextual prefix


def title_from_headings(blocks: list[Block]) -> str:
    """A heading is only a trustworthy document title when it is the sole level-1 heading."""
    h1 = [b.text for b in blocks if b.kind == "heading" and b.level == 1]
    return h1[0] if len(h1) == 1 else ""
