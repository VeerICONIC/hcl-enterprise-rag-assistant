"""Parser selection by file extension."""

from __future__ import annotations

from pathlib import PurePath

from app.ingestion.errors import UnsupportedFormatError
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.html_parser import HtmlParser
from app.ingestion.parsers.markdown_parser import MarkdownParser
from app.ingestion.parsers.pdf_parser import PdfParser
from app.ingestion.parsers.text_parser import TextParser

_PARSERS: tuple[DocumentParser, ...] = (PdfParser(), HtmlParser(), MarkdownParser(), TextParser())
_BY_EXTENSION: dict[str, DocumentParser] = {ext: p for p in _PARSERS for ext in p.extensions}


def file_extension(filename: str) -> str:
    return PurePath(filename).suffix.lower().lstrip(".")


def get_parser(filename: str, allowed: frozenset[str] | None = None) -> DocumentParser:
    """Return the parser for ``filename``; raise UnsupportedFormatError if none/not allowed."""
    ext = file_extension(filename)
    parser = _BY_EXTENSION.get(ext)
    if parser is None or (allowed is not None and ext not in allowed):
        supported = ", ".join(sorted(allowed if allowed is not None else _BY_EXTENSION))
        shown = f".{ext}" if ext else "no extension"
        raise UnsupportedFormatError(filename, f"unsupported file type ({shown}); supported: {supported}")
    return parser
