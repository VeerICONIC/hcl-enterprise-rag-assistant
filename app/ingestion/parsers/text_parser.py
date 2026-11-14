"""Plain-text parser: paragraphs split on blank lines; line breaks inside a paragraph kept."""

from __future__ import annotations

import re

from app.ingestion.models import Block, ParsedDocument
from app.ingestion.parsers.base import DocumentParser, decode_text


class TextParser(DocumentParser):
    doc_type = "txt"
    extensions = ("txt",)

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        text = decode_text(data, filename).replace("\r\n", "\n").replace("\r", "\n")
        blocks = [
            Block("paragraph", para.strip("\n"))
            for para in re.split(r"\n[ \t]*\n", text)
            if para.strip()
        ]
        return ParsedDocument(filename=filename, doc_type="txt", blocks=blocks)
