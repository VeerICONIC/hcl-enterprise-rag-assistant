"""Common parser interface and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from app.ingestion.errors import CorruptDocumentError, UnsupportedEncodingError
from app.ingestion.models import ParsedDocument


class DocumentParser(ABC):
    """Turns raw file bytes into the internal ``ParsedDocument`` representation."""

    doc_type: ClassVar[str]
    extensions: ClassVar[tuple[str, ...]]

    @abstractmethod
    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        """Parse ``data``; raise an ``IngestionError`` subclass on failure."""


def decode_text(data: bytes, filename: str) -> str:
    """Decode text bytes: UTF-8 (with/without BOM) or UTF-16 with BOM. Anything else is an error."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            raise UnsupportedEncodingError(filename, "invalid UTF-16 text") from None
    if b"\x00" in data:
        raise CorruptDocumentError(filename, "file contains binary (NUL) data and is not readable text")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise UnsupportedEncodingError(
            filename, "text is not valid UTF-8; re-save the file as UTF-8 and upload again"
        ) from None
