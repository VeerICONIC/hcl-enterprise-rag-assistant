"""Ingestion errors. All carry the filename and a user-safe reason."""

from __future__ import annotations


class IngestionError(Exception):
    """Base ingestion failure (contract: ``IngestionError(filename, reason)``)."""

    code = "ingestion_error"
    #: True when the problem lies with the submitted file (maps to HTTP 400).
    client_error = True

    def __init__(self, filename: str, reason: str):
        super().__init__(f"{filename}: {reason}")
        self.filename = filename
        self.reason = reason


class UnsupportedFormatError(IngestionError):
    code = "unsupported_format"


class FileTooLargeError(IngestionError):
    code = "file_too_large"


class EmptyDocumentError(IngestionError):
    code = "empty_document"


class CorruptDocumentError(IngestionError):
    code = "corrupt_document"


class EncryptedDocumentError(IngestionError):
    code = "encrypted_document"


class OcrRequiredError(IngestionError):
    code = "ocr_required"


class UnsupportedEncodingError(IngestionError):
    code = "unsupported_encoding"
