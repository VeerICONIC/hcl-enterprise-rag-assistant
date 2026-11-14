"""Ingestion orchestrator: validate → parse → preprocess → chunk → embed → index.

Independent of FastAPI. Safety properties for re-ingestion:

* Nothing in the store is touched until the new document has been parsed, chunked
  and fully embedded in memory.
* New chunks are upserted by deterministic ID, and only afterwards are the chunks that
  are no longer part of the document deleted. A failure before the upsert leaves the
  previous version fully intact.
* If the upsert itself fails midway for a *new* document, the partial write is rolled
  back. For an *existing* document the previous chunks may be partly overwritten; this
  is detected (inconsistent fingerprints) and repaired by simply re-ingesting.
* Identical content, version and settings are detected and skipped without any
  embedding calls.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Literal, Optional, Protocol

from app.core.config import Settings, get_settings
from app.embeddings.base import Embedder
from app.ingestion import ids
from app.ingestion.chunking import chunk_document, with_context
from app.ingestion.errors import (
    EmptyDocumentError,
    FileTooLargeError,
    IngestionError,
)
from app.ingestion.parser_factory import get_parser
from app.ingestion.preprocess import preprocess_document
from app.schemas import Chunk, ChunkMetadata, DocumentInfo, IngestResponse
from app.vectorstore.chroma_store import DocumentState, VectorStoreError

logger = logging.getLogger(__name__)

Status = Literal["indexed", "updated", "unchanged"]


class _Store(Protocol):
    def get_document_state(self, doc_id: str) -> Optional[DocumentState]: ...
    def get_document_info(self, doc_id: str) -> Optional[DocumentInfo]: ...
    def upsert(self, chunks, embeddings, extra_metadata=None) -> None: ...
    def delete_ids(self, ids: list[str]) -> int: ...
    def delete_document(self, doc_id: str) -> int: ...
    def list_documents(self) -> list[DocumentInfo]: ...


@dataclass
class FileOutcome:
    filename: str
    doc_id: str
    status: Status
    document: DocumentInfo
    chunks_indexed: int          # chunks written by this call (0 when unchanged)
    chunks_removed: int = 0      # stale chunks deleted
    warnings: list[str] = field(default_factory=list)


@dataclass
class FileFailure:
    filename: str
    code: str
    reason: str
    client_error: bool           # True: problem with the file (HTTP 400); False: infrastructure


@dataclass
class BatchResult:
    outcomes: list[FileOutcome] = field(default_factory=list)
    failures: list[FileFailure] = field(default_factory=list)

    def to_response(self) -> IngestResponse:
        return IngestResponse(
            documents=[o.document for o in self.outcomes],
            chunks_indexed=sum(o.chunks_indexed for o in self.outcomes),
            failures=[{"filename": f.filename, "reason": f.reason} for f in self.failures],
        )


class IngestionService:
    def __init__(self, embedder: Embedder, store: _Store, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.embedder = embedder
        self.store = store
        self._lock = threading.RLock()  # one ingest at a time: check/embed/write must not interleave

    # -- single file -------------------------------------------------------
    def ingest_bytes(self, filename: str, data: bytes, *, version: Optional[str] = None,
                     source: Optional[str] = None) -> FileOutcome:
        """Ingest one file. Raises IngestionError / EmbeddingError / VectorStoreError on failure."""
        name = ids.normalize_filename(filename)
        if not name:
            raise IngestionError(filename or "(unnamed)", "file has no name")
        s = self.settings
        parser = get_parser(name, s.supported_extensions)
        if len(data) > s.max_upload_bytes:
            raise FileTooLargeError(name, f"file is {len(data) / 1048576:.1f} MB; the limit is {s.max_upload_size_mb} MB")
        if not data.strip():
            raise EmptyDocumentError(name, "file is empty")

        ver = (version or "").strip()
        doc_id = ids.doc_id_for(name)
        chash = ids.content_hash(data)
        fingerprint = ids.index_fingerprint(
            content=chash, version=ver, chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap,
            embed_model_id=s.bedrock_embed_model_id, embed_dim=s.bedrock_embed_dim)

        with self._lock:
            state = self.store.get_document_state(doc_id)
            if (state and state.index_fingerprint == fingerprint
                    and state.chunk_count == state.total_chunks):
                info = self.store.get_document_info(doc_id)
                if info is not None:
                    logger.info("ingest %s: unchanged (%d chunks), skipped", name, info.chunk_count)
                    return FileOutcome(name, doc_id, "unchanged", info, 0)

            parsed = preprocess_document(parser.parse(data, name))
            drafts = chunk_document(parsed, s.chunk_size, s.chunk_overlap)
            if not drafts:
                raise EmptyDocumentError(name, "no text content could be extracted")

            title = parsed.title or PurePath(name).stem
            src = source or f"upload:{name}"
            chunks = [
                Chunk(
                    text=with_context(title, d.section, d.body),
                    metadata=ChunkMetadata(
                        chunk_id=ids.chunk_id_for(doc_id, ver, i), doc_id=doc_id, filename=name,
                        title=title, section=d.section, page=d.page, version=ver, source=src,
                        doc_type=parsed.doc_type, ordinal=i),
                )
                for i, d in enumerate(drafts)
            ]

            # Embed everything first: a failure here leaves the store untouched.
            embeddings = self.embedder.embed_documents([c.text for c in chunks])

            extra = {
                "content_hash": chash,
                "index_fingerprint": fingerprint,
                "total_chunks": len(chunks),
                "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            new_ids = [c.metadata.chunk_id for c in chunks]
            try:
                self.store.upsert(chunks, embeddings, extra)
            except VectorStoreError:
                if state is None:
                    try:
                        self.store.delete_ids(new_ids)  # roll back a partial first-time write
                    except VectorStoreError:
                        logger.warning("rollback after failed upsert of %s also failed", name)
                raise

            stale = [i for i in (state.ids if state else []) if i not in set(new_ids)]
            removed = self.store.delete_ids(stale) if stale else 0
            info = DocumentInfo(doc_id=doc_id, filename=name, title=title, version=ver,
                                doc_type=parsed.doc_type, chunk_count=len(chunks))
            logger.info("ingest %s: %s, %d chunks, %d stale removed", name,
                        "updated" if state else "indexed", len(chunks), removed)
            return FileOutcome(name, doc_id, "updated" if state else "indexed", info,
                               len(chunks), removed, list(parsed.warnings))

    def ingest_path(self, path: Path, *, version: Optional[str] = None) -> FileOutcome:
        path = Path(path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise IngestionError(path.name, f"cannot read file: {exc.strerror or exc}") from None
        return self.ingest_bytes(path.name, data, version=version, source=str(path.resolve()))

    # -- batch -------------------------------------------------------------
    def ingest_files(self, files: list[tuple[str, bytes]], *, version: Optional[str] = None) -> BatchResult:
        """Ingest several files; one file's failure never prevents the others, and none is skipped silently."""
        result = BatchResult()
        for filename, data in files:
            try:
                result.outcomes.append(self.ingest_bytes(filename, data, version=version))
            except IngestionError as exc:
                result.failures.append(FileFailure(exc.filename, exc.code, exc.reason, exc.client_error))
            except Exception as exc:  # embedding / vector-store / unexpected
                logger.exception("ingestion of %r failed", filename)
                code, reason = _describe_infrastructure_error(exc)
                result.failures.append(FileFailure(ids.normalize_filename(filename) or filename, code, reason, False))
        return result

    # -- document management ----------------------------------------------
    def list_documents(self) -> list[DocumentInfo]:
        return self.store.list_documents()

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            return self.store.delete_document(doc_id)


def _describe_infrastructure_error(exc: Exception) -> tuple[str, str]:
    from app.core.bedrock import BedrockError  # local import keeps the module graph simple

    if isinstance(exc, BedrockError):
        return "embedding_failed", f"embedding service error: {exc}"
    if isinstance(exc, VectorStoreError):
        return "vector_store_failed", str(exc)
    return "internal_error", "unexpected internal error while ingesting this file (see server logs)"
