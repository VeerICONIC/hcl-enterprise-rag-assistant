"""Persistent ChromaDB wrapper (api_contracts.md §3).

* One collection (``CHROMA_COLLECTION``), created once and reused across restarts.
* Cosine space; embeddings are always supplied by us (``embedding_function=None``),
  so Chroma never loads or runs a model of its own.
* The collection records the embedding model id and dimension it was built with;
  opening it with different settings fails loudly instead of mixing vector spaces.
* Stored metadata = the contract's ``ChunkMetadata`` fields (all scalars) plus
  store-level bookkeeping supplied by the caller (``content_hash``,
  ``index_fingerprint``, ``total_chunks``, ``ingested_at``).
* ``query`` returns cosine similarity (1 - Chroma cosine distance).
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import chromadb

from app.core.config import Settings, get_settings
from app.schemas import Chunk, ChunkMetadata, DocumentInfo

logger = logging.getLogger(__name__)

_META_FIELDS = tuple(ChunkMetadata.model_fields)
_SCALARS = (str, int, float, bool)
_READ_PAGE = 5000


class VectorStoreError(RuntimeError):
    """A vector-store operation failed. Message is user-safe."""


@dataclass
class DocumentState:
    """What the store currently holds for one document."""

    doc_id: str
    ids: list[str] = field(default_factory=list)
    content_hash: str = ""
    index_fingerprint: str = ""
    version: str = ""
    total_chunks: int = -1
    ingested_at: str = ""

    @property
    def chunk_count(self) -> int:
        return len(self.ids)


@contextmanager
def _guard(op: str) -> Iterator[None]:
    try:
        yield
    except VectorStoreError:
        raise
    except Exception as exc:  # chromadb raises many exception types
        logger.exception("vector store failure during %s", op)
        raise VectorStoreError(f"vector store error during {op}: {exc}") from exc


class ChromaStore:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        s = settings or get_settings()
        self._path = str(Path(path or s.chroma_path))
        self._name = collection_name or s.chroma_collection
        self._model_id = s.bedrock_embed_model_id
        self._dim = s.bedrock_embed_dim
        self._client: Any = None
        self._col: Any = None
        self._revision = 0

    # -- lifecycle ---------------------------------------------------------
    @property
    def collection(self):
        """Open (or create) the collection lazily, verifying its embedding configuration."""
        if self._col is None:
            with _guard("opening the collection"):
                Path(self._path).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(path=self._path)
                col = self._client.get_or_create_collection(
                    name=self._name,
                    metadata={"hnsw:space": "cosine", "embed_model_id": self._model_id,
                              "embed_dim": self._dim},
                    embedding_function=None,
                )
                meta = col.metadata or {}
                if (meta.get("embed_model_id"), meta.get("embed_dim")) != (self._model_id, self._dim):
                    raise VectorStoreError(
                        f"collection '{self._name}' was built with embedding model "
                        f"{meta.get('embed_model_id')!r} at dimension {meta.get('embed_dim')!r}, but the "
                        f"current configuration is {self._model_id!r} at {self._dim}. Mixing vector spaces "
                        "is not allowed: use a new CHROMA_COLLECTION or re-index."
                    )
                self._col = col
        return self._col

    @property
    def revision(self) -> int:
        """Counter bumped by every write/delete made through this object (cache invalidation key)."""
        return self._revision

    def close(self) -> None:
        """Drop handles and Chroma's shared client cache (releases file locks on Windows)."""
        self._col = None
        self._client = None
        try:
            from chromadb.api.shared_system_client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except Exception:  # best effort
            pass

    def health(self) -> dict[str, Any]:
        with _guard("health check"):
            return {"collection": self._name, "path": self._path, "chunks": self.count(),
                    "embed_model_id": self._model_id, "embed_dim": self._dim}

    # -- writes ------------------------------------------------------------
    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]],
               extra_metadata: Optional[Mapping[str, Any]] = None) -> None:
        """Insert or replace chunks by ``chunk_id``. ``extra_metadata`` is added to every chunk."""
        if len(chunks) != len(embeddings):
            raise VectorStoreError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return
        extra = dict(extra_metadata or {})
        bad = [k for k, v in extra.items() if k in _META_FIELDS or not isinstance(v, _SCALARS)]
        if bad:
            raise VectorStoreError(f"invalid extra metadata keys (scalars only, no contract fields): {bad}")
        for emb in embeddings:
            if len(emb) != self._dim:
                raise VectorStoreError(f"embedding dimension {len(emb)} != configured {self._dim}")
        ids = [c.metadata.chunk_id for c in chunks]
        self._revision += 1
        if len(set(ids)) != len(ids):
            raise VectorStoreError("duplicate chunk_id values in one upsert")
        metadatas = [{**c.metadata.model_dump(), **extra} for c in chunks]
        with _guard("upserting chunks"):
            step = max(1, min(self._client_batch_limit(), 1000))
            for i in range(0, len(ids), step):
                self.collection.upsert(
                    ids=ids[i:i + step],
                    embeddings=embeddings[i:i + step],
                    documents=[c.text for c in chunks[i:i + step]],
                    metadatas=metadatas[i:i + step],
                )

    def _client_batch_limit(self) -> int:
        _ = self.collection
        try:
            return int(self._client.get_max_batch_size())
        except Exception:
            return 1000

    def delete_ids(self, ids: list[str]) -> int:
        if not ids:
            return 0
        self._revision += 1
        with _guard("deleting chunks"):
            self.collection.delete(ids=list(ids))
        return len(ids)

    def delete_document(self, doc_id: str) -> int:
        """Delete every chunk of one document; other documents are untouched."""
        with _guard("deleting a document"):
            existing = self.get_document_state(doc_id)
            if existing is None:
                return 0
            self._revision += 1
            self.collection.delete(ids=existing.ids)
            return existing.chunk_count

    # -- reads -------------------------------------------------------------
    def count(self) -> int:
        with _guard("counting chunks"):
            return int(self.collection.count())

    def _get_all(self, *, where: Optional[dict] = None, include: list[str]) -> Iterator[dict[str, Any]]:
        offset = 0
        while True:
            page = self.collection.get(where=where, include=include, limit=_READ_PAGE, offset=offset)
            ids = page["ids"]
            if not ids:
                return
            for idx, cid in enumerate(ids):
                row: dict[str, Any] = {"id": cid}
                if "metadatas" in include:
                    row["metadata"] = page["metadatas"][idx]
                if "documents" in include:
                    row["document"] = page["documents"][idx]
                yield row
            if len(ids) < _READ_PAGE:
                return
            offset += _READ_PAGE

    def get_document_state(self, doc_id: str) -> Optional[DocumentState]:
        """IDs and bookkeeping for one document, or None if it is not indexed."""
        with _guard("reading document state"):
            rows = list(self._get_all(where={"doc_id": doc_id}, include=["metadatas"]))
        if not rows:
            return None
        rows.sort(key=lambda r: r["metadata"].get("ordinal", 0))
        first = rows[0]["metadata"]
        fingerprints = {str(r["metadata"].get("index_fingerprint", "")) for r in rows}
        return DocumentState(
            doc_id=doc_id,
            ids=[r["id"] for r in rows],
            content_hash=str(first.get("content_hash", "")),
            # "" when chunks disagree, i.e. a previous write was interrupted: never trusted
            index_fingerprint=fingerprints.pop() if len(fingerprints) == 1 else "",
            version=str(first.get("version", "")),
            total_chunks=int(first.get("total_chunks", -1)),
            ingested_at=str(first.get("ingested_at", "")),
        )

    def list_documents(self) -> list[DocumentInfo]:
        with _guard("listing documents"):
            docs: dict[str, DocumentInfo] = {}
            for row in self._get_all(include=["metadatas"]):
                m = row["metadata"]
                info = docs.get(m["doc_id"])
                if info is None:
                    docs[m["doc_id"]] = DocumentInfo(
                        doc_id=m["doc_id"], filename=m["filename"], title=m.get("title", ""),
                        version=m.get("version", ""), doc_type=m["doc_type"], chunk_count=1)
                else:
                    info.chunk_count += 1
        return sorted(docs.values(), key=lambda d: (d.filename.lower(), d.doc_id))

    def get_document_info(self, doc_id: str) -> Optional[DocumentInfo]:
        with _guard("reading document info"):
            count, first = 0, None
            for row in self._get_all(where={"doc_id": doc_id}, include=["metadatas"]):
                count += 1
                first = first or row["metadata"]
        if first is None:
            return None
        return DocumentInfo(doc_id=doc_id, filename=first["filename"], title=first.get("title", ""),
                            version=first.get("version", ""), doc_type=first["doc_type"], chunk_count=count)

    @staticmethod
    def _to_chunk(row: dict[str, Any]) -> Chunk:
        meta = {k: row["metadata"][k] for k in _META_FIELDS}
        return Chunk(text=row["document"], metadata=ChunkMetadata(**meta))

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        with _guard("reading chunks"):
            rows = list(self._get_all(where={"doc_id": doc_id}, include=["metadatas", "documents"]))
            chunks = [self._to_chunk(r) for r in rows]
        return sorted(chunks, key=lambda c: c.metadata.ordinal)

    def get_all_chunks(self, where: Optional[dict] = None) -> list[Chunk]:
        """All chunks (optionally restricted by a Chroma ``where`` filter), ordered by chunk_id."""
        with _guard("reading all chunks"):
            rows = self._get_all(where=where or None, include=["metadatas", "documents"])
            chunks = [self._to_chunk(r) for r in rows]
        return sorted(chunks, key=lambda c: c.metadata.chunk_id)

    def get_embeddings(self, ids: list[str]) -> dict[str, list[float]]:
        """Stored embeddings for the given chunk IDs (unknown IDs are omitted)."""
        if not ids:
            return {}
        with _guard("reading embeddings"):
            res = self.collection.get(ids=list(ids), include=["embeddings"])
            embs = res["embeddings"]
            return {cid: [float(x) for x in embs[i]] for i, cid in enumerate(res["ids"])}

    def query(self, embedding: list[float], top_k: int,
              where: Optional[dict] = None) -> list[tuple[Chunk, float]]:
        """Nearest chunks to ``embedding`` as ``(chunk, cosine_similarity)``, best first.

        Chroma returns cosine *distance* (``1 - cos``); it is converted to similarity
        (``1 - distance``) and clamped to [-1, 1] to absorb float32 rounding. Ties are
        broken by chunk_id so results are deterministic. Records that cannot be turned
        into a valid ``Chunk`` are skipped with a warning rather than failing the query.
        """
        self._validate_query_embedding(embedding)
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise VectorStoreError(f"top_k must be a positive integer, got {top_k!r}")
        with _guard("querying"):
            total = self.count()
            if total == 0:
                return []
            res = self.collection.query(
                query_embeddings=[[float(x) for x in embedding]],
                n_results=min(top_k, total),
                where=where or None,
                include=["documents", "metadatas", "distances"],
            )
        results: list[tuple[Chunk, float]] = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            try:
                if dist is None or not math.isfinite(dist):
                    raise ValueError(f"invalid distance {dist!r}")
                chunk = self._to_chunk({"id": cid, "document": doc, "metadata": meta})
            except Exception as exc:  # malformed stored record
                logger.warning("skipping malformed chunk %s in query results: %s", cid, exc)
                continue
            results.append((chunk, max(-1.0, min(1.0, 1.0 - float(dist)))))
        results.sort(key=lambda r: (-r[1], r[0].metadata.chunk_id))
        return results

    def _validate_query_embedding(self, embedding: list[float]) -> None:
        if embedding is None or len(embedding) == 0:
            raise VectorStoreError("query embedding is empty")
        if len(embedding) != self._dim:
            raise VectorStoreError(f"query embedding dimension {len(embedding)} != configured {self._dim}")
        try:
            vals = [float(x) for x in embedding]
        except (TypeError, ValueError):
            raise VectorStoreError("query embedding contains non-numeric values") from None
        if not all(math.isfinite(x) for x in vals):
            raise VectorStoreError("query embedding contains non-finite values")
        if not any(vals):
            raise VectorStoreError("query embedding is the zero vector; cosine similarity is undefined")
