"""Deterministic identifiers.

* ``doc_id``   identifies *which document* (derived from the normalized filename),
               so a re-upload with changed content replaces the right document.
* ``content_hash`` identifies *which content* (SHA-256 of the file bytes).
* ``chunk_id`` is ``{doc_id}:v{version}:{ordinal:05d}`` (api_contracts.md §1).
* ``index_fingerprint`` changes whenever anything that affects the stored vectors
  changes (content, version, chunking settings, embedding model/dimension).
"""

from __future__ import annotations

import hashlib
import unicodedata

CHUNKER_VERSION = "1"  # bump when chunking logic changes in a way that alters output


def normalize_filename(filename: str) -> str:
    name = unicodedata.normalize("NFC", filename.replace("\\", "/").rsplit("/", 1)[-1]).strip()
    return name


def doc_id_for(filename: str) -> str:
    key = normalize_filename(filename).lower()
    return "doc_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_id_for(doc_id: str, version: str, ordinal: int) -> str:
    return f"{doc_id}:v{version}:{ordinal:05d}"


def index_fingerprint(*, content: str, version: str, chunk_size: int, chunk_overlap: int,
                      embed_model_id: str, embed_dim: int) -> str:
    parts = [content, version, str(chunk_size), str(chunk_overlap), embed_model_id, str(embed_dim),
             CHUNKER_VERSION]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
