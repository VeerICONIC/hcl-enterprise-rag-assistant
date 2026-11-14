"""Test helpers: deterministic fake embedder and synthetic document builders."""

from __future__ import annotations

import hashlib
import math

import pymupdf

from app.embeddings.base import EmbeddingError


class FakeEmbedder:
    """Deterministic, offline embedder. Counts calls; can be told to fail."""

    model_id = "amazon.titan-embed-text-v2:0"

    def __init__(self, dimension: int = 256):
        self.dimension = dimension
        self.calls = 0
        self.texts: list[str] = []
        self.fail_with: Exception | None = None

    def _vec(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode()).digest()
        raw = [seed[i % len(seed)] - 127.5 for i in range(self.dimension)]
        norm = math.sqrt(sum(x * x for x in raw))
        return [x / norm for x in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail_with:
            raise self.fail_with
        for t in texts:
            if not t.strip():
                raise EmbeddingError("cannot embed empty text")
        self.calls += 1
        self.texts.extend(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_pdf(pages: list[list[tuple[str, float, bool]]], title: str = "") -> bytes:
    """Build a PDF. Each page is a list of (text, fontsize, bold) lines drawn top to bottom."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for text, size, bold in lines:
            page.insert_text((72, y), text, fontsize=size, fontname="hebo" if bold else "helv")
            y += size * 2.2
    if title:
        doc.set_metadata({"title": title})
    data = doc.tobytes()
    doc.close()
    return data


def make_scanned_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(pymupdf.Rect(72, 72, 272, 272), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return data


def make_encrypted_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret text", fontsize=11)
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="userpw", owner_pw="ownerpw")
    doc.close()
    return data


SAMPLE_MD = b"""# Employee Leave Policy

Intro paragraph about leave.

## Annual Leave

Employees receive 20 days of annual leave. Unused days expire on 31 March.

### Carry-over

- Up to 5 days carry over
- Approval needed from a manager

## Sick Leave

| Type | Days |
|------|------|
| Short | 5 |
| Long | 30 |
"""


class KeywordEmbedder(FakeEmbedder):
    """Offline embedder where texts sharing words get similar vectors (hashed bag of words)."""

    def _vec(self, text: str) -> list[float]:
        import re
        v = [0.0] * self.dimension
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:4], "big")
            v[h % self.dimension] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_query(self, text: str) -> list[float]:
        if self.fail_with:
            raise self.fail_with
        if not text.strip():
            raise EmbeddingError("cannot embed empty text")
        return self._vec(text)
