"""Shared lexical helpers (tokenization) for BM25 and the conflict detector."""

from __future__ import annotations

import re

# Words with no retrieval value. Deliberately small and English-only (a documented limitation).
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
may might must of on or our shall should so than that the their them then there these they this those to
was we were what when where which who whom why will with would you your not no any all each per via also
my me mine his her him she he us ours yours
""".split())

# word characters, optionally joined by - or . so identifiers like "e-4021" or "v2.1" survive intact
_TOKEN = re.compile(r"[^\W_]+(?:[-.][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-cased content tokens. Punctuation-only input yields no tokens.

    Compound identifiers are kept whole *and* split into parts, so both ``e-4021`` and
    ``4021`` match. Stop words and single letters are dropped.
    """
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        parts = re.split(r"[-.]", tok)
        if len(parts) > 1:
            out.append(tok)
        out.extend(parts)
    return [t for t in out if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


_PREFIX = re.compile(r"\A(?:Document: [^\n]*\n)?(?:Section: [^\n]*\n)?\n?")


def strip_context_prefix(text: str) -> str:
    """Remove the ``Document:``/``Section:`` header that ingestion prepends to chunks."""
    return _PREFIX.sub("", text, count=1)
