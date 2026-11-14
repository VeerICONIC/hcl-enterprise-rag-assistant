"""Deterministic, explainable conflict detection (FR-06). Heuristic; NOT semantic understanding.

Only chunks judged *relevant* by the evidence analyzer take part, and only pairs from *different*
documents (a document disagreeing with itself is out of scope). Two kinds of signal:

1. ``version_difference``: metadata/filename based.
   Two different documents that look like versions of one document (same "family": normalized
   title/filename with version, date and words like "final"/"draft" removed) and that cover the same
   subject (same section name, or overlapping content terms).
   * Description starts "METADATA CONFLICT" when both documents carry different non-empty ``version``
     metadata (a fact recorded at ingestion).
   * Otherwise it starts "POSSIBLE": the versions are only suggested by filenames, or are unknown.
   Which version is current is never decided: filename order and ingestion time are not treated as
   authoritative, and missing version information is stated explicitly.

2. ``contradiction`` / ``date_difference``: textual, always heuristic ("POSSIBLE ...").
   Numbers with a unit (``20 days``, ``5%``, ``$10``) or dates are extracted. A pair is flagged when
   both chunks mention the same unit (or both mention dates) with different values AND the text around
   the two values shares at least two content words (so "annual leave 20 days" vs "annual leave 25 days"
   matches, while unrelated numbers do not).

Known limits: no understanding of negation, paraphrase, or policy statements without numbers/dates;
unit/date regexes are English-centric; wording differences are deliberately not treated as conflicts;
false positives and false negatives are both expected. The output is a prompt for M3 to show both
sides, never a verdict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePath
from typing import Iterable, Optional, Sequence

from app.retrieval.text import STOPWORDS, strip_context_prefix, tokenize
from app.schemas import ConflictGroup, ConflictReport, RetrievedChunk

_MIN_CONTEXT_OVERLAP = 2
_MIN_SUBJECT_JACCARD = 0.2
_MAX_EVIDENCE_IN_DESCRIPTION = 3

_VERSION_FOOTER = (" Which version is current is NOT determined here (filename order and ingestion time "
                   "are not treated as authoritative).")
_TEXT_FOOTER = (" This is a heuristic comparison of numbers/dates in similar context; it can be a false "
                "positive or miss other disagreements. Verify against the cited sources.")

# ---- facts ------------------------------------------------------------------
_MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
           "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH = (r"(?P<mon>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
_DATE_ISO = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b")
_DATE_DMY = re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+{_MONTH}\b\.?(?:,?\s+(?P<y>\d{{4}}))?")
_DATE_MDY = re.compile(rf"\b{_MONTH}\b\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(?P<y>\d{{4}}))?")
_NUMBER = re.compile(
    r"(?<![\w.\-/])(?P<cur>[$€£])?(?P<num>\d[\d,]*(?:\.\d+)?)(?P<pct>%)?(?:[ \t]*(?P<unit>[A-Za-z]{2,}))?")
_ORDINAL_SUFFIX = {"st", "nd", "rd", "th"}


@dataclass(frozen=True)
class _Fact:
    value: float
    unit: str
    context: frozenset[str]
    raw: str


@dataclass(frozen=True)
class _Date:
    key: tuple[Optional[int], int, int]   # (year or None, month, day)
    context: frozenset[str]
    raw: str


def _context(text: str, start: int, end: int, exclude: str = "") -> frozenset[str]:
    toks = tokenize(text[max(0, start - 70):start] + " " + text[end:end + 50])
    return frozenset(t for t in toks if not t.isdigit() and t != exclude)


def _blank(text: str, span: tuple[int, int]) -> str:
    return text[:span[0]] + " " * (span[1] - span[0]) + text[span[1]:]


def extract_dates(text: str) -> tuple[list[_Date], str]:
    """Return dates found in ``text`` and the text with those dates blanked out."""
    dates: list[_Date] = []
    for pat in (_DATE_ISO, _DATE_DMY, _DATE_MDY):
        for m in list(pat.finditer(text)):
            gd = m.groupdict()
            month = int(gd["m"]) if "m" in gd and gd.get("m") else _MONTHS[gd["mon"][:3].lower()]
            day = int(gd["d"])
            year = int(gd["y"]) if gd.get("y") else None
            if not (1 <= month <= 12 and 1 <= day <= 31):
                continue
            dates.append(_Date((year, month, day), _context(text, m.start(), m.end()), m.group(0)))
        for m in list(pat.finditer(text)):
            text = _blank(text, m.span())
    return dates, text


def extract_facts(text: str) -> list[_Fact]:
    """Numbers carrying a unit (word), currency symbol or percent sign."""
    facts: list[_Fact] = []
    for m in _NUMBER.finditer(text):
        unit = (m.group("unit") or "").lower()
        if m.group("cur"):
            unit, shown = "$" if m.group("cur") == "$" else m.group("cur"), m.group(0)
        elif m.group("pct"):
            unit, shown = "%", m.group(0)
        else:
            if not unit or unit in STOPWORDS or unit in _ORDINAL_SUFFIX:
                continue
            if len(unit) > 3 and unit.endswith("s"):
                unit = unit[:-1]
            shown = m.group(0).strip()
        try:
            value = float(m.group("num").replace(",", ""))
        except ValueError:
            continue
        facts.append(_Fact(value, unit, _context(text, m.start(), m.end(), unit), shown))
    return facts


# ---- document family / versions ----------------------------------------------
_VERSION_TOKEN = re.compile(
    r"\bv\d+(?:\.\d+)*\b|\b\d+(?:\.\d+)+\b|\b(?:19|20)\d{2}(?:-\d{2}-\d{2})?\b|\brev\d*\b|\b\d{8}\b", re.I)
_NOISE_WORD = re.compile(r"\b(?:final|draft|copy|old|new|latest|updated|current)\b", re.I)


def family_key(filename: str, title: str) -> str:
    base = (title or PurePath(filename).stem).lower().replace("_", " ")
    base = _VERSION_TOKEN.sub(" ", base)
    base = _NOISE_WORD.sub(" ", base)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", base).split())


def filename_versions(filename: str) -> frozenset[str]:
    return frozenset(m.group(0).lower() for m in _VERSION_TOKEN.finditer(PurePath(filename).stem.replace("_", " ")))


def _section_leaf(section: str) -> str:
    return section.split(">")[-1].strip().lower() if section else ""


def _normalized_body(chunk: RetrievedChunk) -> str:
    return " ".join(strip_context_prefix(chunk.chunk.text).lower().split())


def _content_terms(chunk: RetrievedChunk) -> frozenset[str]:
    return frozenset(t for t in tokenize(strip_context_prefix(chunk.chunk.text)) if not t.isdigit())


def _same_subject(a: RetrievedChunk, b: RetrievedChunk) -> bool:
    la, lb = _section_leaf(a.chunk.metadata.section), _section_leaf(b.chunk.metadata.section)
    if la and lb:
        return la == lb  # both sections are named: they must be the same section (a documented limitation)
    ta, tb = _content_terms(a), _content_terms(b)
    return bool(ta and tb) and len(ta & tb) / len(ta | tb) >= _MIN_SUBJECT_JACCARD


def _label(c: RetrievedChunk) -> str:
    m = c.chunk.metadata
    ver = f", version {m.version}" if m.version else ", version unknown"
    sec = f" § {m.section}" if m.section else ""
    pg = f" p.{m.page}" if m.page != -1 else ""
    return f"{m.filename}{sec}{pg}{ver}"


# ---- detection ---------------------------------------------------------------
@dataclass
class _Edge:
    kind: str
    a: str
    b: str
    topic: str
    text: str


def _version_edge(a: RetrievedChunk, b: RetrievedChunk) -> Optional[_Edge]:
    ma, mb = a.chunk.metadata, b.chunk.metadata
    fam_a, fam_b = family_key(ma.filename, ma.title), family_key(mb.filename, mb.title)
    if not fam_a or fam_a != fam_b or not _same_subject(a, b):
        return None
    if _normalized_body(a) == _normalized_body(b):
        return None  # identical passages in two versions are not a conflict
    if ma.version and mb.version:
        if ma.version == mb.version:
            return None  # same recorded version: not a version conflict
        text = (f"METADATA CONFLICT: '{ma.filename}' (version {ma.version}) and '{mb.filename}' "
                f"(version {mb.version}) appear to be different versions of the same document "
                f"('{fam_a}') and both match the question.")
    else:
        fa, fb = filename_versions(ma.filename), filename_versions(mb.filename)
        missing = [m.filename for m in (ma, mb) if not m.version]
        if fa and fb and fa != fb:
            hint = f"filenames suggest different versions ({', '.join(sorted(fa))} vs {', '.join(sorted(fb))})"
        else:
            hint = "they share the same normalized title"
        text = (f"POSSIBLE version difference: '{ma.filename}' and '{mb.filename}' may be versions of the "
                f"same document ('{fam_a}'); {hint}. No version metadata for: {', '.join(missing)}.")
    return _Edge("version_difference", ma.chunk_id, mb.chunk_id, fam_a, text)


def _text_edges(a: RetrievedChunk, b: RetrievedChunk,
                cache: dict[str, tuple[list[_Date], list[_Fact]]]) -> list[_Edge]:
    if not _same_subject(a, b):
        return []
    edges: list[_Edge] = []
    ida, idb = a.chunk.metadata.chunk_id, b.chunk.metadata.chunk_id
    dates_a, facts_a = cache[ida]
    dates_b, facts_b = cache[idb]

    def overlap(x: frozenset[str], y: frozenset[str]) -> frozenset[str]:
        return x & y

    # Per unit, flag only when the two chunks' value sets are disjoint (so "10 days ... 3 days" vs
    # "10 days ... 3 days" agree), using the pair with the most shared surrounding words as evidence.
    for unit in sorted({f.unit for f in facts_a} & {f.unit for f in facts_b}):
        fa_u = [f for f in facts_a if f.unit == unit]
        fb_u = [f for f in facts_b if f.unit == unit]
        if {f.value for f in fa_u} & {f.value for f in fb_u}:
            continue
        ctx, fa, fb = max(((overlap(x.context, y.context), x, y) for x in fa_u for y in fb_u), key=lambda r: len(r[0]))
        if len(ctx) >= _MIN_CONTEXT_OVERLAP:
            topic = " ".join(sorted(ctx)[:3])
            edges.append(_Edge("contradiction", ida, idb, f"{topic} ({unit})", (
                f"POSSIBLE contradiction on '{topic}': '{fa.raw}' in {_label(a)} vs '{fb.raw}' in {_label(b)}.")))
            break

    for da in dates_a:
        hit = False
        for db in dates_b:
            ctx = overlap(da.context, db.context)
            ya, yb = da.key[0], db.key[0]
            differs = da.key[1:] != db.key[1:] or (ya is not None and yb is not None and ya != yb)
            if differs and len(ctx) >= _MIN_CONTEXT_OVERLAP:
                topic = " ".join(sorted(ctx)[:3])
                edges.append(_Edge("date_difference", ida, idb, f"{topic} (date)", (
                    f"POSSIBLE date difference on '{topic}': '{da.raw}' in {_label(a)} vs '{db.raw}' in {_label(b)}."
                )))
                hit = True
                break
        if hit:
            break
    return edges


def _components(edges: Iterable[_Edge], order: dict[str, int]) -> list[tuple[list[str], list[_Edge]]]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = list(edges)
    for e in edges:
        parent[find(e.a)] = find(e.b)
    comp: dict[str, tuple[set[str], list[_Edge]]] = {}
    for e in edges:
        ids, es = comp.setdefault(find(e.a), (set(), []))
        ids.update((e.a, e.b))
        es.append(e)
    return sorted(((sorted(ids, key=lambda i: order[i]), es) for ids, es in comp.values()),
                  key=lambda r: order[r[0][0]])


def detect_conflicts(chunks: Sequence[RetrievedChunk], relevant_ids: Optional[set[str]] = None) -> ConflictReport:
    """Detect potential conflicts among ``chunks`` (best-first). ``relevant_ids`` limits the candidates."""
    pool = [c for c in chunks if relevant_ids is None or c.chunk.metadata.chunk_id in relevant_ids]
    if len(pool) < 2:
        return ConflictReport(has_conflict=False)
    order = {c.chunk.metadata.chunk_id: i for i, c in enumerate(chunks)}
    cache = {}
    for c in pool:
        dates, rest = extract_dates(strip_context_prefix(c.chunk.text))
        cache[c.chunk.metadata.chunk_id] = (dates, extract_facts(rest))

    edges: list[_Edge] = []
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if a.chunk.metadata.doc_id == b.chunk.metadata.doc_id:
                continue
            v = _version_edge(a, b)
            if v:
                edges.append(v)
            edges.extend(_text_edges(a, b, cache))

    groups: list[ConflictGroup] = []
    for kind in ("version_difference", "contradiction", "date_difference"):
        for ids, es in _components([e for e in edges if e.kind == kind], order):
            texts = list(dict.fromkeys(e.text for e in es))[:_MAX_EVIDENCE_IN_DESCRIPTION]
            footer = _VERSION_FOOTER if kind == "version_difference" else _TEXT_FOOTER
            groups.append(ConflictGroup(topic=es[0].topic, chunk_ids=ids, kind=kind,
                                        description=" ".join(texts) + footer))
    return ConflictReport(has_conflict=bool(groups), groups=groups)
