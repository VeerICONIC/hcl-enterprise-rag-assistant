"""HTML parser (BeautifulSoup, stdlib ``html.parser`` backend).

Drops scripts, styles and page chrome (nav/footer/aside/forms), keeps headings,
paragraphs, lists and tables in document order, and keeps absolute link targets
as ``text (url)``.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from app.ingestion.errors import CorruptDocumentError
from app.ingestion.models import Block, ParsedDocument, title_from_headings
from app.ingestion.parsers.base import DocumentParser

_REMOVE = ["script", "style", "noscript", "template", "svg", "iframe", "canvas", "nav",
           "footer", "aside", "form", "button", "select", "head"]
_HEADINGS = {f"h{i}": i for i in range(1, 7)}
_PARAGRAPH = {"p", "blockquote", "figcaption", "summary", "dt", "dd", "address", "caption"}
_INLINE = {"a", "span", "b", "strong", "em", "i", "u", "code", "small", "sup", "sub", "abbr",
           "mark", "label", "cite", "q", "kbd", "var", "samp", "time", "font", "br", "s", "del", "ins"}
_BR = ""  # private-use marker for <br>, survives whitespace collapsing


def _collapse(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return "\n".join(part.strip() for part in text.split(_BR)).strip()


def _text(tag: Tag) -> str:
    return _collapse(tag.get_text())


def _list_lines(lst: Tag, depth: int = 0) -> list[str]:
    lines: list[str] = []
    ordered = lst.name == "ol"
    for idx, li in enumerate(lst.find_all("li", recursive=False), 1):
        own = "".join(
            c.get_text() if isinstance(c, Tag) and c.name not in ("ul", "ol") else
            (str(c) if isinstance(c, NavigableString) else "")
            for c in li.children
        )
        text = _collapse(own)
        marker = f"{idx}." if ordered else "-"
        if text:
            lines.append(f"{'  ' * depth}{marker} {text}")
        for nested in li.find_all(["ul", "ol"], recursive=False):
            lines.extend(_list_lines(nested, depth + 1))
    return lines


def _table_lines(table: Tag) -> list[str]:
    lines: list[str] = []
    header_done = False
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"])
        texts = [_text(c).replace("|", "/") for c in cells]
        if not any(texts):
            continue
        lines.append("| " + " | ".join(texts) + " |")
        if not header_done:
            header_done = True
            if all(c.name == "th" for c in cells):
                lines.append("| " + " | ".join("---" for _ in cells) + " |")
    return lines


class HtmlParser(DocumentParser):
    doc_type = "html"
    extensions = ("html", "htm")

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        if b"\x00" in data:
            raise CorruptDocumentError(filename, "file contains binary (NUL) data and is not HTML")
        soup = BeautifulSoup(data, "html.parser")
        title_tag = soup.find("title")
        title = _text(title_tag) if title_tag else ""

        for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
            c.extract()
        for tag in soup.find_all(_REMOVE):
            tag.decompose()
        for tag in soup.find_all(attrs={"role": "navigation"}):
            tag.decompose()
        for br in soup.find_all("br"):
            br.replace_with(NavigableString(_BR))
        for a in soup.find_all("a", href=True):
            href, label = a["href"].strip(), a.get_text().strip()
            if re.match(r"https?://", href, re.I) and label and label != href:
                a.replace_with(NavigableString(f"{label} ({href})"))

        blocks: list[Block] = []
        self._walk(soup.body or soup, blocks)
        if not title:
            title = title_from_headings(blocks)
        return ParsedDocument(filename=filename, doc_type="html", title=title, blocks=blocks)

    def _walk(self, node: Tag, blocks: list[Block]) -> None:
        inline: list[str] = []

        def flush() -> None:
            text = _collapse("".join(inline))
            inline.clear()
            if text:
                blocks.append(Block("paragraph", text))

        for child in node.children:
            if isinstance(child, NavigableString):
                inline.append(str(child))
                continue
            name = child.name
            if name in _HEADINGS:
                flush()
                text = _text(child)
                if text:
                    blocks.append(Block("heading", text, level=_HEADINGS[name]))
            elif name in _PARAGRAPH:
                flush()
                text = _text(child)
                if text:
                    blocks.append(Block("paragraph", text))
            elif name in ("ul", "ol"):
                flush()
                lines = _list_lines(child)
                if lines:
                    blocks.append(Block("list", "\n".join(lines)))
            elif name == "table":
                flush()
                lines = _table_lines(child)
                if lines:
                    blocks.append(Block("table", "\n".join(lines)))
            elif name == "pre":
                flush()
                text = child.get_text().strip("\n")
                if text.strip():
                    blocks.append(Block("code", text))
            elif name in _INLINE:
                inline.append(child.get_text())
            else:
                flush()
                self._walk(child, blocks)
        flush()
