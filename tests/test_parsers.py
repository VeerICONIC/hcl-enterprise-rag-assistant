import pytest

from app.ingestion.errors import (
    CorruptDocumentError, EmptyDocumentError, EncryptedDocumentError, OcrRequiredError,
    UnsupportedEncodingError, UnsupportedFormatError,
)
from app.ingestion.parser_factory import get_parser
from tests.helpers import SAMPLE_MD, make_encrypted_pdf, make_pdf, make_scanned_pdf


def kinds(doc):
    return [(b.kind, b.text) for b in doc.blocks]


# ---- factory ---------------------------------------------------------------
@pytest.mark.parametrize("name,doc_type", [
    ("a.pdf", "pdf"), ("A.PDF", "pdf"), ("a.html", "html"), ("a.htm", "html"),
    ("a.md", "md"), ("a.markdown", "md"), ("a.txt", "txt"),
])
def test_factory_selects_parser(name, doc_type):
    assert get_parser(name).doc_type == doc_type


@pytest.mark.parametrize("name", ["a.docx", "noext", "a.exe"])
def test_factory_rejects_unsupported(name):
    with pytest.raises(UnsupportedFormatError, match="supported"):
        get_parser(name)


def test_factory_respects_allowed_set():
    with pytest.raises(UnsupportedFormatError):
        get_parser("a.pdf", frozenset({"txt"}))


# ---- PDF -------------------------------------------------------------------
def test_pdf_single_page_with_heading():
    data = make_pdf([[("User Guide", 20, True), ("Reset your password from the login page.", 11, False)]])
    doc = get_parser("g.pdf").parse(data, "g.pdf")
    assert doc.doc_type == "pdf" and doc.page_count == 1
    assert doc.blocks[0].kind == "heading" and doc.blocks[0].text == "User Guide"
    assert doc.blocks[1].kind == "paragraph" and "Reset your password" in doc.blocks[1].text
    assert doc.title == "User Guide"


def test_pdf_multi_page_keeps_page_numbers_and_boundaries():
    data = make_pdf([
        [("Billing", 20, True), ("Invoices are monthly.", 11, False)],
        [("Limits", 20, True), ("The API limit is 100 requests per minute.", 11, False)],
        [("Last page body text here.", 11, False)],
    ])
    doc = get_parser("g.pdf").parse(data, "g.pdf")
    pages = {b.text: b.page for b in doc.blocks}
    assert pages["Invoices are monthly."] == 1
    assert pages["The API limit is 100 requests per minute."] == 2
    assert pages["Last page body text here."] == 3


def test_pdf_removes_repeating_header_and_page_numbers():
    pages = [[(f"ACME Confidential", 9, False), (f"Body of page {i} with real content.", 11, False), (f"Page {i}", 9, False)]
             for i in range(1, 5)]
    doc = get_parser("g.pdf").parse(make_pdf(pages), "g.pdf")
    texts = [b.text for b in doc.blocks]
    assert not any("ACME" in t for t in texts)
    assert not any(t.startswith("Page ") for t in texts)
    assert sum("Body of page" in t for t in texts) == 4


def test_pdf_scanned_requires_ocr():
    with pytest.raises(OcrRequiredError, match="OCR"):
        get_parser("s.pdf").parse(make_scanned_pdf(), "s.pdf")


def test_pdf_empty_page_is_empty_error():
    with pytest.raises(EmptyDocumentError):
        get_parser("e.pdf").parse(make_pdf([[]]), "e.pdf")


def test_pdf_encrypted():
    with pytest.raises(EncryptedDocumentError, match="password"):
        get_parser("x.pdf").parse(make_encrypted_pdf(), "x.pdf")


@pytest.mark.parametrize("data", [b"%PDF-1.4 this is not a real pdf", b"hello world", b""])
def test_pdf_corrupted(data):
    with pytest.raises(CorruptDocumentError):
        get_parser("c.pdf").parse(data, "c.pdf")


def test_pdf_warns_about_blank_pages():
    data = make_pdf([[("Some body text on page one.", 11, False)], []])
    doc = get_parser("w.pdf").parse(data, "w.pdf")
    assert any("page(s) [2]" in w for w in doc.warnings)


# ---- HTML ------------------------------------------------------------------
HTML = b"""<html><head><title>Support Portal</title><style>.x{color:red}</style>
<script>var secret = 1;</script></head>
<body><nav><a href="/">Home</a> NAVTEXT</nav>
<h1>Account Access</h1>
<p>Visit the <a href="https://example.com/reset">reset page</a> to change it.<br>Second line.</p>
<script>alert('x')</script>
<h2>Limits</h2>
<ul><li>Free: 10 req/min<ul><li>burst 20</li></ul></li><li>Pro: 100 req/min</li></ul>
<table><tr><th>Plan</th><th>Price</th></tr><tr><td>Pro</td><td>$10</td></tr></table>
<pre>curl -X POST /api</pre>
<footer>FOOTERTEXT</footer></body></html>"""


def test_html_structure_and_cleaning():
    doc = get_parser("p.html").parse(HTML, "p.html")
    joined = "\n".join(b.text for b in doc.blocks)
    for junk in ("secret", "alert", "color:red", "NAVTEXT", "FOOTERTEXT"):
        assert junk not in joined
    assert doc.title == "Support Portal"
    heads = [(b.level, b.text) for b in doc.blocks if b.kind == "heading"]
    assert heads == [(1, "Account Access"), (2, "Limits")]
    para = next(b for b in doc.blocks if b.kind == "paragraph")
    assert "reset page (https://example.com/reset)" in para.text and "\nSecond line." in para.text
    lst = next(b for b in doc.blocks if b.kind == "list")
    assert lst.text.splitlines() == ["- Free: 10 req/min", "  - burst 20", "- Pro: 100 req/min"]
    table = next(b for b in doc.blocks if b.kind == "table")
    assert table.text.splitlines()[:3] == ["| Plan | Price |", "| --- | --- |", "| Pro | $10 |"]
    assert any(b.kind == "code" and "curl -X POST" in b.text for b in doc.blocks)


def test_multiple_h1_do_not_define_the_title():
    md = b"# One\n\ntext\n\n# Two\n\ntext"
    assert get_parser("p.md").parse(md, "p.md").title == ""
    html = b"<body><h1>One</h1><p>t</p><h1>Two</h1><p>t</p></body>"
    assert get_parser("p.html").parse(html, "p.html").title == ""


def test_html_title_falls_back_to_h1():
    doc = get_parser("p.html").parse(b"<body><h1>Only H1</h1><p>text</p></body>", "p.html")
    assert doc.title == "Only H1"


def test_html_binary_is_corrupt():
    with pytest.raises(CorruptDocumentError):
        get_parser("p.html").parse(b"<html>\x00\x01</html>", "p.html")


# ---- Markdown --------------------------------------------------------------
def test_markdown_headings_lists_tables():
    doc = get_parser("p.md").parse(SAMPLE_MD, "p.md")
    heads = [(b.level, b.text) for b in doc.blocks if b.kind == "heading"]
    assert heads == [(1, "Employee Leave Policy"), (2, "Annual Leave"), (3, "Carry-over"), (2, "Sick Leave")]
    assert doc.title == "Employee Leave Policy"
    lst = next(b for b in doc.blocks if b.kind == "list")
    assert lst.text == "- Up to 5 days carry over\n- Approval needed from a manager"
    table = next(b for b in doc.blocks if b.kind == "table")
    assert table.text.splitlines()[0] == "| Type | Days |" and len(table.text.splitlines()) == 4


def test_markdown_code_fence_not_parsed_as_headings():
    md = b"# T\n\n```python\n# not a heading\nx = 1\n```\n\nafter"
    doc = get_parser("p.md").parse(md, "p.md")
    assert [b.kind for b in doc.blocks] == ["heading", "code", "paragraph"]
    assert "# not a heading" in doc.blocks[1].text


def test_markdown_setext_and_front_matter():
    md = b"---\ntitle: Front Title\nversion: 9\n---\nSetext Heading\n===============\n\nBody\n"
    doc = get_parser("p.md").parse(md, "p.md")
    assert doc.title == "Front Title"
    assert (doc.blocks[0].kind, doc.blocks[0].level) == ("heading", 1)
    assert "version" not in " ".join(b.text for b in doc.blocks)


# ---- TXT -------------------------------------------------------------------
def test_txt_paragraphs_and_newline_conventions():
    data = b"First para line one\r\nline two\r\n\r\nSecond para\r\n\r\n\r\nThird"
    doc = get_parser("a.txt").parse(data, "a.txt")
    assert kinds(doc) == [("paragraph", "First para line one\nline two"),
                          ("paragraph", "Second para"), ("paragraph", "Third")]


def test_txt_utf8_bom_and_utf16():
    assert get_parser("a.txt").parse("﻿café ☕".encode("utf-8"), "a.txt").blocks[0].text == "café ☕"
    assert get_parser("a.txt").parse("hello €".encode("utf-16"), "a.txt").blocks[0].text == "hello €"


def test_txt_unsupported_encoding():
    with pytest.raises(UnsupportedEncodingError, match="UTF-8"):
        get_parser("a.txt").parse("café".encode("latin-1"), "a.txt")


def test_txt_binary_is_corrupt():
    with pytest.raises(CorruptDocumentError):
        get_parser("a.txt").parse(b"abc\x00\x01def", "a.txt")
