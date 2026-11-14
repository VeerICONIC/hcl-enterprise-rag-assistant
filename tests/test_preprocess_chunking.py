import pytest

from app.ingestion.chunking import chunk_document, with_context
from app.ingestion.models import Block, ParsedDocument
from app.ingestion.preprocess import normalize_text, preprocess_document


def doc(blocks, title="Policy"):
    return ParsedDocument(filename="p.md", doc_type="md", title=title, blocks=blocks)


# ---- preprocessing ---------------------------------------------------------
def test_whitespace_and_line_endings_normalized():
    assert normalize_text("a  b\t c \r\nline2   \r\n\r\n\r\n\r\nline3\u00a0x\u200b") == "a b c\nline2\n\nline3 x"


def test_meaningful_content_untouched():
    s = "Error E-4021: limit=1,000/min (v2.1) – user@example.com C:\\tmp #tag"
    assert normalize_text(s) == s


def test_preserve_layout_keeps_indentation():
    assert normalize_text("a\n    b   c  ", preserve_layout=True) == "a\n    b   c"


def test_paragraphs_and_headings_preserved_and_empty_dropped():
    d = preprocess_document(doc([
        Block("heading", "  Annual \n Leave ", level=2),
        Block("paragraph", "one  two\n\n\n\nthree"),
        Block("paragraph", "  \n  "),
        Block("table", "| a   | b |\n|---|---|"),
    ]))
    assert [(b.kind, b.text) for b in d.blocks] == [
        ("heading", "Annual Leave"), ("paragraph", "one two\n\nthree"), ("table", "| a   | b |\n|---|---|")]
    assert d.blocks[0].level == 2


def test_empty_document_stays_empty():
    assert preprocess_document(doc([Block("paragraph", "  \n")])).blocks == []


# ---- chunking --------------------------------------------------------------
def test_small_doc_single_chunk_with_section_path():
    d = doc([Block("heading", "Policy", level=1), Block("heading", "Annual Leave", level=2),
             Block("paragraph", "Employees get 20 days.")])
    chunks = chunk_document(d, 500, 50)
    assert len(chunks) == 1
    assert chunks[0].section == "Annual Leave"  # title heading not repeated
    assert chunks[0].body == "Employees get 20 days."


def test_section_path_nesting_and_reset():
    d = doc([Block("heading", "A", level=1), Block("paragraph", "pa"),
             Block("heading", "B", level=2), Block("paragraph", "pb"),
             Block("heading", "C", level=3), Block("paragraph", "pc"),
             Block("heading", "D", level=2), Block("paragraph", "pd")], title="")
    assert [(c.section, c.body) for c in chunk_document(d, 500, 0)] == [
        ("A", "pa"), ("A > B", "pb"), ("A > B > C", "pc"), ("A > D", "pd")]


def test_chunks_respect_size_and_overlap_between_sentences():
    sentences = [f"Sentence number {i} says something useful." for i in range(30)]
    d = doc([Block("paragraph", " ".join(sentences))])
    chunks = chunk_document(d, 200, 60)
    assert len(chunks) > 3
    assert all(0 < len(c.body) <= 200 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:]):
        last_sentence = prev.body.split(". ")[-1].rstrip(".")
        assert last_sentence.split(" says")[0] in nxt.body  # overlap present
        assert len(nxt.body) >= 1
    # every sentence appears somewhere (nothing lost)
    joined = " ".join(c.body for c in chunks)
    assert all(s in joined for s in sentences)


def test_no_overlap_when_disabled_and_no_duplicate_chunks():
    d = doc([Block("paragraph", " ".join(f"Sentence {i} is here." for i in range(40)))])
    chunks = chunk_document(d, 150, 0)
    text = "\n".join(c.body for c in chunks)
    assert text.count("Sentence 7 is here.") == 1
    assert len({c.body for c in chunks}) == len(chunks)


def test_long_unbroken_text_is_split_safely():
    chunks = chunk_document(doc([Block("paragraph", "x" * 1050)]), 200, 20)
    assert all(len(c.body) <= 200 for c in chunks)
    assert sum(len(c.body) for c in chunks) == 1050


def test_long_paragraph_without_punctuation_splits_on_words():
    chunks = chunk_document(doc([Block("paragraph", " ".join(["word"] * 300))]), 100, 10)
    assert all(len(c.body) <= 100 for c in chunks) and len(chunks) >= 12


def test_table_kept_whole_when_it_fits():
    table = "| a | b |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    d = doc([Block("paragraph", "Intro text."), Block("table", table)])
    chunks = chunk_document(d, 500, 50)
    assert len(chunks) == 1 and table in chunks[0].body


def test_oversized_table_splits_by_rows_and_repeats_header():
    rows = [f"| row{i} | value{i} |" for i in range(40)]
    table = "\n".join(["| Name | Value |", "| --- | --- |"] + rows)
    chunks = chunk_document(doc([Block("table", table)]), 200, 20)
    assert len(chunks) > 1
    for c in chunks:
        assert c.body.startswith("| Name | Value |\n| --- | --- |")
        assert len(c.body) <= 200
    body_rows = [l for c in chunks for l in c.body.splitlines() if l.startswith("| row")]
    assert body_rows == rows  # no row lost or duplicated


def test_overlap_never_copies_lists_or_tables():
    lst = "\n".join(f"- item {i}" for i in range(60))
    chunks = chunk_document(doc([Block("list", lst)]), 120, 100)
    items = [l for c in chunks for l in c.body.splitlines()]
    assert items == lst.splitlines()


def test_chunks_do_not_cross_pages():
    d = doc([Block("paragraph", "Page one text.", page=1), Block("paragraph", "Page two text.", page=2)])
    chunks = chunk_document(d, 500, 50)
    assert [(c.page, c.body) for c in chunks] == [(1, "Page one text."), (2, "Page two text.")]


def test_heading_only_and_empty_documents_yield_no_chunks():
    assert chunk_document(doc([Block("heading", "Only heading", level=1)]), 200, 20) == []
    assert chunk_document(doc([]), 200, 20) == []


def test_no_empty_chunks_ever():
    d = doc([Block("paragraph", "a. " * 200), Block("list", "- x\n- y"), Block("code", "```\n" + "z\n" * 300 + "```")])
    assert all(c.body.strip() for c in chunk_document(d, 120, 30))


def test_invalid_parameters():
    with pytest.raises(ValueError):
        chunk_document(doc([]), 100, 100)


def test_with_context_format():
    assert with_context("Employee Leave Policy", "Annual Leave", "Body.") == \
        "Document: Employee Leave Policy\nSection: Annual Leave\n\nBody."
    assert with_context("T", "", "Body.") == "Document: T\n\nBody."
    assert with_context("", "", "Body.") == "Body."
