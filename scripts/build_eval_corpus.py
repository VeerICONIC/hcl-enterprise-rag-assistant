"""Create the one binary file of the synthetic evaluation corpus (api_guide.pdf).

Run from the project root:  python scripts/build_eval_corpus.py
The other corpus files are plain text and live in data/eval/corpus/. All content is fictional.
"""
from pathlib import Path

import pymupdf

PAGES = [
    [("API Guide", 20, True), ("Rate Limits", 14, True),
     ("The Standard plan allows 100 requests per minute.", 11, False),
     ("The Enterprise plan allows 1000 requests per minute.", 11, False)],
    [("Payload and Authentication", 14, True),
     ("The maximum request payload size is 5 MB.", 11, False),
     ("Bearer tokens expire after 60 minutes.", 11, False)],
]

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "eval" / "corpus" / "api_guide.pdf"
    doc = pymupdf.open()
    for lines in PAGES:
        page, y = doc.new_page(), 72
        for text, size, bold in lines:
            page.insert_text((72, y), text, fontsize=size, fontname="hebo" if bold else "helv")
            y += size * 2.2
    doc.set_metadata({"title": "API Guide"})
    doc.save(out)
    print(f"wrote {out}")
