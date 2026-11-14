"""Citations and retrieved-evidence display."""

from __future__ import annotations

import streamlit as st


def citation_title(n: int, c: dict) -> str:
    where = []
    if c.get("section"):
        where.append(c["section"])
    if c.get("page", -1) != -1:
        where.append(f"p.{c['page']}")
    ver = f" · v{c['version']}" if c.get("version") else ""
    return f"[{n}] {c['filename']}" + (f" — {' · '.join(where)}" if where else "") + ver


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.markdown("**Sources**")
    for n, c in enumerate(citations, 1):
        with st.expander(citation_title(n, c)):
            st.markdown(f'<div class="excerpt">“{c["excerpt"]}”</div>', unsafe_allow_html=True)
            st.caption(f"Source: `{c['source']}` · Chunk: `{c['chunk_id']}`" + (f" · Title: {c['title']}" if c.get("title") else ""))


def render_retrieved(chunks: list[dict] | None) -> None:
    """Ranking details for the retrieved passages (available when the request used debug=true)."""
    if not chunks:
        return
    with st.expander(f"Retrieved evidence ({len(chunks)} passages, best first)"):
        rows = []
        for rc in chunks:
            m = rc["chunk"]["metadata"]
            rows.append({"Rank": rc["rank"], "Document": m["filename"], "Section": m["section"] or "—",
                         "Page": str(m["page"]) if m["page"] != -1 else "—", "Version": m["version"] or "—",
                         "Cosine": round(rc["vector_score"], 3), "Fusion score": round(rc["score"], 3)})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Cosine = semantic similarity. Fusion score = rank agreement between semantic and keyword search "
                   "(not a probability).")
