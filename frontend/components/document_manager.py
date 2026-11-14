"""Upload, list and delete documents via the backend."""

from __future__ import annotations

from pathlib import PurePath

import streamlit as st

from frontend import api_client, config
from frontend.components import ui

RECENT_KEY = "recent_ingests"


def validate_upload(name: str, size: int) -> str | None:
    """Client-side pre-check (the backend validates again). Returns an error message or None."""
    ext = PurePath(name).suffix.lower().lstrip(".")
    if ext not in config.SUPPORTED_TYPES:
        return f"{name}: unsupported type .{ext or '?'} (supported: {', '.join(config.SUPPORTED_TYPES)})"
    if size == 0:
        return f"{name}: file is empty"
    if size > config.MAX_UPLOAD_MB * 1024 * 1024:
        return f"{name}: larger than {config.MAX_UPLOAD_MB} MB"
    return None


def render_upload() -> None:
    st.subheader("Upload documents")
    files = st.file_uploader("PDF, HTML, Markdown or TXT files", type=config.SUPPORTED_TYPES,
                             accept_multiple_files=True, key="uploader")
    version = st.text_input("Version label (optional)", help="Applied to every file in this upload, for example 2.1. "
                            "Leave empty if unknown; versions are never guessed.")
    if st.button("Ingest documents", type="primary", disabled=not files):
        ready, problems = [], []
        for f in files:
            data = f.getvalue()
            err = validate_upload(f.name, len(data))
            (problems if err else ready).append(err or (f.name, data, f.type))
        for p in problems:
            st.error(p)
        if ready:
            with st.spinner(f"Parsing, chunking and embedding {len(ready)} file(s)… this can take a while."):
                ok, result = ui.call_api(api_client.get_client().ingest, ready, version.strip() or None)
            if ok:
                _show_result(result)
                st.session_state.setdefault(RECENT_KEY, []).insert(0, {
                    "files": [d["filename"] for d in result["documents"]], "chunks": result["chunks_indexed"],
                    "failed": len(result["failures"])})


def _show_result(result: dict) -> None:
    if result["documents"]:
        st.success(f"Indexed {len(result['documents'])} document(s); {result['chunks_indexed']} new chunk(s) stored. "
                   "(0 new chunks means the file was already indexed unchanged.)")
        st.dataframe([{"Document": d["filename"], "Type": d["doc_type"], "Chunks": d["chunk_count"],
                       "Version": d["version"] or "—"} for d in result["documents"]], hide_index=True,
                     width="stretch")
    for f in result["failures"]:
        st.error(f"**{f['filename']}** was not indexed: {f['reason']}")


FLASH_KEY = "documents_flash"


def render_documents() -> None:
    flash = st.session_state.pop(FLASH_KEY, None)
    if flash:
        st.success(flash)
    top = st.columns([1, 5])
    if top[0].button("🔄 Refresh"):
        st.rerun()
    st.subheader("Indexed documents")
    ok, docs = ui.call_api(api_client.get_client().documents)
    if not ok:
        return
    if not docs:
        st.info("No documents indexed yet. Upload some above.")
        return
    st.caption(f"{len(docs)} document(s), {sum(d['chunk_count'] for d in docs)} chunks")
    for d in docs:
        cols = st.columns([4, 1, 1, 1, 1])
        cols[0].markdown(f"**{d['filename']}**" + (f"  \n<span class='subtle'>{d['title']}</span>" if d["title"] else ""),
                         unsafe_allow_html=True)
        cols[1].write(d["doc_type"].upper())
        cols[2].write(f"{d['chunk_count']} chunks")
        cols[3].write(f"v{d['version']}" if d["version"] else "no version")
        if cols[4].button("🗑 Delete", key=f"del_{d['doc_id']}"):
            ok2, res = ui.call_api(api_client.get_client().delete_document, d["doc_id"])
            if ok2:
                st.session_state[FLASH_KEY] = f"Deleted {d['filename']} ({res['deleted_chunks']} chunks)."
                st.rerun()
