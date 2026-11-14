"""Conflict warnings."""

from __future__ import annotations

import streamlit as st

_KIND = {"version_difference": "Version difference", "contradiction": "Possible contradiction",
         "date_difference": "Possible date difference"}


def render_conflicts(conflicts: dict | None, citations: list[dict]) -> None:
    if not conflicts or not conflicts.get("has_conflict"):
        return
    names = {c["chunk_id"]: c["filename"] for c in citations}
    st.warning("**The sources may disagree.** The answer shows each side; it does not pick a winner. "
               "Detection is automatic and heuristic, so check the cited passages.")
    for g in conflicts.get("groups", []):
        files = sorted({names[i] for i in g["chunk_ids"] if i in names})
        label = _KIND.get(g["kind"], g["kind"])
        with st.expander(f"{label}: {g['topic']}" + (f" ({', '.join(files)})" if files else "")):
            st.write(g["description"])
