"""Chat UI: history in session state, every answer comes from POST /query."""

from __future__ import annotations

import streamlit as st

from frontend import api_client
from frontend.components import ui
from frontend.components.conflict_viewer import render_conflicts
from frontend.components.source_viewer import render_citations, render_retrieved

HISTORY_KEY = "chat_messages"
_CONF = {"high": "ok", "medium": "warn", "low": "bad"}


def history() -> list[dict]:
    return st.session_state.setdefault(HISTORY_KEY, [])


def render_answer(resp: dict) -> None:
    ans = resp["answer"]
    status = ans["status"]
    if status == "not_found":
        st.warning("**Not enough evidence.** " + ans["text"])
    elif status == "needs_clarification":
        st.info("**I need a bit more detail.** " + (ans.get("clarifying_question") or ans["text"]))
    else:
        st.markdown(ans["text"])
    st.markdown(ui.badge(f"Confidence: {ans['confidence']}", _CONF.get(ans["confidence"], "info")) + "&nbsp;"
                + ui.badge(f"Status: {status.replace('_', ' ')}", "info")
                + (f"&nbsp;<span class='subtle'>{resp['latency_ms']['total'] / 1000:.1f}s</span>" if resp.get("latency_ms") else ""),
                unsafe_allow_html=True)
    render_conflicts(ans.get("conflicts"), ans.get("citations", []))
    render_citations(ans.get("citations", []))
    for w in ans.get("warnings", []):
        st.caption(f"⚠️ {w}")
    render_retrieved(resp.get("retrieved"))


def render_history() -> None:
    for msg in history():
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            elif msg.get("response"):
                render_answer(msg["response"])
            else:
                st.error(msg["content"])


def ask(question: str, *, top_k: int, doc_ids: list[str] | None) -> None:
    """Send ``question`` to the backend and append both turns to the history."""
    history().append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents and drafting a grounded answer…"):
            try:
                resp = api_client.get_client().query(question, top_k=top_k, doc_ids=doc_ids, debug=True)
            except api_client.ApiConnectionError as exc:
                msg = f"🔌 {exc}"
                st.error(msg)
                history().append({"role": "assistant", "content": msg})
                return
            except api_client.ApiError as exc:
                msg = f"⚠️ {exc.message} ({exc.code})"
                st.error(msg)
                history().append({"role": "assistant", "content": msg})
                return
        render_answer(resp)
    history().append({"role": "assistant", "content": resp["answer"]["text"], "response": resp})
