"""Conversational Q&A over the indexed documents."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from frontend import api_client  # noqa: E402
from frontend.components import chat_interface, ui  # noqa: E402

ui.setup_page("Ask Documents", "💬", "Answers use only your indexed documents and cite their sources.")

with st.sidebar:
    st.subheader("Search options")
    top_k = st.slider("Passages to retrieve", 1, 10, 5)
    ok, docs = ui.call_api(api_client.get_client().documents)
    options = {f"{d['filename']}" + (f" (v{d['version']})" if d["version"] else ""): d["doc_id"] for d in docs} if ok else {}
    chosen = st.multiselect("Limit to documents (optional)", list(options))
    if st.button("Clear conversation"):
        st.session_state[chat_interface.HISTORY_KEY] = []
        st.rerun()

if ok and not docs:
    st.info("No documents are indexed yet. Upload some on the **Documents** page first.")

chat_interface.render_history()
question = st.chat_input("Ask a question about your documents…")
if question and question.strip():
    chat_interface.ask(question.strip(), top_k=top_k, doc_ids=[options[c] for c in chosen] or None)
