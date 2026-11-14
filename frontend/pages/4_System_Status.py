"""Operational status. No credentials are ever displayed."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from frontend import api_client  # noqa: E402
from frontend.components import ui  # noqa: E402

ui.setup_page("System Status", "🩺", "Backend, vector store and model configuration.")
client = api_client.get_client()
t0 = time.perf_counter()
ok, health = ui.call_api(client.health)
latency = (time.perf_counter() - t0) * 1000
if ok:
    ok_docs, docs = ui.call_api(client.documents)
    c1, c2, c3 = st.columns(3)
    c1.metric("FastAPI", health["status"].upper())
    c2.metric("ChromaDB chunks", health["chroma_count"])
    c3.metric("Health latency", f"{latency:.0f} ms")
    st.subheader("Configuration (from the backend)")
    st.table({"Setting": ["API base URL", "Embedding model", "LLM model", "Indexed documents"],
              "Value": [client.base_url, health["embed_model"], health["llm_model"], str(len(docs)) if ok_docs else "n/a"]})
    st.info("`/health` does not call AWS, so it cannot confirm that Bedrock model access works. Verify that with "
            "`python scripts/test_bedrock.py --all`. A failed upload or question shows the exact AWS error.")
    if ok_docs and docs:
        st.subheader("Documents by type")
        counts: dict[str, int] = {}
        for d in docs:
            counts[d["doc_type"]] = counts.get(d["doc_type"], 0) + 1
        st.bar_chart(counts)
