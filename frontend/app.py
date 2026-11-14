"""Overview page. Run with: streamlit run frontend/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from frontend import api_client  # noqa: E402
from frontend.components import ui  # noqa: E402
from frontend.components.document_manager import RECENT_KEY  # noqa: E402

ui.setup_page("Overview", "🏠", "Upload enterprise documents, ask questions, and inspect the evidence behind every answer.")

client = api_client.get_client()
ok, health = ui.call_api(client.health)
ok_docs, docs = ui.call_api(client.documents) if ok else (False, None)

if ok:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indexed documents", len(docs) if ok_docs else "n/a")
    c2.metric("Indexed chunks", health["chroma_count"])
    c3.metric("Backend", health["status"].upper())
    c4.metric("Vector store", "HEALTHY")
    st.subheader("Model configuration")
    m1, m2 = st.columns(2)
    m1.markdown(f"**Embedding model**  \n`{health['embed_model']}`")
    m2.markdown(f"**Answer model**  \n`{health['llm_model']}`")
    st.caption("Reported by the backend configuration. This does not prove that AWS has granted access to the models; "
               "see System Status.")

    st.subheader("Recent ingestion (this session)")
    recent = st.session_state.get(RECENT_KEY, [])
    if recent:
        st.dataframe([{"Files": ", ".join(r["files"]) or "—", "New chunks": r["chunks"], "Failed": r["failed"]}
                      for r in recent[:10]], hide_index=True, width="stretch")
    else:
        st.info("Nothing ingested in this browser session yet. Go to **Documents** to upload files.")

    st.subheader("How it works")
    st.markdown("1. **Documents**: upload PDF, HTML, Markdown or TXT files.  \n"
                "2. **Ask Documents**: ask in natural language; answers cite their sources.  \n"
                "3. Conflicting or version-dependent sources are shown side by side, and unsupported questions get a clear "
                "“not enough evidence” reply.")
else:
    st.info("Start the backend, then refresh this page:  `uvicorn app.api.main:app --port 8000`")
