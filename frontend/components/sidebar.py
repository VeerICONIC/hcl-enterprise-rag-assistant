"""Sidebar: backend status at a glance and navigation hints."""

from __future__ import annotations

import streamlit as st

from frontend import api_client, config
from frontend.components import ui


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(f"### {config.APP_TITLE}")
        st.caption("Grounded answers from your own documents, with citations.")
        try:
            h = api_client.get_client().health()
            st.markdown(ui.badge("Backend online", "ok") + "&nbsp;" + ui.badge(f"{h.get('chroma_count', 0)} chunks", "info"),
                        unsafe_allow_html=True)
        except api_client.ApiConnectionError:
            st.markdown(ui.badge("Backend offline", "bad"), unsafe_allow_html=True)
        except api_client.ApiError:
            st.markdown(ui.badge("Backend error", "warn"), unsafe_allow_html=True)
        st.caption(f"API: {api_client.get_client().base_url}")
        st.divider()
        st.caption("Use the pages above: Documents → Ask Documents → Evaluation → System Status.")
