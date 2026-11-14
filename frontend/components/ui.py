"""Shared page setup, styling and small widgets."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# make `frontend.*` and `app.*` importable however Streamlit was launched
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from frontend import api_client, config  # noqa: E402

_CSS = """
<style>
  .block-container {padding-top: 2rem; max-width: 1200px;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  .badge {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;}
  .badge-ok {background:#e3f6ea; color:#136c36;}
  .badge-warn {background:#fff3d6; color:#8a5a00;}
  .badge-bad {background:#fde4e4; color:#9b1c1c;}
  .badge-info {background:#e4effd; color:#1d4f91;}
  .subtle {color: #6b7280; font-size: 0.9rem;}
  div[data-testid="stMetric"] {background: rgba(128,128,128,0.08); border-radius: 12px; padding: 12px 16px;}
  .excerpt {border-left: 3px solid #4f8bf9; padding: 4px 12px; margin: 6px 0; background: rgba(79,139,249,0.06);}
</style>
"""

BADGE_KINDS = {"ok": "badge-ok", "warn": "badge-warn", "bad": "badge-bad", "info": "badge-info"}


def badge(text: str, kind: str = "info") -> str:
    return f'<span class="badge {BADGE_KINDS.get(kind, "badge-info")}">{text}</span>'


def setup_page(title: str, icon: str = "📄", subtitle: str = "") -> None:
    """Page config (idempotent per script run), CSS, sidebar and heading."""
    try:
        st.set_page_config(page_title=f"{title} · {config.APP_TITLE}", page_icon=icon, layout="wide")
    except st.errors.StreamlitAPIException:
        pass
    st.markdown(_CSS, unsafe_allow_html=True)
    from frontend.components.sidebar import render_sidebar
    render_sidebar()
    st.title(f"{icon} {title}")
    if subtitle:
        st.markdown(f'<p class="subtle">{subtitle}</p>', unsafe_allow_html=True)


def call_api(fn, *args, **kwargs):
    """Run a backend call, showing a readable error (never a traceback). Returns (ok, result)."""
    try:
        return True, fn(*args, **kwargs)
    except api_client.ApiConnectionError as exc:
        st.error(f"🔌 {exc}")
    except api_client.ApiError as exc:
        st.error(f"⚠️ Backend error ({exc.status} · {exc.code}): {exc.message}")
    return False, None
