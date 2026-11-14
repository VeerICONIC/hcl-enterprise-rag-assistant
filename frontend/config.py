"""Frontend configuration. Independent of the backend package (the two are deployed separately).

Each setting is looked up in this order (first hit wins):
  1. process environment variable            (local shells, ``scripts\\start_ui.ps1``, Streamlit Cloud also exports secrets here)
  2. Streamlit secrets (``st.secrets``)       (Streamlit Community Cloud -> App settings -> Secrets)
  3. the project's local ``.env`` file        (local development only; never present in a cloud deployment)
  4. the built-in default                     (``http://localhost:8000`` for the API URL)

Settings: ``API_BASE_URL``, ``API_KEY`` (optional shared secret sent as ``X-API-Key``),
``API_TIMEOUT_SECONDS`` (default 20), ``MAX_UPLOAD_SIZE_MB`` (default 25).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

SUPPORTED_TYPES = ["pdf", "html", "htm", "md", "markdown", "txt"]
APP_TITLE = "Enterprise Documentation Q&A Assistant"
DEFAULT_API_URL = "http://localhost:8000"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "[::1]", "::1")

_ROOT = Path(__file__).resolve().parents[1]


def _from_streamlit_secrets(name: str) -> Optional[str]:
    try:
        import streamlit as st

        value = st.secrets.get(name)  # raises if there is no secrets file at all
    except Exception:
        return None
    return str(value).strip() if value not in (None, "") else None


def _from_dotenv(name: str) -> Optional[str]:
    """Minimal ``KEY=value`` reader for the local .env (no dependencies; comments and blanks ignored)."""
    try:
        for line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == name:
                value = value.strip().strip('"').strip("'")
                return value or None
    except OSError:
        pass
    return None


def lookup(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name, "").strip()
    return value or _from_streamlit_secrets(name) or _from_dotenv(name) or default


def normalize_url(raw: str) -> str:
    """Add a missing scheme (https for real hosts, http for localhost), drop trailing slashes."""
    url = raw.strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        host = url.split("/", 1)[0].split(":", 1)[0].lower()
        url = ("http://" if host in LOCAL_HOSTS else "https://") + url
    return url


def is_local_url(url: str) -> bool:
    host = url.split("://", 1)[-1].split("/", 1)[0]
    host = host.rsplit("@", 1)[-1]
    host = host.split(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
    return host.lower() in LOCAL_HOSTS


def running_on_streamlit_cloud() -> bool:
    """Heuristic: Streamlit Community Cloud mounts the repository under /mount/src."""
    return Path.cwd().as_posix().startswith("/mount/src") or str(_ROOT).startswith("/mount/src")


def _int(name: str, default: int) -> int:
    try:
        return max(1, int(lookup(name, str(default)) or default))
    except ValueError:
        return default


API_BASE_URL: str = normalize_url(lookup("API_BASE_URL", DEFAULT_API_URL) or DEFAULT_API_URL)
API_KEY: Optional[str] = lookup("API_KEY")
API_TIMEOUT: int = _int("API_TIMEOUT_SECONDS", 20)
MAX_UPLOAD_MB: int = _int("MAX_UPLOAD_SIZE_MB", 25)
IS_LOCAL_API: bool = is_local_url(API_BASE_URL)
#: True when deployed on Streamlit Cloud but the API URL still points at localhost (the usual misconfiguration)
MISCONFIGURED_FOR_CLOUD: bool = running_on_streamlit_cloud() and IS_LOCAL_API
