"""Frontend configuration. The URL comes from API_BASE_URL (environment or .env), default localhost."""

from __future__ import annotations

import os

SUPPORTED_TYPES = ["pdf", "html", "htm", "md", "markdown", "txt"]
APP_TITLE = "Enterprise Documentation Q&A Assistant"


def _backend_defaults() -> tuple[str, int]:
    """Reuse the backend's settings loader so .env is honoured (no AWS access happens here)."""
    try:
        from app.core.config import get_settings
        s = get_settings()
        return s.api_base_url, s.max_upload_size_mb
    except Exception:  # invalid/missing config must not stop the UI from starting
        return "http://localhost:8000", 25


_url, _max_mb = _backend_defaults()
API_BASE_URL: str = os.getenv("API_BASE_URL", _url).rstrip("/")
MAX_UPLOAD_MB: int = _max_mb
