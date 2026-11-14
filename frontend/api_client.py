"""Thin HTTP client for the FastAPI backend: the frontend's only way to reach any data.

Streamlit never touches Bedrock or ChromaDB; every call below is a backend request.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx

from frontend import config


class ApiError(Exception):
    """The backend answered with an error. ``code``/``message`` come from its uniform error body."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


class ApiConnectionError(Exception):
    """The backend could not be reached (not running, wrong URL, timeout)."""


class ApiClient:
    def __init__(self, base_url: Optional[str] = None, client: Optional[httpx.Client] = None):
        self.base_url = (base_url or config.API_BASE_URL).rstrip("/")
        self._http = client or httpx.Client()

    def _request(self, method: str, path: str, *, timeout: float = 15.0, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._http.request(method, url, timeout=timeout, **kwargs)
        except httpx.TimeoutException:
            raise ApiConnectionError(f"The backend at {self.base_url} did not respond in time.") from None
        except httpx.HTTPError:
            raise ApiConnectionError(
                f"Cannot reach the backend at {self.base_url}. Is FastAPI running "
                "(uvicorn app.api.main:app --port 8000)?") from None
        if resp.status_code >= 400:
            try:
                err = resp.json()["error"]
                raise ApiError(resp.status_code, str(err.get("code", "error")), str(err.get("message", "")))
            except (ValueError, KeyError, TypeError, AttributeError):
                raise ApiError(resp.status_code, f"http_{resp.status_code}", resp.text[:300] or "request failed") from None
        return resp.json()

    # ---- endpoints -------------------------------------------------------------
    def health(self) -> dict:
        return self._request("GET", "/health")

    def documents(self) -> list[dict]:
        return self._request("GET", "/documents")

    def ingest(self, files: list[tuple[str, bytes, str]], version: Optional[str] = None) -> dict:
        payload = [("files", (name, data, mime or "application/octet-stream")) for name, data, mime in files]
        return self._request("POST", "/ingest", timeout=900.0, files=payload,
                             data={"version": version} if version else None)

    def delete_document(self, doc_id: str) -> dict:
        return self._request("DELETE", f"/documents/{doc_id}")

    def query(self, question: str, *, top_k: int = 5, doc_ids: Optional[list[str]] = None,
              versions: Optional[list[str]] = None, debug: bool = True) -> dict:
        body: dict[str, Any] = {"question": question, "top_k": top_k, "debug": debug}
        if doc_ids:
            body["doc_ids"] = doc_ids
        if versions:
            body["versions"] = versions
        return self._request("POST", "/query", timeout=180.0, json=body)

    def latest_evaluation(self) -> Optional[dict]:
        """Latest evaluation results, or None if none have been produced yet."""
        try:
            return self._request("GET", "/evaluation/latest")
        except ApiError as exc:
            if exc.status == 404:
                return None
            raise


    def latest_report(self) -> Optional[str]:
        """The Markdown evaluation report text, or None if there is none yet."""
        try:
            return self._request("GET", "/evaluation/report")["markdown"]
        except ApiError as exc:
            if exc.status == 404:
                return None
            raise


_client: Optional[ApiClient] = None


def get_client() -> ApiClient:
    """Process-wide client (pages call this so tests can substitute a fake)."""
    global _client
    if _client is None:
        _client = ApiClient()
    return _client
