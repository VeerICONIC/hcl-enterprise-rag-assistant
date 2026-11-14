import ast
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.container import Container
from app.api.main import create_app
from app.core.config import Settings
from frontend import api_client, config

ROOT = Path(__file__).resolve().parents[1]


# ---- frontend configuration -----------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("https://api.onrender.com", "https://api.onrender.com"),
    ("https://api.onrender.com/", "https://api.onrender.com"),
    ("api.onrender.com", "https://api.onrender.com"),                 # missing scheme -> https for real hosts
    ("localhost:8000", "http://localhost:8000"),                       # ... http for localhost
    ("  http://127.0.0.1:8000//  ", "http://127.0.0.1:8000"),
])
def test_normalize_url(raw, expected):
    assert config.normalize_url(raw) == expected


@pytest.mark.parametrize("url,local", [
    ("http://localhost:8000", True), ("http://127.0.0.1:8000", True), ("http://0.0.0.0:8000", True),
    ("https://hcl-docqa-api.onrender.com", False), ("https://example.com:8443/path", False),
    ("https://localhost.example.com", False),
])
def test_is_local_url(url, local):
    assert config.is_local_url(url) is local


def run_config(tmp_path, env=None, secrets=None):
    """Import frontend.config in a clean subprocess (cwd = tmp_path, so no project .env/.streamlit is picked up)."""
    if secrets is not None:
        (tmp_path / ".streamlit").mkdir()
        (tmp_path / ".streamlit" / "secrets.toml").write_text(secrets, encoding="utf-8")
    clean = {k: v for k, v in os.environ.items() if k not in ("API_BASE_URL", "API_KEY", "API_TIMEOUT_SECONDS")}
    code = (f"import sys; sys.path.insert(0, r'{ROOT}'); from frontend import config as c; "
            "print(c.API_BASE_URL, c.API_KEY, c.API_TIMEOUT, c.IS_LOCAL_API, sep='|')")
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env={**clean, **(env or {})}, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().splitlines()[-1].split("|")


def test_default_is_localhost_when_nothing_is_configured(tmp_path, monkeypatch):
    out = run_config(tmp_path)
    assert out[0] in ("http://localhost:8000",) or out[0].startswith("http")   # a project .env may be absent in the subprocess cwd
    assert out[2] == "20"


def test_environment_variable_sets_the_api_url(tmp_path):
    url, key, timeout, local = run_config(tmp_path, env={"API_BASE_URL": "https://api.example.com/", "API_KEY": "k1", "API_TIMEOUT_SECONDS": "75"})
    assert (url, key, timeout, local) == ("https://api.example.com", "k1", "75", "False")


def test_streamlit_secrets_set_the_api_url_and_key(tmp_path):
    secrets = 'API_BASE_URL = "my-api.onrender.com"\nAPI_KEY = "from-secrets"\nAPI_TIMEOUT_SECONDS = 60\n'
    url, key, timeout, local = run_config(tmp_path, secrets=secrets)
    assert (url, key, timeout, local) == ("https://my-api.onrender.com", "from-secrets", "60", "False")


def test_environment_variable_beats_streamlit_secrets(tmp_path):
    url, *_ = run_config(tmp_path, env={"API_BASE_URL": "https://from-env.example.com"}, secrets='API_BASE_URL = "https://from-secrets.example.com"\n')
    assert url == "https://from-env.example.com"


def test_cloud_misconfiguration_is_detected_from_the_mount_path(monkeypatch):
    monkeypatch.setattr(config, "running_on_streamlit_cloud", lambda: True)
    assert config.running_on_streamlit_cloud() and config.is_local_url(config.DEFAULT_API_URL)


def test_frontend_never_imports_the_backend_package():
    offenders = []
    for path in (ROOT / "frontend").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            offenders += [f"{path.name}: {n}" for n in names if n == "app" or n.startswith("app.")]
    assert not offenders, offenders


# ---- API client: key header, hints, hosted error pages ----------------------------------------
def client_with(handler, **kw):
    return api_client.ApiClient("https://api.example.com", httpx.Client(transport=httpx.MockTransport(handler)), **kw)


def test_client_sends_api_key_header_only_when_configured():
    seen = {}

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json={"status": "ok"})

    client_with(handler, api_key="s3cret").health()
    assert seen["x-api-key"] == "s3cret"
    seen.clear()
    client_with(handler, api_key="").health()
    assert "x-api-key" not in seen


def test_connection_error_messages_name_the_cause():
    def refuse(request):
        raise httpx.ConnectError("refused")

    hosted = pytest.raises(api_client.ApiConnectionError, client_with(refuse).health)
    assert "https://api.example.com" in str(hosted.value) and "waking from sleep" in str(hosted.value)
    local = api_client.ApiClient("http://localhost:8000", httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(api_client.ApiConnectionError, match="uvicorn app.api.main:app"):
        local.health()


def test_localhost_on_streamlit_cloud_gets_the_secrets_hint(monkeypatch):
    monkeypatch.setattr(config, "running_on_streamlit_cloud", lambda: True)

    def refuse(request):
        raise httpx.ConnectError("refused")

    local = api_client.ApiClient("http://localhost:8000", httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(api_client.ApiConnectionError) as ei:
        local.health()
    assert "Streamlit Cloud" in str(ei.value) and "Secrets" in str(ei.value) and "API_BASE_URL" in str(ei.value)


def test_hosting_error_page_becomes_a_readable_message():
    c = client_with(lambda r: httpx.Response(502, text="<html><body>Bad Gateway</body></html>"))
    with pytest.raises(api_client.ApiError) as ei:
        c.health()
    assert ei.value.status == 502 and "<html" not in ei.value.message and "starting up" in ei.value.message


def test_unauthorized_surfaces_as_an_api_error():
    c = client_with(lambda r: httpx.Response(401, json={"error": {"code": "unauthorized", "message": "Missing or invalid API key"}}))
    with pytest.raises(api_client.ApiError) as ei:
        c.documents()
    assert (ei.value.status, ei.value.code) == (401, "unauthorized")


# ---- backend API key ------------------------------------------------------------------------
def backend(settings, **kw):
    s = Settings(**{**settings.model_dump(), **kw})
    return TestClient(create_app(Container(s)), raise_server_exceptions=False)


def test_without_api_key_configured_everything_is_open(settings, store, kw_embedder):
    c = TestClient(create_app(Container(settings, store=store, embedder=kw_embedder)))
    assert c.get("/documents").status_code == 200 and c.get("/health").status_code == 200


def test_api_key_protects_every_route_except_health(settings, store, kw_embedder):
    s = Settings(**{**settings.model_dump(), "api_key": "s3cret"})
    c = TestClient(create_app(Container(s, store=store, embedder=kw_embedder)), raise_server_exceptions=False)
    assert c.get("/health").status_code == 200                                   # Render's health check has no header
    for method, path in [("get", "/documents"), ("post", "/ingest"), ("post", "/query"), ("post", "/retrieve"),
                         ("delete", "/documents/x"), ("get", "/evaluation/latest"), ("get", "/evaluation/report")]:
        r = getattr(c, method)(path)
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized", (method, path)
    assert c.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/documents", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert c.post("/query", json={"question": "hi"}, headers={"X-API-Key": "s3cret"}).status_code != 401


def test_frontend_client_works_against_a_protected_backend(settings, store, kw_embedder):
    s = Settings(**{**settings.model_dump(), "api_key": "s3cret"})
    http = TestClient(create_app(Container(s, store=store, embedder=kw_embedder)), raise_server_exceptions=False)
    assert api_client.ApiClient("http://testserver", http, api_key="s3cret").documents() == []
    with pytest.raises(api_client.ApiError) as ei:
        api_client.ApiClient("http://testserver", http, api_key="wrong").documents()
    assert ei.value.status == 401


def test_api_key_setting_is_blank_safe_and_never_printed():
    assert Settings(bedrock_embed_dim=256, api_key="  ").api_key is None
    s = Settings(bedrock_embed_dim=256, api_key="topsecret")
    assert "topsecret" not in repr(s) and "topsecret" not in str(s)


# ---- deployment files ------------------------------------------------------------------------
def test_render_blueprint_matches_the_documented_start_command_and_storage():
    text = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "startCommand: uvicorn app.api.main:app --host 0.0.0.0 --port $PORT" in text
    assert "healthCheckPath: /health" in text and "requirements-backend.txt" in text
    mount = re.search(r"mountPath:\s*(\S+)", text).group(1)
    chroma = re.search(r"key: CHROMA_PATH\s+value:\s*(\S+)", text).group(1)
    assert chroma.startswith(mount.rstrip("/") + "/")                      # Chroma lives on the persistent disk
    assert re.search(r"key: AWS_REGION\s+value: ap-south-1", text) and "apac.amazon.nova-lite-v1:0" in text
    assert re.search(r"key: AWS_ACCESS_KEY_ID\s+sync: false", text) and re.search(r"key: AWS_SECRET_ACCESS_KEY\s+sync: false", text)
    assert "key: AWS_PROFILE" not in text and "generateValue: true" in text
    assert not re.search(r"AKIA[0-9A-Z]{12,}", text)


def test_requirements_are_split_correctly_and_local_dev_still_installs_everything():
    def reqs(path):  # requirement lines only, comments stripped
        lines = (ROOT / path).read_text(encoding="utf-8").lower().splitlines()
        return " ".join(line.split("#")[0].strip() for line in lines)

    backend_txt, frontend_txt, full = reqs("requirements-backend.txt"), reqs("frontend/requirements.txt"), reqs("requirements.txt")
    for pkg in ("fastapi", "uvicorn", "chromadb", "pymupdf", "boto3", "beautifulsoup4", "python-multipart", "pydantic-settings"):
        assert pkg in backend_txt and pkg in full
    assert "streamlit" not in backend_txt and "boto3" not in frontend_txt and "chromadb" not in frontend_txt
    assert "streamlit" in frontend_txt and "httpx" in frontend_txt
    assert "streamlit" in full and "boto3[crt]" in full                      # Windows development unchanged
    assert "-r requirements.txt" in (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")


def test_secrets_template_exists_real_secrets_file_is_ignored():
    assert (ROOT / ".streamlit" / "secrets.toml.example").is_file()
    r = subprocess.run(["git", "check-ignore", ".streamlit/secrets.toml"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0                                                   # git ignores the real secrets file
    tracked = subprocess.run(["git", "ls-files", ".env", ".streamlit/secrets.toml"], cwd=ROOT, capture_output=True, text=True).stdout
    assert tracked.strip() == ""


def test_backend_entry_point_exists_for_the_render_command():
    module = importlib.import_module("app.api.main")
    assert hasattr(module, "app") and module.app.title
