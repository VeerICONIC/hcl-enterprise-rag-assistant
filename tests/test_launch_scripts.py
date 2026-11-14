import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
NAMES = ["start_api.ps1", "start_ui.ps1", "start_all.ps1", "stop_all.ps1"]
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell not available")
@pytest.mark.parametrize("name", NAMES)
def test_launch_script_parses(name):
    cmd = (f"$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile('{SCRIPTS / name}',[ref]$null,[ref]$e); "
           "if ($e.Count) { $e | ForEach-Object { $_.Message }; exit 1 }")
    r = subprocess.run([POWERSHELL, "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("name", NAMES)
def test_launch_scripts_exist_and_have_no_credentials(name):
    text = (SCRIPTS / name).read_text(encoding="utf-8").lower()
    assert "aws_secret" not in text and "aws_access_key" not in text and "session_token" not in text


def test_two_terminal_launch_commands_are_documented_and_uses_localhost_only():
    ui, api = (SCRIPTS / "start_ui.ps1").read_text(encoding="utf-8"), (SCRIPTS / "start_api.ps1").read_text(encoding="utf-8")
    assert "--server.address localhost" in ui and "--host 127.0.0.1" in api
    assert "app.api.main:app" in api and "frontend/app.py" in ui and "API_BASE_URL" in ui
