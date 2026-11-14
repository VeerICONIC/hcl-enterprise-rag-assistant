# Start the Streamlit dashboard (Terminal 2). The API must be running (scripts\start_api.ps1).
#   powershell -ExecutionPolicy Bypass -File scripts\start_ui.ps1               # http://localhost:8501
#   powershell -ExecutionPolicy Bypass -File scripts\start_ui.ps1 -ApiPort 8100 -Port 8601 -NoBrowser
param([int]$Port = 8501, [int]$ApiPort = 8000, [switch]$NoBrowser)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Virtual environment not found at .venv. Run: py -3.12 -m venv .venv ; .\.venv\Scripts\python -m pip install -r requirements-dev.txt"
    exit 1
}
$env:API_BASE_URL = "http://localhost:$ApiPort"
$headless = if ($NoBrowser) { "true" } else { "false" }
Write-Host "Dashboard: http://localhost:$Port   (backend: $env:API_BASE_URL)   Ctrl+C to stop"
& $py -m streamlit run frontend/app.py --server.address localhost --server.port $Port --server.headless $headless --browser.gatherUsageStats false
