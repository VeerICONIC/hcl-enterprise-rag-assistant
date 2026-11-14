# Start the FastAPI backend (Terminal 1).
#   powershell -ExecutionPolicy Bypass -File scripts\start_api.ps1            # http://localhost:8000
#   powershell -ExecutionPolicy Bypass -File scripts\start_api.ps1 -Port 8100
param([int]$Port = 8000)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Virtual environment not found at .venv. Run: py -3.12 -m venv .venv ; .\.venv\Scripts\python -m pip install -r requirements-dev.txt"
    exit 1
}
Write-Host "FastAPI: http://localhost:$Port   (docs: http://localhost:$Port/docs)   Ctrl+C to stop"
& $py -m uvicorn app.api.main:app --host 127.0.0.1 --port $Port
