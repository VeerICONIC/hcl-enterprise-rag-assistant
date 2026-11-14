# Optional launcher: starts the API and the dashboard in two separate windows, waits for the API, and
# lets the dashboard open your browser. The two-terminal way (start_api.ps1 / start_ui.ps1) still works.
#   powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\start_all.ps1 -Hidden -NoBrowser   # no windows (for testing)
# Stop with: powershell -ExecutionPolicy Bypass -File scripts\stop_all.ps1
param([int]$ApiPort = 8000, [int]$UiPort = 8501, [switch]$Hidden, [switch]$NoBrowser)

$here = $PSScriptRoot
$style = if ($Hidden) { "Hidden" } else { "Normal" }
$shell = (Get-Process -Id $PID).Path

Start-Process $shell -WindowStyle $style -ArgumentList @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$here\start_api.ps1", "-Port", $ApiPort)

Write-Host "Waiting for the API on port $ApiPort ..."
$ok = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$ApiPort/health" -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { Start-Sleep -Seconds 1 }
}
if (-not $ok) { Write-Error "The API did not become healthy within 60 s. Check its window for errors."; exit 1 }
Write-Host "API is up."

$uiArgs = @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$here\start_ui.ps1", "-Port", $UiPort, "-ApiPort", $ApiPort)
if ($NoBrowser) { $uiArgs += "-NoBrowser" }
Start-Process $shell -WindowStyle $style -ArgumentList $uiArgs
Write-Host "Dashboard: http://localhost:$UiPort    Backend: http://localhost:$ApiPort"
