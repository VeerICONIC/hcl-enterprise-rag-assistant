# Stop whatever is listening on the API and dashboard ports (default 8000 and 8501).
#   powershell -ExecutionPolicy Bypass -File scripts\stop_all.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\stop_all.ps1 -Ports "8100,8601"
param([string]$Ports = "8000,8501")

foreach ($p in ($Ports -split "[,; ]+" | Where-Object { $_ })) {
    $conns = Get-NetTCPConnection -LocalPort ([int]$p) -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        $proc = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "Stopping $($proc.ProcessName) (PID $($proc.Id)) on port $p"
            Stop-Process -Id $proc.Id -Force
        }
    }
}
