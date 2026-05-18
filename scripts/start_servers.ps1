$ErrorActionPreference = "Stop"

$root = "C:\Projects\2026-HAI-ProjectTeam3"
$powershell = "C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe"

Start-Process `
  -FilePath $powershell `
  -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$root\scripts\start_backend.ps1" `
  -WorkingDirectory $root `
  -RedirectStandardOutput "$root\outputs\backend.runtime.log" `
  -RedirectStandardError "$root\outputs\backend.err.log" `
  -WindowStyle Hidden

Start-Process `
  -FilePath $powershell `
  -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$root\scripts\start_frontend.ps1" `
  -WorkingDirectory $root `
  -RedirectStandardOutput "$root\outputs\frontend.runtime.log" `
  -RedirectStandardError "$root\outputs\frontend.err.log" `
  -WindowStyle Hidden
