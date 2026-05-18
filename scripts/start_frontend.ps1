$ErrorActionPreference = "Stop"
Set-Location "C:\Projects\2026-HAI-ProjectTeam3"
python -m http.server 3001 --directory frontend
