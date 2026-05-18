$ErrorActionPreference = "Stop"
Set-Location "C:\Projects\2026-HAI-ProjectTeam3"
$env:ROUTING_POLICY_PATH = "outputs/rl_routing_policy.json"
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
