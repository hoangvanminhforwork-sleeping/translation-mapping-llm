# finetune-and-lora / setup.ps1
# Idempotent. Creates ../.venv (the venv is a direct child of finetune-and-lora,
# not of prototype/, as requested), installs pinned requirements, prints the device.
#
#   .\setup.ps1            create if missing, install
#   .\setup.ps1 -Reset     delete and recreate
param([switch]$Reset)

$ErrorActionPreference = "Stop"
$proto = Split-Path -Parent $MyInvocation.MyCommand.Path
$proj = Split-Path -Parent $proto
$venv = Join-Path $proj ".venv"
$py = Join-Path $venv "Scripts\python.exe"

# keep pip temp + cache inside the project (env-22: no temp on C:)
$tmp = Join-Path $proto "agent-work\tmp"
New-Item -ItemType Directory -Force $tmp | Out-Null
$env:TMP = $tmp
$env:TEMP = $tmp
$env:PIP_CACHE_DIR = Join-Path $proto "agent-work\pip-cache"

if ($Reset -and (Test-Path $venv)) {
    Write-Host "resolved for deletion: $((Resolve-Path $venv).Path)"
    Remove-Item -Recurse -Force $venv
}
if (-not (Test-Path $py)) {
    Write-Host "creating venv with python 3.12 ..."
    py -3.12 -m venv $venv
}

& $py -m pip install --upgrade pip -q
& $py -m pip install -r (Join-Path $proto "requirements.txt")

& $py -c "import sys, torch, transformers, peft; print('exe   :', sys.executable); print('torch :', torch.__version__, '| cuda', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('peft  :', peft.__version__)"
