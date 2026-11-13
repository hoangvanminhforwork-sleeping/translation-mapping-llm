# From-zero environment for research-scale-100k. Safe to run twice.
# Fast path used on 2026-11-13: copy research-prototype\.venv instead of installing (same pins).
param([switch]$FromScratch)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "venv exists - reusing (delete .venv to reinstall)"
} else {
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
}

# Kernel registration is idempotent: it overwrites the same kernelspec.
.\.venv\Scripts\python.exe -m ipykernel install --user --name research-scale-100k --display-name "research-scale-100k"

# Shared checkpoint cache (7.6 GB, already downloaded by research-prototype).
$env:HF_HOME = (Resolve-Path "$root\..\research-prototype\models\hf").Path
Write-Host "HF_HOME=$env:HF_HOME"

.\.venv\Scripts\python.exe -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
