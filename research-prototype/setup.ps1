# research-prototype / setup.ps1
# Idempotent environment setup. Safe to run twice.
#   .\setup.ps1            -> create .venv if missing, install requirements
#   .\setup.ps1 -Reset     -> delete .venv first, then recreate (recovery)
# Never writes outside this folder. HF cache lives in .\models\hf.
param([switch]$Reset)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root ".venv"
$py   = Join-Path $venv "Scripts\python.exe"

# pip unpacks wheels into %TEMP%; point it inside the project (env-22: no temp on C:)
$tmp = Join-Path $root "agent-work\tmp"
New-Item -ItemType Directory -Force $tmp | Out-Null
$env:TMP = $tmp; $env:TEMP = $tmp
$env:PIP_CACHE_DIR = Join-Path $root "agent-work\pip-cache"

if ($Reset -and (Test-Path $venv)) {
    Write-Host "resolved for deletion: $((Resolve-Path $venv).Path)"
    Remove-Item -Recurse -Force $venv
}

if (-not (Test-Path $py)) {
    Write-Host "creating venv with python 3.12 ..."
    py -3.12 -m venv $venv
}

& $py -m pip install --upgrade pip -q
& $py -m pip install -r (Join-Path $root "requirements.txt")

# kernelspec inside the venv, so `python -m nbconvert --ExecutePreprocessor.kernel_name=research-prototype`
# works without touching the user-level Jupyter directory on C:
& $py -m ipykernel install --prefix $venv --name research-prototype --display-name "Python 3.12 (research-prototype)" | Out-Null

# proof that the venv is the one running, and that the GPU is visible
& $py -c "import sys, torch; print('python  :', sys.version.split()[0]); print('exe     :', sys.executable); print('prefix  :', sys.prefix); print('torch   :', torch.__version__); print('cuda    :', torch.version.cuda, torch.cuda.is_available()); print('device  :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('bf16    :', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
