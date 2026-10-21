# Idempotent environment build. Re-running DELETES .venv and rebuilds it (agents.md §3).
# Torch comes from the CUDA 12.8 wheel index; the RTX 3050 (sm_86) + driver 617.42 run it fine.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root '.venv'
$py   = Join-Path $venv 'Scripts\python.exe'

if (Test-Path $venv) { Write-Host '[1/4] removing existing .venv'; Remove-Item -Recurse -Force $venv }
Write-Host '[2/4] creating venv'
python -m venv $venv
Write-Host '[3/4] installing (this is the slow step)'
& $py -m pip install --upgrade pip
& $py -m pip install -r (Join-Path $root 'requirements.txt') `
    --index-url https://download.pytorch.org/whl/cu128 `
    --extra-index-url https://pypi.org/simple
Write-Host '[4/4] freezing exact versions'
& $py -m pip freeze | Out-File -Encoding utf8 (Join-Path $root 'requirements.lock.txt')
& $py -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
