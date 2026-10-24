# Phase 5 GPU setup for RTX 50-series (Blackwell, sm_120) on Windows. This is the only supported torch install path.
# Installs the official prebuilt PyTorch 2.14.1 built against CUDA 13.2 from the hash-locked file
# phase5\torch-cu132.lock (byte-identical to the Seraphina desktop install kit's lock). The CUDA runtime is
# bundled inside the wheel, so no separate CUDA toolkit install is needed. Any NVIDIA driver that reports
# CUDA 13.2 or newer in nvidia-smi (e.g. 13.5) runs it.
# Needs Python 3.13 or 3.14 (the lock pins cp313/cp314 win_amd64 wheels).
# Usage (venv active, from the repo root):  .\phase5\setup_gpu.ps1
# Every native call is checked; the script exits non-zero on the first failure.
$ErrorActionPreference = "Stop"
$Index = "https://download.pytorch.org/whl/cu132"
$Lock = Join-Path $PSScriptRoot "torch-cu132.lock"
$LockSha256 = "2a5d73b2e53bdcedb033a5e4cb8097eb2a9c2b38da9e910740a849125ce16d5c"
$Check = Join-Path $PSScriptRoot "check_gpu.py"

function Invoke-Checked {
    param([string]$What, [scriptblock]$Command)
    & $Command
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        Write-Error "FAILED ($What): exit code $code" -ErrorAction Continue
        exit $(if ($code -gt 0) { $code } else { 1 })
    }
}

if (-not (Test-Path -LiteralPath $Lock)) { Write-Error "missing lock file: $Lock" -ErrorAction Continue; exit 1 }
$actual = (Get-FileHash -LiteralPath $Lock -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $LockSha256) {
    Write-Error "lock file hash mismatch: $Lock is $actual, expected $LockSha256" -ErrorAction Continue
    exit 1
}

Invoke-Checked "python 3.13/3.14 check" { python -c "import sys; v = sys.version_info[:2]; print('python %d.%d' % v); sys.exit(0 if v in ((3, 13), (3, 14)) else 1)" }
Invoke-Checked "pip uninstall torch" { python -m pip uninstall -y torch torchvision torchaudio }
Invoke-Checked "pip install torch (hash-locked)" {
    python -m pip install --require-hashes --only-binary=:all: --index-url $Index --extra-index-url https://pypi.org/simple -r $Lock
}
Invoke-Checked "pip install -e .[test,local-nli2]" { python -m pip install -e ".[test,local-nli2]" }
Invoke-Checked "GPU check" { python $Check }

Write-Output "GPU ready. Next: `$env:PYTHONPATH = 'src'; python phase5\train_pair_checker.py --out phase5\models\pairs-v1"
exit 0
