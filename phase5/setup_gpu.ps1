# WINDOWS ONLY (win_amd64). phase5\torch-cu132.lock is for 64-bit Windows with 64-bit Python 3.13/3.14; the script
# refuses to run anywhere else (exit 7) before any pip call.
# Phase 5 GPU setup for RTX 50-series (Blackwell, sm_120). This is the only supported torch install path.
# Installs the official prebuilt PyTorch 2.14.1 built against CUDA 13.2 from the hash-locked file
# phase5\torch-cu132.lock (byte-identical to the Seraphina desktop install kit's lock). The CUDA runtime is
# bundled inside the wheel, so no separate CUDA toolkit install is needed. Any NVIDIA driver that reports
# CUDA 13.2 or newer in nvidia-smi (e.g. 13.5) runs it.
# The editable repo install is constrained to torch==2.14.1+cu132 (phase5\torch-cu132.constraints.txt), so a
# dependency that wants a different torch fails at resolve time instead of swapping it.
# Usage (venv active, from the repo root):  .\phase5\setup_gpu.ps1
# Every native call is checked; the script exits non-zero on the first failure.
# Exit codes (full table with what to do: PHASE5_DESKTOP.md, "setup_gpu.ps1 exit codes"):
#   0 GPU ready | 1 lock file missing/hash mismatch, or a pip step failed | 2-6 from phase5\check_gpu.py
#   7 not Windows / not AMD64 / Python not win-amd64 | 8 Python is not 3.13 or 3.14
$ErrorActionPreference = "Stop"
$Index = "https://download.pytorch.org/whl/cu132"
$Lock = Join-Path $PSScriptRoot "torch-cu132.lock"
$LockSha256 = "2a5d73b2e53bdcedb033a5e4cb8097eb2a9c2b38da9e910740a849125ce16d5c"
$Constraints = Join-Path $PSScriptRoot "torch-cu132.constraints.txt"
$Check = Join-Path $PSScriptRoot "check_gpu.py"

function Exit-Setup {
    param([int]$Code, [string]$Message)
    Write-Error "FAILED: $Message (exit $Code)" -ErrorAction Continue
    exit $Code
}

# Runs a native command and stops on a non-zero exit. With -FailCode the script exits with that code;
# without it, the command's own exit code is passed through (used for the python checks, codes 2-8).
function Invoke-Checked {
    param([string]$What, [scriptblock]$Command, [int]$FailCode = 0)
    & $Command
    $code = $LASTEXITCODE
    if ($code -ne 0) {
        $out = if ($FailCode -ne 0) { $FailCode } elseif ($code -gt 0) { $code } else { 1 }
        Exit-Setup $out "$What returned exit code $code"
    }
}

# 1. Platform: Windows on AMD64 only, checked before anything else runs. $env:OS works on Windows PowerShell 5.1
#    and pwsh; PROCESSOR_ARCHITEW6432 is set when a 32-bit shell runs on 64-bit Windows.
$arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($env:OS -ne "Windows_NT" -or $arch -ne "AMD64") {
    Exit-Setup 7 "this setup is Windows-only (win_amd64): found OS='$env:OS', PROCESSOR_ARCHITECTURE='$arch'. phase5\torch-cu132.lock has no wheels for this platform."
}

# 2. Lock file integrity.
if (-not (Test-Path -LiteralPath $Lock)) { Exit-Setup 1 "missing lock file: $Lock" }
$actual = (Get-FileHash -LiteralPath $Lock -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $LockSha256) { Exit-Setup 1 "lock file hash mismatch: $Lock is $actual, expected $LockSha256" }
if (-not (Test-Path -LiteralPath $Constraints)) { Exit-Setup 1 "missing constraints file: $Constraints" }

# 3. Python: 64-bit Windows build (7), version 3.13 or 3.14 (8). Still before any pip call.
Invoke-Checked "python platform/version check" {
    python -c "import sys, sysconfig; p = sysconfig.get_platform(); v = sys.version_info[:2]; print('python %d.%d' % v, p); sys.exit(7 if p != 'win-amd64' else (0 if v in ((3, 13), (3, 14)) else 8))"
}

# 4. Installs. Any pip failure exits 1 so it never collides with check_gpu.py's codes.
Invoke-Checked "pip uninstall torch" { python -m pip uninstall -y torch torchvision torchaudio } -FailCode 1
Invoke-Checked "pip install torch (hash-locked)" {
    python -m pip install --require-hashes --only-binary=:all: --index-url $Index --extra-index-url https://pypi.org/simple -r $Lock
} -FailCode 1
Invoke-Checked "pip install -e .[test,local-nli2] (torch constrained)" {
    python -m pip install -c $Constraints --extra-index-url $Index -e ".[test,local-nli2]"
} -FailCode 1

# 5. GPU check (exit codes 2-6 passed through).
Invoke-Checked "GPU check" { python $Check }

Write-Output "GPU ready. Next: `$env:PYTHONPATH = 'src'; python phase5\train_pair_checker.py --out phase5\models\pairs-v1"
exit 0
