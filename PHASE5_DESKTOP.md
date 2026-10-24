# Phase 5 desktop runbook: train and test the meaning-change checker on "SynerGro.Ai" (Windows + NVIDIA GPU)

This trains a small NLI cross-encoder on COVID-Fact true/fake claim pairs. It then tests the checker once, as an
accept-only veto on the frozen rule, against the held-out half B.

**Windows only (win_amd64).** The GPU setup (`phase5\setup_gpu.ps1` with `phase5\torch-cu132.lock`) supports only
64-bit Windows on an AMD64/x64 CPU, with 64-bit Python 3.13 or 3.14 from python.org. `phase5\torch-cu132.lock` is only
complete for win_amd64, so the script refuses to run anywhere else (exit 7) before any pip call.

- **No xAI calls, no spend.** Everything runs on the local GPU.
- **Internal data.** The bundle holds COVID-Fact and HealthVer text. It stays on your machines. Never commit, upload
  or share it. Only aggregate results go into git.
- **B is scored once.** The thresholds are fixed and committed from A_dev before B is opened. The scripts refuse to
  overwrite either file.

## 0. Bring the data bundle over (one file, about 5.4 MB)

The box built `phase5/bundle/pairs_bundle.json` and `pairs_bundle.json.sha256`. Copy both files yourself, by USB or
your own private storage, to:

```
C:\src\roman-wheel-node\phase5\bundle\pairs_bundle.json
C:\src\roman-wheel-node\phase5\bundle\pairs_bundle.json.sha256
```

Expected sha256: `fe71efd2ff5370c52084a46d5d530b4c57c6663c259790178b69bb77757baf56`. The scripts check this hash
against the training record, so the file must be byte-identical.


> **GPU torch:** `.\phase5\setup_gpu.ps1` is the only supported way to install torch. It installs the official
> PyTorch 2.14.1 built for CUDA 13.2 from the hash-locked `phase5\torch-cu132.lock` (`--require-hashes`, wheels only),
> then checks for a working GPU. That build has native sm_120 kernels for RTX 50-series cards (e.g. 5070 Ti) and runs
> on any NVIDIA driver reporting CUDA 13.2 or newer (13.5 included). Do not install torch any other way, and do not
> fall back to CPU.

## 1. Set up (PowerShell, one time, about 10 to 20 minutes, mostly downloads)

```powershell
# needs: Python 3.13 or 3.14 from python.org, Git for Windows, an NVIDIA driver (nvidia-smi should list the GPU)
cd C:\src
if (-not (Test-Path roman-wheel-node)) { git clone https://github.com/SynerGro-AI/roman-wheel-node.git }
cd roman-wheel-node
git fetch origin
git checkout phase5-pairs
git pull
py -3.13 -m venv C:\venvs\rwn5
C:\venvs\rwn5\Scripts\Activate.ps1
python -m pip install --upgrade pip
.\phase5\setup_gpu.ps1                                                       # about 2.5 GB; must end with "GPU ready"
$env:PYTHONPATH = "src"; python -m pytest -q                                # all tests should pass
(Get-FileHash phase5\bundle\pairs_bundle.json -Algorithm SHA256).Hash        # must equal the hash in step 0
```

`setup_gpu.ps1` stops at the first failed step with a `FAILED: ...` line and a non-zero exit code. It installs torch
from the hash-locked lock, then the repo (`pip install -e ".[test,local-nli2]"`, constrained to torch 2.14.1+cu132),
then runs `phase5\check_gpu.py`. Check the code with `$LASTEXITCODE` right after it finishes.

### setup_gpu.ps1 exit codes

| Code | From | Meaning | What to do |
|---|---|---|---|
| 0 | setup_gpu.ps1 | GPU ready | Go on to step 2. |
| 1 | setup_gpu.ps1 | `phase5\torch-cu132.lock` or `phase5\torch-cu132.constraints.txt` is missing, either file's sha256 does not match, the constraints file is not exactly `torch==2.14.1+cu132`, or a pip step failed (uninstall, hash-locked torch install, or the constrained repo install) | Restore the files with `git checkout -- phase5/torch-cu132.lock phase5/torch-cu132.constraints.txt`, then rerun. For a pip failure, read the pip error above the `FAILED` line: a hash error means a download did not match the lock, so do not edit the lock and report it. A `ResolutionImpossible` on the repo install means a dependency wanted a different torch, so report it rather than loosening the constraint. |
| 2 | check_gpu.py | torch is not importable | Rerun `.\phase5\setup_gpu.ps1` inside the activated `C:\venvs\rwn5` venv. |
| 3 | check_gpu.py | Wrong torch build: not exactly 2.14.1+cu132 (CUDA 13.2), e.g. a CPU or older CUDA build replaced it | Rerun `.\phase5\setup_gpu.ps1`. Do not install torch any other way. |
| 4 | check_gpu.py | CUDA not available | Update the NVIDIA driver until `nvidia-smi` lists the GPU and shows CUDA 13.2 or newer, reboot, then rerun. |
| 5 | check_gpu.py | This torch build has no kernels for the GPU's architecture (e.g. sm_120 missing from the arch list) | Do not fall back to CPU or another build. Report the printed GPU name, capability and arch list. |
| 6 | check_gpu.py | The fixed-tensor GPU matmul self-test differs from the CPU result | Do not train. Rerun once; if it still fails, report the output (driver or hardware problem). |
| 7 | setup_gpu.ps1 | Not Windows on AMD64, or Python is not a 64-bit Windows build (`win-amd64`) | Use 64-bit Windows with 64-bit Python 3.13 or 3.14 from python.org. |
| 8 | setup_gpu.ps1 | Python is not 3.13 or 3.14 | Recreate the venv with `py -3.13 -m venv C:\venvs\rwn5`, activate it, then rerun. |
| 9 | setup_gpu.ps1 or check_gpu.py | Unexpected Python error in the platform/version check or the GPU check (not a pip failure); the full traceback is printed with `unexpected error, paste the traceback` | Do not retry blindly. Copy the whole traceback and report it. |

> **Known limit:** the editable repo install (`pip install -e ".[test,local-nli2]"`, which pulls transformers,
> sentencepiece, protobuf and the test tools) is not hash-locked. Only torch and its dependencies are, through
> `phase5\torch-cu132.lock`. That step is protected by the torch constraint (`-c phase5\torch-cu132.constraints.txt`,
> which pins `torch==2.14.1+cu132`, so a torch swap fails at resolve time) and by the exit-3 check in `check_gpu.py`.

## 2. Train (about 3 to 10 minutes on a recent GPU)

```powershell
cd C:\src\roman-wheel-node
C:\venvs\rwn5\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
$env:HF_HOME = "D:\hf"          # optional: where the 0.7 GB base checkpoint is downloaded and cached
python phase5\train_pair_checker.py --out phase5\models\pairs-v1 --hf-cache "$env:HF_HOME"
```

- Defaults: DeBERTa-v3-base-mnli-fever-anli, 2 epochs, lr 1e-5, batch 8 x grad-accum 2, max_len 320, fp16 on CUDA,
  seed 20261022.
- It trains on the 1,473 A_train pairs only and prints A_dev accuracy per epoch.
- If you run out of GPU memory: `--batch-size 4 --grad-accum 4`. That gives the same effective batch.
- If you skip `$env:HF_HOME`, drop the `--hf-cache` argument too.

## 3. Pre-register the thresholds on A_dev, and commit them BEFORE looking at B (about 1 minute)

```powershell
python phase5\eval_pair_checker.py prereg --model phase5\models\pairs-v1
git add results\phase5\prereg_pairs.json results\phase5\train_meta_pairs-v1.json
git commit -m "Phase 5: pre-register pair-checker thresholds (A_dev only)"
git push origin phase5-pairs
```

Pushing first gives a timestamp in git showing the thresholds were fixed before B was scored.

## 4. Score the held-out half B, once (about 1 to 2 minutes)

```powershell
python phase5\eval_pair_checker.py heldout --model phase5\models\pairs-v1
git add results\phase5\heldout_B_pairs.json results\phase5\heldout_B_pairs.txt
git commit -m "Phase 5: held-out B result for the pair checker"
git push origin phase5-pairs
type results\phase5\heldout_B_pairs.txt
```

Commit and push only `results\phase5\` files. `phase5\bundle\`, `phase5\models\` and `phase5\runs\` hold internal
data, weights and per-item scores, and they are gitignored.

## What the result means (pre-registered in `results/phase5/protocol.json`)

- **Primary:** panel + Phase 4b structural veto + checker at tau r0.90, on the 1,856 panel-judged B items.
- **Certification:**
  - Uses LTT fixed-sequence over r0.80 -> r0.90 -> r0.95, with Clopper-Pearson at delta 0.05, for alpha 1%, 2% and 5%.
  - A 1% certificate is impossible on B whatever the checker does. It needs 299 error-free accepts, and the base
    rule accepts only 289 B items.
  - 2% needs at least 149 error-free accepts, and 5% needs 59.
- **No-regression:** on SciFact dev, HealthVer test, Climate-FEVER and hv_test_unjudged, precision must not drop. It
  may lose at most max(1, 5%) of the correct accepts.
- **SUCCESS** means the primary is certified at 5% and every no-regression set passes. A FAIL is still a valid
  result: report it as is, and do not re-tune on B.

Total time with setup is about 20 to 35 minutes. Without setup it is about 5 to 12 minutes.
