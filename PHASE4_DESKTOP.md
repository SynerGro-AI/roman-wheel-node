# Phase 4 desktop runbook: shadow mode on Windows with a GPU (machine "SynerGro.Ai")

This runbook adds the Roman Wheel Node in **shadow mode** to a Seraphina checkout (the public repo, or your private
one), runs it on the GPU, and collects fresh claims for a later 1% / 2% certificate.

- **Shadow mode never changes what the agent returns.**
- **Everything runs locally.** The package makes no paid or hosted-model API calls; nothing below spends money.

## 1. Install (PowerShell)

```powershell
# one-time: Python 3.10+ from python.org, Git for Windows, and an NVIDIA driver (check with: nvidia-smi)
cd C:\src
git clone https://github.com/SynerGro-AI/roman-wheel-node.git
cd roman-wheel-node
git checkout phase4-shadow
py -3 -m venv C:\venvs\rwn
C:\venvs\rwn\Scripts\Activate.ps1
pip install -e ".[test,local-nli,local-nli2]"
# GPU builds (pick the CUDA wheel that matches your driver; cu124 works on recent drivers)
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
pip uninstall -y onnxruntime; pip install onnxruntime-gpu
python -c "import torch; print(torch.cuda.is_available())"          # should print True
$env:PYTHONPATH = "src"; python -m pytest -q                          # all tests should pass
```

If you'd rather not clone it, install from the branch directly:
`pip install "roman-wheel-node[local-nli,local-nli2] @ git+https://github.com/SynerGro-AI/roman-wheel-node.git@phase4-shadow"`

## 2. Apply the hook to your PRIVATE Seraphina repo

The hook is one self-contained file, `seraphina/rwn_shadow.py`, with no required dependencies. It is on the public
Seraphina.AGI branch `feat/rwn-shadow-hook`. Copy it into the private repo, or cherry-pick the commit:

```powershell
cd C:\src\Seraphina-private
git remote add public https://github.com/SynerGro-AI/Seraphina.AGIv1.0.8.git   # once
git fetch public feat/rwn-shadow-hook
git checkout -b rwn-shadow
git checkout public/feat/rwn-shadow-hook -- seraphina/rwn_shadow.py tests/test_rwn_shadow.py docs/RWN_SHADOW.md
pip install -e .
```

Then wrap the agent's response function at the one place where it returns text. This is the integration point:

```python
from seraphina.rwn_shadow import ShadowHook
hook = ShadowHook()                          # local profile, shadow
respond = hook.wrap(respond)                 # returns exactly what respond() returned
```

## 3. Run with the GPU

```powershell
$env:RWN_DEVICE = "cuda"
$env:RWN_NLI2 = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"     # the 'local' profile needs this (~0.7 GB download)
$env:RWN_HF_CACHE = "D:\hf"                                        # optional: where models are cached
$env:SERAPHINA_RWN_CORPUS = "D:\data\trusted_corpus.jsonl"         # optional: {"doc_id": "...", "sentences": ["..."]}
# optional, profile local-lite: the ONNX DeBERTa-v3-small model directory (config.json, tokenizer.json, model*.onnx)
# $env:RWN_NLI_DIR = "D:\models\nli-deberta-v3-small"
python -m seraphina.rwn_shadow "Vitamin D supplementation reduces respiratory infections."
```

You can also run it without Seraphina, on a JSONL stream with one `{"claim": ..., "evidence": [...]}` per line:

```powershell
python -m roman_wheel_node.shadow claims.jsonl --log-dir D:\rwn_shadow --device cuda --nli2 MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli --trust-given-evidence
```

The exact-copy gate only passes evidence that is byte-identical to a sentence in a trusted store. Use
`--trust-given-evidence` only if your pipeline already verified that evidence, for example by quoting it from a
document store. Without a trusted store, every claim is logged as `BLOCK`. Those claims are still useful for labelling.

## 4. Where everything goes

| What | Default location |
|---|---|
| Seraphina hook logs | `%LOCALAPPDATA%\Seraphina\rwn_shadow\` (override with `$env:SERAPHINA_RWN_LOG_DIR`) |
| CLI logs | `--log-dir` (default `.\rwn_shadow\`) |
| `audit.jsonl` | Hash-chained ClaimTrace. Every decision recomputes from it, and `replay()` verifies it. |
| `fresh_claims.jsonl` | Claim, evidence, raw source scores and outcome per claim. This is the input for labelling. |
| `labels.jsonl`, `prereg.json` | Written by the labelling and pre-registration steps below. |

These files hold claim text. Keep them out of git (`*.jsonl` and `rwn_shadow/` are gitignored), and commit only
aggregate numbers.

## 5. Label and certify (after a few hundred claims)

```powershell
$d = "$env:LOCALAPPDATA\Seraphina\rwn_shadow"
python -m roman_wheel_node.fresh prereg  --log-dir $d     # FIRST: freezes policy + profile + alphas, before any label
python -m roman_wheel_node.fresh label   --log-dir $d     # s = SUPPORT, n = NOT, u = UNSURE, q = quit
python -m roman_wheel_node.fresh certify --log-dir $d
```

To certify a false-accept rate at a given alpha, you need this many **ACCEPTed** claims with zero false accepts
(one-sided Clopper-Pearson, δ = 0.05):

| Alpha | ACCEPTed claims needed |
|---|---|
| 1% | 299 |
| 2% | 149 |
| 5% | 59 |

The number of claims you have to label is that count divided by the node's ACCEPT rate. `certify` prints it for your
own stream.

## 6. What the GPU speeds up

| Component | CPU on the box (8 cores) | GPU |
|---|---|---|
| nli2 (DeBERTa-v3-base, torch) | about 3–9 claims/s, slower under load | typically 10–50× faster. This is the main cost of the `local` profile. |
| nli (DeBERTa-v3-small int8 ONNX) | fast, but **not deterministic on the box's CPU**: the same input gave different outputs | run an fp32/fp16 ONNX export on `CUDAExecutionProvider` instead. The loader prefers `model.onnx` or `model_fp16.onnx` when present. |
| Rerankers (MedCPT cross-encoder, Phase 3 rv3 retrieval) | slow, so they are not in the default sources | practical per claim |
| A bigger local judge (for example an open 7–8B instruct model in 4-bit) | impractical | possible as a free extra source. It would need its own calibration and pre-registration before it counts toward any certificate. |

## 7. What is and isn't certified

The shipped profiles (`local`, `local-lite`, policy 4.3.0) are **not certified at any level**. On held-out SciFact dev,
`local` accepted 58 of 300 claims with 6 false accepts (89.7% precision). The Phase 4 fresh-set results in
`results/phase4/` measured the withdrawn policy 4.0.0 and do not apply.

To fit a policy on claims you have labelled yourself (for example the fresh claims collected here), use
`python -m roman_wheel_node.fit labels.jsonl -o my-policy.json --nli2 MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
and load it with `ShadowRunner(..., policy=load_policy("my-policy.json", expected_sha256=...))`. See the README,
"Fit your own policy".
