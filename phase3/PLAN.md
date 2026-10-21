# Phase 3 plan: real claims, real sources (scaffolding only; nothing has been run on real data)

## Goal
Measure Roman Wheel Node on real fact-verification claims with gold labels. Use the gated trace (`ClaimTrace`, `mode="bayes"`,
`variant="ess"`), learning online from the labels, and report the same metrics as Phase 2:
- accuracy, true-accept, false-block, ECE;
- per-source accuracy;
- the learned dependence between sources;
- cost per claim.

No Bayes-optimal ceiling exists for real data. The references are each source alone, majority vote, naive `bayes-hard`,
and a held-out logistic-regression stack fitted on the same sources (an upper reference that sees the labels in advance).

## Dataset: SciFact first (FEVER later)
| | SciFact (allenai) | FEVER dev |
|---|---|---|
| Size | 1,409 claims (dev 300), corpus of 5,183 abstracts | 19,998 claims; corpus is ~5.4M Wikipedia pages |
| Download | 3.1 MB tarball | several GB for the wiki dump plus a retrieval index |
| Fits this box? | trivially | possible but heavy; a retrieval subset would be needed |
| Caveat | dev labels are public; test labels are hidden | some NLI models (e.g. `*-mnli-fever-anli`) were **trained on FEVER**, so they leak |

Labels: SUPPORT is true and CONTRADICT is false. NOT-ENOUGH-INFO claims are reported separately, as "should not be accepted".
Use the train split (809 claims) to warm up the ledgers, and dev for evaluation.

## Sources (each gives a score in (0, 1) = P(supported))
1. **`nli`**: a local cross-encoder NLI model on CPU via ONNX Runtime. The score is the best (claim, evidence sentence)
   pair, entail / (entail + contradict).
2. **`overlap`**: lexical overlap between the claim and the evidence (implemented; weak, topical only).
3. **`xai:<model>`**: 1–3 LLM judges via the xAI API (`XAI_API_KEY`). The interface is implemented, with response caching.
   Different models are likely *correlated*, which is exactly what `variant="ess"` discounts. Phase 3 reports the learned φ between judges.

Retrieval is BM25 over the abstracts (top-3), plus an oracle-evidence condition (gold abstracts), so source quality
can be separated from retrieval quality. Exact-copy gate: `a = 1` iff every evidence passage is byte-identical
(sha256) to a corpus passage. That is trivial for retrieved text, but it matters once an LLM agent supplies quotes itself.

## What fits on this box (measured by `python phase3/check_box.py --online`)
- **Hardware:** 16.4 GB RAM (about 6 GB free while other work runs), 118 GB free disk, 8 vCPUs with AVX2, AVX-512F and AVX-512 VNNI, no GPU.
- **Software:** `onnxruntime` and `numpy` are installed for the system Python 3.13. `torch`, `transformers` and `tokenizers` are not.
  The plan needs only `onnxruntime` plus `tokenizers` (~10 MB); torch is not needed.

| NLI model candidate | fp32 | int8 ONNX (AVX-512 VNNI) | Note |
|---|---|---|---|
| `cross-encoder/nli-deberta-v3-small` | 568 MB | **173 MB** | **Recommended**: strong for its size, not FEVER-trained |
| `cross-encoder/nli-deberta-v3-xsmall` | 283 MB | 87 MB | fallback if latency matters |
| `cross-encoder/nli-MiniLM2-L6-H768` | 328 MB | 83 MB | fastest, weaker |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | 369 MB (safetensors only) | none | strongest, but FEVER-trained (SciFact only) |

CPU cost (estimate, to be measured): int8 deberta-v3-small takes ~20–40 ms per pair at 256 tokens. 300 claims × 3 passages × ~10 sentences is about 9k pairs, so a few minutes.

## LLM judges (xAI)
- **Smoke-tested** (2 calls, ~270 tokens each): `grok-4.20-0309-non-reasoning` returned 0.99 for a supported claim
  and 0.01 for a contradicted one. The API reported `cost_in_usd_ticks` = 2,143,500 per call, which would be
  ~$0.0002/call *if* a tick is 1e-10 USD (unconfirmed).
- **Models listed:** grok-4.20-0309-non-reasoning, grok-4.20-0309-reasoning, grok-4.20-multi-agent-0309, grok-4.3, grok-4.5,
  grok-4.6, grok-4.7, grok-build-0.1 (plus image/video models).
- **Budget for a full SciFact dev run** with 2–3 judges: ~900 calls at ~1k tokens each. Confirm the price per token before running.
- **Determinism:** temperature 0 is not a guarantee, so every response is cached by (model, prompt sha256). Reruns read the cache, which makes the benchmark reproducible.

## Steps
1. Download SciFact (3.1 MB) into `phase3/data/`; implement `load_scifact` and `BM25Retriever` (pure Python).
2. Download the int8 ONNX deberta-v3-small and its tokenizer (~175 MB), `pip install tokenizers`, and implement `NLIScorer`.
3. Run local-only sources (nli + overlap) on dev, with both oracle evidence and BM25. No key is needed.
4. Add 1, then 2–3 xAI judges (after cost approval). Compare `bayes-hard` vs `bayes-ess` and report the judge correlation.
5. Write `phase3/results/` with the same JSON/text/chart format and a `--check` against the cached responses.

## Interfaces (in `factcheck/`)
- `interfaces.py`:
  - `Passage`, `ClaimItem`, `Retriever` and `Scorer` protocols, and `exact_copy_gate`.
  - `OverlapScorer` (implemented) and `XAIJudge` (implemented, cached; reads `XAI_API_KEY`, never logs it).
  - `BM25Retriever` and `NLIScorer` (stubs).
- `harness.py`: `load_scifact` (stub) and `run()`, the gated loop with online feedback, ledgers and trace verification.
- `smoke_judge.py`: the 2-call smoke test above. `check_box.py`: the resource check.
