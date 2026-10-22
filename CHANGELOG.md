# Changelog

## Unreleased: Phase 3d (step 2), larger calibration pool and out-of-sample tests
- `phase3/s2_data.py`, `s2_features.py`, `s2_judges.py` (6-judge panel, hard ledger cap, stops on xAI billing errors),
  `s2_eval.py` (calibration-only CV selection, LTT at 1/2/5/10%, plus an LTT over the whole candidate family).
- Calibration pool 2,798 claims (SciFact train 799 + HealthVer train 1,999). Tests: HealthVer test 1,515 of 1,823
  (internal, aggregates only), Climate-FEVER 1,500 (transfer), SciFact dev 300.
- Full-coverage node: HealthVer 78.6%, Climate-FEVER 78.8%, SciFact 87.3%. Better than J1 on HealthVer/Climate-FEVER, but
  it has more false accepts (8.2% / 1.7% / 9.7% of unsupported claims).
- LTT: **≤ 1% and ≤ 2% certify nothing (0% coverage, 0 accepts)**. Only 10% is certifiable, and only with the family
  supplement (4.5% calibration coverage).
- Fix: `evaluate.cp_upper` now works in log space (`math.comb` overflowed float for n in the thousands).
- API spend $216.60 of $220.

## Unreleased: Phase 3c step 1, strict proof verifier
- J11: a strict proof verifier (prompt V: quote first, no extrapolation; the quote must pass the byte-exact gate),
  usable as a node source, a hard ACCEPT veto (`node_run(veto=...)`), or ACCEPT-only-with-proof → NOT_SURE.
  `phase3/p3c_select.py` chooses on train CV only; `judges.py --split`.
- Result: it is too strict for SciFact. As a veto it cuts false accepts (rv3 12 → 4) but costs about 9 accuracy points,
  so it is off by CV. As a source it adds +0.3 CV points; rv3 dev stays at 87.3%. Spend $93.63 of $100.
- `phase3/STEP2_PLAN.md`: scoping for a larger certification set (datasets, licences, costs, CP sample sizes).

## Unreleased: Phase 3 (SciFact) and SAFETY.md
### Added
- `SAFETY.md`: guarantees by construction (fail-closed, NOT_SURE never accepted, zero-propagation, non-bypassable gate,
  hash-chained audit), each mapped to its tests, and measured guarantees with Clopper-Pearson bounds and caveats.
- `ClaimTrace(not_sure_band=...)` and `submit(..., not_sure=True)`: a distinct NOT_SURE outcome. It is never admitted,
  invisible like a block, logged and replayed. A gate failure stays BLOCK. `ThresholdLearner.band()` for selective mode.
- `tests/test_fail_closed.py` (hypothesis property and fuzz tests of the fail-closed guarantee), NOT_SURE tests, and a
  byte-exact quote-gate test. Tests: 147 → 164 (162 pass, 2 opt-in skips).
- Phase 3 pipeline: SciFact loader, BM25, ONNX/torch NLI, xAI judges J1–J10 with a hard-capped spend ledger, rv3
  retrieval (RRF of BM25/MedCPT/bge + MedCPT cross-encoder), exact band search, LTT (Bonferroni) NOT_SURE,
  risk-coverage chart, label-review list.
### Results (held-out SciFact dev, n = 300)
- gold evidence: node 91.3% [88.0, 94.3]; false accepts 19 → 6 of 176 (p = 0.0002).
- rv3 retrieved evidence: node 87.3% [83.3, 91.0] (ties best judges; < 90%); false accepts 29 → 12 (p < 0.0001).
- Selective, gold: 0/56 errors at 18.7% coverage (upper 5.21%). LTT ≤ 1%: not certifiable with 350 train claims, so it abstains.
- API spend $80.82 of $100.

## Unreleased (proposed 1.2.0, on top of 1.1.0): correlation-aware combining
### Added
- `roman_wheel_node.dependence.DependenceLearner`: an append-only, replayable ledger of each source's correctness per claim,
  with decay on replay (`half_life=150`). It learns pairwise dependence (decayed 2×2 tables, G-test at p < 0.01, φ).
- New bayes variants that need `dependence=`: **`ess`** (recommended; effective-sample-size discount), `tan`
  (tree-augmented naive Bayes) and `stack` (online logistic stacking). `give_feedback(..., dependence=)` and
  `ClaimTrace(..., dependence=)` log the exact model snapshot, so `replay()` recomputes decisions exactly.
- Generator: copy groups can have a latent leader (`leader=None`, `latent_accuracy`) or switch leaders (`switch=(k, name)`).
  Three new scenarios (`partial_corr`, `two_clusters`, `leader_switch`) have exact joint-model oracles.
- `python -m roman_wheel_node.bench --suite dependence`, with results in `results/dependence/` (20 seeds).
- `phase3/`: the fact-verification plan and scaffolding (SciFact, local ONNX NLI, xAI judges); no real-data runs yet.
- Tests: 108 → 147 (including offline tests of the phase3 scaffolding).

### Results (20 seeds)
- copycats: bayes-hard 67.7% → bayes-ess **79.0%** (ceiling 79.1%). ECE 0.172 → 0.021.
- leader_switch +3.4 points, two_clusters +1.9, partial_corr +0.5 (paired, all significant).
- Independent scenarios: no change beyond noise (paired difference ≤ 0.01 points, 19–20/20 seeds identical).

### Unchanged
- The six standard scenario definitions, every existing mode/variant, and `results/` (still byte-identical via `--check`).

## 1.1.0 (unreleased): Phase 1 gated trace + Phase 2 reproducible benchmark
### Added
- `roman_wheel_node.trace`: `ClaimTrace` / `Claim` / `Decision` / `replay` / `TamperError`.
  - `submit()` is the only way into context; every call goes through `roman_wheel_node`.
  - Claims are immutable and can't be constructed outside the trace; the trace has no public mutators.
  - Zero-propagation through parents (identity-checked), containment of blocked claims, and HEDGE off by default.
  - Every call is written to an append-only, hash-chained (optionally HMAC) JSONL audit log. `replay()` verifies the
    chain and recomputes every decision; `verify()` also checks the in-memory claims.
- `roman_wheel_node.bench`:
  - The documented claim-stream generator (`python -m roman_wheel_node.bench.generator`) and six standard named scenarios.
  - A runner with 13 policies, 95% Student-t CIs over ≥20 seeds, learned-reliability error, ECE/Brier calibration of
    eff, paired product-vs-geomean head-to-heads, and charts.
  - `python -m roman_wheel_node.bench --seeds 20 [--check results]`, whose output is byte-identical (tested).
- `results/`: committed 20-seed results and charts.
- Tests grew from 58 to 108, including property-based tests (hypothesis, optional) and a randomized fuzz loop.

### Changed
- Headline numbers are now over 20 seeds (they were 5): baseline bayes-hard 92.2% ± 0.3 against a 92.4% ceiling. The ceiling is
  now estimated from complete streams (≥200k claims), which makes the drift ceiling 87.8% (was 87.6%).
- `benchmarks/run_benchmark.py` is removed (superseded by `python -m roman_wheel_node.bench`). The 1.0.0 results are
  kept in `benchmarks/results_v1.0.0.*`. Seeds 0–4 of the new generator reproduce the 1.0.0 streams exactly.
- No changes to `core.py`: every 1.0.0 call gives identical results.

### Findings
- With the fixed v1 bands, product mode rejects almost everything (2.4% of true claims accepted on baseline). Geomean is equal
  or better in 5 of 6 scenarios even when both use a learned threshold. Keep geomean or bayes as the default.

## 1.0.0 (2026-10-21)
First packaged release (algorithm generation v3).
- `mode="bayes"`: the posterior P(true) from the sum of reliability-weighted log-likelihood ratios, with
  `hard` / `soft` / `logit` variants (`hard` is the default). Liars are inverted automatically.
- `ThresholdLearner`: an append-only `(eff, truth)` ledger per mode; the accuracy-maximising ACCEPT cut-off is chosen online with decay.
- `LiveWeights`: the default `half_life` is now 150 (`None` = never forget).
- `give_feedback()` helper, typed public API, `py.typed`.
- Backward compatible: the geomean/product paths are bit-identical to v2, and the six original reference calls are unchanged.
