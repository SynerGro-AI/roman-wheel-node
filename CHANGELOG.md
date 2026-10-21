# Changelog

## Unreleased (proposed 1.1.0): Phase 1 gated trace + Phase 2 reproducible benchmark
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
