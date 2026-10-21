# Changelog

## 1.0.0 (2026-10-21)
First packaged release (algorithm generation v3).
- `mode="bayes"`: the posterior P(true) from the sum of reliability-weighted log-likelihood ratios, with
  `hard` / `soft` / `logit` variants (`hard` is the default). Liars are inverted automatically.
- `ThresholdLearner`: an append-only `(eff, truth)` ledger per mode; the accuracy-maximising ACCEPT cut-off is chosen online with decay.
- `LiveWeights`: the default `half_life` is now 150 (`None` = never forget).
- `give_feedback()` helper, typed public API, `py.typed`.
- Backward compatible: the geomean/product paths are bit-identical to v2, and the six original reference calls are unchanged.
