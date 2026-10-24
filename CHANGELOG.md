# Changelog

## 1.2.0 (unreleased)
Everything since 1.0.0. Version 1.1.0 was never published, so its changes are included here.

### Breaking changes
- **Shipped policy is now `rwn-policy-4.2.0`** (`src/roman_wheel_node/policies/rwn-policy-4.2.0.json`, sha256
  `6f7bb9e3…6c2a`), fitted only on SciFact train (claims CC BY 4.0, abstracts ODC-By 1.0). Policies 4.0.0 and 4.1.0
  were fitted partly on HealthVer and are removed from the package, together with `POLICY_4_1_*` and
  `--policy 4.0.0 / 4.1.0`. Their sha256s are listed in the 4.2.0 file as withdrawn.
- The `rwn-fp2` exact-copy fingerprint (NFC, invisible-character removal, quote/dash normalization, whitespace
  collapse; not NFKC, no case folding) is the gate in every shipped profile. It is a strict superset of fp1; elisions
  such as "..." or "> " still fail.
- `.github/FUNDING.yml` is removed (it only held placeholders).

### Measured (policy 4.2.0, held-out SciFact dev, n = 300)
- `panel` (paid, opt-in): 108 ACCEPTs, 7 false (precision 93.5%), 81.5% of supported claims accepted. Certified
  accept precision ≤ 10% (280 accepts, 17 false, out-of-fold on SciFact train, upper bound 9.0%, δ = 0.05).
- `local` (default, free): 64 ACCEPTs, 9 false (85.9%). `local-lite`: 16 ACCEPTs, 1 false (93.8%). Not certified.
- **Nothing is certified at 5%, 2% or 1%.** Numbers from the earlier research fits and the withdrawn policies are not
  carried into this release. Details: `SAFETY.md` (b), `DATA_LICENSES.md`, `results/phase6/refit_42.txt` (repository).

### Added
- `DATA_LICENSES.md`: the licence, source and commercial-use status of every dataset used to fit a shipped file.
- Shadow mode (`roman_wheel_node.shadow`): `ShadowRunner` / `FrozenNode` decide claims under the frozen, sha256-pinned
  policy and return the agent's output untouched, with an opt-in enforce mode. Outcomes are ACCEPT, NOT_SURE and BLOCK;
  the policy is accept-only and anything uncertain or erroring is NOT_SURE.
- `roman_wheel_node.policy` (`load_policy`, pinned sha256, refuses edited files), `roman_wheel_node.sources` (exact-copy
  gate, BM25, local NLI on CPU or CUDA; `XAIPanel` off by default, needs `enable=True` and `cap_usd`),
  `roman_wheel_node.fingerprint`, and the opt-in `roman_wheel_node.structure` veto (not part of any policy).
- `roman_wheel_node.fresh`: `prereg` / `label` / `import-labels` / `certify` with one-sided Clopper-Pearson on one
  pre-registered rule.
- NOT_SURE outcome in `ClaimTrace(not_sure_band=...)` / `submit(..., not_sure=True)`: never admitted, logged and
  replayed. `ThresholdLearner.band()`.
- `roman_wheel_node.trace`: `ClaimTrace` / `Claim` / `Decision` / `replay` / `TamperError`. `submit()` is the only way
  into context; zero-propagation through parents; an append-only, hash-chained (optionally HMAC) JSONL audit log that
  `replay()` verifies and recomputes.
- Correlation-aware combining: `roman_wheel_node.dependence.DependenceLearner` and the bayes variants `ess`
  (recommended), `tan` and `stack`, with exact replay of the dependence snapshot.
- `roman_wheel_node.bench`: the documented synthetic claim-stream generator, standard and dependence scenario suites,
  a runner with 95% Student-t CIs over 20 seeds, and byte-identical `--check`.
- Command line: `roman-wheel-node` console script and `python -m roman_wheel_node` with `--help`, `--version` and a
  demo; bad arguments exit non-zero.
- `SAFETY.md`: guarantees by construction, each mapped to its tests, kept apart from measured error rates.

### Changed
- The package version has one source, `src/roman_wheel_node/_version.py`; `pyproject.toml` reads it dynamically and a
  test checks that nothing else hard-codes it.
- Files opened by the package and tests are closed deterministically (no `ResourceWarning`; the suite
  passes under `pytest -W error`).
- Code style: `ruff` (E, F, W, I; line length 120), configured in `pyproject.toml`; `ruff check .` is clean.
- `benchmarks/run_benchmark.py` is removed (superseded by `python -m roman_wheel_node.bench`); seeds 0–4 of the new
  generator reproduce the 1.0.0 streams exactly. `core.py` gives identical results for every 1.0.0 call.
- The sdist ships `CHANGELOG.md`, `SAFETY.md`, `DATA_LICENSES.md` and the tests, so the suite runs from an unpacked
  sdist against the installed wheel; tests that need repo-only research folders skip with a stated reason.

### Synthetic benchmark (simulated claim streams, not real data; 20 seeds)
- Baseline: bayes-hard 92.2% ± 0.3 against a 92.4% Bayes-optimal ceiling.
- copycats: bayes-hard 67.7% → bayes-ess 79.0% (ceiling 79.1%); ECE 0.172 → 0.021.
- With the fixed v1 bands, product mode accepts only 2.4% of true claims on baseline; keep geomean or bayes.

## 1.0.0 (2026-10-21)
First packaged release (algorithm generation v3).
- `mode="bayes"`: the posterior P(true) from the sum of reliability-weighted log-likelihood ratios, with
  `hard` / `soft` / `logit` variants (`hard` is the default). Liars are inverted automatically.
- `ThresholdLearner`: an append-only `(eff, truth)` ledger per mode; the accuracy-maximising ACCEPT cut-off is chosen online with decay.
- `LiveWeights`: the default `half_life` is now 150 (`None` = never forget).
- `give_feedback()` helper, typed public API, `py.typed`.
- Backward compatible: the geomean/product paths are bit-identical to v2, and the six original reference calls are unchanged.
