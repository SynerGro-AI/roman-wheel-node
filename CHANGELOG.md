# Changelog

## 1.2.1 (UNRELEASED)
Fixes found after the 1.2.0 upload. No policy, weight or decision changes: the shipped policy is still
`rwn-policy-4.3.0`, and every decision on the same inputs is unchanged.

### Fixed
- **Non-ASCII claims on Windows (and any non-UTF-8 locale).** Every text file the package, tests and scripts open,
  read or write now uses `encoding="utf-8"` (it used the locale encoding, cp1252 on Windows, so a claim with `≥` or
  `α` could fail to log and audit logs were not portable between systems). The shadow CLI reads stdin and writes
  stdout as UTF-8. Regression test: `tests/test_encoding.py` (runs the package under a non-UTF-8 locale with
  `EncodingWarning` as an error).
- **Errors are no longer lost silently.** If `errors.jsonl` cannot be written, the node error is reported on stderr
  (the error used to be swallowed silently).
- **Model load failures are named.** A local source that fails to load now says which model could not be loaded, or
  that torch/transformers (or onnxruntime) is not installed, with the pip extra to install. `ShadowRunner` says once
  on stderr which profile source is unavailable and why, and verdicts carry `error:missing_source:<src>` instead of
  `error:KeyError`.
- **`fit` input checks.** `folds` must be an integer of at least 2 (`--folds 1` was an IndexError; on the CLI it is
  now a usage error, exit 2). The "both classes in every fold" check is now really per fold. An unreadable,
  non-UTF-8 or invalid examples file is a usage error and an unwritable output file is a clean `fit failed`, not a
  traceback. A deeply nested JSON line in the examples file is a usage error too (it was an uncaught
  `RecursionError`), and the "too few examples" message now says what is counted
  (`need at least 50 examples for 5-fold cross-fitting; got 12`).
- **Shadow CLI input.** A claim containing Unicode line separators (U+2028, U+0085, form feed and so on) is one claim:
  stdin and `--claims` are split on `\n` only (`str.splitlines()` split them into several claims). Non-UTF-8 or
  invalid JSON on stdin, in the claims file or in the corpus file (including deeply nested JSON, and valid JSON whose
  `\uXXXX` escapes leave a lone surrogate, which is not Unicode text) is a clear usage error naming the file and line,
  exit 2, instead of a traceback. Reading claims from `-` with stdin closed is a usage error too.
- **Malformed scores are labelled as such.** A source score that is present but not a number in [0, 1] gives
  `error:malformed_score:<src>`; `error:missing_source:<src>` now means the score is really absent.
- **CHANGELOG:** 1.2.0 is dated (it said "unreleased" after it was uploaded); a test keeps only the current version
  undated.

### Changed
- Tests: no test skips itself any more in a clean checkout. The two full-benchmark byte-identity tests run by default
  (no `RWN_FULL_BENCH`); the 4.3.0 refit test (`tests/test_repo_data.py`, repo-only) reads SciFact data built by the
  new `scripts/fetch_test_data.py` from pinned public sources and fails, printing that command, when it is missing;
  CI and the release workflow run the fetch step. Hypothesis is imported unconditionally (it is in the `test` extra),
  so missing it is an error, not silently fewer tests. The COVID-Fact manifest check, which needs an internal cache that
  is never distributed, moved out of pytest to `phase5/check_manifest_from_data.py`.
- CI: runners pinned to `ubuntu-24.04`, Python 3.14 added to the matrix, a `windows-2025` job, a pytest run with
  `-X warn_default_encoding -W error::EncodingWarning`, a ruff job, `twine check --strict`, actions pinned to SHAs.
- Release build: also runs `ruff format --check .` and the EncodingWarning pytest run, fetches the SciFact test data,
  and fails if any test is skipped.
- Package metadata: Python 3.9-3.14 and Development Status classifiers.
- README: "no API calls" qualified (the optional `nli2` model is downloaded from Hugging Face by name on first use),
  the private `phase3/cache/step2` feature cache needed to reproduce the published dev numbers is stated, and the test
  counts are re-measured.
- Tests use fixed deterministic data instead of seeded random generators in `tests/test_fit.py`.

### Known issues (planned for 1.2.2, target 2026-11-06)
These predate 1.2.1 and are not fixed in it. None changes a decision: each is a crash or a wrong exit code on bad
input, never an ACCEPT.
- `--trust-given-evidence` with an `evidence` value that is not a list, or a list with non-text items, gives a
  traceback instead of a usage error (`shadow.py` main, `sources.py` exact-copy hashing).
- A corpus line whose `sentences` list holds non-text items gives a traceback (`sources.py` BM25 tokenising).
- A missing, binary or deeply nested `--policy-file` gives a `PolicyError` or `RecursionError` traceback instead of
  exit 2 (`policy.py` `load_policy`).
- CR-only line endings (`\r` with no `\n`) are a usage error (exit 2) on stdin, which is read as one line, but a
  claims file opened in text mode treats each `\r` as a line break and runs (exit 0); the two should agree.
- `tests/test_test_deps.py` catches `import hypothesis` inside `try`, but not `importlib.import_module("hypothesis")`
  or `__import__("hypothesis")` inside `try`.

### Release provenance
- 1.2.0 on PyPI was a manual `twine` upload from PR commit `13725f7` (same tree as merge `9dcdb4f`) before the merge,
  with no tag and no attestations. 1.2.1 is the first release to go through `release.yml` from a `v1.2.1` tag.

## 1.2.0 (2026-10-24)
Everything since 1.0.0. Version 1.1.0 was never published, so its changes are included here.

### Breaking changes
- **Shipped policy is now `rwn-policy-4.3.0`** (`src/roman_wheel_node/policies/rwn-policy-4.3.0.json`, sha256
  `6f60f07b…e38a2`): two profiles, `local` (default; nli2 + overlap) and `local-lite` (nli + overlap), fitted only on
  SciFact train (claims CC BY 4.0, abstracts ODC-By 1.0) from free local open-model features, with the public
  `roman_wheel_node.fit` API. **No xAI/Grok (or other hosted-model) output is used in any shipped weight, calibrator,
  threshold or margin, and the package makes no hosted-model calls.**
- Removed from the package: the paid `panel` profile and its judge calibrators, the Grok judge code paths
  (`sources.XAIPanel`, the judge and quote-gate options of `shadow`, `fingerprint.quote_verbatim_v2`, the
  `--enable-xai` / `--cap-usd` flags) and every earlier policy file (4.0.0 and 4.1.0 were fitted partly on HealthVer;
  4.2.0 carried the xAI-derived panel and was never published), together with `POLICY_4_1_*` and
  `--policy 4.0.0 / 4.1.0`. The loader now accepts only the local sources `nli`, `nli2` and `overlap`. The panel's
  measured results are kept in the repository only, as a benchmark comparison not used for fitting
  (`results/phase6/xai_panel_ledger.md`).
- The `rwn-fp2` exact-copy fingerprint (NFC, invisible-character removal, quote/dash normalization, whitespace
  collapse; not NFKC, no case folding) is the gate in every shipped profile. It is a strict superset of fp1; elisions
  such as "..." or "> " still fail.
- `.github/FUNDING.yml` is removed (it only held placeholders).

### Measured (policy 4.3.0, held-out SciFact dev, n = 300)
- `local` (default, free): 58 ACCEPTs, 6 false (precision 89.7%, upper bound on the false-accept share 19.4%), 41.9% of
  supported claims accepted; 74.0% of claims handled correctly with NOT_SURE counted as a rejection.
- `local-lite`: 16 ACCEPTs, 1 false (93.8%, upper bound 26.4%), 12.1% of supported claims accepted; 63.3% handled
  correctly.
- **No certificate.** Learn-Then-Test (δ = 0.05) on out-of-fold SciFact train certifies neither profile at 10%, 5%, 2%
  or 1%. Numbers from the research fits and the withdrawn policies are not carried into this release. Details:
  `SAFETY.md` (b), `DATA_LICENSES.md`, `results/phase6/refit_43.txt` (repository).

### Added
- `roman_wheel_node.fit` (`fit_policy`, `write_policy`, `load_examples`, `score_examples`, `fit_profile`,
  `build_policy`, `isotonic_fit`; command line `python -m roman_wheel_node.fit`): turn your own labelled
  (claim, evidence, label) JSONL into a frozen, loadable policy file with the same method as the shipped one.
  Local sources only, no API calls, byte-identical output for the same input.
- `DATA_LICENSES.md`: the licence, source and commercial-use status of every dataset used to fit a shipped file.
- Shadow mode (`roman_wheel_node.shadow`): `ShadowRunner` / `FrozenNode` decide claims under the frozen, sha256-pinned
  policy and return the agent's output untouched, with an opt-in enforce mode. Outcomes are ACCEPT, NOT_SURE and BLOCK;
  the policy is accept-only and anything uncertain or erroring is NOT_SURE.
- `roman_wheel_node.policy` (`load_policy`, pinned sha256, refuses edited files), `roman_wheel_node.sources` (exact-copy
  gate, BM25, local NLI on CPU or CUDA),
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
- Release workflow (`.github/workflows/release.yml`) hardened: runners pinned to `ubuntu-24.04`, job timeouts (build
  30 min, publish 15 min), per-job concurrency (publish never cancelled mid-upload), a tag must point at a commit on
  `main`, and publishing refuses to run unless the `pypi` environment has a required-reviewers rule. See
  `RELEASING.md`.

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
