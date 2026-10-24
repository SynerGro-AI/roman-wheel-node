# SAFETY: what is guaranteed, what is measured, and what is not

The project's north star is "100% safe AI". This file keeps two kinds of statement strictly apart:

* **(a) Guarantees by construction.** Properties of the code path, each enforced by named tests. They hold for every
  input under the stated threat model, and they say nothing about whether a claim is *true*.
* **(b) Measured guarantees.** Error rates of the shipped policy 4.3.0 on real claims (SciFact). These are statistical
  estimates with confidence bounds on one dataset. They are not proofs and they do not transfer automatically to other domains.

Nothing here claims "100% correct". The strongest honest statement is: **the gate fails closed by construction, and
on held-out data its decided-claim error rate is bounded as in (b).**

## (a) Guarantees by construction

Threat model: API-level. Code that calls `ClaimTrace` through its public methods cannot break these. In-process Python
can always reach private state (`object.__setattr__`, `gc`, importing `_TOKEN`). That kind of tampering is not
prevented but **detected**: `ClaimTrace.verify()` / `replay()` recompute every decision and compare. See the module
docstring of `src/roman_wheel_node/trace.py`.

| # | Guarantee | How it is enforced | Tests |
|---|---|---|---|
| G1 | **Fail-closed.** An unverified claim is never emitted as verified. A claim enters context only if the exact-copy gate passed (`a == 1`, exactly), every source score is non-zero, every parent is an admitted claim of the same trace, the status is `ACCEPT` (or `HEDGE` when the caller opted in with `allow_hedge=True`), and it is not `NOT_SURE`. Every other outcome, including errors, missing sources and malformed scores, results in no claim object. | `ClaimTrace.submit` → `roman_wheel_node` → `_apply_not_sure` → `_admit`. | `tests/test_fail_closed.py::test_property_fail_closed` (hypothesis: random gates, scores, bands, NOT_SURE flags, parents, modes), `::test_fail_closed_fuzz_loop`; `tests/test_trace.py::test_only_exact_a_equal_1_passes`, `::test_block_and_zero_score_never_enter`, `::test_property_invariants`, `::test_randomized_fuzz_loop`; `tests/test_roman_wheel.py::test_gate_blocks_non_exact_copy`, `::test_empty_scores_block`, `::test_hard_zero_veto_even_from_trusted_or_flipped_source`; `tests/test_dependence.py::test_gate_empty_and_veto_still_block` |
| G1b | **NOT_SURE never propagates as accepted.** `NOT_SURE` (posterior inside the logged band, or caller-flagged disagreement) is never admitted, whatever `allow_hedge` says. It is invisible like a block, and no descendant can enter. It is logged, and replay recomputes it. | `_admit("NOT_SURE") -> (False, "not_sure")`; band and flag are stored in the hash-chained record. | `tests/test_fail_closed.py` (both), `tests/test_core_band.py::test_not_sure_band_is_never_admitted_even_with_hedge`, `::test_forced_not_sure_and_zero_propagation`, `::test_not_sure_replays_and_tampering_is_caught`, `::test_low_posterior_inside_band_is_not_sure_but_gate_block_stays_block`, `::test_bad_band_rejected` |
| G1c | **Exact-copy gate is exact.** Evidence shown to the node must be trusted units: the raw sha256 (fp1) or the versioned `rwn-fp2` fingerprint (formatting-only normalization: NFC, invisible characters, quote/dash variants, whitespace; not NFKC, no case folding) must be in the trusted store. A content change, a case change or an untrusted unit closes the gate (BLOCK). | `sources.exact_gate`, `fingerprint.exact_gate_v2`, `FrozenNode.decide`. | `tests/test_fingerprint_fp2.py::test_gate_v2_superset_and_formatting_only`, `::test_v2_never_blocks_what_v1_admits`, `::test_shadow_runner_fp2_vs_fp1`; `tests/test_phase4_shadow.py::test_fail_closed_cases` |
| G2 | **Zero-propagation.** A claim whose parent is blocked, NOT_SURE, unknown, forged or from another trace is evaluated with `a = 0` and blocks. Descendants of an unverified claim can never enter, at any depth. | parent identity check in `ClaimTrace.submit`. | `tests/test_trace.py::test_zero_propagation_through_generations`, `::test_blocked_is_invisible`, `::test_forged_claim_object_cannot_become_parent`, `::test_claim_from_another_trace_is_not_a_parent`; `tests/test_core_band.py::test_forced_not_sure_and_zero_propagation`; parent clause of `tests/test_fail_closed.py` |
| G3 | **Non-bypassable gate.** `ClaimTrace.submit` is the only way to create a `Claim`. `Claim` cannot be constructed directly (private token), is immutable, and cannot be pickled or copied into a new object. The trace has no public mutators, and context is a tuple. | `Claim.__init__` token check, `__slots__`, `__setattr__`, `__reduce__`. | `tests/test_trace.py::test_claim_cannot_be_constructed_directly`, `::test_claim_is_immutable`, `::test_claim_cannot_be_pickled_or_copied_into_a_new_object`, `::test_trace_has_no_public_mutators`, `::test_raw_string_content_must_be_str_and_scored`, `::test_private_list_injection_is_caught_by_verify` |
| G4 | **Tamper-evident, hash-chained audit.** Every call, admitted or not, appends one JSONL record with the full audit tuple (inputs, reliabilities, dependence model, cutoff, band, eff, opcode, status, parents, time). The record carries `prev` and `hash = H(prev ‖ canonical(record))`, optionally keyed with HMAC. `replay()` verifies the chain **and** recomputes every decision. Editing, deleting, reordering, truncating or forging a record (even with a recomputed hash, unless keyed) is detected. | `trace.py` `_chain`, `replay`, `ClaimTrace.verify`. | `tests/test_trace.py::test_every_call_logged_with_full_audit_tuple`, `::test_replay_reconstructs_exactly`, `::test_editing_a_logged_decision_is_detected`, `::test_deleting_or_reordering_records_is_detected`, `::test_forged_entry_with_recomputed_hash_but_wrong_decision_is_detected`, `::test_forged_consistent_entry_detected_with_hmac`, `::test_truncated_or_garbage_line_detected`, `::test_in_memory_tampering_is_caught_by_verify`, `::test_deleted_log_is_caught_by_verify`; `tests/test_dependence.py::test_trace_tampered_dependence_model_detected` |
| G5 | **Frozen, pinned policy.** A deployed node runs from one policy file whose sha256 is pinned in the loader. An edited, re-hashed or swapped file is refused. Only three outcomes exist, `ACCEPT`, `NOT_SURE` and `BLOCK`, and the policy is accept-only: it never rejects on its posterior. | `policy.load_policy`, `policy_sha256`, `DEFAULT_POLICY_SHA256`. | `tests/test_phase4_shadow.py::test_default_policy_is_pinned_and_honest`, `::test_tampered_policy_refused`, `::test_policy_never_rejects_on_posterior` |
| G6 | **Shadow mode never alters agent outputs.** `ShadowRunner.wrap` returns the same object the agent returned, and the agent's exceptions propagate unchanged. Every node, source, extractor or logging failure is caught and logged as NOT_SURE. | `shadow.ShadowRunner.wrap`, `observe` guard, `_jsonable`. | `tests/test_phase4_shadow.py::test_shadow_returns_the_agent_output_unchanged` (hypothesis: random outputs, raw scores incl. NaN/inf/strings, crashing sources, gates), `::test_agent_exceptions_propagate_unchanged`, `::test_extract_crash_does_not_touch_output`, `::test_cli_default_is_local` |
| G7 | **Deployed node fails closed.** A missing source, malformed score or source crash leads to `NOT_SURE`. A failed exact-copy gate, missing evidence, or no trusted store leads to `BLOCK`. Enforce mode (opt-in) releases an output only if every claim is ACCEPTed. | `FrozenNode.decide` (two independent decision paths must agree, else NOT_SURE), `exact_gate`. | `::test_fail_closed_cases`, `::test_enforce_releases_only_accepts`, `::test_retrieval_feeds_the_gate`, `::test_shadow_returns_the_agent_output_unchanged` |
| G8 | **Local only; no hosted-model output ships.** The package calls no hosted model and reads no API key. Every shipped profile uses only the open local sources `nli`, `nli2` and `overlap`, and the loader refuses any other source, any unknown profile key and any unknown disagreement rule, so a judge-based profile cannot be loaded or selected. No file in the package references a fit made on xAI/Grok output. | `policy._profile` (`LOCAL_SOURCES`, `PROFILE_KEYS`, `DISAGREEMENT_RULES`), `fit.fit_profile`. | `tests/test_no_xai_in_package.py` (scans every installed package file; loader refusals; the only shipped policy is 4.3.0), `tests/test_phase4_shadow.py::test_no_api_key_is_read`, `::test_cli_default_is_local` |
| G9 | **Fit API is deterministic and its output is checked like the shipped policy.** `roman_wheel_node.fit` turns labelled (claim, evidence, label) JSONL into a policy file; the same examples in the same order give a byte-identical file, and the result loads only through the same pinned-hash loader. | `fit.fit_policy`, `fit.write_policy`, `policy.load_policy`. | `tests/test_fit.py` (synthetic, deterministically generated examples; real overlap scorer, loader, FrozenNode and ShadowRunner; no mocks) |

Limits of (a):

* An **unkeyed** chain can be rewritten wholesale by someone with write access to the log. Use `hmac_key=` or anchor
  `ClaimTrace.head()` externally.
* (a) guarantees that nothing *unverified* is emitted. "Verified" means it passed the gate and the learned posterior
  cutoff. It does **not** mean "true": the NLI models, the overlap feature and retrieval can all be wrong together. That is what (b) measures.
* The policy 4.3.0 evaluation decides every held-out claim through `roman_wheel_node.shadow.FrozenNode`, the shipped
  shadow decision path with its trace (`phase6/refit_policy_43.py:evaluate`), so the measured numbers below come from the
  guarded path, not from a re-implementation.

## (b) Measured guarantees (shipped policy 4.3.0, SciFact, held-out dev)

Setup: SciFact dev, n = 300 held-out claims, gold evidence. The binary task is "supported vs not". "Upper" is the
one-sided 95% Clopper-Pearson bound.

Script: `phase6/refit_policy_43.py` (fits with the public `roman_wheel_node.fit` API and checks that it reproduces the
research pipeline exactly). Report: `results/phase6/refit_43.txt`. Licences: `DATA_LICENSES.md`.

* **Data.** Both shipped profiles are fitted on SciFact train only (799 claims; CC BY 4.0 claims, ODC-By 1.0
  abstracts) from local open-model features. HealthVer, Climate-FEVER and COVID-Fact are not used. **No xAI/Grok (or
  other hosted-model) output is used in any shipped weight, calibrator, threshold or margin.** Held-out SciFact dev
  (300 claims) was evaluated once.
* **Held-out SciFact dev.**
  * `local` (default; nli2 + overlap): 58 ACCEPTs (19.3%), 6 false, precision 89.7% (upper bound on the false-accept
    share 19.4%), 41.9% of supported claims accepted. With NOT_SURE counted as a rejection, 74.0% of dev claims are
    handled correctly.
  * `local-lite` (nli + overlap): 16 ACCEPTs (5.3%), 1 false, precision 93.8% (upper bound 26.4%), 12.1% of supported
    claims accepted; 63.3% handled correctly.
  * No BLOCKs (gold evidence passes the exact-copy gate), no source errors, no replay mismatches.
* **Certificate: none.** Learn-Then-Test fixed-sequence testing (δ = 0.05, Bonferroni over 2 profiles) on out-of-fold
  SciFact train posteriors certifies neither profile at 10%, 5%, 2% or 1%. Each profile's margin is therefore its
  pre-registered one (lowest out-of-fold upper bound): `local` 135 out-of-fold accepts, 21 false, upper 21.6%;
  `local-lite` 54 accepts, 8 false, upper 25.1%. The policy file says "No profile is certified at any level."
* **Structural veto.** `structure.FROZEN` stays opt-in and outside the policy. On the `local` profile's out-of-fold
  SciFact accepts it catches 1 false accept and loses 3 correct ones; no field meets the pre-declared bar
  (`results/phase6/structural_scifact_43.json`).
* **Net:** the guarantees in (a) are unchanged. The shipped profiles carry **no** certified error bound; on held-out
  SciFact dev about 1 in 10 `local` ACCEPTs was wrong. Treat ACCEPT as "passed a calibrated local check", not as true.

### Research record (not a property of this package)

Earlier research phases (3 to 3e), the policy 4.0.0 fresh-set run (Phase 4) and the withdrawn policies 4.0.0 to 4.2.0
measured fits that are not shipped. Some were calibrated on HealthVer, which the commercial-use licensing rule now excludes, and some used
xAI/Grok judge outputs, which the owner removed from everything shipped (kept only as a benchmark ledger,
`results/phase6/xai_panel_ledger.md`, not used for fitting). Their
numbers are therefore not repeated here and must not be read as properties of 1.2.0. The repository keeps them as a
research record (`results/phase3/`, `results/phase4/`). Lessons from that work that do not depend on any one fit:

* Certifying a tight error level needs many decided claims. With zero errors, a one-sided Clopper-Pearson bound at
  δ = 0.05 needs 59 accepts for 5%, 149 for 2% and 299 for 1%, and a search over rules needs more.
* Label noise in a benchmark puts a floor under the measurable error, whatever the system does.
* A certificate earned on one dataset does not transfer to another; it has to be re-earned on the target data.
