# SAFETY: what is guaranteed, what is measured, and what is not

The project's north star is "100% safe AI". This file keeps two kinds of statement strictly apart:

* **(a) Guarantees by construction.** Properties of the code path, each enforced by named tests. They hold for every
  input under the stated threat model, and they say nothing about whether a claim is *true*.
* **(b) Measured guarantees.** Error rates on real claims (SciFact). These are statistical estimates with
  confidence bounds on one dataset. They are not proofs and they do not transfer automatically to other domains.

Nothing here claims "100% correct". The strongest honest statement is: **the gate fails closed by construction, and
on held-out data its decided-claim error rate is bounded as in (b).**

## (a) Guarantees by construction

Threat model: API-level. Code that calls `ClaimTrace` through its public methods cannot break these. In-process Python
can always reach private state (`object.__setattr__`, `gc`, importing `_TOKEN`). That kind of tampering is not
prevented but **detected**: `ClaimTrace.verify()` / `replay()` recompute every decision and compare. See the module
docstring of `src/roman_wheel_node/trace.py`.

| # | Guarantee | How it is enforced | Tests |
|---|---|---|---|
| G1 | **Fail-closed.** An unverified claim is never emitted as verified. A claim enters context only if the exact-copy gate passed (`a == 1`, exactly), every source score is non-zero, every parent is an admitted claim of the same trace, the status is `ACCEPT` (or `HEDGE` when the caller opted in with `allow_hedge=True`), and it is not `NOT_SURE`. Every other outcome, including errors, missing judges and unparseable replies, results in no claim object. | `ClaimTrace.submit` → `roman_wheel_node` → `_apply_not_sure` → `_admit`. | `tests/test_fail_closed.py::test_property_fail_closed` (hypothesis: random gates, scores, bands, NOT_SURE flags, parents, modes), `::test_fail_closed_fuzz_loop`; `tests/test_trace.py::test_only_exact_a_equal_1_passes`, `::test_block_and_zero_score_never_enter`, `::test_property_invariants`, `::test_randomized_fuzz_loop`; `tests/test_roman_wheel.py::test_gate_blocks_non_exact_copy`, `::test_empty_scores_block`, `::test_hard_zero_veto_even_from_trusted_or_flipped_source`; `tests/test_dependence.py::test_gate_empty_and_veto_still_block` |
| G1b | **NOT_SURE never propagates as accepted.** `NOT_SURE` (posterior inside the logged band, or caller-flagged disagreement) is never admitted, whatever `allow_hedge` says. It is invisible like a block, and no descendant can enter. It is logged, and replay recomputes it. | `_admit("NOT_SURE") -> (False, "not_sure")`; band and flag are stored in the hash-chained record. | `tests/test_fail_closed.py` (both), `tests/test_core_band.py::test_not_sure_band_is_never_admitted_even_with_hedge`, `::test_forced_not_sure_and_zero_propagation`, `::test_not_sure_replays_and_tampering_is_caught`, `::test_low_posterior_inside_band_is_not_sure_but_gate_block_stays_block`, `::test_bad_band_rejected` |
| G1c | **Exact-copy gate is byte-exact.** Evidence shown to the node must be corpus sentences (sha256 match). When the gate judge commits to SUPPORT/CONTRADICT, its quote must occur byte-for-byte inside a shown sentence whose sha256 is in the corpus set. A paraphrased quote, a one-byte edit or an invented quote closes the gate. | `factcheck.interfaces.exact_copy_gate`, `phase3/evaluate.py:quote_verbatim`. | `tests/test_phase3_scaffold.py::test_exact_copy_gate`, `::test_quote_gate_is_byte_exact` |
| G2 | **Zero-propagation.** A claim whose parent is blocked, NOT_SURE, unknown, forged or from another trace is evaluated with `a = 0` and blocks. Descendants of an unverified claim can never enter, at any depth. | parent identity check in `ClaimTrace.submit`. | `tests/test_trace.py::test_zero_propagation_through_generations`, `::test_blocked_is_invisible`, `::test_forged_claim_object_cannot_become_parent`, `::test_claim_from_another_trace_is_not_a_parent`; `tests/test_core_band.py::test_forced_not_sure_and_zero_propagation`; parent clause of `tests/test_fail_closed.py` |
| G3 | **Non-bypassable gate.** `ClaimTrace.submit` is the only way to create a `Claim`. `Claim` cannot be constructed directly (private token), is immutable, and cannot be pickled or copied into a new object. The trace has no public mutators, and context is a tuple. | `Claim.__init__` token check, `__slots__`, `__setattr__`, `__reduce__`. | `tests/test_trace.py::test_claim_cannot_be_constructed_directly`, `::test_claim_is_immutable`, `::test_claim_cannot_be_pickled_or_copied_into_a_new_object`, `::test_trace_has_no_public_mutators`, `::test_raw_string_content_must_be_str_and_scored`, `::test_private_list_injection_is_caught_by_verify` |
| G4 | **Tamper-evident, hash-chained audit.** Every call, admitted or not, appends one JSONL record with the full audit tuple (inputs, reliabilities, dependence model, cutoff, band, eff, opcode, status, parents, time). The record carries `prev` and `hash = H(prev ‖ canonical(record))`, optionally keyed with HMAC. `replay()` verifies the chain **and** recomputes every decision. Editing, deleting, reordering, truncating or forging a record (even with a recomputed hash, unless keyed) is detected. | `trace.py` `_chain`, `replay`, `ClaimTrace.verify`. | `tests/test_trace.py::test_every_call_logged_with_full_audit_tuple`, `::test_replay_reconstructs_exactly`, `::test_editing_a_logged_decision_is_detected`, `::test_deleting_or_reordering_records_is_detected`, `::test_forged_entry_with_recomputed_hash_but_wrong_decision_is_detected`, `::test_forged_consistent_entry_detected_with_hmac`, `::test_truncated_or_garbage_line_detected`, `::test_in_memory_tampering_is_caught_by_verify`, `::test_deleted_log_is_caught_by_verify`; `tests/test_dependence.py::test_trace_tampered_dependence_model_detected` |

Limits of (a):

* An **unkeyed** chain can be rewritten wholesale by someone with write access to the log. Use `hmac_key=` or anchor
  `ClaimTrace.head()` externally.
* (a) guarantees that nothing *unverified* is emitted. "Verified" means it passed the gate and the learned posterior
  cutoff. It does **not** mean "true": the judges, NLI and retrieval can all be wrong together. That is what (b) measures.
* The SciFact evaluation decides every held-out claim through a real `ClaimTrace` and calls `verify()` on its log
  (`phase3/evaluate.py:node_run`), so the measured numbers below come from the guarded path, not from a re-implementation.

## (b) Measured guarantees (SciFact, held-out dev)

Setup: SciFact dev, n = 300 held-out claims (124 SUPPORT, 64 CONTRADICT, 112 NEI). The binary task is "supported vs not",
so 176 claims are unsupported. Every reliability, dependence estimate, calibration, cutoff and abstain band was learned
on 350 SciFact **train** claims (5-fold CV for selection). Dev was used once, for reporting. CIs are 95% bootstrap
(10,000 resamples), "upper" is the one-sided 95% Clopper-Pearson bound, and paired tests are exact McNemar.
Source for b1–b4: the Phase 3b run (commit f70cc79), kept as `results/phase3/scifact_p3b.txt`. `results/phase3/scifact.txt` / `scifact.json`
now hold the Phase 3c re-run with the strict verifier J11 added (section b5). Chart: `results/phase3/risk_coverage.png` (3c run).

### b1. Decide-everything mode (no abstention)

| Evidence condition | Node (train-CV selected) | Best single judge | False accepts, J1 unwrapped → node (of 176) |
|---|---|---|---|
| gold abstracts | **91.3%** [88.0, 94.3] (bayes-logit) | J4 unwrapped 90.0% (+1.3 pts, p = 0.42) | 19 → 6 (3.4%), p = 0.0002 |
| rv3 retrieval: RRF(BM25, MedCPT, bge) → MedCPT cross-encoder, top 5 | **87.3%** [83.3, 91.0] (bayes-ess) | J4 85.7%; calibrated J9 88.7%, J10 88.0% | **29 → 12** (16.5% → 6.8%), **p < 0.0001** |
| retrieved5 (BM25 top 20 → MiniLM, top 5) | 87.3% [83.3, 91.0] | J4 / J9 87.0% | 27 → 19, p = 0.057 (not significant) |

Honest reading:

* **On realistic retrieved evidence the node does not reach 90%.** It ties the best single judges (differences are
  within ±2 points, McNemar p > 0.3). Its measurable gain is **safety**: it accepts 59% fewer unsupported claims than
  the unwrapped judge (17 claims only J1 accepted, 0 only the node accepted), and it is 4.3 points more accurate than J1
  (p = 0.029).
* Better retrieval did not translate into accuracy. Dev recall@5 of a gold abstract rose from .883 (BM25) to .936
  (MiniLM) to **.963** (rv3), yet node accuracy stayed at 87.3%. Only 4 of the 38 rv3 errors are retrieval misses. The
  rest are judgement errors: 12 supported claims rejected by panel consensus, 10 NEI claims accepted, and 17 errors where
  the whole panel unanimously disagrees with gold (see label noise below).

### b2. Selective mode ("not sure"): abstain band picked on train out-of-fold posteriors, applied to dev unchanged

| Condition / policy | Train target | Dev coverage | Dev errors / decided | Dev error rate, 95% upper bound |
|---|---|---|---|---|
| gold, bayes-ess_cal | 1.00 (zero errors) | 18.7% | **0 / 56** | **5.21%** |
| gold, bayes-ess_cal | 0.99 | 31.7% | 3 / 95 | 7.96% |
| gold, bayes-ess_cal | 0.98 | 33.3% | 5 / 100 | 10.23% |
| gold, bayes-ess_cal | 0.97 | 77.7% | 12 / 233 (94.8% acc.) | 8.21% |
| rv3, bayes-ess_cal | 0.97 | 20.0% | 3 / 60 (95.0%) | 12.42% |
| rv3 and retrieved5, any policy | ≥ 0.98 | no train-certified band: abstain on everything | – | – |

* **Zero held-out errors is achievable only at low coverage, and only with gold evidence.** 56 of 300 decided with 0
  errors still leaves a 95% upper bound of 5.21% (rule of three, ≈ 3/56). "0 errors" is not "100% safe".
* With retrieved evidence, no band reaching ≥ 98% on train exists, so the node fails closed (everything NOT_SURE).

### b3. Risk-controlled NOT_SURE (Learn-Then-Test, Bonferroni over 20 margins, δ = 0.05)

The claim to certify is: decided-claim error ≤ α with 95% confidence, calibrated on train OOF only. Disagreement
(judges split 2+/2+, or NLI contradicts an accept) also forces NOT_SURE.

* **α = 1% (and 0.5%, 2%, 5%): nothing can be certified.** Coverage collapses to about 0. The only decided dev claims
  are 2 exact-copy-gate BLOCKs, and nothing is accepted, in every condition. This is a sample-size limit, not a tuning
  failure. With 350 training claims and confident errors already among the 25 most confident, certifying ≤ 1% would need
  roughly 300 or more decided errorless calibration claims (≈ 3/α, more after Bonferroni). Abstaining is the correct
  fail-closed result.
* α = 10% is certifiable on gold only (bayes-ess_cal): train OOF 12/271 errors (upper 9.15%), dev coverage 81.0%,
  15 errors / 243 decided (upper 9.35%), 4 false accepts of 90 accepted. It is not certifiable on retrieved evidence.

### b4. Dataset caveats, including label noise

* **Label noise caps measurable accuracy.** `phase3/label_review.md` lists 9 dev and 8 train claims where every judge
  unanimously and confidently (p ≥ 0.8) contradicts the gold label on gold evidence. Excluding the 9 dev claims
  (sensitivity only, never a headline, and the test set was not relabelled), accuracy is 93.5% (gold node), 89.3% (rv3
  node), 89.7% (rv3 bayes-ess_cal) and 87.6% (J4 on rv3). If those labels are wrong, ~3% of dev is unreachable for any
  honest system. That and the confident-error tail are why 99% is not supportable on SciFact at this sample size.
* One dataset (biomedical abstracts, 300 dev claims). Shifts in domain, claim style or retrieval corpus will change all
  of (b). Re-calibrate before relying on these numbers elsewhere.
* The judges are LLMs behind a paid API. Their behaviour can change across model versions. All responses are cached so
  this run is reproducible, but a re-run against live models is not guaranteed to match.
* 1 judge call (dev:971, J5, gold) timed out repeatedly and is treated as an abstaining (uniform) vote.
* The binary task merges CONTRADICT and NEI. The 3-way accuracy is lower (rv3: node 79.7% [75.0, 84.0], best judge J10 81.7%).

### b5. Phase 3c: strict proof verifier (J11), aimed at NEI claims accepted as supported

J11 (grok-4.20 reasoning, prompt V) must copy one evidence sentence and answer PROVEN only when that quote directly
entails the claim, with the same population, outcome and direction and no hedge. A PROVEN verdict also needs the quote
to pass the byte-exact gate. Prompt and use were chosen on train CV only (`phase3/p3c_select.py`,
`results/phase3/p3c_selection.json`). A second prompt (W) was dropped after 47 calls for budget.

* **J11 is very strict.** On train it gives a verified proof for only 42% (gold) / 55% (rv3) of SUPPORT claims, but for
  just 1.5% / 3.8% of NEI claims. SciFact labels paraphrased or generalised findings as SUPPORT, so strict entailment
  under-accepts.
* **As a hard ACCEPT veto it fails the accuracy test.** Train CV: gold .911 → .754, rv3 .889 → .797, so the veto is
  OFF in the headline policy. Dev confirms it: with the veto, false accepts are rv3 4/176 (2.3%, against 12) and gold
  1/176 (0.6%), but accuracy falls to 78.7% / 77.3%.
* **As an extra source it changes little.** Train CV +0.3 pts. On dev, rv3 node 87.3% [83.3, 91.0] (unchanged), false
  accepts 29 → 12 (unchanged), and still 10 NEI claims accepted. Gold: train CV now selects bayes-ess with J11, at
  90.3% [87.0, 93.7] and 19 → 8 false accepts (p = 0.0034). Before J11, bayes-logit scored 91.3%, and bayes-ess_cal
  still scores 91.3% [88.0, 94.3].
* **"ACCEPT only with proof, otherwise NOT_SURE"** (fail-closed; node rejects stay REJECT): on rv3 dev, coverage 86.0%,
  4 false accepts among 68 accepted (upper 12.96%), and 30 errors among 258 decided (mostly supported claims the node
  rejected). On gold dev, 1 false accept among 58 accepted (upper 7.92%).
* **Selective bands with J11 (3c run):** on gold, the selected node at target 0.99/1.00 decides 47/300 (15.7%) with 0
  errors (upper 6.18%), and ess_cal at target 1.00 decides 63 with 1 error (upper 7.31%). On rv3, no band reaches 98% or
  higher at usable coverage (ess_cal decides 12 claims with 0 errors, upper 22.1%). LTT at α ≤ 5% still certifies nothing.

Net: a strict verifier makes accepted claims cleaner (the false-accept rate falls about 5×) at the cost of rejecting
about half of the true supports. It does not raise accuracy on SciFact, and it does not change the certification
picture. That still needs the larger, cleaner test set scoped in `phase3/STEP2_PLAN.md`.

**Bottom line.** By construction (a), nothing unverified and nothing NOT_SURE is ever emitted as verified. Measured (b),
when the node does decide on held-out data its error rate is bounded at about 5% (gold evidence, 19% coverage) to
about 9–12% (broader coverage). It cannot yet certify ≤ 1%. Getting there needs a much larger calibration set and cleaner labels.

### b6. Phase 3d: larger calibration pool (2,798 claims) and three out-of-sample test sets

Full details: `results/phase3/RESULTS.md` (last section), `results/phase3/step2.txt`. Calibration was SciFact train plus
HealthVer train, and every choice was fit on that pool only. HealthVer is internal (aggregates only).

* **α = 1% and 2%: still nothing certifiable, with 8× more calibration data.** Coverage is 0%, and no test claim is
  accepted, so there are 0 false accepts. That is the fail-closed outcome. The reason is now the confident-error tail, not
  only sample size: about 16% of the most confident calibration decisions disagree with gold. Most are accepts of claims
  that HealthVer labels NEI or CONTRADICT.
* α = 10% can be certified only through the family supplement (Bonferroni over 10 configurations × 20 margins), with 4.5%
  calibration coverage. On test: HealthVer 4/106 errors (upper 8.4%), SciFact 2/65 (upper 9.4%). Climate-FEVER has no
  calibration data and exceeds the bound (15/142, 10.6%). This is a measured example of why certificates do not transfer
  across sources.
* Decide-everything accuracy: HealthVer 78.6%, Climate-FEVER 78.8%, SciFact 87.3%. False accepts of unsupported claims are
  8.2% / 1.7% / 9.7%, higher than the best single judge's. The full-coverage node is not a safety mode; NOT_SURE is.
* Net: 99% accuracy on decided claims is **not** demonstrated on any real dataset. The supported safety statement is
  structural: an uncertain claim becomes NOT_SURE and is never accepted (section a). The measured bound is ≤ 10% on
  exchangeable claims, at small coverage.

### b7. Phase 3e: audit, stricter accept rules and per-dataset calibration (no new API calls)

Full details: `results/phase3/RESULTS.md` (Phase 3e) and `results/phase3/step3e.txt`.

* **Audit of the confident errors** (22 in the Phase 3d top 139; 27 in the whole tied tier, all false accepts): 7 (10)
  label noise, 6 (7) ambiguous, 9 (10) real node errors, 0 retrieval misses. The node errors are over-inference, for
  example from a recommendation to a risk finding, from one gene to a general rule, or from "most" to "not all".
* **α = 1% and 2%: still not certifiable**, under any of three risk definitions and either calibration scheme, so the
  node stays fail-closed (NOT_SURE).
  * SciFact is limited by sample size. Even zero errors would need at least 402 decided claims at 2% and 808 at 1% for
    this family, against 799 calibration claims, of which only 328 are SUPPORT.
  * HealthVer is limited by label noise: a unanimous 6/6 panel disagrees with gold on 12.5% of claims.
* **What is now certified (δ = 0.05, calibration out-of-fold, SciFact population only):**
  * Accept precision ≤ 5% (accept-only mode, 19% calibration coverage, 0/152 errors). On SciFact dev: 63 accepted,
    1 unsupported.
  * Selective error ≤ 10% with BLOCK reported separately (91% coverage). On SciFact dev: 92.6% accuracy at 94.3%
    coverage.
  * Nothing is certified for HealthVer. Climate-FEVER is transfer and not covered.
* **Per-dataset calibration** brings SciFact full-coverage accuracy back to 89.7% (Phase 3d pooled: 87.3%) and cuts
  false accepts from 29 to 10 of 176.
* **Caveats:** the rule family was designed after looking at calibration errors; the two extra risk definitions were
  added post hoc (δ ≤ 0.15 jointly); the certificates are out-of-fold.
* **Net:** no claim of 99% or 100% accuracy is supported on real data. The structural guarantee in (a) is unchanged: an
  uncertain claim is never accepted.
