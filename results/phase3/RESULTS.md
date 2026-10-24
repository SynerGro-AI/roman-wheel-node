# Phase 3 results: SciFact through the Roman Wheel Node

Sections up to "Spend" are the Phase 3b run (commit f70cc79, `scifact_p3b.txt`). `scifact.txt`/`scifact.json` are now the
Phase 3c re-run with the strict verifier J11 (last section).

Protocol: 350 SciFact train claims (stratified, hashed order) were used for every learned quantity: calibration
(isotonic), reliabilities, dependence, cutoff, model selection (5-fold CV), abstain bands and LTT margins. Reporting is
on all 300 dev claims, used once. Binary task: SUPPORT vs (CONTRADICT or NEI); 176 dev claims are unsupported.
95% bootstrap CIs (10,000), exact McNemar, one-sided Clopper-Pearson upper bounds. Full numbers: `scifact.txt`/`scifact.json`.

## Retrieval (dev, 188 claims with gold evidence; pipeline chosen on train recall/MRR only)
| Pipeline | R@1 | R@3 | R@5 | R@10 | MRR@20 |
|---|---|---|---|---|---|
| BM25 | .723 | .856 | .883 | .931 | .800 |
| bge-small (dense) | .782 | .894 | .920 | .968 | .848 |
| MedCPT (dense) | .729 | .899 | .947 | .984 | .825 |
| RRF(BM25, MedCPT, bge) | .782 | .920 | .941 | .979 | .852 |
| **rv3 = MedCPT cross-encoder over RRF top 20** | **.824** | **.936** | **.963** | .968 | **.881** |
| retrieved5 (MiniLM over BM25 top 20), R@5 | | | .936 | | |

On train (220 evidence claims), rv3 R@5 was .982 against .914 for BM25 and MiniLM(BM25).

## Binary accuracy, selected node vs judges
| Condition | Node | J1 unwrapped | Best judge (train-chosen) | Node − best judge |
|---|---|---|---|---|
| gold | 91.3 [88.0, 94.3] | 86.7 | J4 90.0 | +1.3 (p = 0.42) |
| rv3 | 87.3 [83.3, 91.0] | 83.0 | J4 85.7 | +1.7 (p = 0.53) |
| retrieved5 | 87.3 [83.3, 91.0] | 83.0 | J4/J9 87.0 | +0.3 (n.s.) |
| retrieved (BM25 top 3) | 84.7 [80.3, 88.7] | 84.3 | J1 84.3 | +0.3 (n.s.) |

rv3 extras: bayes-ess_cal 87.7, logistic stack (not gated) 88.7, calibrated J9 alone 88.7, J10 (NEI-focused prompt) 88.0.
3-way on rv3: node 79.7%, best judge J10 81.7%.

## False accepts (unsupported claims accepted, of 176) vs J1 unwrapped
gold 19 → 6 (p = 0.0002); **rv3 29 → 12 (p < 0.0001; 17 only-J1, 0 only-node)**; retrieved5 27 → 19 (p = 0.057).

## rv3 error analysis (38 dev errors)
4 retrieval misses, 10 NEI accepted, 2 CONTRADICT accepted, 12 SUPPORT rejected by panel consensus, 9 SUPPORT rejected with a
split panel, 1 by the gate. 17 errors are cases where the whole panel unanimously disagrees with gold. Retrieval is no longer the bottleneck.

## Selective / NOT_SURE (bands from train OOF only)
| | target | coverage | errors/decided | upper 95% |
|---|---|---|---|---|
| gold ess_cal | 1.00 | 18.7% | 0/56 | 5.21% |
| gold ess_cal | 0.99 | 31.7% | 3/95 | 7.96% |
| gold ess_cal | 0.98 | 33.3% | 5/100 | 10.23% |
| gold ess_cal | 0.97 | 77.7% | 12/233 | 8.21% |
| rv3 ess_cal | 0.97 | 20.0% | 3/60 | 12.42% |
| rv3 / retrieved5 | ≥ 0.98 | 0 (no train-certified band; abstain) | – | – |

LTT (Bonferroni over 20 log-odds margins, δ = 0.05) with disagreement → NOT_SURE: α ≤ 5% certifies nothing in any
condition, so coverage is ≈ 0 and nothing is accepted. This is the sample-size limit of 350 calibration claims. α = 10% on gold:
81.0% coverage, 15/243 errors (upper 9.35%). Chart: `risk_coverage.png`.

## Label noise
`phase3/label_review.md`: 9 dev + 8 train claims where every judge confidently contradicts gold. Excluding the 9 dev
claims (sensitivity only): gold node 93.5%, rv3 node 89.3%, rv3 ess_cal 89.7%, rv3 J4 87.6%.

## Spend
Phase 3b: $80.82. Phase 3c: +$12.81 (J11 1,300 calls, J12 47 calls) → **$93.63 of the $100 cap** (`phase3/cache/spend.jsonl`).

## Phase 3c: strict proof verifier J11 (prompt V, grok-4.20 reasoning), selection on train CV only
| | gold before → after | rv3 before → after |
|---|---|---|
| train-CV, no verifier → J11 as source | .9114 → .9143 | .8886 → .8914 |
| train-CV with J11 hard veto | .754 | .797 (so the veto is OFF) |
| J11 verified-proof rate on train S / C / NEI | .42 / .013 / .015 | .55 / .027 / .038 |
| selected node, dev accuracy | 91.3 (bayes-logit) → 90.3 [87.0, 93.7] (bayes-ess) | 87.3 → 87.3 [83.3, 91.0] |
| bayes-ess_cal, dev accuracy | 91.0 → 91.3 | 87.7 → 87.3 |
| false accepts of 176, selected node | 6 → 8 | 12 → 12 |
| false accepts, bayes-ess_cal | 7 → 7 | 16 → 11 |
| node + J11 veto: accuracy / false accepts | 77.3% / 1 (0.6%) | 78.7% / 4 (2.3%) |
| proof-or-NOT_SURE (dev): coverage, FA/accepted (upper) | 82.3%, 1/58 (7.92%) | 86.0%, 4/68 (12.96%) |
| selective, zero-error target (dev) | 0/56 (5.21%) → 0/47 (6.18%) | none → none (ess_cal: 0/12, 22.1%) |
| NEI claims accepted (selected node errors) | – | 10 → 10 |

J11 alone: gold 82.7% (calibrated) with 0.6% false accepts; rv3 84.7%. Its error correlation with the other reasoning
judges is high (phi .44–.58 on rv3), so the dependence-aware combiner gives it little extra weight.

## Phase 3d (step 2): larger calibration pool, three out-of-sample test sets, full 6-judge panel
Script `phase3/s2_eval.py`; aggregates only in `results/phase3/step2.txt` / `step2.json`. Every choice (isotonic source
calibration, node variant and source set by 5-fold CV, reliabilities, dependence, cutoff, LTT margins) was fit on the
calibration pool only: SciFact train 350 + 449 unused train claims + HealthVer train 1,999 pairs = **2,798 claims**, gold
evidence, judges J1 J2 J4 J7 J9 J10 + local NLI. Test sets were used once. HealthVer is an internal, unlicensed dataset:
its data and per-item outputs are not redistributed, only aggregate numbers appear here. Climate-FEVER (unanimous-vote
pairs, CC BY-SA) has no calibration data, so it is a transfer test (measured, not certified).

Selected on calibration CV: bayes-hard [judges only] (CV accuracy .823; cutoff 0.065). Best single judge on calibration: J4.

| Test set (binary: supported vs not) | n | Node, full coverage | Unwrapped J1 | J4 | False accepts node / J1 / J4 | McNemar node vs J1 |
|---|---|---|---|---|---|---|
| HealthVer test (internal) | 1,515 | **78.6%** [76.6, 80.7] | 75.2% | 75.8% | 124 / 93 / 76 of 951 (8.2% / 6.1% / 5.0%) | 140 vs 89, p = 0.0009 |
| Climate-FEVER (transfer) | 1,500 | **78.8%** [76.7, 80.9] | 74.5% | 74.3% | 26 / 17 / 13 of 867 (1.7% / 1.1% / 0.9%) | 98 vs 34, p < 1e-7 |
| SciFact dev | 300 | 87.3% [83.3, 91.0] | 86.7% | **90.0%** | 29 / 19 / 15 of 176 (9.7% / 6.3% / 5.0%) | 19 vs 17, p = 0.87 |

At full coverage the node is more accurate than J1 on HealthVer and Climate-FEVER, but it is tuned for accuracy and
accepts more unsupported claims than J1 or J4 on all three sets. On SciFact dev the mixed-domain pool does worse than the
SciFact-only Phase 3b node (91.3% gold); HealthVer dominates the pool.

**LTT NOT_SURE (δ = 0.05):**
* Pre-registered protocol (the CV-selected node, Bonferroni over 20 margins): **nothing is certifiable at α = 1%, 2%, 5%
  or even 10%**. Among the most confident calibration claims, 22 of the top 139 are already errors (mostly confident
  accepts of claims HealthVer labels NEI/CONTRADICT). The node abstains on every non-BLOCK claim. The only decided test
  claims are exact-copy-gate BLOCKs (HealthVer 2, Climate-FEVER 11 with 4 wrong blocks, SciFact 2 with 1 wrong block):
  **0 accepts, so 0 false accepts**.
* Supplement (all 10 candidates × 20 margins, Bonferroni δ/200, chosen on calibration only): α = 10% is certifiable
  (bayes-logit [all]: 127 calibration decided, 1 error, upper 8.2%, 4.5% calibration coverage). α ≤ 5% again certifies nothing.
  Applied to test at α = 10%: HealthVer 7.0% coverage, 4/106 errors (upper 8.4%), 0 false accepts of 19 accepted;
  SciFact 21.7% coverage, 2/65 errors (upper 9.4%), 1 false accept of 63; Climate-FEVER (transfer, not covered by the
  certificate) 9.5% coverage, 15/142 errors (10.6%, upper 15.8%), 0 false accepts of 36.

**Certified coverage at ≤ 1% and ≤ 2% error: 0%** (fail-closed abstention), with 2,798 calibration claims. The
blocker is no longer sample size alone: the confident-error tail (label disagreement with HealthVer/SciFact gold) is
far above 2%, so no margin isolates an error-free region big enough to certify.

HealthVer test coverage: 1,515 of the 1,823 test pairs have all six judges cached; 308 were not judged because judge
collection stopped at the $215 script cap (items are processed in a fixed hash order, so the judged subset is a
pseudo-random sample). The SciFact dev J10 votes (299 calls, $1.38) were collected afterwards under a $218 guard.

Spend: **$216.60** of the $220 hard cap (`phase3/cache/spend.jsonl`).

## Phase 3e: zero-cost fixes (audit, stricter accept rules, per-dataset calibration, fixed-sequence LTT)

No new xAI call: every judge output is read from the Phase 3d cache, and `phase3/s3e_eval.py` refuses to run with
`XAI_API_KEY` set. Spend stays **$216.60** of $220. Details: `results/phase3/step3e.txt` / `step3e.json`,
`step3e_audit.json`. HealthVer is internal and unlicensed: aggregates only.

**1. Audit of the confident calibration errors** (Phase 3d node, out-of-fold; the top-confidence tier is 184 decided with
27 errors, and the "top 139" quoted for Phase 3d holds 22 of them). All 27 are false accepts. Per-claim notes are local only.

| Class | Top 139 (22) | Whole tier (27) |
|---|---|---|
| Label noise (the shown evidence supports the claim; gold says NEI/CONTRADICT) | 7 | 10 (HealthVer 8, SciFact 2) |
| Ambiguous (reasonable annotators could differ) | 6 | 7 (HealthVer 5, SciFact 2) |
| Real node error (gold right; the panel over-infers) | 9 | 10 (HealthVer 6, SciFact 4) |
| Retrieval miss | 0 | 0 (calibration uses gold evidence) |

8 of the 27 had all six judges voting SUPPORT. J11 was cached on 4 of them (all SciFact) and rejected all 4.

**2. Calibration-set precision of the stricter accept filters** (non-BLOCK claims passing, gold SUPPORT / n):
SciFact: 6/6 unanimous 194/204, unanimous + NLI entailment 108/109, J11-verified + unanimous 51/52.
HealthVer: unanimous 70/80 (12.5% disagree with gold), unanimous + NLI 30/35. The exact-copy gate BLOCKs 7 SciFact
calibration claims, 4 of them gold SUPPORT (HealthVer 9, 3 SUPPORT).

**3–4. Certification.** The family is pre-specified per population: 2 calibration schemes (per-dataset, pooled) × 3 nodes
× 6 accept rules (+2 J11 rules on SciFact) × 4 accept/reject asymmetries. It uses fixed-sequence LTT along each margin
path, Bonferroni over the family and δ = 0.05 per population (SciFact: 168 tests; HealthVer: 144).

| Risk definition | Population | 10% | 5% | 2% | 1% |
|---|---|---|---|---|---|
| Selective error, BLOCK counted as a decided REJECT (Phase 3d accounting, primary) | SciFact | none | none | none | none |
| | HealthVer | none | none | none | none |
| Selective error, BLOCK reported as its own fail-closed outcome | SciFact | **91.0%** cov. (39/727, UB 8.8%) | none | none | none |
| | HealthVer | none | none | none | none |
| Accept precision (accept-only; share of ACCEPTs that are unsupported) | SciFact | 33.4% cov. (12/267, UB 9.7%) | **19.0% cov. (0/152, UB 4.3%)** | none | none |
| | HealthVer | none | none | none | none |

Held-out tests (frozen rules; Climate-FEVER uses the SciFact certificate, so it is transfer, measured and not certified):
* SciFact dev, accept precision at 5%: 63 of 300 accepted (21%), 1 false accept (1.6%; CP UB 7.3%). At 10%: 104
  accepted, 6 false. With BLOCK separate at 10%: 94.3% coverage, 92.6% [89.4, 95.4] accuracy, 7/176 false accepts.
* Climate-FEVER, accept precision at 5%: 36 accepted, 0 false. At 10%: 121 accepted, 1 false. BLOCK separate at 10%:
  91.7% coverage but 71.4% accuracy (it transfers poorly on rejects), 2/867 false accepts.
* HealthVer test: nothing certified, so every non-BLOCK claim is NOT_SURE (0 accepts).
* Phase 3d for comparison: 1/2% gave 0% coverage everywhere; the 10% family certificate covered HealthVer 7.0%,
  SciFact 21.7% and Climate-FEVER 9.5%.

**Full coverage with per-dataset calibration** (CV on each population):
* SciFact dev: **89.7%** [86.0, 93.0], 10/176 false accepts (Phase 3d pooled: 87.3%, 29/176).
* HealthVer test: 78.5%, 137/951 false accepts (Phase 3d: 78.6%, 124/951).
* Climate-FEVER with SciFact calibration: 67.9%, 3/867 false accepts (Phase 3d pooled: 78.8%, 26/867). The
  SciFact-calibrated node rarely accepts on Climate-FEVER.

**Why 1% and 2% are still not certifiable.**
1. **Sample size (SciFact).** With this family, 2% needs at least 402 decided calibration claims (accept-only: 334)
   with zero errors, and 1% needs 808 (670). SciFact calibration has 799 claims, only 328 of them SUPPORT. An
   accept-only 2% certificate is therefore impossible even with perfect accepts. A single rule registered in advance
   would still need 149 (2%) or 299 (1%) error-free decisions.
2. **Label noise (HealthVer).** Even a unanimous 6/6 panel disagrees with HealthVer gold on 12.5% of calibration claims,
   and 17 of the 27 audited errors are label noise or ambiguous. No accept rule can certify below that floor against
   these labels.
3. **Gate BLOCKs on supported claims** (4 of 7 on SciFact) put a floor under the selective risk when BLOCK counts as a
   decided REJECT.

**Caveats.**
* The rule family was designed after the audit of calibration errors and is certified on the same calibration data. The
  held-out tests are the independent check.
* The two extra risk definitions were added after the primary failed. Each uses δ = 0.05, so δ ≤ 0.15 jointly.
* Certificates use out-of-fold posteriors, while the deployed node is refit on the whole population (as in Phase 3d).
