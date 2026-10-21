# Phase 3c step 2: a bigger test set (scoping only; $0 spent)

Goal: certify a decided-claim error rate of at most 1% (one-sided 95% Clopper-Pearson upper bound) at useful coverage,
on data the node never tuned on.

## How many claims certification needs
CP upper bound ≤ 1% at 95%, with k errors among n decided claims (computed with `evaluate.cp_upper`):

| errors k | 0 | 1 | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|---|
| decided n needed (one test, δ = 0.05) | 299 | 473 | 628 | 773 | 1,049 | 1,693 |
| decided n needed (LTT calibration, Bonferroni K = 20, δ/K) | 597 | 818 | 1,009 | – | 1,511 | – |

The true decided error rate must sit well below 1% for this to be reachable. If it is 0.25%, the test needs about 480
decided claims and LTT calibration about 1,190. If it is 0.5%, those become about 1,050 and 2,680.
At 50% coverage that is **about 2,400 calibration + about 1,000 test claims** (0.25% case). Label noise counts as error,
so the gold labels themselves must be wrong in well under 0.25% of claims. SciFact's unanimous-panel disagreement rate is
about 3% of dev, which is why 1% is out of reach there: labels need adjudication, or a dataset with multiple votes.

## Candidate public datasets (counts from the downloaded files or the papers; licences from the GitHub API, 2026-10-21)
| Dataset | Size / labels | Gold evidence condition? | Licence | Notes |
|---|---|---|---|---|
| **SciFact, unused train claims** | 459 more train claims (809 total, we used 350) | yes (cited abstracts + rationales) | claims CC BY 4.0, abstracts ODC-By | same distribution, cheapest calibration add; still has SciFact label noise |
| **HealthVer** | 14,330 evidence–claim pairs, 1,855 claims. Pairs: train 3,782 S / 2,411 R / 4,397 N; dev 533 / 391 / 993; test 671 / 425 / 727. Claim level (test): 74 S, 39 R, 47 N, 70 mixed | yes (annotated evidence statement per pair) | **none** (repo has no licence; open issue #4 asks for one) | the pair-level task is cheap (one passage per call), but internal evaluation only until it is licensed |
| **COVID-Fact** | 4,086 claims: 1,296 SUPPORTED / 2,790 REFUTED (no NEI) | yes (evidence sentences) | **none** (no licence in repo) | refuted claims are auto-generated counter-claims (artefacts); no NEI, so it does not test the main error mode |
| **Climate-FEVER** | 1,535 claims (655 S / 253 R / 474 NEI / 153 DISPUTED), 7,675 pairs with per-pair annotator votes | yes (5 Wikipedia sentences per claim) | original: unknown (HF card "unknown"); BEIR mirror CC BY-SA 4.0 | the votes allow a low-noise subset (unanimous pairs) |
| **SciFact-Open** | 279 claims, 500K-abstract corpus | pooled annotations only (open-domain) | repo: none; SciFact parts CC BY 4.0 / ODC-By | too small to certify; useful as an open-retrieval stress test |
| COVID-FACT (MultiVerS abstract version, `dwadden/covidfact_entailment`) | abstract-level verdicts | yes | MultiVerS code MIT; data licence unclear | optional |

## Per-claim cost (measured in phase3/cache/spend.jsonl)
Mean $/call: grok-4.20 non-reasoning 0.0016, grok-4.3 0.0030, grok-4.20 reasoning 0.0062 (J4/J9/J10) and 0.010 (J11, longer output),
grok-4.6 0.0135, grok-4.7 0.0087. These are on SciFact evidence of 1–5 abstracts.

* Lean panel (J1 gate + J4 + J9 + J10 + J11 + J2): about **$0.034 per claim** per condition (5-abstract evidence). With
  single-passage gold evidence (HealthVer pairs, Climate-FEVER sentences) the prompt is about 5–10× shorter, so about $0.02 per claim.
* Minimal panel (J1 + J4 + J11): about $0.018 per claim.

## Plan and dollar estimate
1. Calibration: SciFact unused train (459) + HealthVer train pairs, sampled to about 2,000 → about 2,450 items.
2. Held-out test: HealthVer test (1,823 pairs) + a unanimous-vote Climate-FEVER subset (about 1,000 pairs) → about 2,800 items.
   Never tuned on.
3. Gold-evidence condition only for HealthVer/Climate-FEVER (their evidence is given). SciFact keeps gold + rv3.
4. Cost: about 5,250 items × $0.02 (single passage) ≈ **$105 with the lean panel**, or ≈ **$60–70 with the minimal panel**
   (J1 + J4 + J11). Adding rv3 for the SciFact part: +$16.
5. A new budget is needed: Phase 3 has $6.37 left under the $100 cap ($93.63 spent). Recommend approving **$75** for the minimal panel
   (with a 10% margin), or $120 for the lean panel.
6. Before spending: confirm licences (HealthVer and COVID-Fact have none), and adjudicate or filter label noise (Climate-FEVER votes).
