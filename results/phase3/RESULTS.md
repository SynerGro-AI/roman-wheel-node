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
