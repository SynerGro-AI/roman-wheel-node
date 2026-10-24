# Phase 4 fresh-set results (aggregates only)

Fresh = never used to select or tune any rule: HealthVer test pairs left unjudged in Phase 3d (`hv_test_unjudged`), COVID-Fact (`covidfact`), HealthVer dev (`hv_dev`, local profiles only). HealthVer and COVID-Fact are internal, unlicensed datasets: no claim, evidence or per-item output is committed. Labels: dataset gold (SUPPORT vs not).

Risk = false accepts among ACCEPTs; one-sided Clopper-Pearson, delta = 0.05; one pre-registered fixed rule per profile (no search). Precision CI = two-sided 95% Clopper-Pearson. 'needs' = labelled claims needed at the observed ACCEPT rate if every accept were correct (299 / 149 / 59 error-free accepts for 1% / 2% / 5%).

| profile | set | labelled | ACCEPT (coverage) | false accepts | precision [95% CI] | UB on false-accept share | 1% | 2% | 5% | needs for 1% / 2% |
|---|---|---|---|---|---|---|---|---|---|---|
| local (free: nli2 + overlap) | all | 6311 | 633 (10.0%) | 204 | 67.8% [64.0%, 71.4%] | 35.4% | no | no | no | 2982 / 1486 |
| local (free: nli2 + overlap) | hv_test_unjudged | 308 | 0 (0.0%) | 0 | n/a | 100.0% | no | no | no | n/a / n/a |
| local (free: nli2 + overlap) | covidfact | 4086 | 612 (15.0%) | 203 | 66.8% [62.9%, 70.6%] | 36.4% | no | no | no | 1997 / 995 |
| local (free: nli2 + overlap) | hv_dev | 1917 | 21 (1.1%) | 1 | 95.2% [76.2%, 99.9%] | 20.7% | no | no | no | 27295 / 13602 |
| local-lite (free: ONNX nli + overlap) | all | 6311 | 393 (6.2%) | 147 | 62.6% [57.6%, 67.4%] | 41.6% | no | no | no | 4802 / 2393 |
| local-lite (free: ONNX nli + overlap) | hv_test_unjudged | 308 | 0 (0.0%) | 0 | n/a | 100.0% | no | no | no | n/a / n/a |
| local-lite (free: ONNX nli + overlap) | covidfact | 4086 | 392 (9.6%) | 146 | 62.8% [57.8%, 67.6%] | 41.5% | no | no | no | 3117 / 1554 |
| local-lite (free: ONNX nli + overlap) | hv_dev | 1917 | 1 (0.1%) | 1 | 0.0% [0.0%, 97.5%] | 100.0% | no | no | no | 573183 / 285633 |
| panel (frozen Phase 3e rule: 6 xAI judges + local) | all | 3829 | 559 (14.6%) | 148 | 73.5% [69.7%, 77.1%] | 29.7% | no | no | no | 2049 / 1021 |
| panel (frozen Phase 3e rule: 6 xAI judges + local) | hv_test_unjudged | 308 | 4 (1.3%) | 0 | 100.0% [39.8%, 100.0%] | 52.7% | no | no | no | 23023 / 11473 |
| panel (frozen Phase 3e rule: 6 xAI judges + local) | covidfact | 3521 | 555 (15.8%) | 148 | 73.3% [69.4%, 77.0%] | 29.9% | no | no | no | 1897 / 946 |
