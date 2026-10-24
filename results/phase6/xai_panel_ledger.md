# xAI/Grok judge panel: performance ledger (benchmark comparison only)

**Not used for fitting.** These numbers come from xAI/Grok judge outputs. None of them, and none of those outputs,
is used to fit, select or tune anything that ships: the shipped policy 4.3.0 and the fit API
(`roman_wheel_node.fit`) use local open-model features only (`phase6/refit_policy_43.py`). This file lives in the
repository only. It is not in the sdist or the wheel (`tests/test_no_xai_in_package.py` checks the package).

Source: the policy 4.2.0 run (git `69aa8db`, `phase6/refit_policy_42.py`, now historical), aggregates only, from
`results/phase6/refit_42.json` and `results/phase6/structural_scifact.json`. Machine-readable copy:
`xai_panel_ledger.json`.

## Held-out SciFact dev (n = 300, gold evidence)

| Configuration | ACCEPT | False accepts | Precision | FA upper95 (CP) | Coverage of supported | BLOCK | NOT_SURE | Handled correctly (NOT_SURE = reject) |
|---|---|---|---|---|---|---|---|---|
| xAI panel (withdrawn 4.2.0 `panel`; J1 J2 J4 J7 J9 J10 + nli nli2 overlap) | 108 | 7 | 93.5% | 11.8% | 81.5% | 2 | 190 | 90.0% |
| shipped 4.3.0 `local` (nli2 + overlap) | 58 | 6 | 89.7% | 19.4% | 41.9% | 0 | 242 | 74.0% |
| shipped 4.3.0 `local-lite` (nli + overlap) | 16 | 1 | 93.8% | 26.4% | 12.1% | 0 | 284 | 63.3% |

The panel's 2 BLOCKs come from its J1 verbatim-quote gate. The local profiles have no such gate (a = 1 on gold
evidence).

## Panel certificate on SciFact train out-of-fold (n = 799, LTT fixed sequence, delta 0.05, 1 test)

* alpha 10%: certified, 280 accepted / 17 false, upper 9.0%
* alpha 5%: not certified
* alpha 2%: not certified
* alpha 1%: not certified

The shipped local profiles are not certified at any of these levels (`results/phase6/refit_43.txt`).

## Structural veto fields on the panel's out-of-fold accepts

280 accepts, 17 false; fields meeting the bar: none; frozen rule:
0 false accepts caught, 7 correct accepts lost.
The same check on the shipped `local` profile is `results/phase6/structural_scifact_43.json`.
