"""Re-check the Phase 4b structural veto fields on commercial-use data only (zero API calls).

structure.FROZEN's veto fields were chosen on half A of COVID-Fact, which has no licence at its primary source
(DATA_LICENSES.md). This script applies the same pre-declared bar (>= 3 false accepts caught, and >= 3 caught per
correct accept lost) to the policy 4.2.0 panel's out-of-fold ACCEPTs on SciFact train (CC BY 4.0), at the margin
the 4.2.0 policy ships. Evidence = the gold rationale sentences; judge quotes are not used. Aggregates only.

    env -u XAI_API_KEY python3 phase6/structural_scifact.py
"""

from __future__ import annotations

import json
import os
import sys

if os.environ.get("XAI_API_KEY"):
    sys.exit("refusing to run with XAI_API_KEY set: this script makes no API call (use env -u XAI_API_KEY)")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "phase4b"), os.path.join(ROOT, "phase3"), os.path.join(ROOT, "src"), HERE):
    sys.path.insert(0, p)
import evaluate as E  # noqa: E402
import refit_policy_42 as RF  # noqa: E402
import run_structural as RS  # noqa: E402
import s2_eval as S2  # noqa: E402
import s3e_eval as S3E  # noqa: E402
import s3e_rules as R  # noqa: E402
from features import CACHE  # noqa: E402

from roman_wheel_node.policy import load_policy  # noqa: E402

HISTORICAL_43 = (
    "superseded: policy 4.2.0 no longer ships (its panel profile was fitted on xAI judge outputs and its local margins "
    "used the J1 quote gate). Shipped policy: 4.3.0 (phase6/refit_policy_43.py). This run's aggregates are kept as a "
    "benchmark ledger only (results/phase6/xai_panel_ledger.md); reproduce it from git commit 69aa8db."
)


def main() -> None:
    sys.exit(HISTORICAL_43)
    rows, _ = S2.build_rows()
    F = S3E.features(rows)
    pool = [r for r in rows if r["ds"] in RF.FIT_DS]
    items = {i["id"]: i for i in json.load(open(os.path.join(CACHE, "step2", "items.json"), encoding="utf-8"))}
    margin = load_policy().profile("panel").accept_margin
    o, d, c = E.oof_effs(pool, RF.COND, RF.PROFILES["panel"], "logit")
    feats = [dict(F[r["id"]], dis=d[i]) for i, r in enumerate(pool)]
    its = R.to_items(feats, o, c, R.make_rules(j11=False)["base"], False)
    srows = []
    for r, it in zip(pool, its):
        dec = R.decide(it, margin, R.INF)
        srows.append(
            {
                "outcome": "ACCEPT" if dec is True else "NOT_SURE",
                "claim": items[r["id"]]["claim"],
                "units": items[r["id"]]["units"],
                "y": r["label3"] == "SUPPORT",
            }
        )
    acc = [s for s in srows if s["outcome"] == "ACCEPT"]
    tab = RS._field_table(srows)
    res = RS._before_after(RS._apply(srows))
    out = {
        "population": f"SciFact train out-of-fold (n={len(pool)}), policy 4.2.0 panel margin {margin:.4f}",
        "accepts": len(acc),
        "false_accepts": sum(1 for s in acc if not s["y"]),
        "bar": RS.BAR,
        "fields": tab,
        "fields_meeting_bar": [k for k, v in tab.items() if v["meets_bar"]],
        "current_frozen_veto_fields": list(RS.ST.FROZEN["veto_fields"]),
        "frozen_rule_effect": {k: res[k] for k in ("false_accepts_caught", "correct_accepts_lost", "veto_reasons")},
    }
    os.makedirs(RF.RES, exist_ok=True)
    json.dump(out, open(os.path.join(RF.RES, "structural_scifact.json"), "w", encoding="utf-8"), indent=1, default=list)
    print(json.dumps(out, indent=1, default=list))


if __name__ == "__main__":
    main()
