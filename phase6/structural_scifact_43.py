"""Re-check the structural veto fields on the shipped local profile of policy 4.3.0 (zero API calls).

Same pre-declared bar as phase6/structural_scifact.py (>= 3 false accepts caught, and >= 3 caught per correct accept
lost), applied to the out-of-fold ACCEPTs of profile ``local`` (nli2 + overlap) on SciFact train (CC BY 4.0) at the
margin policy 4.3.0 ships. Out-of-fold posteriors come from ``roman_wheel_node.fit`` (the same code that fitted the
policy). Evidence = the gold rationale sentences. Local open-model features only. Aggregates only.

    env -u XAI_API_KEY python3 phase6/structural_scifact_43.py
"""

from __future__ import annotations

import json
import math
import os
import sys

if os.environ.get("XAI_API_KEY"):
    sys.exit("refusing to run with XAI_API_KEY set: this script makes no API call (use env -u XAI_API_KEY)")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "phase4b"), os.path.join(ROOT, "phase3"), os.path.join(ROOT, "src"), HERE):
    sys.path.insert(0, p)
import refit_policy_43 as RF  # noqa: E402
import run_structural as RS  # noqa: E402

from roman_wheel_node import fit as FIT  # noqa: E402
from roman_wheel_node.policy import load_policy  # noqa: E402

PROFILE = "local"


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    pool, _ = RF.load_rows()
    items = {i["id"]: i for i in _load(os.path.join(RF.CACHE, "items.json"))}
    prof = load_policy().profile(PROFILE)
    effs, cuts, dis = FIT._oof(pool, prof.sources, 5, dict(prof.disagreement))
    srows = []
    for r, e, c, d in zip(pool, effs, cuts, dis):
        it = FIT._items([r], [e], [c], [d])[0]
        ok = it["accept_side"] and it["z"] >= prof.accept_margin and math.isfinite(it["z"])
        srows.append(
            {
                "outcome": "ACCEPT" if ok else "NOT_SURE",
                "claim": items[r["id"]]["claim"],
                "units": items[r["id"]]["units"],
                "y": r["label"] == "SUPPORT",
            }
        )
    acc = [s for s in srows if s["outcome"] == "ACCEPT"]
    tab = RS._field_table(srows)
    res = RS._before_after(RS._apply(srows))
    out = {
        "population": f"SciFact train out-of-fold (n={len(pool)}), policy 4.3.0 profile {PROFILE} margin "
        f"{prof.accept_margin:.4f}",
        "accepts": len(acc),
        "false_accepts": sum(1 for s in acc if not s["y"]),
        "bar": RS.BAR,
        "fields": tab,
        "fields_meeting_bar": [k for k, v in tab.items() if v["meets_bar"]],
        "current_frozen_veto_fields": list(RS.ST.FROZEN["veto_fields"]),
        "frozen_rule_effect": {k: res[k] for k in ("false_accepts_caught", "correct_accepts_lost", "veto_reasons")},
    }
    with open(os.path.join(RF.RES, "structural_scifact_43.json"), "w") as f:
        json.dump(out, f, indent=1, default=list)
        f.write("\n")
    print(json.dumps(out, indent=1, default=list))


if __name__ == "__main__":
    main()
