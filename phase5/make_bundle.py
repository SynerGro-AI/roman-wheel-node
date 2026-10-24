"""Phase 5: build the INTERNAL data bundle the desktop needs (box only; the output is gitignored, never commit it).

    python3 phase5/make_bundle.py            ->  phase5/bundle/pairs_bundle.json  (+ .sha256)

Contents (all from local caches, no API call):
* covidfact: every COVID-Fact item with claim, evidence units, label, split role, and the frozen decisions already
  logged for it: panel (rwn-policy-4.0.0 panel profile, Phase 4 log), panel_4b (panel + the frozen Phase 4b
  structural veto, using the logged judge quotes), local (rwn-policy-4.0.0 local profile, Phase 4 local log).
  Items the panel never judged have panel = null and are excluded from panel-based numbers (as in Phase 4b).
* noreg: SciFact dev, HealthVer test, Climate-FEVER and hv_test_unjudged rows with panel and panel_4b decisions.
"""

from __future__ import annotations

import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import pairs_common as C  # noqa: E402

from roman_wheel_node import structure as ST  # noqa: E402

CACHE = os.path.join(C.ROOT, "phase3", "cache")


def main() -> None:
    items = [
        i
        for i in json.load(open(os.path.join(CACHE, "phase4", "items.json"), encoding="utf-8"))
        if i["ds"] == "covidfact"
    ]
    roles = C.check_roles(items, C.load_manifest())
    rows = json.load(open(os.path.join(CACHE, "phase4b", "rows.json"), encoding="utf-8"))
    if rows["policy"] != "254804f8ac9b500ef1951879fb64328b0f54303b8bef20e38b1684545f876eb1":
        sys.exit("rows.json is not from rwn-policy-4.0.0")
    panel = {r["id"]: r for r in rows["rows"]}
    local = {}
    for line in open(os.path.join(CACHE, "phase4", "shadow_local", "fresh_claims.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        local[r["meta"]["id"]] = r["outcome"]

    def p4b(r):
        return ST.veto_accept(r["outcome"], r["claim"], r["units"], r["quotes"])[0]

    cf = []
    for it in items:
        r = panel.get(it["id"])
        cf.append(
            {
                "id": it["id"],
                "role": roles[it["id"]],
                "y": it["label3"] == "SUPPORT",
                "claim": it["claim"],
                "units": C.lst(it["units"]),
                "panel": r["outcome"] if r else None,
                "panel_4b": p4b(r) if r else None,
                "local": local.get(it["id"]),
            }
        )
    noreg = [
        {
            "id": r["id"],
            "ds": r["ds"],
            "y": bool(r["y"]),
            "claim": r["claim"],
            "units": r["units"],
            "panel": r["outcome"],
            "panel_4b": p4b(r),
            "local": local.get(r["id"]),
        }
        for r in rows["rows"]
        if r["ds"] in C.NOREG_SETS
    ]
    doc = {
        "format": "rwn-phase5-bundle",
        "note": "INTERNAL: contains COVID-Fact and HealthVer text. Do not commit, upload or share.",
        "policy": rows["policy"],
        "structural_rule": ST.frozen_rule_sha256(),
        "manifest_sha256": C.load_manifest()["sha256"],
        "covidfact": cf,
        "noreg": noreg,
    }
    out = C.BUNDLE
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, sort_keys=True)
    h = C.file_sha(out)
    open(out + ".sha256", "w").write(f"{h}  {os.path.basename(out)}\n")
    c = collections.Counter((x["role"], x["panel"] is not None) for x in cf)
    print("covidfact", len(cf), dict(c))
    print("noreg", dict(collections.Counter(x["ds"] for x in noreg)))
    print(out, os.path.getsize(out), "bytes, sha256", h)


if __name__ == "__main__":
    main()
