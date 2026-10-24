"""Phase 5 / Task B: what the fp2 (NFC-normalized) exact-copy gate changes on the cached Phase 4 / 4b decisions.

Cache-only and zero API calls: judge quotes are read from the Phase 3 judge cache, and the run refuses if
XAI_API_KEY is set. Every logged panel decision (the Phase 4 fresh set plus the Phase 4b replay of SciFact dev,
HealthVer test and Climate-FEVER) is re-gated under fp1 (it must reproduce the logged ``a``) and under fp2. Each claim
the fp2 gate un-blocks is re-decided by the frozen panel profile of policy 4.1.0, which is byte-identical to 4.0.0
apart from the gate, using its logged raw scores. The Phase 4b structural veto is then applied as well.
Output: aggregates only (results/phase5/fp2_gate.json/.txt). Run from a worktree with RWN_P3 pointing at a phase3/
that has data/ and cache/ if needed.

    env -u XAI_API_KEY python3 phase5/measure_fp2.py
"""
from __future__ import annotations

import ast
import collections
import json
import os
import sys
import tempfile

if os.environ.get("XAI_API_KEY"):
    sys.exit("refusing to run with XAI_API_KEY set: this run makes no API call (use env -u XAI_API_KEY)")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.environ.get("RWN_P3", os.path.join(ROOT, "phase3")))
from features import CACHE                                                       # noqa: E402
from roman_wheel_node import structure as ST                                     # noqa: E402
from roman_wheel_node.fingerprint import exact_gate_v2, quote_verbatim_v2, trusted_set  # noqa: E402
from roman_wheel_node.policy import POLICY_4_1_PATH, load_policy                 # noqa: E402
from roman_wheel_node.shadow import FrozenNode, quote_verbatim    # noqa: E402
from roman_wheel_node.sources import exact_gate, sha256                          # noqa: E402

PANEL = ("J1", "J2", "J4", "J7", "J9", "J10")
OUT = os.path.join(ROOT, "results", "phase5")
NOTE = "HealthVer and COVID-Fact: internal, unlicensed datasets; aggregates only, no text or per-item output"
LOGS = {"phase4_fresh": os.path.join(CACHE, "phase4", "shadow_panel"),
        "phase4b_replay": os.path.join(CACHE, "phase4b", "replay_1792687785")}


def main() -> None:
    from factcheck.interfaces import SpendLedger
    from judges import make_judges
    J = {n: j for n, j in make_judges(SpendLedger(os.path.join(CACHE, "spend.jsonl"), 1e-9)).items() if n in PANEL}
    pol41 = load_policy(POLICY_4_1_PATH)
    assert pol41.fingerprint == "fp2"
    node = FrozenNode(os.path.join(tempfile.mkdtemp(prefix="rwn_fp2_"), "audit.jsonl"), pol41, "panel")
    gold = {}
    for line in open(os.path.join(LOGS["phase4_fresh"], "labels.jsonl")):
        r = json.loads(line); gold[("phase4_fresh", r["id"])] = r["label"] == "SUPPORT"
    lst = lambda x: x if isinstance(x, list) else ast.literal_eval(x)
    items = {i["id"]: i for i in json.load(open(os.path.join(CACHE, "step2", "items.json")))}
    items.update({i["id"]: i for i in json.load(open(os.path.join(CACHE, "phase4", "items.json")))})
    s2 = {k: i["label3"] == "SUPPORT" for k, i in items.items()}
    tot = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    for log, d in LOGS.items():
        recs = [json.loads(line) for line in open(os.path.join(d, "fresh_claims.jsonl"))]
        # as in phase4/run_fresh.py and phase4b collect: the trusted store is every evidence unit of the run's items,
        # and the judge cache key is the item's Phase 3d evidence text
        units = [u for r in recs for u in lst(items[r["meta"]["id"]]["units"])]
        hs1 = {sha256(u) for u in units}
        hs2 = trusted_set(units, "fp2")
        for r in recs:
            it = items[r["meta"]["id"]]
            ds = r["meta"].get("ds", "?")
            y = gold.get((log, r["id"])) if log == "phase4_fresh" else s2.get(r["meta"].get("id"))
            ev = r["evidence"]
            a1, a2 = exact_gate(ev, hs1), exact_gate_v2(ev, hs2)
            jev = it["evidence"]
            p, q = J["J1"].judge(r["claim"], jev) if J["J1"].cached(r["claim"], jev) else (None, None)
            if p is not None:
                commits = max(("SUPPORT", "CONTRADICT", "NEI"), key=lambda k: p[k]) != "NEI"
                if commits:
                    a1 = a1 if quote_verbatim(q or "", ev, hs1) else 0
                    a2 = a2 if quote_verbatim_v2(q or "", ev, hs2) else 0
            c = per[ds]
            c["records"] += 1
            c["fp1_reproduces_log"] += int(a1 == r["a"])
            c["fp1_blocked"] += int(a1 == 0)
            c["fp2_blocked"] += int(a2 == 0)
            if a1 == 1 and a2 == 0:
                c["fp2_newly_blocks"] += 1
            if a1 == 0 and a2 == 1:
                v = node.decide(r["claim"], r["raw"], 1)
                lab = "true" if y else "false"
                c[f"unblocked_{lab}"] += 1
                c[f"unblocked_{lab}_now_{v.outcome}"] += 1
                if v.outcome == "ACCEPT":
                    quotes = [J[n].judge(r["claim"], jev)[1] or "" for n in PANEL if J[n].cached(r["claim"], jev)]
                    o, _ = ST.veto_accept("ACCEPT", r["claim"], ev, quotes)
                    c[f"unblocked_{lab}_ACCEPT_after_4b_veto_{o}"] += 1
            if a1 == 0 and a2 == 0:
                c[f"still_blocked_{'true' if y else 'false'}"] += 1
    for c in per.values():
        tot.update(c)
    res = {"policy_4_1_0": pol41.sha256, "profile": "panel", "logs": {k: os.path.relpath(v, CACHE) for k, v in LOGS.items()},
           "by_dataset": {k: dict(v) for k, v in sorted(per.items())}, "total": dict(tot), "note": NOTE}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "fp2_gate.json"), "w"), indent=1, sort_keys=True)
    lines = [f"fp2 exact-copy gate on cached panel decisions (policy 4.1.0 {pol41.sha256[:12]}, profile panel)", ""]
    for k, v in list(sorted(per.items())) + [("TOTAL", tot)]:
        lines.append(f"{k}: records {v['records']}, fp1 blocked {v['fp1_blocked']} (log reproduced {v['fp1_reproduces_log']}/"
                     f"{v['records']}), fp2 blocked {v['fp2_blocked']}, newly blocked {v['fp2_newly_blocks']}; un-blocked "
                     f"true {v['unblocked_true']} (now ACCEPT {v['unblocked_true_now_ACCEPT']}), un-blocked false "
                     f"{v['unblocked_false']} (now ACCEPT {v['unblocked_false_now_ACCEPT']}, after 4b veto still ACCEPT "
                     f"{v['unblocked_false_ACCEPT_after_4b_veto_ACCEPT']})")
    lines += ["", NOTE]
    open(os.path.join(OUT, "fp2_gate.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
