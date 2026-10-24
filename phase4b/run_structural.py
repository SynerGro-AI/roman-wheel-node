"""Phase 4b: deterministic structural veto on ACCEPTs (src/roman_wheel_node/structure.py), held-out protocol.

    env -u XAI_API_KEY python phase4b/run_structural.py collect   # cache-only: frozen-panel decisions + quotes
    env -u XAI_API_KEY python phase4b/run_structural.py design    # half A of COVID-Fact only: per-field table
    env -u XAI_API_KEY python phase4b/run_structural.py prereg    # freeze: rule hash + timestamp, BEFORE half B
    env -u XAI_API_KEY python phase4b/run_structural.py score     # half B once (refuses a second run) +
    SciFact/HealthVer

COVID-Fact is split 50/50 by evidence group: sha256(SEED:group) parity, where the group is the hash of the evidence
text. All counter-claims made from one source share that evidence, so they all land in the same half and there is
no leakage between the halves. Decisions come from the frozen Phase 4 panel policy (254804f8ac9b). The Phase 4
fresh sets are read from the shadow_panel log. SciFact dev, HealthVer test and Climate-FEVER are replayed from the
judge cache through the same ShadowRunner. A cache miss fails closed, and this script never makes an API call.
Per-item data stays in phase3/cache/phase4b/ (gitignored). Only aggregates are written to results/phase4b/.
Set RWN_P3 to another checkout's phase3/ (same code, with its data/ and cache/) when running from a worktree.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import json
import os
import sys
import time

if os.environ.get("XAI_API_KEY"):
    sys.exit("refusing to run with XAI_API_KEY set: this run makes no API call (use env -u XAI_API_KEY)")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.environ.get("RWN_P3", os.path.join(ROOT, "phase3")))
from features import CACHE  # noqa: E402

from roman_wheel_node import fresh as FR  # noqa: E402
from roman_wheel_node import structure as ST  # noqa: E402

HISTORICAL = (
    "superseded: rwn-policy-4.0.0/4.1.0 were fitted partly on HealthVer and no longer ship (policy 4.2.0, "
    "phase6/refit_policy_42.py). Reproduce this historical run from git commit 3857154."
)

SEED = "rwn-phase4b-structural-2026-10-22"
P4B = os.path.join(CACHE, "phase4b")
OUT = os.path.join(ROOT, "results", "phase4b")
PANEL = ("J1", "J2", "J4", "J7", "J9", "J10")
NOTE = "HealthVer and COVID-Fact: internal, unlicensed datasets; data and per-item outputs are not redistributed"
# pre-declared bar for a field to veto (applied on half A only)
BAR = {"min_false_accepts_caught": 3, "min_caught_per_correct_lost": 3}


def half(group: str) -> str:
    return "A" if int(hashlib.sha256(f"{SEED}:{group}".encode()).hexdigest(), 16) % 2 == 0 else "B"


def _lst(x):
    return x if isinstance(x, list) else ast.literal_eval(x)


def collect() -> None:
    from roman_wheel_node.policy import DEFAULT_POLICY_SHA256

    if DEFAULT_POLICY_SHA256 != "254804f8ac9b500ef1951879fb64328b0f54303b8bef20e38b1684545f876eb1":
        sys.exit(HISTORICAL)
    from factcheck.interfaces import SpendLedger
    from judges import make_judges

    from roman_wheel_node.shadow import ShadowRunner
    from roman_wheel_node.sources import sha256

    J = {n: j for n, j in make_judges(SpendLedger(os.path.join(CACHE, "spend.jsonl"), 1e-9)).items() if n in PANEL}

    def quotes(claim, ev):
        return [J[n].judge(claim, ev)[1] or "" for n in PANEL if J[n].cached(claim, ev)]

    def row(it, outcome):
        return {
            "id": it["id"],
            "ds": it["ds"],
            "y": it["label3"] == "SUPPORT",
            "outcome": outcome,
            "claim": it["claim"],
            "units": _lst(it["units"]),
            "group": hashlib.sha256(it["evidence"].encode()).hexdigest()[:16],
            "quotes": quotes(it["claim"], it["evidence"]) if outcome == "ACCEPT" else [],
        }

    os.makedirs(P4B, exist_ok=True)
    p4 = {i["id"]: i for i in json.load(open(os.path.join(CACHE, "phase4", "items.json"), encoding="utf-8"))}
    rows = []
    for line in open(os.path.join(CACHE, "phase4", "shadow_panel", "fresh_claims.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        rows.append(row(p4[r["meta"]["id"]], r["outcome"]))
    items = [
        i
        for i in json.load(open(os.path.join(CACHE, "step2", "items.json"), encoding="utf-8"))
        if i["ds"] in ("scifact_dev", "healthver_test", "climatefever")
    ]
    local = json.load(open(os.path.join(CACHE, "step2", "local.json"), encoding="utf-8"))
    items = [i for i in items if i["id"] in local and all(J[n].cached(i["claim"], i["evidence"]) for n in PANEL)]
    cur = {}

    def judges(claim, ev):
        raw, qs = {}, {}
        for n in PANEL:
            if not J[n].cached(claim, cur["evidence"]):
                raise LookupError(f"{n} not judged")
            p, q = J[n].judge(claim, cur["evidence"])
            raw[n] = {"S": p["SUPPORT"], "C": p["CONTRADICT"], "N": p["NEI"]}
            qs[n] = q or ""
        return raw, qs

    log = os.path.join(P4B, f"replay_{int(time.time())}")
    runner = ShadowRunner(
        log,
        profile="panel",
        sources=lambda c, s: cur["raw"],
        judges=judges,
        trusted_hashes={sha256(u) for i in items for u in _lst(i["units"])},
    )
    for it in items:
        lf = local[it["id"]]
        cur["raw"] = {
            "nli": {"S": lf["nli_entail"], "C": lf["nli_contra"]},
            "nli2": {"S": lf["nli2_entail"], "C": lf["nli2_contra"]},
            "overlap": {"S": lf["overlap"]},
        }
        cur["evidence"] = it["evidence"]
        rows.append(
            row(it, runner.observe(it["claim"], _lst(it["sents"]), meta={"id": it["id"], "ds": it["ds"]}).outcome)
        )
    runner.verify()
    json.dump(
        {"policy": runner.node.policy.sha256, "rows": rows}, open(os.path.join(P4B, "rows.json"), "w", encoding="utf-8")
    )
    print("collected", len(rows), "policy", runner.node.policy.sha256[:12], runner.counts, "errors", runner.errors)


def _rows():
    return json.load(open(os.path.join(P4B, "rows.json"), encoding="utf-8"))


def _cvf(d, h):
    return [r for r in d["rows"] if r["ds"] == "covidfact" and half(r["group"]) == h]


def _apply(rows, rule=None):
    """(outcome_before, outcome_after, y, reasons, pattern) per row; the veto only sees ACCEPTs."""
    out = []
    for r in rows:
        o, c = ST.veto_accept(r["outcome"], r["claim"], r["units"], r.get("quotes"), rule)
        out.append((r["outcome"], o, r["y"], c.reasons if c else (), c))
    return out


def _field_table(rows):
    acc = [r for r in rows if r["outcome"] == "ACCEPT"]
    tests = {
        "identifier_mutation": lambda c: c.identifier_mutation,
        "fragment": lambda c: c.fragment,
        "near_copy_swap": lambda c: c.near_copy_swap,
        "numbers_missing": lambda c: c.numbers_missing,
        "negation_inserted": lambda c: c.negation[0] > 0 and c.negation[1] == 0,
        "negation_dropped": lambda c: c.negation[0] == 0 and c.negation[1] > 0,
        "direction_conflict": lambda c: c.direction_conflict,
        "absolutes_unbacked": lambda c: c.absolutes_unbacked,
    }
    chk = [(r["y"], ST.check(r["claim"], r["units"], r.get("quotes"))) for r in acc]
    tab = {}
    for k, fn in tests.items():
        fa = sum(1 for y, c in chk if fn(c) and not y)
        ta = sum(1 for y, c in chk if fn(c) and y)
        ok = fa >= BAR["min_false_accepts_caught"] and fa >= BAR["min_caught_per_correct_lost"] * ta
        tab[k] = {"false_accepts_caught": fa, "correct_accepts_lost": ta, "meets_bar": ok}
    return tab


def _before_after(res):
    before = FR.stats([(b, y) for b, _, y, _, _ in res])
    after = FR.stats([(a, y) for _, a, y, _, _ in res])
    caught = sum(1 for b, a, y, _, _ in res if b == "ACCEPT" and a != "ACCEPT" and not y)
    lost = sum(1 for b, a, y, _, _ in res if b == "ACCEPT" and a != "ACCEPT" and y)
    why = collections.Counter("+".join(rs) for b, a, y, rs, _ in res if rs)
    why_fa = collections.Counter("+".join(rs) for b, a, y, rs, _ in res if rs and not y)
    return {
        "before": before,
        "after": after,
        "vetoed": caught + lost,
        "false_accepts_caught": caught,
        "correct_accepts_lost": lost,
        "coverage_lost_points": before["coverage"] - after["coverage"],
        "veto_reasons": dict(why),
        "veto_reasons_on_false_accepts": dict(why_fa),
    }


def design() -> None:
    d = _rows()
    A = _cvf(d, "A")
    tab = _field_table(A)
    res = _before_after(_apply(A))
    os.makedirs(OUT, exist_ok=True)
    rep = {
        "half": "A",
        "seed": SEED,
        "groups": len({r["group"] for r in A}),
        "bar": BAR,
        "fields": tab,
        "frozen_veto_fields": list(ST.FROZEN["veto_fields"]),
        "frozen_rule": ST.FROZEN,
        "result_A": res,
        "note": NOTE,
    }
    json.dump(rep, open(os.path.join(OUT, "design_A.json"), "w", encoding="utf-8"), indent=1, default=list)
    print(json.dumps(tab, indent=1))
    print("A:", {k: res[k] for k in ("false_accepts_caught", "correct_accepts_lost")})


def prereg() -> None:
    p = os.path.join(OUT, "prereg.json")
    if os.path.exists(p):
        sys.exit(f"{p} exists: the rule is already frozen")
    d = _rows()
    B = _cvf(d, "B")
    rec = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rule": ST.FROZEN,
        "rule_sha256": ST.frozen_rule_sha256(),
        "panel_policy": d["policy"],
        "split_seed": SEED,
        "split": "sha256(seed:evidence-group) parity; A even, B odd",
        "half_B_items": len(B),
        "half_B_groups": len({r["group"] for r in B}),
        "alphas": [0.01, 0.02, 0.05],
        "delta": 0.05,
        "risk": "false accepts among ACCEPTs (one-sided Clopper-Pearson)",
        "primary": "half B of COVID-Fact, scored once; secondary: SciFact dev, HealthVer test, Climate-FEVER, "
        "hv_test_unjudged",
        "note": "rule designed on half A only; half B not scored before this file",
    }
    json.dump(rec, open(p, "w", encoding="utf-8"), indent=1, default=list)
    print(json.dumps(rec, indent=1, default=list))


def score() -> None:
    pre = json.load(open(os.path.join(OUT, "prereg.json"), encoding="utf-8"))
    if pre["rule_sha256"] != ST.frozen_rule_sha256():
        sys.exit("structure.py changed since pre-registration: refusing to score")
    if os.path.exists(os.path.join(OUT, "heldout_B.json")):
        sys.exit("half B was already scored once")
    d = _rows()
    sets = {"covidfact_B": _cvf(d, "B"), "covidfact_A_design": _cvf(d, "A")}
    for ds in ("scifact_dev", "healthver_test", "climatefever", "hv_test_unjudged"):
        sets[ds] = [r for r in d["rows"] if r["ds"] == ds]
    res = {"prereg": pre, "scored": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": NOTE, "sets": {}}
    for k, rows in sets.items():
        res["sets"][k] = _before_after(_apply(rows))
    # anonymised patterns (counts per field only, no text): what the caught/missed false accepts on half B look like
    B = _apply(sets["covidfact_B"])
    missed = [c.pattern() for b, a, y, rs, c in B if b == "ACCEPT" and a == "ACCEPT" and not y]
    res["sets"]["covidfact_B"]["missed_false_accept_patterns"] = dict(
        collections.Counter(
            "+".join(k for k, v in p.items() if v and not k.startswith("negation")) or "no-structural-signal"
            for p in missed
        )
    )
    json.dump(res, open(os.path.join(OUT, "heldout_B.json"), "w", encoding="utf-8"), indent=1, default=list)
    lines = [
        f"Phase 4b structural veto, rule {pre['rule_sha256'][:12]} pre-registered {pre['created']}, scored "
        f"{res['scored']}",
        f"panel policy {pre['panel_policy'][:12]}; split seed {SEED}; veto fields "
        f"{', '.join(pre['rule']['veto_fields'])}",
        "",
    ]
    for k, s in res["sets"].items():
        b, a = s["before"], s["after"]

        def f(x):
            return "n/a" if x is None else f"{100 * x:.1f}%"

        lines.append(
            f"{k}: labelled {b['labelled']}; ACCEPT {b['accepted']} -> {a['accepted']}; false accepts "
            f"{b['false_accepts']} -> {a['false_accepts']}; precision {f(b['precision'])} -> {f(a['precision'])}; "
            f"FA-share UB {f(b['false_accept_share_upper'])} -> {f(a['false_accept_share_upper'])}; coverage "
            f"{f(b['coverage'])} -> {f(a['coverage'])}; correct accepts lost {s['correct_accepts_lost']}"
        )
        lines.append(
            "   certified after veto: "
            + ", ".join(
                f"{al}: {v['certified']} (needs {v['accepts_needed_zero_errors']} error-free)"
                for al, v in a["alphas"].items()
            )
        )
    lines += ["", NOTE]
    open(os.path.join(OUT, "heldout_B.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"collect": collect, "design": design, "prereg": prereg, "score": score}[sys.argv[1]]()
