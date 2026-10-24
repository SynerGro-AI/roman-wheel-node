"""Phase 4: run the frozen policy in SHADOW mode end to end on the fresh sets, pre-register, import gold labels, certify.

    env -u XAI_API_KEY python phase4/run_fresh.py --profile local        # free: live ONNX NLI + DeBERTa-base + overlap
    env -u XAI_API_KEY python phase4/run_fresh.py --profile local-lite   # free, ONNX only (diagnostic)
    env -u XAI_API_KEY python phase4/run_fresh.py --profile panel        # judge panel from the CACHE only (the paid calls
                                                                         # were made by fresh_judges.py; any miss -> NOT_SURE)
Each profile gets its own log dir under phase3/cache/phase4/ (gitignored: claims, evidence, labels stay local) and its
own prereg.json written BEFORE any label is imported. The gate's trusted store is each dataset's own evidence text, so
a = 1 iff the evidence shown is an exact copy of it (and, for the panel, J1's quote is verbatim, as in Phase 3d).
Aggregates only -> results/phase4/fresh_<profile>.json / .txt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

if os.environ.get("XAI_API_KEY"):
    sys.exit("refusing to run with XAI_API_KEY set: this run makes no API call (use env -u XAI_API_KEY)")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "phase3"))
from roman_wheel_node import fresh as FR                          # noqa: E402
from roman_wheel_node.shadow import ShadowRunner                   # noqa: E402
from roman_wheel_node.sources import LocalSources, sha256          # noqa: E402
from features import CACHE                                         # noqa: E402
HISTORICAL = ("superseded: rwn-policy-4.0.0/4.1.0 were fitted partly on HealthVer and no longer ship (policy 4.2.0, phase6/refit_policy_42.py). Reproduce this historical run from git commit 3857154.")

P4 = os.path.join(CACHE, "phase4")
NOTE = "HealthVer and COVID-Fact: internal, unlicensed datasets; data and per-item outputs are not redistributed"


def cached_panel():
    from factcheck.interfaces import SpendLedger
    from judges import make_judges
    J = {n: j for n, j in make_judges(SpendLedger(os.path.join(CACHE, "spend.jsonl"), 1e-9)).items()
         if n in ("J1", "J2", "J4", "J7", "J9", "J10")}

    def panel(claim, ev):
        raw, quotes = {}, {}
        for n, j in J.items():
            if not j.cached(claim, ev):
                raise LookupError(f"{n} not judged")           # never calls the API: claim fails closed
            p, q = j.judge(claim, ev)
            raw[n] = {"S": p["SUPPORT"], "C": p["CONTRADICT"], "N": p["NEI"]}; quotes[n] = q or ""
        return raw, quotes
    panel.complete = lambda claim, ev: all(j.cached(claim, ev) for j in J.values())
    return panel


def main() -> None:
    from roman_wheel_node.policy import DEFAULT_POLICY_SHA256
    if DEFAULT_POLICY_SHA256 != "254804f8ac9b500ef1951879fb64328b0f54303b8bef20e38b1684545f876eb1":
        sys.exit(HISTORICAL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=("local", "local-lite", "panel"))
    ap.add_argument("--ds", default="hv_test_unjudged,covidfact,hv_dev")
    a = ap.parse_args()
    items = [i for i in json.load(open(os.path.join(P4, "items.json"))) if i["ds"] in a.ds.split(",")]
    log = os.path.join(P4, f"shadow_{a.profile}")
    if os.path.exists(os.path.join(log, "fresh_claims.jsonl")):
        sys.exit(f"{log} already has a run; move it away to rerun")
    m = os.path.join(ROOT, "phase3", "models")
    judges = None
    if a.profile == "panel":
        local = json.load(open(os.path.join(P4, "local.json")))
        src = lambda claim, sents: _cur["raw"]                       # cached local features of the current item
        panel = cached_panel()
        judges = lambda claim, ev: panel(claim, _cur["evidence"])    # the exact Phase 3d evidence text (cache key)
        items = [i for i in items if i["id"] in local and panel.complete(i["claim"], i["evidence"])]   # fully judged only
        print("fully judged fresh items:", len(items), flush=True)
    else:
        src = LocalSources(nli_dir=os.path.join(m, "nli-deberta-v3-small") if a.profile == "local-lite" else None,
                           nli2="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli" if a.profile == "local" else None,
                           hf_cache=os.path.join(m, "hf"))
        print("available sources:", src.available, src.errors, flush=True)
    _cur = {}
    runner = ShadowRunner(log, profile=a.profile, sources=src, judges=judges,
                          trusted_hashes={sha256(u) for i in items for u in i["units"]})
    t0 = time.time()
    for n, it in enumerate(items):
        if a.profile == "panel":
            lf = local[it["id"]]
            _cur["raw"] = {"nli": {"S": lf["nli_entail"], "C": lf["nli_contra"]}, "nli2": {"S": lf["nli2_entail"], "C": lf["nli2_contra"]},
                           "overlap": {"S": lf["overlap"]}}
            _cur["evidence"] = it["evidence"]
        runner.observe(it["claim"], it["sents"], meta={"id": it["id"], "ds": it["ds"]})
        if n % 500 == 0:
            print(n, f"{time.time() - t0:.0f}s", runner.counts, "errors", runner.errors, flush=True)
    runner.verify()
    pol = runner.node.policy.sha256
    FR.prereg(log, pol, a.profile, note=f"Phase 4 fresh sets {a.ds}; decisions were logged before this file and before any label")
    with open(os.path.join(log, "labels.jsonl"), "a") as f:           # dataset gold labels (the user cannot label by hand)
        for it in items:
            f.write(json.dumps({"id": it["id"], "label": "SUPPORT" if it["label3"] == "SUPPORT" else "NOT", "ts": time.time(),
                                "labeller": "dataset-gold"}) + "\n")
    rows = FR.load_jsonl(os.path.join(log, "fresh_claims.jsonl"))
    lab = {json.loads(l)["id"]: json.loads(l)["label"] for l in open(os.path.join(log, "labels.jsonl"))}
    with open(os.path.join(log, "labels.jsonl"), "w") as f:            # rewrite keyed by the logged claim id
        for r in rows:
            f.write(json.dumps({"id": r["id"], "label": lab[r["meta"]["id"]], "ts": time.time(), "labeller": "dataset-gold"}) + "\n")
    res = FR.certify(log, by="ds")
    res["note"] = NOTE
    res["counts"] = runner.counts; res["errors"] = runner.errors
    res["available_sources"] = list(getattr(src, "available", ())) if a.profile != "panel" else ["J1", "J2", "J4", "J7", "J9", "J10", "nli", "nli2", "overlap"]
    out = os.path.join(ROOT, "results", "phase4")
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, f"fresh_{a.profile}.json"), "w"), indent=1)
    txt = FR.report(res) + f"\n{NOTE}\nshadow counts {runner.counts}, node errors {runner.errors}, elapsed {time.time() - t0:.0f}s\n"
    open(os.path.join(out, f"fresh_{a.profile}.txt"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
