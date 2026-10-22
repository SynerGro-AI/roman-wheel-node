"""Phase 4: build the FRESH labelled sets (never used to select or tune any Phase 3 rule).

* hv_test_unjudged: the HealthVer test pairs that Phase 3d did not finish judging (no Phase 3 number used them).
* covidfact: COVID-Fact (Saakyan et al. 2021), all 4,086 claim-evidence pairs. Never used in Phase 3.
* hv_dev: HealthVer dev, 1,917 pairs. Never used in Phase 3 (local-only run; judged only if budget remains).
HealthVer and COVID-Fact have no licence: INTERNAL ONLY. The items file is written under phase3/cache/ (gitignored);
only aggregate numbers may be committed.
Output: phase3/cache/phase4/items.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
P3 = os.path.join(os.path.dirname(HERE), "phase3")
sys.path.insert(0, P3)
from features import CACHE            # noqa: E402
from s2_data import SPLIT, doc_text   # noqa: E402

OUT = os.path.join(CACHE, "phase4")
COVIDFACT = os.environ.get("RWN_COVIDFACT", "/workspace/p3c_datasets/covidfact.jsonl")
RAW = os.path.join(CACHE, "step2", "raw")
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
INTERNAL_NOTE = "internal, unlicensed dataset, data not redistributed"


def main() -> None:
    sys.path.insert(0, P3)
    from factcheck.interfaces import SpendLedger
    from judges import make_judges
    J = make_judges(SpendLedger(os.path.join(CACHE, "spend.jsonl"), 1e-9))     # cache reads only
    panel = ("J1", "J2", "J4", "J7", "J9", "J10")
    s2 = json.load(open(os.path.join(CACHE, "step2", "items.json")))
    items = []
    for it in s2:
        if it["ds"] == "healthver_test" and not all(J[n].cached(it["claim"], it["evidence"]) for n in panel):
            items.append(dict(it, ds="hv_test_unjudged", role="fresh"))
    lab = {"SUPPORTED": "SUPPORT", "REFUTED": "CONTRADICT"}
    for n, line in enumerate(open(COVIDFACT)):
        r = json.loads(line)
        sents = [s.strip() for s in r["evidence"] if s.strip()]
        items.append({"id": f"cvf:{n}:{H(r['claim'])[:10]}", "ds": "covidfact", "role": "fresh", "claim": r["claim"].strip(),
                      "label3": lab[r["label"]], "evidence": doc_text("E1", "Evidence", sents), "units": sents, "sents": sents})
    hl = {"Supports": "SUPPORT", "Refutes": "CONTRADICT", "Neutral": "NEI"}
    for r in csv.DictReader(open(os.path.join(RAW, "hv_dev.csv"))):
        ev = r["evidence"].strip()
        sents = [s.strip() for s in SPLIT.split(ev) if s.strip()]
        items.append({"id": f"hv:dev:{r['id']}", "ds": "hv_dev", "role": "fresh", "claim": r["claim"].strip(),
                      "label3": hl[r["label"]], "evidence": doc_text("E1", "Evidence statement", sents), "units": sents, "sents": sents})
    seen, out = set(), []
    for it in items:
        if it["id"] not in seen and it["sents"]:
            seen.add(it["id"]); out.append(it)
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "items.json"), "w"))
    from collections import Counter
    print(Counter((i["ds"], i["label3"]) for i in out))


if __name__ == "__main__":
    main()
