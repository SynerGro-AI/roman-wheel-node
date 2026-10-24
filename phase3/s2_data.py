"""Phase 3d (step 2): build the calibration and test item sets. Output: cache/step2/items.json (gitignored).

Calibration (learning, node selection by CV, LTT certification):
  * SciFact train, the 350 claims used so far          (gold evidence; ds = scifact_train350)
  * SciFact train, all other claims except the 10 J9 few-shot examples (gold evidence; ds = scifact_train_unused)
  * HealthVer train, 2,000 claim-evidence pairs, stratified by label in hashed order   (ds = healthver_train)
Test (never used for any choice):
  * HealthVer test, all 1,823 pairs                     (ds = healthver_test)   INTERNAL ONLY: unlicensed, not
    redistributed
  * Climate-FEVER, claim-evidence pairs whose annotator votes are unanimous (>= 2 votes), stratified sample (ds =
    climatefever)
  * SciFact dev, 300 claims (already held out; gold evidence)  (ds = scifact_dev)

Raw files (downloaded to cache/step2/raw/): HealthVer CSVs (github.com/sarrouti/HealthVer, no licence: internal use
only) and Climate-FEVER jsonl (github.com/tdiggelm/climate-fever-dataset; CC BY-SA 4.0 via BEIR). Labels map to SUPPORT
/ CONTRADICT / NEI; the binary task is SUPPORT vs not.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factcheck.scifact import load_claims, load_corpus  # noqa: E402
from features import CACHE, DATA, evidence_text  # noqa: E402

S2 = os.path.join(CACHE, "step2")
RAW = os.path.join(S2, "raw")
N_HV_TRAIN = 2000
N_CF = int(os.environ.get("RWN_S2_NCF", "1500"))


def H(s):
    return hashlib.sha256(s.encode()).hexdigest()


SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def doc_text(doc_id: str, title: str, sentences) -> str:
    """Same layout as features.evidence_text (one document)."""
    return f"[doc {doc_id}] {title}\n" + "\n".join(f"  [{doc_id}.{i}] {s}" for i, s in enumerate(sentences))


def stratified(items, n, key):
    """Proportional stratified sample in hashed order (deterministic)."""
    by = {}
    for it in sorted(items, key=lambda x: H("s2" + x["id"])):
        by.setdefault(it[key], []).append(it)
    tot = len(items)
    out = []
    for lab, xs in sorted(by.items()):
        out += xs[: round(n * len(xs) / tot)]
    return out


def main() -> None:
    from fewshot import build as fewshot_build
    from judges import selected

    corpus = load_corpus(DATA)
    feats = json.load(open(os.path.join(CACHE, "local_features.json"), encoding="utf-8"))
    used = set(selected(feats))
    few = {f"train:{c}" for c in fewshot_build()[1]}
    items = []
    for split in ("train", "dev"):
        for c in load_claims(DATA, split):
            k = f"{split}:{c.claim_id}"
            docs = list(feats[k]["gold"]["docs"])
            if not docs or k in few:
                continue
            ds = "scifact_dev" if split == "dev" else ("scifact_train350" if k in used else "scifact_train_unused")
            sents = [s for d in docs for s in corpus[d].sentences]
            items.append(
                {
                    "id": f"sf:{k}",
                    "ds": ds,
                    "role": "test" if split == "dev" else "calib",
                    "claim": c.claim,
                    "label3": c.label3,
                    "evidence": evidence_text(corpus, docs),
                    "units": sents + [corpus[d].title for d in docs],
                    "sents": sents,
                }
            )
    lab = {"Supports": "SUPPORT", "Refutes": "CONTRADICT", "Neutral": "NEI"}
    for f, role in (("train", "calib"), ("test", "test")):
        rows = list(csv.DictReader(open(os.path.join(RAW, f"hv_{f}.csv"), encoding="utf-8")))
        its = []
        for r in rows:
            ev = r["evidence"].strip()
            sents = [s.strip() for s in SPLIT.split(ev) if s.strip()]
            its.append(
                {
                    "id": f"hv:{f}:{r['id']}",
                    "ds": f"healthver_{f}",
                    "role": role,
                    "claim": r["claim"].strip(),
                    "label3": lab[r["label"]],
                    "evidence": doc_text("E1", "Evidence statement", sents),
                    "units": sents,
                    "sents": sents,
                }
            )
        items += stratified(its, N_HV_TRAIN, "label3") if f == "train" else its
    cl = {"SUPPORTS": "SUPPORT", "REFUTES": "CONTRADICT", "NOT_ENOUGH_INFO": "NEI"}
    its = []
    for line in open(os.path.join(RAW, "cfever.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        for e in r["evidences"]:
            v = [x for x in e["votes"] if x]
            if len(v) >= 2 and len(set(v)) == 1:
                its.append(
                    {
                        "id": f"cf:{r['claim_id']}:{e['evidence_id']}",
                        "ds": "climatefever",
                        "role": "test",
                        "claim": r["claim"].strip(),
                        "label3": cl[e["evidence_label"]],
                        "evidence": doc_text("W1", e["article"], [e["evidence"].strip()]),
                        "units": [e["evidence"].strip(), e["article"]],
                        "sents": [e["evidence"].strip()],
                    }
                )
    items += stratified(its, N_CF, "label3")
    os.makedirs(S2, exist_ok=True)
    json.dump(items, open(os.path.join(S2, "items.json"), "w", encoding="utf-8"))
    from collections import Counter

    print(Counter((i["ds"], i["label3"]) for i in items))
    print({d: sum(1 for i in items if i["ds"] == d) for d in sorted({i["ds"] for i in items})})


if __name__ == "__main__":
    main()
