"""Few-shot examples for judge J9, drawn ONLY from SciFact train claims that are NOT in the 350-claim learning subset
(and never from dev). Deterministic: sha256 order, stratified 4 SUPPORT / 3 CONTRADICT / 3 NEI."""

from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factcheck.scifact import load_claims, load_corpus  # noqa: E402
from features import CACHE, DATA  # noqa: E402

QUOTA = {"SUPPORT": 4, "CONTRADICT": 3, "NEI": 3}


def build() -> str:
    from judges import selected

    feats = json.load(open(os.path.join(CACHE, "local_features.json")))
    used = set(selected(feats))
    corpus = load_corpus(DATA)
    pool = [c for c in load_claims(DATA, "train") if f"train:{c.claim_id}" not in used]
    pool.sort(key=lambda c: hashlib.sha256(c.claim_id.encode()).hexdigest())
    picked, got = [], {k: 0 for k in QUOTA}
    for c in pool:
        if got[c.label3] < QUOTA[c.label3] and c.cited_doc_ids:
            picked.append(c)
            got[c.label3] += 1
    order = sorted(picked, key=lambda c: hashlib.sha256(("o" + c.claim_id).encode()).hexdigest())
    out = []
    for c in order:
        d = c.cited_doc_ids[0]
        idx = sorted({i for dd, i in c.rationale if dd == d}) or [0, 1, 2]
        sents = " ".join(corpus[d].sentences[i] for i in idx[:3] if i < len(corpus[d].sentences))
        lab = {"SUPPORT": "SUPPORTS", "CONTRADICT": "REFUTES", "NEI": "NOINFO"}[c.label3]
        out.append(f"Claim: {c.claim}\nAbstract excerpt ({corpus[d].title}): {sents}\nLabel: {lab}")
    return "\n\n".join(out), [c.claim_id for c in order]


if __name__ == "__main__":
    t, ids = build()
    print(t)
    print(ids)
