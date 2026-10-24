"""Stage 1: per-claim evidence and local (no-API) source features for SciFact train + dev, cached to JSON.

Conditions:
* ``retrieved``: BM25 top-3 abstracts (the realistic open setting).
* ``gold``: the claim's cited abstracts (SciFact's "oracle abstract" setting). NEI claims' cited docs are
  non-evidential.
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factcheck.interfaces import BM25Retriever, NLIScorer, OverlapScorer, Passage  # noqa: E402
from factcheck.scifact import load_claims, load_corpus  # noqa: E402

DATA = os.path.join(HERE, "data", "data")
CACHE = os.path.join(HERE, "cache")
K = 3


def evidence_text(corpus, doc_ids) -> str:
    """The judges' view: numbered sentences grouped by abstract (sentence text is the exact corpus string)."""
    parts = []
    for d in doc_ids:
        doc = corpus[d]
        parts.append(f"[doc {d}] {doc.title}\n" + "\n".join(f"  [{d}.{i}] {s}" for i, s in enumerate(doc.sentences)))
    return "\n\n".join(parts)


def main() -> None:
    os.makedirs(CACHE, exist_ok=True)
    out_path = os.path.join(CACHE, "local_features.json")
    corpus = load_corpus(DATA)
    bm25 = BM25Retriever({d: doc.title + " " + " ".join(doc.sentences) for d, doc in corpus.items()})
    nli = NLIScorer(os.path.join(HERE, "models", "nli-deberta-v3-small"))
    ov = OverlapScorer()
    feats = {}
    t0 = time.time()
    for split in ("train", "dev"):
        for n, c in enumerate(load_claims(DATA, split)):
            row = {
                "split": split,
                "claim": c.claim,
                "label3": c.label3,
                "cited": list(c.cited_doc_ids),
                "rationale": [list(x) for x in c.rationale],
            }
            row["retrieved"] = bm25.retrieve(c.claim, K)
            for cond, docs in (("retrieved", row["retrieved"]), ("gold", list(c.cited_doc_ids))):
                sents = [(d, i, s) for d in docs for i, s in enumerate(corpus[d].sentences)]
                ps = nli.probs(c.claim, [s for _, _, s in sents]) if sents else []
                ie = max(range(len(ps)), key=lambda j: ps[j]["entailment"]) if ps else None
                ic = max(range(len(ps)), key=lambda j: ps[j]["contradiction"]) if ps else None
                row[cond] = {
                    "docs": docs,
                    "nli_entail": ps[ie]["entailment"] if ps else 0.0,
                    "nli_contra": ps[ic]["contradiction"] if ps else 0.0,
                    "nli_entail_at": [sents[ie][0], sents[ie][1]] if ps else None,
                    "nli_contra_at": [sents[ic][0], sents[ic][1]] if ps else None,
                    "overlap": ov.score(c.claim, [Passage(d, s) for d, _, s in sents]),
                    "evidence_chars": len(evidence_text(corpus, docs)),
                }
            feats[f"{split}:{c.claim_id}"] = row
            if n % 100 == 0:
                print(split, n, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(feats, open(out_path, "w"), sort_keys=True)
    print("wrote", out_path, len(feats))


if __name__ == "__main__":
    main()
