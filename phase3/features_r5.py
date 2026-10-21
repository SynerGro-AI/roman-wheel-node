"""Stage 1b: the "retrieved5" condition: BM25 top-20, reranked by a cross-encoder (ms-marco-MiniLM-L-6-v2), top-5 kept.

On 250 training claims, recall of a gold abstract rose from 0.868 (BM25@3) to 0.924 (reranked@5).
Local sources are recomputed on the reranked evidence. Output: cache/local_features_r5.json (merged by evaluate.py).
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factcheck.interfaces import BM25Retriever, NLIScorer, OverlapScorer, Passage   # noqa: E402
from factcheck.scifact import load_corpus                                            # noqa: E402
from features import CACHE, DATA, evidence_text                                      # noqa: E402
from judges import selected                                                          # noqa: E402

RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI2 = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(8)
    hf = os.path.join(HERE, "models", "hf")
    rt = AutoTokenizer.from_pretrained(RERANKER, cache_dir=hf)
    rm = AutoModelForSequenceClassification.from_pretrained(RERANKER, cache_dir=hf).eval()
    nt = AutoTokenizer.from_pretrained(NLI2, cache_dir=hf)
    nm = AutoModelForSequenceClassification.from_pretrained(NLI2, cache_dir=hf).eval()
    lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
    corpus = load_corpus(DATA)
    bm25 = BM25Retriever({d: doc.title + " " + " ".join(doc.sentences) for d, doc in corpus.items()})
    nli = NLIScorer(os.path.join(HERE, "models", "nli-deberta-v3-small"))
    ov = OverlapScorer()
    feats = json.load(open(os.path.join(CACHE, "local_features.json")))
    out_path = os.path.join(CACHE, "local_features_r5.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    t0 = time.time()
    for n, k in enumerate(selected(feats)):
        if k in out:
            continue
        claim = feats[k]["claim"]
        cand = bm25.retrieve(claim, 20)
        enc = rt([claim] * len(cand), [corpus[d].title + " " + " ".join(corpus[d].sentences) for d in cand],
                 truncation=True, max_length=384, padding=True, return_tensors="pt")
        with torch.inference_mode():
            s = rm(**enc).logits[:, 0].tolist()
        docs = [d for _, d in sorted(zip(s, cand), key=lambda x: (-x[0], x[1]))][:5]
        sents = [(d, i, t) for d in docs for i, t in enumerate(corpus[d].sentences)]
        ps = nli.probs(claim, [t for _, _, t in sents])
        e2 = c2 = 0.0
        for b in range(0, len(sents), 32):
            ch = sents[b:b + 32]
            x = nt([t for _, _, t in ch], [claim] * len(ch), truncation=True, max_length=256, padding=True, return_tensors="pt")
            with torch.inference_mode():
                p = torch.softmax(nm(**x).logits.float(), dim=-1)
            e2 = max(e2, float(p[:, lab["entailment"]].max())); c2 = max(c2, float(p[:, lab["contradiction"]].max()))
        out[k] = {"docs": docs, "rerank_scores": [round(v, 4) for v in sorted(s, reverse=True)[:5]],
                  "nli_entail": max(d["entailment"] for d in ps), "nli_contra": max(d["contradiction"] for d in ps),
                  "nli2_entail": round(e2, 6), "nli2_contra": round(c2, 6),
                  "overlap": ov.score(claim, [Passage(d, t) for d, _, t in sents]),
                  "evidence_chars": len(evidence_text(corpus, docs))}
        if n % 10 == 0:
            print(n, f"{time.time() - t0:.0f}s", flush=True)
            json.dump(out, open(out_path, "w"), sort_keys=True)
    json.dump(out, open(out_path, "w"), sort_keys=True)
    print("wrote", out_path, len(out))


if __name__ == "__main__":
    main()
