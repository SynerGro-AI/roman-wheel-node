"""Stage 1c: the "rv3" condition (retrieval v3), chosen on TRAIN recall/MRR only (retrieval_lab.py):
RRF(BM25, MedCPT dense, bge-small dense) top-20, reranked by ncbi/MedCPT-Cross-Encoder, top-5 abstracts kept.

Train (220 evidence claims), recall of any gold abstract @5: BM25 .914, MiniLM-rerank(BM25) .914, RRF(all3) .950,
MedCPT-CE(RRF) .982 (MRR@20 .814 -> .910). Dev runs the same frozen pipeline.
Local sources are recomputed on the new evidence. Output: cache/local_features_rv3.json (merged by evaluate.py).
Usage: python features_rv3.py [train|dev ...]
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factcheck.interfaces import NLIScorer, OverlapScorer, Passage   # noqa: E402
from factcheck.scifact import load_corpus                            # noqa: E402
from features import CACHE, DATA, evidence_text                      # noqa: E402
from judges import selected                                          # noqa: E402

PIPE = "ce_medcpt(rrf_all3)"
CE = "ncbi/MedCPT-Cross-Encoder"
NLI2 = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
K = 5


def main(splits) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(8)
    hf = os.path.join(HERE, "models", "hf")
    ct = AutoTokenizer.from_pretrained(CE, cache_dir=hf)
    cm = AutoModelForSequenceClassification.from_pretrained(CE, cache_dir=hf).eval()
    nt = AutoTokenizer.from_pretrained(NLI2, cache_dir=hf)
    nm = AutoModelForSequenceClassification.from_pretrained(NLI2, cache_dir=hf).eval()
    lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
    corpus = load_corpus(DATA)
    nli = NLIScorer(os.path.join(HERE, "models", "nli-deberta-v3-small"))
    ov = OverlapScorer()
    feats = json.load(open(os.path.join(CACHE, "local_features.json")))
    out_path = os.path.join(CACHE, "local_features_rv3.json")
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    t0 = time.time()
    for split in splits:
        keys = [k for k in selected(feats) if k.startswith(split)]
        ranks = json.load(open(os.path.join(CACHE, f"ranks_{split}.json")))[PIPE]
        assert len(ranks) == len(keys)
        for n, (k, rk) in enumerate(zip(keys, ranks)):
            if k in out:
                continue
            claim = feats[k]["claim"]
            docs = rk[:K]
            x = ct([claim], [corpus[docs[0]].title + " " + " ".join(corpus[docs[0]].sentences)],
                   truncation=True, max_length=512, return_tensors="pt")
            with torch.inference_mode():
                top = float(cm(**x).logits[0, 0])
            sents = [(d, i, t) for d in docs for i, t in enumerate(corpus[d].sentences)]
            ps = nli.probs(claim, [t for _, _, t in sents])
            e2 = c2 = 0.0
            for b in range(0, len(sents), 32):
                ch = sents[b:b + 32]
                x = nt([t for _, _, t in ch], [claim] * len(ch), truncation=True, max_length=256, padding=True, return_tensors="pt")
                with torch.inference_mode():
                    p = torch.softmax(nm(**x).logits.float(), dim=-1)
                e2 = max(e2, float(p[:, lab["entailment"]].max())); c2 = max(c2, float(p[:, lab["contradiction"]].max()))
            out[k] = {"docs": docs, "rerank_scores": [round(top, 4)],
                      "nli_entail": max(d["entailment"] for d in ps), "nli_contra": max(d["contradiction"] for d in ps),
                      "nli2_entail": round(e2, 6), "nli2_contra": round(c2, 6),
                      "overlap": ov.score(claim, [Passage(d, t) for d, _, t in sents]),
                      "evidence_chars": len(evidence_text(corpus, docs))}
            if n % 25 == 0:
                print(split, n, f"{time.time() - t0:.0f}s", flush=True)
                json.dump(out, open(out_path, "w"), sort_keys=True)
    json.dump(out, open(out_path, "w"), sort_keys=True)
    print("wrote", out_path, len(out), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["train"])
