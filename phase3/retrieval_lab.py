"""Retrieval study on SciFact TRAIN claims only (gold doc ids from train are the only labels used).

Ranks: BM25; dense bge-small-en-v1.5; dense MedCPT (PubMed-trained query/article encoders); RRF fusions; and
cross-encoder reranking (ms-marco-MiniLM-L-6-v2, MedCPT-Cross-Encoder) of a fused top-N. Writes cache/ranks_<split>.json
so the chosen pipeline can then be applied to dev once, unchanged.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factcheck.interfaces import BM25Retriever          # noqa: E402
from factcheck.scifact import load_corpus                # noqa: E402
from features import CACHE, DATA                         # noqa: E402
from judges import selected                              # noqa: E402

HF = os.path.join(HERE, "models", "hf")


def doc_text(d):
    return d.title + " " + " ".join(d.sentences)


def dense(model_q, model_d, claims, corpus_ids, corpus, pool="cls", qprefix="", max_len=512):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = os.path.join(CACHE, f"emb_{model_d.replace('/', '_')}.npy")
    tq, mq = AutoTokenizer.from_pretrained(model_q, cache_dir=HF), AutoModel.from_pretrained(model_q, cache_dir=HF).eval()

    def enc(tok, mod, texts, ml, pairs=False):
        out = []
        for i in range(0, len(texts), 32):
            b = texts[i:i + 32]
            x = tok([t[0] for t in b], [t[1] for t in b], truncation=True, max_length=ml, padding=True, return_tensors="pt") if pairs \
                else tok(b, truncation=True, max_length=ml, padding=True, return_tensors="pt")
            with torch.inference_mode():
                h = mod(**x).last_hidden_state
            v = h[:, 0] if pool == "cls" else (h * x["attention_mask"][..., None]).sum(1) / x["attention_mask"].sum(1, keepdim=True)
            out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        return np.concatenate(out)
    if os.path.exists(path):
        D = np.load(path)
    else:
        td, md = AutoTokenizer.from_pretrained(model_d, cache_dir=HF), AutoModel.from_pretrained(model_d, cache_dir=HF).eval()
        if "MedCPT" in model_d:
            D = enc(td, md, [(corpus[i].title, " ".join(corpus[i].sentences)) for i in corpus_ids], max_len, pairs=True)
        else:
            D = enc(td, md, [doc_text(corpus[i]) for i in corpus_ids], max_len)
        np.save(path, D)
    Q = enc(tq, mq, [qprefix + c for c in claims], 64)
    S = Q @ D.T
    return [[corpus_ids[j] for j in np.argsort(-row)[:100]] for row in S]


def rrf(*lists, k=60, n=100):
    sc = {}
    for L in lists:
        for r, d in enumerate(L):
            sc[d] = sc.get(d, 0.0) + 1.0 / (k + r + 1)
    return [d for d, _ in sorted(sc.items(), key=lambda x: (-x[1], x[0]))][:n]


def rerank(model, claim_docs, corpus, topn=20):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, cache_dir=HF)
    mod = AutoModelForSequenceClassification.from_pretrained(model, cache_dir=HF).eval()
    out = []
    for claim, docs in claim_docs:
        cand = docs[:topn]
        x = tok([claim] * len(cand), [doc_text(corpus[d]) for d in cand], truncation=True, max_length=512, padding=True, return_tensors="pt")
        with torch.inference_mode():
            s = mod(**x).logits[:, 0].tolist()
        out.append([d for _, d in sorted(zip(s, cand), key=lambda z: (-z[0], z[1]))] + docs[topn:])
    return out


def main(split="train", methods=None):
    import torch
    torch.set_num_threads(8)
    corpus = load_corpus(DATA)
    ids = sorted(corpus)
    feats = json.load(open(os.path.join(CACHE, "local_features.json")))
    keys = [k for k in selected(feats) if k.startswith(split)]
    claims = [feats[k]["claim"] for k in keys]
    path = os.path.join(CACHE, f"ranks_{split}.json")
    R = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    if "bm25" not in R:
        bm = BM25Retriever({d: doc_text(corpus[d]) for d in ids})
        R["bm25"] = [bm.retrieve(c, 100) for c in claims]
    if "bge" not in R:
        R["bge"] = dense("BAAI/bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", claims, ids, corpus,
                         qprefix="Represent this sentence for searching relevant passages: ")
    if "medcpt" not in R:
        R["medcpt"] = dense("ncbi/MedCPT-Query-Encoder", "ncbi/MedCPT-Article-Encoder", claims, ids, corpus)
    json.dump(R, open(path, "w"))
    R["rrf_bm25_medcpt"] = [rrf(a, b) for a, b in zip(R["bm25"], R["medcpt"])]
    R["rrf_all3"] = [rrf(a, b, c) for a, b, c in zip(R["bm25"], R["medcpt"], R["bge"])]
    for base in ("rrf_all3",):
        rerankers = (("ncbi/MedCPT-Cross-Encoder", "ce_medcpt"), ("cross-encoder/ms-marco-MiniLM-L-6-v2", "ce_minilm"))
        if split != "train":   # the reranker was selected on train recall/MRR; dev runs only the chosen pipeline
            rerankers = rerankers[:1]
        for m, tag in rerankers:
            key = f"{tag}({base})"
            if key not in R:
                R[key] = rerank(m, list(zip(claims, R[base])), corpus)
                json.dump(R, open(path, "w"))
            print(key, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(R, open(path, "w"))
    ev = [i for i, k in enumerate(keys) if feats[k]["label3"] != "NEI"]
    print(f"{split}: {len(ev)} claims with evidence; recall@k of any gold doc, and mean #uncited docs shown at k")
    for name, L in R.items():
        rec = {k: round(sum(any(d in L[i][:k] for d in feats[keys[i]]["cited"]) for i in ev) / len(ev), 3) for k in (1, 2, 3, 5, 10, 20)}
        mrr = sum(next((1 / (r + 1) for r, d in enumerate(L[i][:20]) if d in feats[keys[i]]["cited"]), 0) for i in ev) / len(ev)
        print(f"  {name:32s} {rec}  MRR@20 {mrr:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
