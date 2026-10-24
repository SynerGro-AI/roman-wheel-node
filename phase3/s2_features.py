"""Phase 3d: local sources for every step-2 item (nli = deberta-v3-small int8 ONNX, nli2 = deberta-v3-base
mnli-fever-anli,
overlap). Max entailment / contradiction over the evidence sentences. Output: cache/step2/local.json (gitignored)."""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factcheck.interfaces import NLIScorer, OverlapScorer, Passage  # noqa: E402
from features import CACHE  # noqa: E402

NLI2 = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def main(items_path: str = "", out_path: str = "") -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(6)
    hf = os.path.join(HERE, "models", "hf")
    nt = AutoTokenizer.from_pretrained(NLI2, cache_dir=hf)
    nm = AutoModelForSequenceClassification.from_pretrained(NLI2, cache_dir=hf).eval()
    lab = {v.lower(): int(k) for k, v in nm.config.id2label.items()}
    nli = NLIScorer(os.path.join(HERE, "models", "nli-deberta-v3-small"))
    ov = OverlapScorer()
    items = json.load(open(items_path or os.path.join(CACHE, "step2", "items.json")))
    path = out_path or os.path.join(CACHE, "step2", "local.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for n, it in enumerate(items):
        if it["id"] in out:
            continue
        claim, sents = it["claim"], it["sents"]
        ps = nli.probs(claim, sents)
        e2 = c2 = 0.0
        for b in range(0, len(sents), 32):
            ch = sents[b : b + 32]
            x = nt(ch, [claim] * len(ch), truncation=True, max_length=256, padding=True, return_tensors="pt")
            with torch.inference_mode():
                p = torch.softmax(nm(**x).logits.float(), dim=-1)
            e2 = max(e2, float(p[:, lab["entailment"]].max()))
            c2 = max(c2, float(p[:, lab["contradiction"]].max()))
        out[it["id"]] = {
            "nli_entail": max(d["entailment"] for d in ps),
            "nli_contra": max(d["contradiction"] for d in ps),
            "nli2_entail": round(e2, 6),
            "nli2_contra": round(c2, 6),
            "overlap": ov.score(claim, [Passage("e", s) for s in sents]),
        }
        if n % 200 == 0:
            print(n, f"{time.time() - t0:.0f}s", flush=True)
            json.dump(out, open(path, "w"))
    json.dump(out, open(path, "w"))
    print("wrote", path, len(out), flush=True)


if __name__ == "__main__":
    main()
