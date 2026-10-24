"""Optional stronger local NLI source: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (184M params, ~0.7 GB) on CPU.

It needs torch (CPU) and transformers. It adds ``nli2_entail`` / ``nli2_contra`` (max over every sentence of the shown
abstracts) to the local features cache. It was trained on MNLI/FEVER/ANLI, not on SciFact, so it cannot leak SciFact
labels.
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factcheck.scifact import load_corpus  # noqa: E402
from features import CACHE, DATA  # noqa: E402

MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=os.path.join(HERE, "models", "hf"))
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, cache_dir=os.path.join(HERE, "models", "hf")
    ).eval()
    lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    corpus = load_corpus(DATA)
    path = os.path.join(CACHE, "local_features.json")
    feats = json.load(open(path, encoding="utf-8"))
    t0 = time.time()
    for n, (k, f) in enumerate(sorted(feats.items())):
        for cond in ("retrieved", "gold"):
            sents = [(d, i, s) for d in f[cond]["docs"] for i, s in enumerate(corpus[d].sentences)]
            best_e = best_c = (0.0, None)
            for b in range(0, len(sents), 32):
                chunk = sents[b : b + 32]
                enc = tok(
                    [s for _, _, s in chunk],
                    [f["claim"]] * len(chunk),
                    truncation=True,
                    max_length=256,
                    padding=True,
                    return_tensors="pt",
                )
                with torch.inference_mode():
                    p = torch.softmax(model(**enc).logits.float(), dim=-1)
                for j, (d, i, _) in enumerate(chunk):
                    e, c = float(p[j, lab["entailment"]]), float(p[j, lab["contradiction"]])
                    if e > best_e[0]:
                        best_e = (e, [d, i])
                    if c > best_c[0]:
                        best_c = (c, [d, i])
            f[cond]["nli2_entail"], f[cond]["nli2_entail_at"] = round(best_e[0], 6), best_e[1]
            f[cond]["nli2_contra"], f[cond]["nli2_contra_at"] = round(best_c[0], 6), best_c[1]
        if n % 100 == 0:
            print(n, f"{time.time() - t0:.0f}s", flush=True)
    json.dump(feats, open(path + ".tmp", "w", encoding="utf-8"), sort_keys=True)
    os.replace(path + ".tmp", path)
    print("updated", path)


if __name__ == "__main__":
    main()
