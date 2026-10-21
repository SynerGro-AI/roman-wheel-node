"""Write label_review.md: claims where EVERY judge on gold (cited-abstract) evidence disagrees with the gold label.

For a human reviewer only. Gold labels are never changed, and no number in the report uses this list.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factcheck.interfaces import XAIJudge                   # noqa: E402
from factcheck.scifact import load_corpus                    # noqa: E402
from features import CACHE, DATA, evidence_text              # noqa: E402
from judges import JUDGES, _register_fewshot, selected       # noqa: E402

LAB = {"SUPPORT": "SUPPORT", "CONTRADICT": "CONTRADICT", "NEI": "NEI"}


def main() -> None:
    _register_fewshot()
    corpus = load_corpus(DATA)
    feats = json.load(open(os.path.join(CACHE, "local_features.json")))
    js = {n: XAIJudge(c["model"], style=c["style"], temperature=c["temperature"], samples=c["samples"],
                      cache_path=os.path.join(CACHE, f"judge_{n}.jsonl")) for n, c in JUDGES.items() if "gold" in c["conditions"]}
    out = {"dev": [], "train": []}
    for k in selected(feats):
        f = feats[k]
        ev = evidence_text(corpus, f["gold"]["docs"])
        verdicts, quotes, conf = {}, {}, []
        for n, j in js.items():
            if not j.cached(f["claim"], ev):
                continue
            p, q = j.judge(f["claim"], ev)
            v = max(p, key=p.get); verdicts[n] = v; conf.append(p[v])
            if q:
                quotes[n] = q
        vs = set(verdicts.values())
        if len(verdicts) >= 8 and len(vs) == 1 and vs != {f["label3"]}:
            rat = [corpus[d].sentences[i] for d, i in f["rationale"] if d in corpus and i < len(corpus[d].sentences)]
            out[f["split"]].append({"key": k, "claim": f["claim"], "gold": f["label3"], "panel": vs.pop(),
                                    "min_conf": min(conf), "quote": quotes.get("J4") or quotes.get("J1") or "",
                                    "rationale": rat[:2], "doc": f["gold"]["docs"][0] if f["gold"]["docs"] else "",
                                    "title": corpus[f["gold"]["docs"][0]].title if f["gold"]["docs"] else ""})
    L = ["# SciFact label review (for a human; nothing here changes any label or any reported number)", "",
         "Claims where **every** Grok judge (8-9 judges, 5 models, 6 prompt styles) reading the *cited* abstracts gave the",
         "same verdict, and that verdict differs from the SciFact gold label. Unanimity is not proof: the panel is",
         "correlated (error φ 0.4-0.9), so a shared misreading is possible. Each case needs a domain expert.", "",
         f"Dev (held-out): {len(out['dev'])} of 300 claims ({100 * len(out['dev']) / 300:.1f}%). "
         f"Train subset: {len(out['train'])} of 350 ({100 * len(out['train']) / 350:.1f}%).", ""]
    for split in ("dev", "train"):
        L.append(f"## {split}")
        L.append("")
        for x in sorted(out[split], key=lambda x: -x["min_conf"]):
            L.append(f"### {x['key']}: gold **{x['gold']}**, panel **{x['panel']}** (min judge confidence {x['min_conf']:.2f})")
            L.append(f"- Claim: {x['claim']}")
            L.append(f"- Cited abstract {x['doc']}: *{x['title']}*")
            if x["quote"]:
                L.append(f"- Judge quote (verbatim from the abstract): \"{x['quote']}\"")
            for r in x["rationale"]:
                L.append(f"- Gold rationale sentence: \"{r}\"")
            if not x["rationale"]:
                L.append("- Gold rationale: none (NEI)")
            L.append("- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous")
            L.append("")
    open(os.path.join(HERE, "label_review.md"), "w").write("\n".join(L) + "\n")
    json.dump(out, open(os.path.join(os.path.dirname(HERE), "results", "phase3", "label_review.json"), "w"), indent=1)
    print({s: len(v) for s, v in out.items()})


if __name__ == "__main__":
    main()
