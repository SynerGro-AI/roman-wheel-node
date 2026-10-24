"""Write label_review.md: claims where EVERY judge on gold (cited-abstract) evidence disagrees with the gold label.

For a human reviewer only. Gold labels are never changed, and no number in the report uses this list.

SciFact is licensed for non-commercial use, so both outputs carry SciFact IDs only (claim id, doc id, sentence index),
never claim text, titles, abstract sentences or judge quotes. Look the IDs up in your own copy of the dataset.
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factcheck.interfaces import XAIJudge  # noqa: E402
from factcheck.scifact import load_corpus  # noqa: E402
from features import CACHE, DATA, evidence_text  # noqa: E402
from judges import JUDGES, _register_fewshot, selected  # noqa: E402

LAB = {"SUPPORT": "SUPPORT", "CONTRADICT": "CONTRADICT", "NEI": "NEI"}


def _ws(s: str) -> str:
    return " ".join(s.split())


def quote_sentences(corpus, docs, quote):
    """Locate a judge quote as [doc_id, sentence index] pairs instead of storing its text.

    Returns (pairs, match): "exact" if the quote is a substring of (or contains) corpus sentences, "near" if one
    sentence holds >= 80% of the quote's word tokens (a judge that lightly reworded), "none" otherwise.
    """
    q = _ws(quote)
    hits = [[d, i] for d in docs for i, s in enumerate(corpus[d].sentences) if _ws(s) and (q in _ws(s) or _ws(s) in q)]
    if hits:
        return hits, "exact"
    qt, best = set(re.findall(r"\w+", q.lower())), (0.0, None)
    for d in docs:
        for i, s in enumerate(corpus[d].sentences):
            r = len(qt & set(re.findall(r"\w+", s.lower()))) / max(1, len(qt))
            if r > best[0]:
                best = (r, [d, i])
    return ([best[1]], "near") if best[0] >= 0.8 else ([], "none")


def _sents(pairs) -> str:
    return ", ".join(f"doc {d} sentence {i}" for d, i in pairs)


def render_md(out) -> str:
    L = [
        "# SciFact label review (for a human; nothing here changes any label or any reported number)",
        "",
        "Claims where **every** Grok judge (8-9 judges, 5 models, 6 prompt styles) reading the *cited* abstracts gave "
        "the",
        "same verdict, and that verdict differs from the SciFact gold label. Unanimity is not proof: the panel is",
        "correlated (error φ 0.4-0.9), so a shared misreading is possible. Each case needs a domain expert.",
        "",
        "SciFact is licensed for non-commercial use, so this file lists SciFact IDs only (claim id, abstract doc id,",
        "0-based abstract sentence index), never the claim, title or abstract text. Look them up in your own copy of",
        "SciFact (`claims_{split}.jsonl` by `id`, `corpus.jsonl` by `doc_id`, sentence index into `abstract`).",
        "",
        f"Dev (held-out): {len(out['dev'])} of 300 claims ({100 * len(out['dev']) / 300:.1f}%). "
        f"Train subset: {len(out['train'])} of 350 ({100 * len(out['train']) / 350:.1f}%).",
        "",
    ]
    for split in ("dev", "train"):
        L.append(f"## {split}")
        L.append("")
        for x in sorted(out[split], key=lambda x: -x["min_conf"]):
            L.append(
                f"### {x['key']}: gold **{x['gold']}**, panel **{x['panel']}** (min judge confidence "
                f"{x['min_conf']:.2f})"
            )
            L.append(f"- Claim: SciFact {split} claim id {x['claim_id']}")
            L.append(f"- Cited abstract: doc {x['doc']}")
            if x["quote_sentences"]:
                how = "verbatim" if x["quote_match"] == "exact" else "lightly reworded"
                L.append(f"- Judge quote ({how}): {_sents(x['quote_sentences'])}")
            if x["rationale"]:
                L.append(f"- Gold rationale: {_sents(x['rationale'])}")
            else:
                L.append("- Gold rationale: none (NEI)")
            L.append("- Reviewer verdict: [ ] gold correct  [ ] gold wrong  [ ] ambiguous")
            L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    _register_fewshot()
    corpus = load_corpus(DATA)
    feats = json.load(open(os.path.join(CACHE, "local_features.json")))
    js = {
        n: XAIJudge(
            c["model"],
            style=c["style"],
            temperature=c["temperature"],
            samples=c["samples"],
            cache_path=os.path.join(CACHE, f"judge_{n}.jsonl"),
        )
        for n, c in JUDGES.items()
        if "gold" in c["conditions"]
    }
    out = {"dev": [], "train": []}
    for k in selected(feats):
        f = feats[k]
        ev = evidence_text(corpus, f["gold"]["docs"])
        verdicts, quotes, conf = {}, {}, []
        for n, j in js.items():
            if not j.cached(f["claim"], ev):
                continue
            p, q = j.judge(f["claim"], ev)
            v = max(p, key=p.get)
            verdicts[n] = v
            conf.append(p[v])
            if q:
                quotes[n] = q
        vs = set(verdicts.values())
        if len(verdicts) >= 8 and len(vs) == 1 and vs != {f["label3"]}:
            docs = f["gold"]["docs"]
            rat = [[d, i] for d, i in f["rationale"] if d in corpus and i < len(corpus[d].sentences)]
            q = quotes.get("J4") or quotes.get("J1") or ""
            qs, qm = quote_sentences(corpus, docs, q) if q else ([], "no quote")
            out[f["split"]].append(
                {
                    "key": k,
                    "claim_id": int(k.split(":")[1]),
                    "gold": f["label3"],
                    "panel": vs.pop(),
                    "min_conf": min(conf),
                    "doc": docs[0] if docs else "",
                    "quote_sentences": qs,
                    "quote_match": qm,
                    "rationale": rat[:2],
                }
            )
    open(os.path.join(HERE, "label_review.md"), "w").write(render_md(out))
    json.dump(out, open(os.path.join(os.path.dirname(HERE), "results", "phase3", "label_review.json"), "w"), indent=1)
    print({s: len(v) for s, v in out.items()})


if __name__ == "__main__":
    main()
