"""Stage 2: Grok judges on every selected claim x condition, with a cost estimate, a hard budget cap and a response cache.

    python phase3/judges.py --estimate        # prints the projected cost, spends nothing
    python phase3/judges.py --go              # runs (skips cached responses); stops at the cap

Claims: all 300 SciFact dev claims plus a fixed training subset (the first N_TRAIN train claims ordered by
sha256(claim id), chosen before any results were seen) to learn reliabilities, dependence and the threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factcheck.interfaces import XAI_PRICES, BudgetExceeded, SpendLedger, XAIJudge   # noqa: E402
from factcheck.scifact import load_corpus                                            # noqa: E402
from features import CACHE, DATA, evidence_text                                      # noqa: E402
from factcheck.interfaces import JUDGE_PROMPTS                                        # noqa: E402

N_TRAIN = 350
CONDITIONS = ("retrieved", "gold", "retrieved5")
CAP_USD = 100.0            # hard cap on total spend (raised from $10 by the user), including all earlier calls
ALL = ("retrieved", "gold", "retrieved5")


def J(model, style, max_tokens, out_tokens, conditions=ALL, temperature=0.0, samples=1, note=""):
    return dict(model=model, style=style, max_tokens=max_tokens, out_tokens=out_tokens, conditions=conditions,
                temperature=temperature, samples=samples, note=note)


# out_tokens = expected output incl. reasoning tokens (from probe calls); used only for the cost estimate
JUDGES = {
    "J1": J("grok-4.20-0309-non-reasoning", "A", 300, 75, note="fact-checker, 3-way + verbatim quote"),
    "J2": J("grok-4.3", "B", 2000, 450, note="skeptical peer reviewer"),
    "J3": J("grok-4.20-0309-non-reasoning", "C", 60, 20, note="plain QA framing"),
    "J4": J("grok-4.20-0309-reasoning", "D", 3000, 900, ("gold", "retrieved5"), note="SciFact annotation guidelines"),
    "J5": J("grok-4.7", "D", 4000, 700, ("gold", "retrieved5"), note="SciFact guidelines, strongest model"),
    "J6": J("grok-4.5", "F", 3000, 300, ("gold", "retrieved5"), note="adversarial devil's advocate"),
    "J7": J("grok-4.6", "E", 4000, 1300, ("gold", "retrieved5"), note="two-sided support-vs-refute"),
    "J9": J("grok-4.20-0309-reasoning", "G", 3000, 900, ("gold", "retrieved5"), note="few-shot (10 train examples) + guidelines"),
    "J8": J("grok-4.20-0309-non-reasoning", "A", 300, 75, ("gold", "retrieved5"), temperature=1.0, samples=3,
            note="self-consistency: 3 samples at T=1, averaged"),
}
def _register_fewshot() -> None:
    """Style G = style D's guidelines preceded by 10 labelled examples from train claims outside the learning subset
    (see fewshot.py). The examples come first, so the shared prefix is billed at the cached-input rate."""
    if "G" in JUDGE_PROMPTS or not os.path.exists(os.path.join(CACHE, "local_features.json")):
        return
    from fewshot import build
    ex, _ = build()
    d = JUDGE_PROMPTS["D"]
    head, tail = d.split("Claim: {claim}", 1)
    JUDGE_PROMPTS["G"] = head + "Worked examples (labels from the SciFact annotators):\n\n" + ex + "\n\nNow the new case.\n\nClaim: {claim}" + tail


CHARS_PER_TOKEN = 3.4      # measured 3.45-3.57 prompt chars per prompt_token on calibration calls; 3.4 is conservative


def selected(feats: dict) -> list:  # noqa: D401
    train = sorted((k for k in feats if k.startswith("train:")), key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    return sorted(k for k in feats if k.startswith("dev:")) + train[:N_TRAIN]


def make_judges(spend: SpendLedger) -> dict:
    _register_fewshot()
    return {n: XAIJudge(c["model"], name=n, cache_path=os.path.join(CACHE, f"judge_{n}.jsonl"), style=c["style"],
                        spend=spend, max_tokens=c["max_tokens"], temperature=c["temperature"], samples=c["samples"])
            for n, c in JUDGES.items()}


def docs_for(feats, r5, k, cond):
    return r5[k]["docs"] if cond == "retrieved5" else feats[k][cond]["docs"]


def jobs(feats, r5, corpus, judges):
    for k in selected(feats):
        f = feats[k]
        for cond in CONDITIONS:
            if cond == "retrieved5" and k not in r5:
                continue
            ev = evidence_text(corpus, docs_for(feats, r5, k, cond))
            for n, j in judges.items():
                if cond not in JUDGES[n]["conditions"]:
                    continue
                yield k, cond, n, j, f["claim"], ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-separated judge names to run (default: all)")
    ap.add_argument("--conds", default="", help="comma-separated conditions (default: all)")
    args = ap.parse_args()
    feats = json.load(open(os.path.join(CACHE, "local_features.json")))
    corpus = load_corpus(DATA)
    spend = SpendLedger(os.path.join(CACHE, "spend.jsonl"), CAP_USD)
    judges = make_judges(spend)
    if args.only:
        judges = {k: v for k, v in judges.items() if k in args.only.split(",")}
    r5p = os.path.join(CACHE, "local_features_r5.json")
    r5 = json.load(open(r5p)) if os.path.exists(r5p) else {}
    todo = [j for j in jobs(feats, r5, corpus, judges) if not j[3].cached(j[4], j[5])]
    if args.conds:
        todo = [j for j in todo if j[1] in args.conds.split(",")]
    todo.sort(key=lambda j: (j[2] == "J2", j[0], j[1], j[2]))     # cheap fixed-cost judges first, J2 (reasoning) last
    est = 0.0
    per = {}
    for k, cond, n, j, claim, ev in todo:
        c = JUDGES[n]
        pin, _, pout = XAI_PRICES[c["model"]]
        ratio = 2.0 if c["model"] in ("grok-4.7", "grok-4.6", "grok-4.5") else 1.0   # these bill ~2x prompt tokens (probe)
        usd = c["samples"] * (len(j._render(claim, ev)) / CHARS_PER_TOKEN * ratio * pin + c["out_tokens"] * pout) / 1e6
        est += usd; per[n] = per.get(n, 0.0) + usd
    print(f"spent so far ${spend.total:.4f} over {spend.calls} calls; {len(todo)} uncached calls; "
          f"estimated ${est:.2f} ({', '.join(f'{n} ${v:.2f}' for n, v in sorted(per.items()))}); "
          f"projected total ${spend.total + est:.2f} vs cap ${CAP_USD:.2f}")
    if not args.go:
        return
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(j.judge, claim, ev) for _, _, _, j, claim, ev in todo]
        try:
            for fu in as_completed(futs):
                fu.result(); done += 1
                if done % 200 == 0:
                    print(f"{done}/{len(todo)} calls, spent ${spend.total:.4f}", flush=True)
        except BudgetExceeded as e:
            for fu in futs:
                fu.cancel()
            print("STOPPED:", e)
    print(f"done {done}; total spend ${spend.total:.4f} over {spend.calls} calls")


if __name__ == "__main__":
    main()
