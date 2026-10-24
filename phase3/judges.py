"""Stage 2: Grok judges on every selected claim x condition, with a cost estimate, a hard budget cap and a response
cache.

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
from factcheck.interfaces import (  # noqa: E402
    JUDGE_PROMPTS,  # noqa: E402
    XAI_PRICES,
    BudgetExceeded,
    SpendLedger,
    XAIJudge,
)
from factcheck.scifact import load_corpus  # noqa: E402
from features import CACHE, DATA, evidence_text  # noqa: E402

N_TRAIN = 350
CONDITIONS = ("retrieved", "gold", "retrieved5", "rv3")
CAP_USD = 100.0  # hard cap on total spend (raised from $10 by the user), including all earlier calls
ALL = ("retrieved", "gold", "retrieved5")
# conditions whose evidence lists live in their own feature file (key -> {"docs": [...], local scores})
COND_FILES = {"retrieved5": "local_features_r5.json", "rv3": "local_features_rv3.json"}


def J(model, style, max_tokens, out_tokens, conditions=ALL, temperature=0.0, samples=1, note=""):
    return dict(
        model=model,
        style=style,
        max_tokens=max_tokens,
        out_tokens=out_tokens,
        conditions=conditions,
        temperature=temperature,
        samples=samples,
        note=note,
    )


# out_tokens = expected output incl. reasoning tokens (from probe calls); used only for the cost estimate
# J5, J6, J8 run on gold only: on retrieved5 the budget went to the judges with the best TRAIN accuracy per dollar
# on gold (J4 .903, J9 .900, J7 .891 vs J5 .840, J6 .831, J8 .860), plus the cheap J1-J3.
JUDGES = {
    "J1": J(
        "grok-4.20-0309-non-reasoning",
        "A",
        300,
        75,
        ALL + ("rv3",),
        note="fact-checker, 3-way + verbatim quote (exact-copy gate; unwrapped baseline)",
    ),
    "J2": J("grok-4.3", "B", 2000, 450, ALL + ("rv3",), note="skeptical peer reviewer"),
    "J3": J("grok-4.20-0309-non-reasoning", "C", 60, 20, note="plain QA framing"),
    "J4": J(
        "grok-4.20-0309-reasoning", "D", 3000, 900, ("gold", "retrieved5", "rv3"), note="SciFact annotation guidelines"
    ),
    "J5": J("grok-4.7", "D", 4000, 700, ("gold",), note="SciFact guidelines, strongest model"),
    "J6": J("grok-4.5", "F", 3000, 300, ("gold",), note="adversarial devil's advocate"),
    "J7": J("grok-4.6", "E", 4000, 1300, ("gold", "retrieved5", "rv3"), note="two-sided support-vs-refute"),
    "J9": J(
        "grok-4.20-0309-reasoning",
        "G",
        3000,
        900,
        ("gold", "retrieved5", "rv3"),
        note="few-shot (10 train examples) + guidelines",
    ),
    "J10": J("grok-4.20-0309-reasoning", "H", 3000, 900, ("rv3",), note="primary-abstract / NEI-focused"),
    "J11": J(
        "grok-4.20-0309-reasoning", "V", 3000, 900, ("gold", "rv3"), note="Phase 3c strict proof verifier (quote first)"
    ),
    "J12": J(
        "grok-4.20-0309-reasoning",
        "W",
        3500,
        1100,
        (),
        note="Phase 3c component-wise verifier: dropped after 47 calls (measured $0.010/call, 2x the estimate; budget)",
    ),
    "J8": J(
        "grok-4.20-0309-non-reasoning",
        "A",
        300,
        75,
        ("gold",),
        temperature=1.0,
        samples=3,
        note="self-consistency: 3 samples at T=1, averaged",
    ),
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
    JUDGE_PROMPTS["G"] = (
        head
        + "Worked examples (labels from the SciFact annotators):\n\n"
        + ex
        + "\n\nNow the new case.\n\nClaim: {claim}"
        + tail
    )


CHARS_PER_TOKEN = 3.4  # measured 3.45-3.57 prompt chars per prompt_token on calibration calls; 3.4 is conservative


def selected(feats: dict) -> list:  # noqa: D401
    train = sorted((k for k in feats if k.startswith("train:")), key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    return sorted(k for k in feats if k.startswith("dev:")) + train[:N_TRAIN]


def make_judges(spend: SpendLedger) -> dict:
    _register_fewshot()
    return {
        n: XAIJudge(
            c["model"],
            name=n,
            cache_path=os.path.join(CACHE, f"judge_{n}.jsonl"),
            style=c["style"],
            spend=spend,
            max_tokens=c["max_tokens"],
            temperature=c["temperature"],
            samples=c["samples"],
        )
        for n, c in JUDGES.items()
    }


def load_cond_files() -> dict:
    out = {}
    for c, fn in COND_FILES.items():
        p = os.path.join(CACHE, fn)
        out[c] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    return out


def docs_for(feats, extra, k, cond):
    return extra[cond][k]["docs"] if cond in COND_FILES else feats[k][cond]["docs"]


def jobs(feats, extra, corpus, judges):
    for k in selected(feats):
        f = feats[k]
        for cond in CONDITIONS:
            if cond in COND_FILES and k not in extra[cond]:
                continue
            ev = evidence_text(corpus, docs_for(feats, extra, k, cond))
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
    ap.add_argument("--split", default="", help="train or dev: only claims of that split")
    ap.add_argument("--shard", default="0/1", help="i/n: only claims whose sha256(key) mod n == i (parallel runs)")
    args = ap.parse_args()
    feats = json.load(open(os.path.join(CACHE, "local_features.json"), encoding="utf-8"))
    corpus = load_corpus(DATA)
    spend = SpendLedger(os.path.join(CACHE, "spend.jsonl"), CAP_USD)
    judges = make_judges(spend)
    if args.only:
        judges = {k: v for k, v in judges.items() if k in args.only.split(",")}
    todo = [j for j in jobs(feats, load_cond_files(), corpus, judges) if not j[3].cached(j[4], j[5])]
    if args.conds:
        todo = [j for j in todo if j[1] in args.conds.split(",")]
    if args.split:
        todo = [j for j in todo if j[0].startswith(args.split)]
    si, sn = (int(x) for x in args.shard.split("/"))
    todo = [j for j in todo if int(hashlib.sha256(j[0].encode()).hexdigest(), 16) % sn == si]
    todo.sort(key=lambda j: (j[2] == "J2", j[0], j[1], j[2]))  # cheap fixed-cost judges first, J2 (reasoning) last
    est = 0.0
    per = {}
    for k, cond, n, j, claim, ev in todo:
        c = JUDGES[n]
        pin, _, pout = XAI_PRICES[c["model"]]
        ratio = (
            2.0 if c["model"] in ("grok-4.7", "grok-4.6", "grok-4.5") else 1.0
        )  # these bill ~2x prompt tokens (probe)
        usd = c["samples"] * (len(j._render(claim, ev)) / CHARS_PER_TOKEN * ratio * pin + c["out_tokens"] * pout) / 1e6
        est += usd
        per[n] = per.get(n, 0.0) + usd
    print(
        f"spent so far ${spend.total:.4f} over {spend.calls} calls; {len(todo)} uncached calls; "
        f"estimated ${est:.2f} ({', '.join(f'{n} ${v:.2f}' for n, v in sorted(per.items()))}); "
        f"projected total ${spend.total + est:.2f} vs cap ${CAP_USD:.2f}"
    )
    if not args.go:
        return
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(j.judge, claim, ev) for _, _, _, j, claim, ev in todo]
        try:
            for fu in as_completed(futs):
                fu.result()
                done += 1
                if done % 200 == 0:
                    print(f"{done}/{len(todo)} calls, spent ${spend.total:.4f}", flush=True)
        except BudgetExceeded as e:
            for fu in futs:
                fu.cancel()
            print("STOPPED:", e)
    print(f"done {done}; total spend ${spend.total:.4f} over {spend.calls} calls")


if __name__ == "__main__":
    main()
