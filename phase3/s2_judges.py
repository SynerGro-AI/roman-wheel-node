"""Phase 3d (step 2): run the full 6-judge panel (J1, J2, J4, J7, J9, J10) on cache/step2/items.json.

Same judge models, prompts and caches as Phase 3, so SciFact items whose evidence was already judged are cache hits.
Hard cap: CAP_USD over the whole of cache/spend.jsonl (user cap $220, minus about $5 spare). A billing, credit or auth
error from xAI stops every worker and exits with code 3.
Usage: s2_judges.py --ds climatefever,healthver_train [--limit N] [--estimate|--go] [--shard i/n]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factcheck.interfaces import XAI_PRICES, BillingError, BudgetExceeded, SpendLedger  # noqa: E402
from features import CACHE  # noqa: E402
from judges import CHARS_PER_TOKEN, JUDGES, make_judges  # noqa: E402

CAP_USD = 215.0
PANEL = ("J1", "J2", "J4", "J7", "J9", "J10")
# measured $/call on Phase 3 SciFact evidence relative to the char-based estimate (reasoning tokens are unpredictable)
RATIO = {"grok-4.20-0309-non-reasoning": 1.0, "grok-4.3": 1.3, "grok-4.20-0309-reasoning": 1.9, "grok-4.6": 2.2}


def load_items(dss, limit=0, role=""):
    items = json.load(open(os.path.join(CACHE, "step2", "items.json"), encoding="utf-8"))
    out = []
    for ds in dss:
        xs = [i for i in items if i["ds"] == ds and (not role or i["role"] == role)]
        xs.sort(key=lambda i: hashlib.sha256(("ord" + i["id"]).encode()).hexdigest())  # random but fixed order
        out += xs[:limit] if limit else xs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--judges", default=",".join(PANEL))
    args = ap.parse_args()
    spend = SpendLedger(os.path.join(CACHE, "spend.jsonl"), CAP_USD)
    J = {n: j for n, j in make_judges(spend).items() if n in args.judges.split(",")}
    items = load_items(args.ds.split(","), args.limit)
    si, sn = (int(x) for x in args.shard.split("/"))
    todo = [
        (it, n)
        for it in items
        for n in J
        if not J[n].cached(it["claim"], it["evidence"])
        and int(hashlib.sha256(it["id"].encode()).hexdigest(), 16) % sn == si
    ]
    per = {}
    for it, n in todo:
        c = JUDGES[n]
        pin, _, pout = XAI_PRICES[c["model"]]
        usd = (len(J[n]._render(it["claim"], it["evidence"])) / CHARS_PER_TOKEN * pin + c["out_tokens"] * pout) / 1e6
        per[n] = per.get(n, 0.0) + usd * RATIO.get(c["model"], 1.5)
    est = sum(per.values())
    print(
        f"spent ${spend.total:.4f}; {len(items)} items, {len(todo)} uncached calls; estimated ${est:.2f} "
        f"({', '.join(f'{k} ${v:.2f}' for k, v in sorted(per.items()))}); projected ${spend.total + est:.2f} vs cap "
        f"${CAP_USD}",
        flush=True,
    )
    if not args.go:
        return 0
    stop = threading.Event()
    done = [0]

    def run(it, n):
        if stop.is_set():
            return
        J[n].judge(it["claim"], it["evidence"])

    rc = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(run, it, n) for it, n in todo]
        for f in as_completed(futs):
            try:
                f.result()
                done[0] += 1
            except BillingError as e:
                stop.set()
                rc = 3
                print("STOP: billing/credit/auth error from xAI:", e, flush=True)
            except BudgetExceeded as e:
                stop.set()
                rc = 2
                print("STOP: spend cap reached:", e, flush=True)
            except Exception as e:  # timeouts etc.: leave uncached, a later sweep retries
                print("call failed:", type(e).__name__, str(e)[:120], flush=True)
            if done[0] % 200 == 0:
                print(f"done {done[0]}; spend ${spend.total:.2f}", flush=True)
    print(f"finished: done {done[0]}/{len(todo)}; spend ${spend.total:.4f}; rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
