"""Phase 4: run the FROZEN Phase 3d 6-judge panel (J1 J2 J4 J7 J9 J10; same models, prompts and caches, no re-tuning)
on the fresh sets in phase3/cache/phase4/items.json. Paid and opt-in: this script is never run by shadow mode.

Hard cap: CAP_USD over the whole of phase3/cache/spend.jsonl (user cap $320 total, stop at $315). A billing, credit or
auth error from xAI stops every worker and exits with code 3 (top up xAI, then rerun: the cache keeps progress).
Usage: fresh_judges.py --ds hv_test_unjudged [--limit N] [--estimate|--go] [--shard i/n] [--workers W]
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "phase3"))
import s2_judges as S  # noqa: E402
from features import CACHE  # noqa: E402

CAP_USD = float(os.environ.get("RWN_P4_CAP", "315.0"))


def load_items(dss, limit=0, role=""):
    items = json.load(open(os.path.join(CACHE, "phase4", "items.json"), encoding="utf-8"))
    out = []
    for ds in dss:
        xs = [i for i in items if i["ds"] == ds]
        xs.sort(key=lambda i: hashlib.sha256(("ord" + i["id"]).encode()).hexdigest())  # random but fixed order
        out += xs[:limit] if limit else xs
    return out


if __name__ == "__main__":
    if CAP_USD > 315.0:
        sys.exit("cap above the user's $315 stop")
    S.CAP_USD = CAP_USD
    S.load_items = load_items
    sys.exit(S.main())
