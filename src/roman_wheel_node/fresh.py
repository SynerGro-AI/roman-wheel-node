"""Fresh-data pipeline (Phase 4): label the claims shadow mode logged, then certify ONE pre-registered fixed rule.

    python -m roman_wheel_node.fresh prereg  --log-dir rwn_shadow            # freeze policy+profile+alphas BEFORE labels
    python -m roman_wheel_node.fresh label   --log-dir rwn_shadow            # interactive: SUPPORT / NOT / UNSURE
    python -m roman_wheel_node.fresh import-labels --log-dir rwn_shadow --labels gold.jsonl   # {id|meta_key, label}
    python -m roman_wheel_node.fresh certify --log-dir rwn_shadow [--by ds] [--json out.json]

Risk (as in the Phase 3e certificate): the share of unsupported claims among ACCEPTs (BLOCK and NOT_SURE are not
accepts). For each alpha the one-sided Clopper-Pearson upper bound at level delta (default 0.05) must be <= alpha.
There is no search: the rule is the frozen policy profile that made the logged decisions, fixed in ``prereg.json``
before any label, so a single binomial test per alpha is valid (the alphas are nested claims about one rule).
With 0 false accepts, ceil(log(delta) / log(1 - alpha)) accepts certify alpha: 299 for 1%, 149 for 2%, 59 for 5%.
Labels and claims stay local (``*.jsonl``, gitignored): commit only aggregate numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["cp_upper", "cp_lower", "min_accepts_for", "labelled_needed", "certify", "load_jsonl"]

LABELS = ("SUPPORT", "NOT", "UNSURE")
ALPHAS = (0.01, 0.02, 0.05)


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 1.0
    if p >= 1:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    return sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
               for i in range(k + 1))


def cp_upper(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound: largest p with P(Bin(n, p) <= k) >= delta. n == 0 -> 1."""
    if n <= 0 or k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _binom_cdf(k, n, mid) > delta else (lo, mid)
    return hi


def cp_lower(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided Clopper-Pearson lower bound (0 when k == 0)."""
    if n <= 0 or k <= 0:
        return 0.0
    return 1.0 - cp_upper(n - k, n, delta)


def min_accepts_for(alpha: float, delta: float = 0.05) -> int:
    """Accepts needed with ZERO false accepts to certify alpha: ceil(log(delta) / log(1 - alpha))."""
    return math.ceil(math.log(delta) / math.log1p(-alpha))


def accepts_needed_at(rate: float, alpha: float, delta: float = 0.05, n_max: int = 200000) -> Optional[int]:
    """Smallest number of accepts n such that a false-accept share of ``rate`` (k = floor(rate * n)) certifies alpha."""
    if rate >= alpha:
        return None
    n = min_accepts_for(alpha, delta)
    while n <= n_max:
        if cp_upper(int(math.floor(rate * n)), n, delta) <= alpha:
            return n
        n = int(n * 1.05) + 1
    return None


def labelled_needed(alpha: float, coverage: float, fa_share: float = 0.0, delta: float = 0.05) -> Optional[int]:
    """Labelled fresh claims needed at an ACCEPT rate ``coverage`` and false-accept share ``fa_share``."""
    na = min_accepts_for(alpha, delta) if fa_share <= 0 else accepts_needed_at(fa_share, alpha, delta)
    if na is None or coverage <= 0:
        return None
    return math.ceil(na / coverage)


def load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _labels(log_dir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for r in load_jsonl(os.path.join(log_dir, "labels.jsonl")):     # append-only; the last label for an id wins
        if r.get("label") in LABELS:
            out[r["id"]] = r["label"]
    return out


def stats(rows: Sequence[Tuple[str, bool]], alphas: Sequence[float] = ALPHAS, delta: float = 0.05) -> dict:
    """rows: (outcome, gold_is_support) for every labelled (non-UNSURE) claim."""
    n = len(rows)
    acc = [y for o, y in rows if o == "ACCEPT"]
    na, fa = len(acc), sum(1 for y in acc if not y)
    uns = sum(1 for _, y in rows if not y)
    sup = n - uns
    cov = na / n if n else 0.0
    share = fa / na if na else None
    out = {"labelled": n, "supported": sup, "unsupported": uns, "accepted": na, "false_accepts": fa,
           "blocked": sum(1 for o, _ in rows if o == "BLOCK"), "not_sure": sum(1 for o, _ in rows if o == "NOT_SURE"),
           "coverage": cov, "precision": (1 - share) if share is not None else None,
           "precision_ci95": [cp_lower(na - fa, na, 0.025), 1 - cp_lower(fa, na, 0.025)] if na else None,
           "false_accept_share_upper": cp_upper(fa, na, delta), "delta": delta,
           "false_accept_rate_of_unsupported": (fa / uns) if uns else None,
           "false_accept_rate_of_unsupported_upper": cp_upper(fa, uns, delta),
           "recall_of_supported": ((na - fa) / sup) if sup else None, "alphas": {}}
    for al in alphas:
        ub = out["false_accept_share_upper"]
        out["alphas"][f"{al}"] = {
            "certified": bool(na > 0 and ub <= al), "upper": ub,
            "accepts_needed_zero_errors": min_accepts_for(al, delta),
            "labelled_needed_at_observed_coverage_zero_errors": labelled_needed(al, cov, 0.0, delta),
            "labelled_needed_at_observed_coverage_and_error_rate": labelled_needed(al, cov, share or 0.0, delta)}
    return out


def certify(log_dir: str, by: Optional[str] = None, alphas: Optional[Sequence[float]] = None,
            delta: Optional[float] = None) -> dict:
    pre = json.load(open(os.path.join(log_dir, "prereg.json")))
    alphas = tuple(alphas or pre.get("alphas", ALPHAS))
    delta = float(delta or pre.get("delta", 0.05))
    labels = _labels(log_dir)
    groups: Dict[str, List[Tuple[str, bool]]] = {"all": []}
    skipped = {"other_rule": 0, "unlabelled": 0, "unsure": 0}
    for r in load_jsonl(os.path.join(log_dir, "fresh_claims.jsonl")):
        if r.get("policy") != pre["policy"] or r.get("profile") != pre["profile"]:
            skipped["other_rule"] += 1; continue
        lab = labels.get(r["id"])
        if lab is None:
            skipped["unlabelled"] += 1; continue
        if lab == "UNSURE":
            skipped["unsure"] += 1; continue
        row = (r["outcome"], lab == "SUPPORT")
        groups["all"].append(row)
        if by:
            groups.setdefault(f"{by}={(r.get('meta') or {}).get(by)}", []).append(row)
    return {"prereg": pre, "skipped": skipped, "groups": {g: stats(rs, alphas, delta) for g, rs in groups.items()}}


def report(res: dict) -> str:
    p = res["prereg"]
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    L = [f"Fresh-set certification: policy {p['policy'][:12]}, profile {p['profile']}, risk = false accepts among ACCEPTs, "
         f"one-sided Clopper-Pearson, delta {p.get('delta', 0.05)}; pre-registered {p.get('created', '?')}",
         f"skipped: {res['skipped']}"]
    for g, s in res["groups"].items():
        ci = s["precision_ci95"]
        L.append(f"== {g}: {s['labelled']} labelled ({s['supported']} supported), {s['accepted']} ACCEPT "
                 f"(coverage {pct(s['coverage'])}), {s['false_accepts']} false accepts, precision {pct(s['precision'])}"
                 + (f" [95% CI {pct(ci[0])}, {pct(ci[1])}]" if ci else "") +
                 f", false-accept share UB {pct(s['false_accept_share_upper'])}; BLOCK {s['blocked']}, NOT_SURE {s['not_sure']}; "
                 f"false accepts / unsupported {pct(s['false_accept_rate_of_unsupported'])} (UB {pct(s['false_accept_rate_of_unsupported_upper'])})")
        for al, a in s["alphas"].items():
            L.append(f"   alpha {float(al):.0%}: {'CERTIFIED' if a['certified'] else 'not certified'} (UB {pct(a['upper'])}); "
                     f"needs {a['accepts_needed_zero_errors']} error-free accepts = ~{a['labelled_needed_at_observed_coverage_zero_errors']} "
                     f"labelled claims at this coverage; at the observed error rate: {a['labelled_needed_at_observed_coverage_and_error_rate'] or 'impossible (rate >= alpha)'}")
    return "\n".join(L) + "\n"


def prereg(log_dir: str, policy: str, profile: str, alphas=ALPHAS, delta: float = 0.05, note: str = "") -> dict:
    path = os.path.join(log_dir, "prereg.json")
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: a pre-registration is never overwritten")
    if os.path.exists(os.path.join(log_dir, "labels.jsonl")):
        raise RuntimeError("labels already exist: pre-register before labelling")
    doc = {"policy": policy, "profile": profile, "alphas": list(alphas), "delta": delta,
           "risk": "false accepts among ACCEPTs (one-sided Clopper-Pearson)", "search": "none (single fixed rule)",
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "note": note}
    os.makedirs(log_dir, exist_ok=True)
    with open(path, "x") as f:
        json.dump(doc, f, indent=1)
    return doc


def _label_cli(log_dir: str, labeller: str, inp=sys.stdin, out=sys.stdout) -> int:
    done = _labels(log_dir)
    todo = [r for r in load_jsonl(os.path.join(log_dir, "fresh_claims.jsonl")) if r["id"] not in done]
    out.write(f"{len(todo)} claims to label. Keys: s = SUPPORT (the evidence supports it), n = NOT, u = UNSURE, q = quit. "
              "The node's outcome is hidden so it can't bias you.\n")
    k = 0
    with open(os.path.join(log_dir, "labels.jsonl"), "a", encoding="utf-8") as f:
        for r in todo:
            out.write("\n" + "=" * 70 + f"\nCLAIM: {r['claim']}\nEVIDENCE:\n" + "\n".join(f"  - {e}" for e in r["evidence"]) + "\n> ")
            out.flush()
            ans = (inp.readline() or "q").strip().lower()[:1]
            if ans == "q":
                break
            lab = {"s": "SUPPORT", "n": "NOT", "u": "UNSURE"}.get(ans)
            if lab is None:
                out.write("skipped\n"); continue
            f.write(json.dumps({"id": r["id"], "label": lab, "ts": time.time(), "labeller": labeller}) + "\n"); k += 1
    out.write(f"\nlabelled {k}\n")
    return 0


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m roman_wheel_node.fresh")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("prereg", "label", "import-labels", "certify"):
        s = sub.add_parser(c); s.add_argument("--log-dir", default="rwn_shadow")
    sub.choices["prereg"].add_argument("--note", default="")
    sub.choices["label"].add_argument("--labeller", default=os.environ.get("USER", "human"))
    il = sub.choices["import-labels"]
    il.add_argument("--labels", required=True, help="JSONL with label in SUPPORT/NOT/UNSURE and either id or --key")
    il.add_argument("--key", default=None, help="match on fresh_claims meta[key] instead of id")
    il.add_argument("--labeller", default="import")
    ce = sub.choices["certify"]
    ce.add_argument("--by", default=None); ce.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "prereg":
        rs = load_jsonl(os.path.join(a.log_dir, "fresh_claims.jsonl"))
        rules = sorted({(r["policy"], r["profile"]) for r in rs})
        if len(rules) != 1:
            print(f"need exactly one (policy, profile) in the log, found {rules}", file=sys.stderr); return 2
        print(json.dumps(prereg(a.log_dir, rules[0][0], rules[0][1], note=a.note), indent=1)); return 0
    if a.cmd == "label":
        return _label_cli(a.log_dir, a.labeller)
    if a.cmd == "import-labels":
        idx = {}
        if a.key:
            for r in load_jsonl(os.path.join(a.log_dir, "fresh_claims.jsonl")):
                idx[str((r.get("meta") or {}).get(a.key))] = r["id"]
        n = 0
        with open(os.path.join(a.log_dir, "labels.jsonl"), "a", encoding="utf-8") as f:
            for r in load_jsonl(a.labels):
                cid = idx.get(str(r.get(a.key))) if a.key else r.get("id")
                if cid and r.get("label") in LABELS:
                    f.write(json.dumps({"id": cid, "label": r["label"], "ts": time.time(), "labeller": a.labeller}) + "\n"); n += 1
        print(f"imported {n} labels"); return 0
    res = certify(a.log_dir, by=a.by)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    print(report(res)); return 0


if __name__ == "__main__":
    sys.exit(_main())
