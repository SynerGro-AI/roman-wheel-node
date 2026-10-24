"""Phase 3e: stricter accept rules + asymmetric fixed-sequence LTT (pure functions, no I/O, no network).

A calibration item for the selective rule is a dict with
  side    "A" (node accepts), "R" (node rejects on its posterior) or "B" (exact-copy gate BLOCK: always decided, REJECT)
  z       |logit(eff) - logit(cutoff)|, the node's confidence
  acc_ok  the accept filter passes (only read when side == "A")
  rej_ok  the reject filter passes (only read when side == "R")
  y       gold is SUPPORT
A claim is decided iff it is a BLOCK, or an accept passing the accept filter with z >= m_a, or a reject passing the
reject filter with z >= m_r. Everything else is NOT_SURE (fails closed, never admitted).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

INF = float("inf")


def cp_upper(k: int, n: int, delta: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound (log space) on a binomial rate; n == 0 or k >= n gives 1."""
    if n == 0 or k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lp, lq = math.log(mid), math.log1p(-mid)
        cdf = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
                  for i in range(k + 1))
        lo, hi = (mid, hi) if cdf > delta else (lo, mid)
    return hi


def min_decided_for(alpha: float, delta: float) -> int:
    """Smallest n for which even 0 errors in n decided claims certifies risk <= alpha at level delta."""
    return math.ceil(math.log(delta) / math.log1p(-alpha))


def judge_votes(src: dict, judges: Sequence[str]) -> int:
    """Number of judges whose own top label is SUPPORT (raw outputs, no calibration)."""
    return sum(1 for n in judges if max(("S", "C", "N"), key=lambda x: src[n][x]) == "S")


def decide(item: dict, m_a: float, m_r: float) -> Optional[bool]:
    """True = ACCEPT, False = REJECT, None = NOT_SURE (or, for side "X", a BLOCK reported as its own outcome)."""
    if item["side"] == "B":
        return False
    if item["side"] == "X":
        return None
    if item["side"] == "A":
        return True if (item["acc_ok"] and item["z"] >= m_a) else None
    return False if (item["rej_ok"] and item["z"] >= m_r) else None


def _thr(zs_desc: List[float], frac: float) -> float:
    """z threshold that decides the top `frac` of a side (ties at the threshold are all decided); INF decides none."""
    n = int(round(frac * len(zs_desc)))
    return INF if n <= 0 else zs_desc[n - 1]


def path(items: Sequence[dict], rho: float, K: int) -> List[Tuple[float, float]]:
    """The pre-ordered sequence of (m_a, m_r): step q decides the top q/K of the filtered accepts and the top
    min(1, rho*q/K) of the filtered rejects. rho = 0 is accept-only (plus BLOCKs)."""
    za = sorted((it["z"] for it in items if it["side"] == "A" and it["acc_ok"]), reverse=True)
    zr = sorted((it["z"] for it in items if it["side"] == "R" and it["rej_ok"]), reverse=True)
    return [(_thr(za, q / K), _thr(zr, min(1.0, rho * q / K))) for q in range(1, K + 1)]


def risk(items: Sequence[dict], m_a: float, m_r: float, precision: bool = False) -> Tuple[int, int, int, int]:
    """(decided, errors, accepted, false accepts). precision=True: the risk is false accepts among accepted, so
    (decided, errors) are returned as (accepted, false accepts)."""
    n = k = na = fa = 0
    for it in items:
        d = decide(it, m_a, m_r)
        if d is None:
            continue
        n += 1; k += int(d != it["y"])
        if d:
            na += 1; fa += int(not it["y"])
    return (na, fa, na, fa) if precision else (n, k, na, fa)


def ltt_fixed_sequence(items: Sequence[dict], alpha: float, delta: float, rho: float, K: int = 40,
                       precision: bool = False) -> dict:
    """Learn-Then-Test with fixed-sequence testing along the pre-ordered path (most conservative first): test step q at
    level delta and stop at the first step whose Clopper-Pearson upper bound exceeds alpha. FWER <= delta for any
    ordering fixed before the data, so every passed step is certified. Steps that decide fewer claims than
    min_decided_for(alpha, delta) are skipped without spending a test: they cannot pass even with 0 errors, and the
    count of decided claims at a step does not use any label, so skipping keeps the sequence fixed in advance.
    Returns the last passed step, or an abstain-on-everything result (m_a = m_r = INF) when none passes."""
    best = {"certified": False, "m_a": INF, "m_r": INF, "step": 0}
    best.update(dict(zip(("decided", "errors", "accepted", "false_accepts"), risk(items, INF, INF, precision))))
    best["upper"] = cp_upper(best["errors"], best["decided"], delta)
    n_min = min_decided_for(alpha, delta)
    for q, (m_a, m_r) in enumerate(path(items, rho, K), 1):
        n, k, na, fa = risk(items, m_a, m_r, precision)
        if n < n_min:
            continue
        ub = cp_upper(k, n, delta)
        if ub > alpha:
            break
        best = {"certified": True, "m_a": m_a, "m_r": m_r, "step": q, "decided": n, "errors": k, "accepted": na,
                "false_accepts": fa, "upper": ub}
    return best


def certify_family(family: Dict[str, Sequence[dict]], alphas: Sequence[float], delta: float = 0.05,
                   rhos: Sequence[float] = (0.0, 0.5, 1.0, 2.0), K: int = 40, precision: bool = False) -> dict:
    """Bonferroni over every (config, rho) in the family (delta / (|family| * |rhos|)), fixed sequence along each path.
    Per alpha, the certified (config, rho) with the most decided calibration claims is chosen (valid: all were tested)."""
    m = len(family) * len(rhos)
    d_each = delta / m
    out = {"n_tests": m, "delta_each": d_each, "min_decided_zero_errors": {}, "alphas": {}, "all": {}}
    # smallest alpha any (config, rho) could certify (its first path step), with that step's calibration coverage
    tight = None
    for name, items in family.items():
        for rho in rhos:
            for m_a, m_r in path(items, rho, K):        # best single step (diagnostic only, not a certificate)
                n, k, _, _ = risk(items, m_a, m_r, precision)
                ub = cp_upper(k, n, d_each)
                if tight is None or ub < tight["upper"]:
                    tight = {"config": name, "rho": rho, "decided": n, "errors": k, "upper": ub, "n": len(items)}
    out["tightest_first_step"] = tight
    for al in alphas:
        out["min_decided_zero_errors"][f"{al}"] = min_decided_for(al, d_each)
        best = None
        for name, items in family.items():
            for rho in rhos:
                r = ltt_fixed_sequence(items, al, d_each, rho, K, precision)
                out["all"][f"{al}|{name}|{rho}"] = r
                if r["certified"] and (best is None or r["decided"] > best["decided"]):
                    best = dict(r, config=name, rho=rho, n=len(items))
        out["alphas"][f"{al}"] = best or {"certified": False}
    return out


# accept / reject filters (pre-specified family; dis = the Phase 3 pre-registered disagreement rule)
def make_rules(j11: bool) -> Dict[str, Tuple[Callable[[dict], bool], Callable[[dict], bool]]]:
    R = {
        "base":     (lambda f: not f["dis"], lambda f: not f["dis"]),
        "U5":       (lambda f: not f["dis"] and f["votes"] >= 5, lambda f: not f["dis"]),
        "U6":       (lambda f: not f["dis"] and f["votes"] == 6, lambda f: not f["dis"]),
        "U6+NLIe":  (lambda f: not f["dis"] and f["votes"] == 6 and f["nli2_entail"] >= 0.5, lambda f: not f["dis"]),
        "U6+noNLIc": (lambda f: not f["dis"] and f["votes"] == 6 and max(f["nli_contra"], f["nli2_contra"]) < 0.5,
                      lambda f: not f["dis"]),
        "U6/U0":    (lambda f: not f["dis"] and f["votes"] == 6, lambda f: not f["dis"] and f["votes"] == 0),
    }
    if j11:
        R["J11"] = (lambda f: not f["dis"] and bool(f["j11"]), lambda f: not f["dis"])
        R["J11+U6"] = (lambda f: not f["dis"] and bool(f["j11"]) and f["votes"] == 6, lambda f: not f["dis"])
    return R


def to_items(feats: Sequence[dict], effs: Sequence[float], cuts: Sequence[float], rule,
             blocks_decided: bool = True) -> List[dict]:
    """feats[i] carries y, blocked, dis, votes, nli*, j11; effs/cuts are the node posterior and its cutoff.
    blocks_decided=False reports an exact-copy BLOCK as its own fail-closed outcome (side "X") instead of a REJECT."""
    acc_f, rej_f = rule
    out = []
    for f, e, c in zip(feats, effs, cuts):
        if f["blocked"]:
            side, z = ("B" if blocks_decided else "X"), INF
        else:
            e2 = min(1 - 1e-9, max(1e-9, e)); c2 = min(1 - 1e-9, max(1e-9, c))
            side = "A" if e >= c else "R"
            z = abs(math.log(e2 / (1 - e2)) - math.log(c2 / (1 - c2)))
        out.append({"side": side, "z": z, "acc_ok": acc_f(f), "rej_ok": rej_f(f), "y": f["y"]})
    return out
