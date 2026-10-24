"""Benchmark runner: policies x standard scenarios x seeds, with CIs, calibration and head-to-heads.

Everything here is deterministic: the same arguments produce byte-identical JSON and text output.
(Charts are excluded, since PNG bytes depend on the matplotlib version.)
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
import tempfile
from multiprocessing import Pool
from typing import Dict, List, Optional, Sequence, Tuple

from ..core import LiveWeights, ThresholdLearner, give_feedback, roman_wheel_node
from ..dependence import DependenceLearner
from .generator import ALL_SCENARIOS, SCENARIOS, stream

BENCH_VERSION = 1
ECE_BINS = 10
TRACK_EVERY = 25

# policy -> (live ledger, node kwargs, learned threshold)
POLICIES: Dict[str, Tuple[Optional[str], dict, bool]] = {
    "static-geomean": (None, {}, False),
    "static-product": (None, {"mode": "product"}, False),
    "live": ("none", {}, False),
    "live+flip": ("none", {"flip": True}, False),
    "live+flip+decay": ("150", {"flip": True}, False),
    "live+flip+decay-product": ("150", {"flip": True, "mode": "product"}, False),
    "flip+decay+thr": ("150", {"flip": True}, True),
    "flip+decay+thr-product": ("150", {"flip": True, "mode": "product"}, True),
    "bayes-hard": ("150", {"mode": "bayes", "variant": "hard"}, False),
    "bayes-soft": ("150", {"mode": "bayes", "variant": "soft"}, False),
    "bayes-logit": ("150", {"mode": "bayes", "variant": "logit"}, False),
    "bayes-hard+thr": ("150", {"mode": "bayes", "variant": "hard"}, True),
    "oracle": ("oracle", {}, False),
}
HEAD_TO_HEAD = [
    ("static-geomean", "static-product"),
    ("live+flip+decay", "live+flip+decay-product"),
    ("flip+decay+thr", "flip+decay+thr-product"),
]
CALIBRATED = ["bayes-hard", "bayes-soft", "live+flip+decay", "oracle"]  # eff calibration (ECE) reported for these

# ---- dependence suite (correlation-aware combiners); the standard suite above is frozen -------------
# live ledger "dep" = DependenceLearner(half_life=150); the variant decides which combiner reads it.
DEP_POLICIES: Dict[str, Tuple[Optional[str], dict, bool]] = {
    "bayes-hard": ("150", {"mode": "bayes", "variant": "hard"}, False),
    "bayes-hard+thr": ("150", {"mode": "bayes", "variant": "hard"}, True),
    "bayes-tan": ("dep", {"mode": "bayes", "variant": "tan"}, False),
    "bayes-ess": ("dep", {"mode": "bayes", "variant": "ess"}, False),
    "bayes-stack": ("dep", {"mode": "bayes", "variant": "stack"}, False),
    "bayes-tan+thr": ("dep", {"mode": "bayes", "variant": "tan"}, True),
    "oracle": ("oracle", {}, False),
}
DEP_HEAD_TO_HEAD = [
    ("bayes-tan", "bayes-hard"),
    ("bayes-ess", "bayes-hard"),
    ("bayes-stack", "bayes-hard"),
    ("bayes-tan", "bayes-ess"),
]
DEP_CALIBRATED = ["bayes-hard", "bayes-tan", "bayes-ess", "bayes-stack", "oracle"]
SUITES = {
    "standard": dict(
        policies=POLICIES, h2h=HEAD_TO_HEAD, calibrated=CALIBRATED, scenarios=SCENARIOS, labels=("geomean", "product")
    ),
    "dependence": dict(
        policies=DEP_POLICIES,
        h2h=DEP_HEAD_TO_HEAD,
        calibrated=DEP_CALIBRATED,
        scenarios=ALL_SCENARIOS,
        labels=("a", "b"),
    ),
}

# two-sided 95% Student-t critical values
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def t95(df: int) -> float:
    if df in _T95:
        return _T95[df]
    for k in sorted(_T95):
        if k > df:
            return _T95[k]
    return 1.960


def R(x: float) -> float:
    """Round for output (fixed precision keeps the JSON byte-stable)."""
    return round(float(x), 6)


def ci(xs: Sequence[float]) -> dict:
    m = st.fmean(xs)
    h = t95(len(xs) - 1) * st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
    return {"mean": R(m), "ci95": R(h), "per_seed": [R(x) for x in xs]}


def _rates(acc: List[bool], truth: List[bool]) -> dict:
    n_t = sum(truth)
    n_f = len(truth) - n_t
    return {
        "accuracy": sum(a == t for a, t in zip(acc, truth)) / len(truth),
        "true_accept": sum(a and t for a, t in zip(acc, truth)) / max(1, n_t),
        "false_block": sum((not a) and (not t) for a, t in zip(acc, truth)) / max(1, n_f),
    }


def _calib(effs: List[float], truth: List[bool]) -> Tuple[float, float, List[List[float]]]:
    """ECE (10 equal-width bins), Brier score, and per-bin [n, sum_eff, sum_true]."""
    bins = [[0, 0.0, 0] for _ in range(ECE_BINS)]
    for e, t in zip(effs, truth):
        b = bins[min(ECE_BINS - 1, int(e * ECE_BINS))]
        b[0] += 1
        b[1] += e
        b[2] += int(t)
    n = len(effs)
    ece = sum(abs(b[1] - b[2]) for b in bins if b[0]) / n
    brier = sum((e - int(t)) ** 2 for e, t in zip(effs, truth)) / n
    return ece, brier, bins


def run_one(key: str, seed: int, suite: str = "standard") -> dict:
    """Run every policy of ``suite`` on one (scenario, seed) stream. Returns raw per-run metrics."""
    cfg = SUITES[suite]
    POLICIES, CALIBRATED = cfg["policies"], cfg["calibrated"]
    sc = cfg["scenarios"][key]
    with tempfile.TemporaryDirectory() as tmp:
        lws = {
            "none": LiveWeights(os.path.join(tmp, "w_none.jsonl"), half_life=None),
            "150": LiveWeights(os.path.join(tmp, "w_150.jsonl"), half_life=150),
        }
        dep = DependenceLearner(os.path.join(tmp, "dep.jsonl"), half_life=150) if suite == "dependence" else None
        tl = ThresholdLearner(os.path.join(tmp, "thr.jsonl"), half_life=500)
        acc = {p: [] for p in POLICIES}
        effs = {p: [] for p in CALIBRATED}
        truth: List[bool] = []
        mae = {"150": [], "none": []}
        track = []
        for c in stream(sc, seed):
            truth.append(c.truth)
            for h, lw in lws.items():
                mae[h].append(st.fmean(abs(lw.reliability(k) - a) for k, a in c.true_accuracy.items()))
            if seed == 0 and c.i % TRACK_EVERY == 0:
                track.append(
                    {
                        "i": c.i,
                        "learned": {k: R(lws["150"].reliability(k)) for k in c.scores},
                        "true": {k: R(v) for k, v in c.true_accuracy.items()},
                    }
                )
            step_eff = {}
            for p, (lw, kw, thr) in POLICIES.items():
                if lw == "oracle":
                    z = max(-700.0, min(700.0, c.oracle_llr))
                    eff = 1.0 / (1.0 + math.exp(-z))
                    accepted = c.oracle_llr > 0
                elif lw == "dep":
                    eff, _, s = roman_wheel_node(1, c.scores, None, threshold=tl if thr else None, dependence=dep, **kw)
                    accepted = s == "ACCEPT"
                else:
                    eff, _, s = roman_wheel_node(1, c.scores, lws.get(lw), threshold=tl if thr else None, **kw)
                    accepted = s == "ACCEPT"
                step_eff[p] = eff
                acc[p].append(accepted)
                if p in effs:
                    effs[p].append(eff)
            for p, (lw, kw, thr) in POLICIES.items():
                if thr:
                    give_feedback(
                        c.scores,
                        c.truth,
                        threshold=tl,
                        eff=step_eff[p],
                        mode=kw.get("mode", "geomean"),
                        variant=kw.get("variant", "hard"),
                    )
            for lw in lws.values():
                give_feedback(c.scores, c.truth, live=lw)
            if dep is not None:
                give_feedback(c.scores, c.truth, dependence=dep)
    half = sc.n // 2
    out = {
        "policies": {},
        "reliability_mae": {h: {"mean_over_run": st.fmean(v), "final": v[-1]} for h, v in mae.items()},
        "eff_calibration": {},
        "tracking": track,
    }
    for p in POLICIES:
        m = _rates(acc[p], truth)
        m["train_accuracy"] = _rates(acc[p][:half], truth[:half])["accuracy"]
        m["test_accuracy"] = _rates(acc[p][half:], truth[half:])["accuracy"]
        if sc.sources[0].drift:
            k = sc.sources[0].drift[0]
            m["post_drift_accuracy"] = _rates(acc[p][k:], truth[k:])["accuracy"]
        sw = [g.switch[0] for g in sc.groups if g.switch]
        if sw:
            m["post_switch_accuracy"] = _rates(acc[p][sw[0] :], truth[sw[0] :])["accuracy"]
        out["policies"][p] = m
    for p in CALIBRATED:
        ece, brier, bins = _calib(effs[p], truth)
        out["eff_calibration"][p] = {"ece": ece, "brier": brier, "bins": bins}
    return out


def _job(args):
    return args[:2], run_one(*args)


def ceiling(key: str, n_claims: int = 200_000, seed: int = 12345) -> float:
    """Large-sample Bayes-optimal accuracy over ceil(n_claims / n) complete streams (seeds 12345, 12346, ...),
    so time-varying scenarios (drift) are weighted exactly like a real run."""
    sc = ALL_SCENARIOS[key]
    k = max(1, math.ceil(n_claims / sc.n))
    ok = sum((c.oracle_llr > 0) == c.truth for j in range(k) for c in stream(sc, seed + j))
    return ok / (k * sc.n)


def run(
    seeds: int = 20,
    scenarios: Optional[Sequence[str]] = None,
    ceiling_claims: int = 200_000,
    jobs: Optional[int] = None,
    suite: str = "standard",
) -> dict:
    cfg = SUITES[suite]
    POLICIES, HEAD_TO_HEAD, CALIBRATED = cfg["policies"], cfg["h2h"], cfg["calibrated"]
    la, lb = cfg["labels"]
    keys = list(scenarios or cfg["scenarios"])
    tasks = [(k, s, suite) for k in keys for s in range(seeds)]
    jobs = jobs or os.cpu_count() or 1
    if jobs > 1:
        with Pool(min(jobs, len(tasks))) as pool:
            raw = dict(pool.map(_job, tasks))
            ceils = dict(zip(keys, pool.starmap(ceiling, [(k, ceiling_claims) for k in keys])))
    else:
        raw = dict(map(_job, tasks))
        ceils = {k: ceiling(k, ceiling_claims) for k in keys}
    res = {
        "bench_version": BENCH_VERSION,
        "config": {
            "seeds": seeds,
            "seed_values": list(range(seeds)),
            "ceiling_claims": ceiling_claims,
            "ceiling_seed": 12345,
            "ece_bins": ECE_BINS,
            "policies": {
                p: {"live": lw, "kwargs": kw, "learned_threshold": thr} for p, (lw, kw, thr) in POLICIES.items()
            },
        },
        "scenarios": {},
    }
    if suite != "standard":
        res["config"]["suite"] = suite
    for k in keys:
        runs = [raw[(k, s)] for s in range(seeds)]
        d = {
            "spec": ALL_SCENARIOS[k].spec(),
            "ceiling": R(ceils[k]),
            "policies": {},
            "head_to_head": [],
            "reliability_mae": {},
            "eff_calibration": {},
            "tracking_seed0": runs[0]["tracking"],
        }
        for p in POLICIES:
            d["policies"][p] = {m: ci([r["policies"][p][m] for r in runs]) for m in runs[0]["policies"][p]}
        for g, pr in HEAD_TO_HEAD:
            diffs = [r["policies"][g]["accuracy"] - r["policies"][pr]["accuracy"] for r in runs]
            d["head_to_head"].append(
                {
                    la: g,
                    lb: pr,
                    f"{la}_minus_{lb}": ci(diffs),
                    f"{la}_wins": sum(x > 0 for x in diffs),
                    "ties": sum(x == 0 for x in diffs),
                    f"{lb}_wins": sum(x < 0 for x in diffs),
                }
            )
        for h in ("150", "none"):
            d["reliability_mae"][h] = {
                m: ci([r["reliability_mae"][h][m] for r in runs]) for m in ("mean_over_run", "final")
            }
        for p in CALIBRATED:
            bins = [
                [sum(r["eff_calibration"][p]["bins"][b][j] for r in runs) for j in range(3)] for b in range(ECE_BINS)
            ]
            d["eff_calibration"][p] = {
                "ece": ci([r["eff_calibration"][p]["ece"] for r in runs]),
                "brier": ci([r["eff_calibration"][p]["brier"] for r in runs]),
                "pooled_bins": [[b[0], R(b[1]), b[2]] for b in bins],
            }
        res["scenarios"][k] = d
    return res


def to_json(res: dict) -> str:
    return json.dumps(res, sort_keys=True, indent=1) + "\n"


def to_text(res: dict) -> str:
    suite = res["config"].get("suite", "standard")
    L = [
        f"Roman Wheel Node benchmark v{res['bench_version']}{'' if suite == 'standard' else ' [' + suite + ' suite]'} "
        f" |  {res['config']['seeds']} seeds "
        f"(0..{res['config']['seeds'] - 1})  |  mean +/- 95% CI (Student t)",
        "",
    ]

    def pct(x):
        return f"{100 * x['mean']:5.1f} +/-{100 * x['ci95']:4.1f}"

    for k, d in res["scenarios"].items():
        L.append(
            f"== {d['spec']['title']}  [{k}]  n={d['spec']['n']}  Bayes-optimal ceiling "
            f"({res['config']['ceiling_claims']} claims): {100 * d['ceiling']:.1f}%"
        )
        L.append(f"  {'policy':24} {'accuracy':>13} {'true-accept':>13} {'false-block':>13} {'train':>7} {'test':>7}")
        for p, m in d["policies"].items():
            line = (
                f"  {p:24} {pct(m['accuracy']):>13} {pct(m['true_accept']):>13} {pct(m['false_block']):>13} "
                f"{100 * m['train_accuracy']['mean']:6.1f}% {100 * m['test_accuracy']['mean']:6.1f}%"
            )
            if "post_drift_accuracy" in m:
                line += f"  post-drift {100 * m['post_drift_accuracy']['mean']:5.1f}%"
            if "post_switch_accuracy" in m:
                line += f"  post-switch {100 * m['post_switch_accuracy']['mean']:5.1f}%"
            L.append(line)
        if res["config"].get("suite", "standard") == "standard":
            L.append("  product vs geomean (paired, same streams): accuracy difference geomean - product")
            for h in d["head_to_head"]:
                x = h["geomean_minus_product"]
                L.append(
                    f"    {h['geomean']:18} vs {h['product']:26} {100 * x['mean']:+6.2f} +/-{100 * x['ci95']:4.2f} pts"
                    f"   wins g/tie/p = {h['geomean_wins']}/{h['ties']}/{h['product_wins']}"
                )
        else:
            L.append("  paired comparisons (same streams): accuracy difference a - b")
            for h in d["head_to_head"]:
                x = h["a_minus_b"]
                L.append(
                    f"    {h['a']:18} vs {h['b']:18} {100 * x['mean']:+6.2f} +/-{100 * x['ci95']:4.2f} pts"
                    f"   wins a/tie/b = {h['a_wins']}/{h['ties']}/{h['b_wins']}"
                )
        r = d["reliability_mae"]
        L.append(
            f"  learned-reliability error |r - true accuracy| (mean over sources):"
            f" half_life=150: run-avg {r['150']['mean_over_run']['mean']:.3f}, final {r['150']['final']['mean']:.3f};"
            f" no decay: run-avg {r['none']['mean_over_run']['mean']:.3f}, final {r['none']['final']['mean']:.3f}"
        )
        L.append(
            "  eff calibration: "
            + "; ".join(
                f"{p} ECE {c['ece']['mean']:.3f} +/-{c['ece']['ci95']:.3f} Brier {c['brier']['mean']:.3f}"
                for p, c in d["eff_calibration"].items()
            )
        )
        L.append("")
    return "\n".join(L)
