"""Stage 3: run SciFact claims through the Roman Wheel Node and report held-out dev metrics.

Protocol (nothing is fitted on dev):
1. Every source's raw score is calibrated on the training subset only (isotonic regression, Laplace-smoothed).
2. Reliabilities (LiveWeights), dependence (DependenceLearner) and the ACCEPT threshold (ThresholdLearner) are learned
   from the training subset only, then frozen.
3. Dev claims go through ``ClaimTrace`` (the gated, hash-chained trace) one at a time, and the audit log is verified by replay.
   The "live" policy additionally learns online on dev, prequentially: each claim is decided before its label is fed back.
4. 95% CIs come from a 10,000-resample bootstrap over dev claims. Paired tests are exact McNemar plus a paired bootstrap.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import tempfile
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from roman_wheel_node import DependenceLearner, LiveWeights, ThresholdLearner, give_feedback, roman_wheel_node  # noqa
from roman_wheel_node.trace import ClaimTrace                                         # noqa: E402
from factcheck.interfaces import XAIJudge, parse_judge                                 # noqa: E402
from factcheck.scifact import load_corpus, sentence_hashes, sha                        # noqa: E402
from features import CACHE, DATA, evidence_text                                        # noqa: E402
from judges import COND_FILES, CONDITIONS, JUDGES, _register_fewshot, load_cond_files, selected  # noqa: E402

# judge calls that timed out on every retry (180 s x 5); listed explicitly so nothing else can silently go missing
MISSING_OK = {"dev:971": ("J5",)}
CALIBRATOR = os.environ.get("RWN_P3_CAL", "isotonic")     # isotonic (default) | platt
PARTIAL = os.environ.get("RWN_P3_PARTIAL") == "1"          # development only: skip claims without all responses
if os.environ.get("RWN_P3_JUDGES"):
    JUDGES = {k: v for k, v in JUDGES.items() if k in os.environ["RWN_P3_JUDGES"].split(",")}

OUT = os.path.join(ROOT, "results", "phase3")
LOCAL = ("nli", "nli2", "overlap")     # nli = deberta-v3-small int8 ONNX; nli2 = deberta-v3-base (mnli-fever-anli)
B = int(os.environ.get("RWN_P3_BOOT", "10000"))
EPS = 0.01


# ---------------------------------------------------------------- data assembly
def quote_verbatim(q: str, docs, hashes) -> bool:
    """Exact-copy gate on a judge's quote: it may span several consecutive sentences or be a title, but every piece
    (split at sentence boundaries) must occur byte-for-byte inside one shown sentence or title (>= 20 chars in total),
    and that sentence's sha256 must be in the corpus hash set. A paraphrased or invented quote fails."""
    import re as _re
    if len(q) < 20:
        return False
    units = [u for d in docs for u in (d.title,) + d.sentences]
    for piece in [x for x in _re.split(r"(?<=[.!?])\s+", q.strip()) if x]:
        hit = [u for u in units if piece in u]
        if not hit or sha(hit[0]) not in hashes:
            return False
    return True



def load_rows() -> Dict[str, dict]:
    _register_fewshot()
    feats = json.load(open(os.path.join(CACHE, "local_features.json")))
    extra = load_cond_files()
    r5 = extra["retrieved5"]
    corpus = load_corpus(DATA)
    hashes = sentence_hashes(corpus)
    judges = {n: XAIJudge(c["model"], name=n, cache_path=os.path.join(CACHE, f"judge_{n}.jsonl"), style=c["style"],
                          temperature=c["temperature"], samples=c["samples"])
              for n, c in JUDGES.items()}
    rows = {}
    for k in selected(feats):
        f = feats[k]
        row = {"split": f["split"], "label3": f["label3"], "claim": f["claim"], "cited": f["cited"],
               "retrieved_docs": f["retrieved"]["docs"], "retrieved5_docs": r5.get(k, {}).get("docs", []), "cond": {}}
        for cond in CONDITIONS:
            if cond in COND_FILES and k not in extra[cond]:
                continue
            lf = extra[cond][k] if cond in COND_FILES else f[cond]
            ev = evidence_text(corpus, lf["docs"])
            src = {"nli": {"S": lf["nli_entail"], "C": lf["nli_contra"]}, "overlap": {"S": lf["overlap"]}}
            if "nli2_entail" in lf:
                src["nli2"] = {"S": lf["nli2_entail"], "C": lf["nli2_contra"]}
            quotes, missing = {}, []
            for n, j in judges.items():
                if cond not in JUDGES[n]["conditions"]:
                    continue
                if not j.cached(f["claim"], ev):
                    if n in MISSING_OK.get(k, ()):
                        # a call that repeatedly timed out: the judge abstains (uniform), and it is counted in the report
                        src[n] = {"S": 1 / 3, "C": 1 / 3, "N": 1 / 3, "unparsed": True, "missing": True}
                        quotes[n] = ""
                        continue
                    missing.append(n); continue
                p, q = j.judge(f["claim"], ev)
                src[n] = {"S": p["SUPPORT"], "C": p["CONTRADICT"], "N": p["NEI"], "unparsed": "_unparsed" in p}
                quotes[n] = q
            if missing and PARTIAL:
                break
            if missing:
                raise SystemExit(f"missing judge responses for {k} {cond}: {missing} (run judges.py --go)")
            # exact-copy gate: every evidence sentence shown is a corpus sentence (true by construction, checked),
            # and if J1 commits to SUPPORT/CONTRADICT its quoted sentence must exist byte-for-byte in the corpus.
            shown_ok = all(sha(s) in hashes for d in lf["docs"] for s in corpus[d].sentences)
            j1 = src["J1"]
            commits = max(("S", "C", "N"), key=lambda x: j1[x]) != "N"
            q = quotes.get("J1", "")
            # the quote must occur byte-for-byte inside one shown sentence (>= 20 chars), and that sentence's sha256
            # must be in the corpus set. A paraphrased, stitched or invented quote closes the gate.
            quote_ok = (not commits) or quote_verbatim(q, [corpus[d] for d in lf["docs"]], hashes)
            row["cond"][cond] = {"src": src, "a": int(shown_ok and quote_ok), "quote_ok": quote_ok,
                                 "j1_commits": commits, "quote_given": bool(q)}
        row["rv3_docs"] = extra["rv3"].get(k, {}).get("docs", [])
        if all(c in row["cond"] for c in CONDITIONS if extra.get(c, True)) or PARTIAL and row["cond"]:
            rows[k] = row
    return rows


# ---------------------------------------------------------------- calibration
def _lg(p: float) -> float:
    p = min(1 - 1e-4, max(1e-4, p))
    return math.log(p / (1 - p))


def _sig(z: float) -> float:
    if z >= 0:
        return 1 / (1 + math.exp(-z))
    e = math.exp(z)
    return e / (1 + e)


def platt_fit(xs: List[float], ys: List[int], l2: float = 1.0) -> Tuple[float, float]:
    """Logistic regression y ~ sigmoid(a * logit(x) + b) by Newton's method (deterministic)."""
    a, b = 1.0, 0.0
    z = [_lg(x) for x in xs]
    for _ in range(100):
        ga = gb = haa = hab = hbb = 0.0
        for zi, yi in zip(z, ys):
            p = _sig(a * zi + b)
            g = p - yi; w = p * (1 - p)
            ga += g * zi; gb += g; haa += w * zi * zi; hab += w * zi; hbb += w
        ga += l2 * a; haa += l2; hbb += 1e-9
        det = haa * hbb - hab * hab
        da = (hbb * ga - hab * gb) / det; db = (haa * gb - hab * ga) / det
        step = min(1.0, 5.0 / max(1e-12, abs(da) + abs(db)))       # damped Newton
        a, b = a - step * da, b - step * db
        if abs(da) + abs(db) < 1e-10:
            break
    return a, b


def iso_fit(xs: List[float], ys: List[int]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Isotonic (pool-adjacent-violators) calibration with Laplace smoothing (pos+1)/(n+2) per block.

    It suits LLM judges, whose outputs sit on a few discrete values (0, 0.9, 0.95, ...), where a single logistic slope
    is dragged around by the point mass at 0. Returns (upper x of each block, calibrated value of each block).
    """
    pts = sorted(zip(xs, ys))
    blocks: List[List[float]] = []                  # [x_max, pos, n]
    i = 0
    while i < len(pts):                             # tie groups first: equal x must share a value
        j = i; pos = 0
        while j < len(pts) and pts[j][0] == pts[i][0]:
            pos += pts[j][1]; j += 1
        blocks.append([pts[i][0], float(pos), float(j - i)])
        while len(blocks) > 1 and (blocks[-2][1] + 1) / (blocks[-2][2] + 2) >= (blocks[-1][1] + 1) / (blocks[-1][2] + 2):
            x, p_, n_ = blocks.pop(); blocks[-1][0] = x; blocks[-1][1] += p_; blocks[-1][2] += n_
        i = j
    return tuple(b[0] for b in blocks), tuple((b[1] + 1) / (b[2] + 2) for b in blocks)


def iso(m, x: float) -> float:
    xs, vs = m
    for xm, v in zip(xs, vs):
        if x <= xm:
            return min(1 - EPS, max(EPS, v))
    return min(1 - EPS, max(EPS, vs[-1]))


def platt(ab, x: float) -> float:
    """Apply a fitted calibrator: isotonic (tuple of tuples) or Platt (a, b)."""
    if isinstance(ab[0], tuple):
        return iso(ab, x)
    p = _sig(ab[0] * _lg(x) + ab[1])
    return min(1 - EPS, max(EPS, p))


# ---------------------------------------------------------------- stats
def boot_ci(vals: List[float], rng_seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(rng_seed); n = len(vals)
    ms = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(vals) / n, ms[int(0.025 * B)], ms[int(0.975 * B) - 1]


def paired_boot(a: List[float], b: List[float]) -> Tuple[float, float, float, float]:
    """Mean of a-b with a 95% CI and the two-sided bootstrap p-value of a-b = 0."""
    d = [x - y for x, y in zip(a, b)]
    m, lo, hi = boot_ci(d, 1)
    rng = random.Random(2); n = len(d)
    ms = [sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B)]
    p = 2 * min(sum(x <= 0 for x in ms), sum(x >= 0 for x in ms)) / B
    return m, lo, hi, min(1.0, p)


def mcnemar(a: List[bool], b: List[bool]) -> Tuple[int, int, float]:
    """Exact McNemar test on paired binary outcomes: discordant counts and the two-sided p-value."""
    n01 = sum(1 for x, y in zip(a, b) if x and not y)
    n10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, p)


def ece(probs: List[float], ys: List[int], bins: int = 10) -> float:
    tot = 0.0
    for i in range(bins):
        idx = [j for j, p in enumerate(probs) if (i / bins <= p < (i + 1) / bins) or (i == bins - 1 and p == 1.0)]
        if idx:
            tot += len(idx) / len(probs) * abs(sum(probs[j] for j in idx) / len(idx) - sum(ys[j] for j in idx) / len(idx))
    return tot


# ---------------------------------------------------------------- the node
def node_run(train, dev, cond, sources, target, cal, variant, gate=True, live_online=False, log_dir=None):
    """Learn on train (frozen), decide every dev claim through a ClaimTrace. Returns (admitted list, eff list, info)."""
    tmp = log_dir or tempfile.mkdtemp(prefix="rwn_p3_")
    y = (lambda r: r["label3"] == "SUPPORT") if target == "S" else (lambda r: r["label3"] == "CONTRADICT")

    def scores(r):
        return {s: platt(cal[(cond, s, target)], r["cond"][cond]["src"][s][target]) for s in sources}

    lw = LiveWeights(os.path.join(tmp, "weights.jsonl"), half_life=None)
    dep = DependenceLearner(os.path.join(tmp, "dependence.jsonl"), half_life=None)
    for r in train:
        give_feedback(scores(r), y(r), live=lw, dependence=dep)
    thr = ThresholdLearner(os.path.join(tmp, "threshold.jsonl"), half_life=None)
    for r in train:
        eff, _, _ = roman_wheel_node(1, scores(r), live=lw, mode="bayes", variant=variant, dependence=dep)
        thr.record(eff, y(r), f"bayes:{variant}")
    trace = ClaimTrace(os.path.join(tmp, "audit.jsonl"), live=lw, mode="bayes", variant=variant, threshold=thr,
                       dependence=dep if variant in ("ess", "tan", "stack", "ess_cal") else None)
    adm, effs = [], []
    for r in dev:
        sc = scores(r)
        d = trace.submit(r["claim"], sc, a=r["cond"][cond]["a"] if gate else 1)
        adm.append(d.admitted); effs.append(d.eff)
        if live_online:
            give_feedback(sc, y(r), live=lw, threshold=thr, eff=d.eff, mode="bayes", variant=variant, dependence=dep)
    trace.verify()
    info = {"cutoff": thr.cutoff(f"bayes:{variant}"), "reliability": {s: round(lw.reliability(s), 4) for s in sources}}
    if variant in ("ess", "ess_cal"):
        info["dependence"] = {f"{a}~{b}": [round(v, 4) for v in dep.dependence(a, b)]
                              for i, a in enumerate(sources) for b in sources[i + 1:]}
    if log_dir is None:
        shutil.rmtree(tmp)
    return adm, effs, info


def static_run(dev, cond, sources, target):
    """Static equal-trust node as shipped (v1 geomean, fixed bands, no learning) on the raw scores."""
    tmp = tempfile.mkdtemp(prefix="rwn_p3s_")
    trace = ClaimTrace(os.path.join(tmp, "audit.jsonl"), mode="geomean")
    adm, effs = [], []
    for r in dev:
        sc = {s: min(1 - EPS, max(EPS, r["cond"][cond]["src"][s][target])) for s in sources}
        d = trace.submit(r["claim"], sc, a=r["cond"][cond]["a"])
        adm.append(d.admitted); effs.append(d.eff)
    trace.verify(); shutil.rmtree(tmp)
    return adm, effs


def stack_reference(train, dev, cond, sources, target, cal):
    """Reference (not the node): a logistic regression over all calibrated logits fitted on train, at cutoff 0.5."""
    y = (lambda r: r["label3"] == "SUPPORT") if target == "S" else (lambda r: r["label3"] == "CONTRADICT")
    X = [[_lg(platt(cal[(cond, s, target)], r["cond"][cond]["src"][s][target])) for s in sources] + [1.0] for r in train]
    Y = [int(y(r)) for r in train]
    w = [0.0] * len(X[0])
    for it in range(3000):                       # plain gradient descent with L2; deterministic
        g = [0.0] * len(w)
        for x, t in zip(X, Y):
            p = _sig(sum(a * b for a, b in zip(w, x)))
            for i, xi in enumerate(x):
                g[i] += (p - t) * xi
        w = [wi - 0.5 * (gi / len(X) + 0.01 * wi * (i < len(w) - 1)) for i, (wi, gi) in enumerate(zip(w, g))]
    out = []
    for r in dev:
        x = [_lg(platt(cal[(cond, s, target)], r["cond"][cond]["src"][s][target])) for s in sources] + [1.0]
        out.append(_sig(sum(a * b for a, b in zip(w, x))))
    return out


def fit_cal(rows, cond, sources) -> dict:
    cal = {}
    for s in sources:
        for t in ("S", "C"):
            if t == "C" and s == "overlap":
                continue
            yy = [int(r["label3"] == ("SUPPORT" if t == "S" else "CONTRADICT")) for r in rows]
            cal[(cond, s, t)] = (iso_fit if CALIBRATOR == "isotonic" else platt_fit)(
                [r["cond"][cond]["src"][s][t] for r in rows], yy)
    return cal


CANDIDATES = [(v, srcname) for v in ("hard", "ess", "soft", "logit", "ess_cal") for srcname in ("all", "judges")]


def cv_select(train, cond, srcsets, folds: int = 5) -> Tuple[Tuple[str, str], Dict[str, float]]:
    """Choose the node configuration by 5-fold cross-validation on the TRAINING subset only."""
    scores = {}
    for v, sn in CANDIDATES:
        ok = 0
        for f in range(folds):
            tr = [r for i, r in enumerate(train) if i % folds != f]
            te = [r for i, r in enumerate(train) if i % folds == f]
            cal = fit_cal(tr, cond, srcsets["all"])
            adm, _, _ = node_run(tr, te, cond, srcsets[sn], "S", cal, v)
            ok += sum(a == (r["label3"] == "SUPPORT") for a, r in zip(adm, te))
        scores[f"bayes-{v} [{sn}]"] = ok / len(train)
    order = sorted(CANDIDATES, key=lambda c: (-scores[f"bayes-{c[0]} [{c[1]}]"], CANDIDATES.index(c)))
    return order[0], scores


# ---------------------------------------------------------------- main
def main() -> None:
    global OUT
    if PARTIAL:
        OUT = tempfile.mkdtemp(prefix="rwn_p3_partial_")
    os.makedirs(OUT, exist_ok=True)
    rows = load_rows()
    train = [r for k, r in sorted(rows.items()) if r["split"] == "train"]
    dev_keys = sorted(k for k in rows if rows[k]["split"] == "dev")
    dev = [rows[k] for k in dev_keys]
    ysup = [int(r["label3"] == "SUPPORT") for r in dev]
    unsup = [i for i, r in enumerate(dev) if r["label3"] != "SUPPORT"]
    judges_all = list(JUDGES)
    report = {"protocol": __doc__.strip(), "n_train": len(train), "n_dev": len(dev),
              "dev_label_counts": {l: sum(r["label3"] == l for r in dev) for l in ("SUPPORT", "CONTRADICT", "NEI")},
              "judges": {n: {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items() if k not in ("max_tokens", "out_tokens")} for n, c in JUDGES.items()},
              "conditions": {}}
    for cond in [c for c in CONDITIONS if c in dev[0]["cond"]]:
        judges = [j for j in judges_all if cond in JUDGES[j]["conditions"]]
        allsrc = [s for s in LOCAL if s in dev[0]["cond"][cond]["src"]] + judges
        cal = fit_cal(train, cond, allsrc)
        srcsets = {"all": allsrc, "judges": judges}
        (sel_v, sel_s), cv = cv_select(train, cond, srcsets)
        sel_name = f"node: train-CV selected = bayes-{sel_v} [{sel_s}]"
        pol: Dict[str, List[bool]] = {}
        prob: Dict[str, List[float]] = {}
        info: Dict[str, dict] = {}
        for s in allsrc:                                   # single sources, calibrated, cutoff 0.5
            prob[f"{s} alone"] = [platt(cal[(cond, s, "S")], r["cond"][cond]["src"][s]["S"]) for r in dev]
            pol[f"{s} alone"] = [p > 0.5 for p in prob[f"{s} alone"]]
        pol["J1 unwrapped (raw argmax, no node)"] = [max(("S", "C", "N"), key=lambda x: r["cond"][cond]["src"]["J1"][x]) == "S" for r in dev]
        pol["judge majority (raw argmax votes)"] = [sum(max(("S", "C", "N"), key=lambda x: r["cond"][cond]["src"][j][x]) == "S" for j in judges) >= 2 for r in dev]
        pol["static node (geomean, equal trust)"], effs = static_run(dev, cond, allsrc, "S"); prob["static node (geomean, equal trust)"] = effs
        for v, name in (("hard", "bayes-hard"), ("ess", "bayes-ess"), ("ess_cal", "bayes-ess_cal")):
            a, e, inf = node_run(train, dev, cond, allsrc, "S", cal, v)
            pol[name], prob[name], info[name] = a, e, inf
        a, e, inf = node_run(train, dev, cond, allsrc, "S", cal, "ess", live_online=True)
        pol["bayes-ess live (prequential)"], prob["bayes-ess live (prequential)"], info["bayes-ess live (prequential)"] = a, e, inf
        a, e, _ = node_run(train, dev, cond, allsrc, "S", cal, "ess", gate=False)
        pol["bayes-ess, gate off (ablation)"], prob["bayes-ess, gate off (ablation)"] = a, e
        a, e, _ = node_run(train, dev, cond, judges, "S", cal, "ess")
        pol["bayes-ess, judges only"], prob["bayes-ess, judges only"] = a, e
        a, e, _ = node_run(train, dev, cond, [s for s in allsrc if s not in judges], "S", cal, "ess")
        pol["bayes-ess, local only (no API)"], prob["bayes-ess, local only (no API)"] = a, e
        a, e, inf = node_run(train, dev, cond, srcsets[sel_s], "S", cal, sel_v)
        pol[sel_name], prob[sel_name], info["selected"] = a, e, dict(inf, variant=sel_v, sources=sel_s, cv_accuracy=cv)
        ref = stack_reference(train, dev, cond, allsrc, "S", cal)
        pol["reference: logistic stack (not gated)"], prob["reference: logistic stack (not gated)"] = [p > 0.5 for p in ref], ref
        # every judge unwrapped (raw argmax); the best single judge is chosen on TRAIN, then compared on dev
        am = lambda d: max(("S", "C", "N"), key=lambda x: d[x])
        tr_acc = {}
        for j in judges:
            pol[f"{j} unwrapped"] = [am(r["cond"][cond]["src"][j]) == "S" for r in dev]
            tr_acc[j] = sum((am(r["cond"][cond]["src"][j]) == "S") == (r["label3"] == "SUPPORT") for r in train) / len(train)
        best_j = max(judges, key=lambda j: (tr_acc[j], -judges.index(j)))
        # binary table
        tab = {}
        for name, adm in pol.items():
            corr = [float(x == bool(y)) for x, y in zip(adm, ysup)]
            m, lo, hi = boot_ci(corr)
            fa = [float(adm[i]) for i in unsup]
            sup = [i for i in range(len(dev)) if ysup[i]]
            tab[name] = {"accuracy": [m, lo, hi], "false_accept": list(boot_ci(fa)),
                         "true_accept": sum(adm[i] for i in sup) / len(sup),
                         "n_accepted": sum(adm),
                         "ece": ece(prob[name], ysup) if name in prob else None}
        # 3-way: S-node then C-node
        three = {}
        y3 = [r["label3"] for r in dev]

        def tri(sv, cv):
            return ["SUPPORT" if s else ("CONTRADICT" if c else "NEI") for s, c in zip(sv, cv)]
        for v, name in (("hard", "bayes-hard"), ("ess", "bayes-ess"), ("ess_cal", "bayes-ess_cal")):
            cadm, _, _ = node_run(train, dev, cond, [s for s in allsrc if s != "overlap"], "C", cal, v)
            three[name] = tri(pol[name], cadm)
        for j in judges:
            three[f"{j} unwrapped (raw argmax)"] = [{"S": "SUPPORT", "C": "CONTRADICT", "N": "NEI"}[max(("S", "C", "N"), key=lambda x: r["cond"][cond]["src"][j][x])] for r in dev]
        three["nli alone (calibrated)"] = tri(pol["nli alone"], [platt(cal[(cond, "nli", "C")], r["cond"][cond]["src"]["nli"]["C"]) > 0.5 for r in dev])
        tab3 = {}
        for name, pred in three.items():
            corr = [float(p == t) for p, t in zip(pred, y3)]
            f1s = []
            for l in ("SUPPORT", "CONTRADICT", "NEI"):
                tp = sum(p == l and t == l for p, t in zip(pred, y3)); fp = sum(p == l and t != l for p, t in zip(pred, y3))
                fn = sum(p != l and t == l for p, t in zip(pred, y3))
                f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
            tab3[name] = {"accuracy": list(boot_ci(corr)), "macro_f1": sum(f1s) / 3,
                          "confusion": {t: {p: sum(1 for a, b in zip(pred, y3) if a == p and b == t) for p in ("SUPPORT", "CONTRADICT", "NEI")} for t in ("SUPPORT", "CONTRADICT", "NEI")}}
        # safety: false accepts, unwrapped vs static vs node
        safety = {}
        base = "J1 unwrapped (raw argmax, no node)"
        for name in ("static node (geomean, equal trust)", "bayes-hard", "bayes-ess", "bayes-ess live (prequential)", "bayes-ess_cal", sel_name, "judge majority (raw argmax votes)"):
            fa_b = [pol[base][i] for i in unsup]; fa_n = [pol[name][i] for i in unsup]
            n01, n10, p = mcnemar(fa_b, fa_n)
            d = paired_boot([float(x) for x in fa_b], [float(x) for x in fa_n])
            ca = [float(pol[name][i] == bool(ysup[i])) for i in range(len(dev))]
            cb = [float(pol[base][i] == bool(ysup[i])) for i in range(len(dev))]
            acc_d = paired_boot(ca, cb)
            _, _, p_acc = mcnemar([x > 0 for x in ca], [x > 0 for x in cb])
            safety[name] = {"false_accepts_baseline": sum(fa_b), "false_accepts_policy": sum(fa_n), "n_unsupported": len(unsup),
                            "only_baseline_false_accepts": n01, "only_policy_false_accepts": n10, "mcnemar_p": p,
                            "fa_drop_paired_boot": d, "accuracy_gain_vs_baseline": acc_d, "accuracy_mcnemar_p": p_acc}
        # does the node beat the best single judge (chosen on train)? paired on dev
        vs_best = {}
        ysb = [bool(y) for y in ysup]
        for name in (sel_name, "bayes-ess", "bayes-ess_cal", "bayes-hard"):
            cn = [float(a == y) for a, y in zip(pol[name], ysb)]; cj = [float(a == y) for a, y in zip(pol[f"{best_j} unwrapped"], ysb)]
            n01, n10, p = mcnemar([x > 0 for x in cn], [x > 0 for x in cj])
            vs_best[name] = {"judge": best_j, "judge_train_acc": tr_acc[best_j], "node_acc": sum(cn) / len(cn),
                             "judge_acc": sum(cj) / len(cj), "diff_paired_boot": paired_boot(cn, cj),
                             "node_only_right": n01, "judge_only_right": n10, "mcnemar_p": p}
        # error analysis of the selected node on dev
        sel_adm = pol[sel_name]
        cats = {"retrieval_miss": [], "nei_accepted": [], "contradict_accepted": [], "support_rejected_gate": [],
                "support_rejected_panel_consensus": [], "support_rejected_split_panel": [], "panel_unanimous_vs_gold": []}
        docs_key = {"retrieved": "retrieved_docs", "retrieved5": "retrieved5_docs", "rv3": "rv3_docs"}.get(cond)
        for i, r in enumerate(dev):
            y = r["label3"] == "SUPPORT"
            votes = [am(r["cond"][cond]["src"][j]) for j in judges]
            if sel_adm[i] == y:
                continue
            k = dev_keys[i]
            if docs_key and r["label3"] != "NEI" and not any(d in r[docs_key] for d in r["cited"]):
                cats["retrieval_miss"].append(k)
            elif not y and r["label3"] == "NEI":
                cats["nei_accepted"].append(k)
            elif not y:
                cats["contradict_accepted"].append(k)
            elif r["cond"][cond]["a"] == 0:
                cats["support_rejected_gate"].append(k)
            elif sum(v == "S" for v in votes) <= len(votes) // 3:
                cats["support_rejected_panel_consensus"].append(k)
            else:
                cats["support_rejected_split_panel"].append(k)
            if len(set(votes)) == 1 and (votes[0] == "S") != y:
                cats["panel_unanimous_vs_gold"].append(k)
        # candidate label errors: every judge's argmax disagrees with gold, each with probability >= 0.8 (gold evidence)
        suspects = []
        if cond == "gold":
            for split_rows, split_keys in ((dev, dev_keys), (train, sorted(k for k in rows if rows[k]["split"] == "train"))):
                for k, r in zip(split_keys, split_rows):
                    gl = {"SUPPORT": "S", "CONTRADICT": "C", "NEI": "N"}[r["label3"]]
                    ps = [r["cond"][cond]["src"][j] for j in judges]
                    tops = [am(p) for p in ps]
                    if len(set(tops)) == 1 and tops[0] != gl and all(p[tops[0]] >= 0.8 for p in ps):
                        suspects.append({"key": k, "claim": r["claim"], "gold": r["label3"],
                                         "panel": {"S": "SUPPORT", "C": "CONTRADICT", "N": "NEI"}[tops[0]],
                                         "cited": r["cited"]})
        # judge correlation (dev, calibrated logits; and error overlap)
        corrm, errov = {}, {}
        lg = {s: [_lg(platt(cal[(cond, s, "S")], r["cond"][cond]["src"][s]["S"])) for r in dev] for s in allsrc}
        wrong = {s: [(p > 0.5) != bool(y) for p, y in zip(prob[f"{s} alone"], ysup)] for s in allsrc}
        for i, a in enumerate(allsrc):
            for b in allsrc[i + 1:]:
                xa, xb = lg[a], lg[b]; ma, mb = sum(xa) / len(xa), sum(xb) / len(xb)
                cov = sum((x - ma) * (y - mb) for x, y in zip(xa, xb))
                corrm[f"{a}~{b}"] = cov / math.sqrt(sum((x - ma) ** 2 for x in xa) * sum((y - mb) ** 2 for y in xb))
                both = sum(x and y for x, y in zip(wrong[a], wrong[b])); ea = sum(wrong[a]); eb = sum(wrong[b])
                n = len(dev); pa, pb = ea / n, eb / n
                phi = (both / n - pa * pb) / math.sqrt(pa * (1 - pa) * pb * (1 - pb)) if 0 < pa < 1 and 0 < pb < 1 else 0.0
                errov[f"{a}~{b}"] = {"both_wrong": both, "a_wrong": ea, "b_wrong": eb, "phi_errors": phi}
        gate = {"gate_closed_dev": sum(1 - r["cond"][cond]["a"] for r in dev),
                "gate_closed_dev_supported": sum(1 - r["cond"][cond]["a"] for r in dev if r["label3"] == "SUPPORT"),
                "j1_commits_without_quote": sum(r["cond"][cond]["j1_commits"] and not r["cond"][cond]["quote_given"] for r in dev),
                "gate_closed_train": sum(1 - r["cond"][cond]["a"] for r in train),
                "unparsed_judge_replies": sum(r["cond"][cond]["src"][j]["unparsed"] for r in dev + train for j in judges),
                "missing_judge_calls": sum(bool(r["cond"][cond]["src"][j].get("missing")) for r in dev + train for j in judges)}
        report["conditions"][cond] = {"selected": sel_name, "cv_train": cv, "binary": tab, "vs_best_single_judge": vs_best,
                                      "judge_train_accuracy": tr_acc, "error_analysis": {k: {"n": len(v), "keys": v} for k, v in cats.items()},
                                      "label_noise_candidates": suspects, "three_way": tab3, "safety_vs_unwrapped_J1": safety,
                                      "score_correlation_dev": corrm, "error_overlap_dev": errov, "gate": gate,
                                      "node_info": info,
                                      "calibration": {f"{s}:{t}": [list(x) if isinstance(x, tuple) else x for x in v] for (c, s, t), v in cal.items()}}
        report["conditions"][cond]["_per_claim"] = {"keys": dev_keys, "label3": y3,
                                                     "pred": {k: [int(x) for x in v] for k, v in pol.items()},
                                                     "pred3": three}
    if cond == "retrieved":
        pass
    report["retrieval"] = {"recall_at_3_supported_or_contradicted": None}
    ev_rows = [r for r in dev if r["label3"] != "NEI"]
    report["retrieval"]["recall_at_3_supported_or_contradicted"] = sum(any(d in r["retrieved_docs"] for d in r["cited"]) for r in ev_rows) / len(ev_rows)
    if any(r["rv3_docs"] for r in ev_rows):
        report["retrieval"]["recall_rv3"] = sum(any(d in r["rv3_docs"] for d in r["cited"]) for r in ev_rows) / len(ev_rows)
        report["retrieval"]["mean_docs_rv3"] = sum(len(r["rv3_docs"]) for r in dev) / len(dev)
    report["retrieval"]["recall_at_5_reranked"] = sum(any(d in r["retrieved5_docs"] for d in r["cited"]) for r in ev_rows) / len(ev_rows)
    json.dump(report, open(os.path.join(OUT, "scifact.json"), "w"), indent=1, sort_keys=True)
    print("wrote", os.path.join(OUT, "scifact.json"))


if __name__ == "__main__":
    main()
