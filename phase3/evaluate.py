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
VERIFIERS = ("J11", "J12")     # Phase 3c strict proof verifiers (prompts V / W)
SELECTION_3C = os.path.join(ROOT, "results", "phase3", "p3c_selection.json")
MISSING_OK = {"dev:971": ("J5",)}
CALIBRATOR = os.environ.get("RWN_P3_CAL", "isotonic")     # isotonic (default) | platt
PARTIAL = os.environ.get("RWN_P3_PARTIAL") == "1"          # development only: skip claims without all responses
if os.environ.get("RWN_P3_JUDGES"):
    JUDGES = {k: v for k, v in JUDGES.items() if k in os.environ["RWN_P3_JUDGES"].split(",")}

OUT = os.path.join(ROOT, "results", "phase3")
LOCAL = ("nli", "nli2", "overlap", "rerank")
S_ONLY = ("overlap", "rerank")          # sources with no contradiction score     # nli = deberta-v3-small int8 ONNX; nli2 = deberta-v3-base (mnli-fever-anli)
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
            if lf.get("rerank_scores"):
                src["rerank"] = {"S": lf["rerank_scores"][0]}     # top cross-encoder relevance (evidence quality)
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
            # Phase 3c verifier proof: SUPPORT is its argmax AND its quote passes the same byte-exact gate
            verify = {n: max(("S", "C", "N"), key=lambda x: src[n][x]) == "S"
                      and quote_verbatim(quotes.get(n, ""), [corpus[d] for d in lf["docs"]], hashes)
                      for n in VERIFIERS if n in src}
            row["cond"][cond] = {"src": src, "a": int(shown_ok and quote_ok), "quote_ok": quote_ok, "verify": verify,
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
def node_run(train, dev, cond, sources, target, cal, variant, gate=True, live_online=False, log_dir=None,
             ltt_margin=None, veto=None):
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
    cut0 = thr.cutoff(f"bayes:{variant}")
    band = None if ltt_margin is None else ltt_interval(cut0, ltt_margin)
    trace = ClaimTrace(os.path.join(tmp, "audit.jsonl"), live=lw, mode="bayes", variant=variant, threshold=thr,
                       dependence=dep if variant in ("ess", "tan", "stack", "ess_cal") else None, not_sure_band=band)
    adm, effs, sts = [], [], []
    for r in dev:
        sc = scores(r)
        ns = False
        if ltt_margin is not None:                 # disagreement needs eff: evaluate without logging first
            e0, _, _ = roman_wheel_node(1, sc, live=lw, mode="bayes", variant=variant, dependence=dep)
            ns = disagree(r, cond, cal, sources, e0, cut0)
        a_in = r["cond"][cond]["a"] if gate else 1
        if veto:                                   # hard veto: ACCEPT needs the verifier's byte-exact proof (else gate closes)
            a_in = int(a_in == 1 and r["cond"][cond]["verify"][veto])
        d = trace.submit(r["claim"], sc, a=a_in, not_sure=ns)
        adm.append(d.admitted); effs.append(d.eff); sts.append(d.status)
        if live_online:
            give_feedback(sc, y(r), live=lw, threshold=thr, eff=d.eff, mode="bayes", variant=variant, dependence=dep)
    trace.verify()
    info = {"cutoff": thr.cutoff(f"bayes:{variant}"), "status": sts, "reliability": {s: round(lw.reliability(s), 4) for s in sources}}
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
            if t == "C" and s in S_ONLY:
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


def disagree(r, cond, cal, sources, eff, cut) -> bool:
    """Pre-registered disagreement rule (fixed before any dev result was seen):
    * judges split: at least 2 judges vote "supported" AND at least 2 vote "not supported" (calibrated P > 0.5), or
    * NLI contradicts the panel: the node leans "supported" but DeBERTa-v3-base NLI calibrated P(contradiction) > 0.5."""
    js = [s for s in sources if s.startswith("J")]
    v = [platt(cal[(cond, s, "S")], r["cond"][cond]["src"][s]["S"]) > 0.5 for s in js]
    split = sum(v) >= 2 and (len(v) - sum(v)) >= 2
    nli_c = ("nli2" in sources and eff >= cut and platt(cal[(cond, "nli2", "C")], r["cond"][cond]["src"]["nli2"]["C"]) > 0.5)
    return bool(split or nli_c)


def oof_effs(train, cond, sources, variant, folds: int = 5, veto=None):
    """Out-of-fold node posteriors, cutoffs and disagreement flags on the training subset
    (each fold's calibration and learners are fit on the other folds only)."""
    out, dis, cuts = [0.0] * len(train), [False] * len(train), [0.5] * len(train)
    for f in range(folds):
        tr = [r for i, r in enumerate(train) if i % folds != f]
        idx = [i for i in range(len(train)) if i % folds == f]
        cal = fit_cal(tr, cond, sources)
        _, effs, inf = node_run(tr, [train[i] for i in idx], cond, sources, "S", cal, variant, veto=veto)
        for i, e in zip(idx, effs):
            out[i] = e; cuts[i] = inf["cutoff"]
            dis[i] = effs[idx.index(i)] > 0.0 and train[i]["cond"][cond]["a"] == 1 and disagree(train[i], cond, cal, sources, e, inf["cutoff"])  # gate BLOCK is decided, as on dev
    return out, dis, cuts


def ltt_interval(cut: float, m: float) -> Tuple[float, float]:
    """NOT_SURE band for a log-odds margin m around the cutoff: [sig(lg(cut) - m), sig(lg(cut) + m)).
    m >= 50 means "decide nothing" (hi above any posterior)."""
    if m >= 50:
        return (0.0, 1.001)
    return (_sig(_lg(cut) - m), _sig(_lg(cut) + m))


def ltt_band(oof, ys, dis, cuts, alpha: float, delta: float = 0.05, K: int = 20):
    """Learn-Then-Test risk control (Bonferroni over a fixed grid) on train OOF posteriors.

    A claim is decided iff it shows no disagreement and its log-odds distance from its own fold's cutoff
    |lg(eff) - lg(cut)| >= m. The K candidate margins are fixed in advance as the distances that decide the top
    1/K, 2/K, ..., K/K of the non-disagreeing train claims. Each candidate's one-sided Clopper-Pearson upper bound on
    the decided-claim error rate is computed at level delta / K (Bonferroni), so with probability >= 1 - delta every
    certified candidate's true risk is <= alpha; we take the certified candidate with the widest coverage.
    Returns (m, n_decided, errors, upper_bound); m = 99 (abstain on everything) when none is certified."""
    z = [abs(_lg(o) - _lg(c)) for o, c in zip(oof, cuts)]
    zs = sorted((z[i] for i in range(len(z)) if not dis[i]), reverse=True)
    best = (99.0, 0, 0, 0.0)
    if not zs:
        return best
    for q in range(1, K + 1):
        m = zs[max(0, int(round(q * len(zs) / K)) - 1)]
        dec = [i for i in range(len(oof)) if not dis[i] and z[i] >= m]
        k = sum((oof[i] >= cuts[i]) != bool(ys[i]) for i in dec)
        ub = cp_upper(k, len(dec), delta / K)
        if ub <= alpha and len(dec) > best[1]:
            best = (round(m, 6), len(dec), k, ub)
    return best


def ltt_eval(train, dev, cond, sources, variant, cal, alphas=(0.10, 0.05, 0.02, 0.01, 0.005), veto=None) -> dict:
    """Fail-closed selective mode: LTT band + disagreement rule, applied to dev through ClaimTrace(NOT_SURE)."""
    oof, dis, cuts = oof_effs(train, cond, sources, variant, veto=veto)
    ys_tr = [r["label3"] == "SUPPORT" for r in train]
    out = {}
    for al in alphas:
        m, n_tr, k_tr, ub_tr = ltt_band(oof, ys_tr, dis, cuts, al)
        # dev: frozen node learned on all of train; NOT_SURE = band cutoff +/- m or disagreement (logged + replayed)
        adm, effs, inf = node_run(train, dev, cond, sources, "S", cal, variant, ltt_margin=m, veto=veto)
        cut = inf["cutoff"]
        status = ["NOT_SURE" if st == "NOT_SURE" else ("ACCEPT" if a_ else "REJECT") for st, a_ in zip(inf["status"], adm)]
        ys = [r["label3"] == "SUPPORT" for r in dev]
        dec = [i for i, st in enumerate(status) if st != "NOT_SURE"]
        errs = sum((status[i] == "ACCEPT") != ys[i] for i in dec)
        acc_i = [i for i in dec if status[i] == "ACCEPT"]
        out[f"{al}"] = {"margin_logodds": m, "band": list(ltt_interval(cut, m)), "train_oof": {"decided": n_tr, "errors": k_tr, "upper95": ub_tr,
                        "coverage": n_tr / len(train), "disagreements": sum(dis)},
                        "dev_coverage": len(dec) / len(dev), "dev_decided": len(dec), "dev_errors_decided": errs,
                        "dev_error_rate_upper95_clopper_pearson": cp_upper(errs, len(dec)),
                        "dev_accepted": len(acc_i), "dev_false_accepts": sum(1 for i in acc_i if not ys[i]),
                        "dev_not_sure_disagreement": sum(1 for r, e in zip(dev, effs) if disagree(r, cond, cal, sources, e, cut)),
                        "dev_status_counts": {k: status.count(k) for k in ("ACCEPT", "REJECT", "NOT_SURE")}}
    return out


def proof_or_not_sure(train, dev, cond, sources, variant, cal, vname) -> dict:
    """Phase 3c fail-closed option: the node may ACCEPT only with the verifier's byte-exact proof. A node-accept without
    proof becomes NOT_SURE (not proven is not disproven); node rejects stay REJECT. Train OOF and dev, same rule."""
    def stats(effs, cuts, rows):
        st = []
        for e, c, r in zip(effs, cuts, rows):
            if e >= c and e > 0.0:
                st.append("ACCEPT" if r["cond"][cond]["verify"][vname] else "NOT_SURE")
            else:
                st.append("REJECT")
        ys = [r["label3"] == "SUPPORT" for r in rows]
        dec = [i for i, x in enumerate(st) if x != "NOT_SURE"]
        err = sum((st[i] == "ACCEPT") != ys[i] for i in dec)
        acc = [i for i in dec if st[i] == "ACCEPT"]
        return {"coverage": len(dec) / len(rows), "decided": len(dec), "errors_decided": err,
                "error_rate_upper95_clopper_pearson": cp_upper(err, len(dec)), "accepted": len(acc),
                "false_accepts": sum(1 for i in acc if not ys[i]),
                "false_accept_upper95_of_accepted": cp_upper(sum(1 for i in acc if not ys[i]), len(acc)),
                "status_counts": {k: st.count(k) for k in ("ACCEPT", "REJECT", "NOT_SURE")}}
    oof, _, cuts = oof_effs(train, cond, sources, variant)
    _, effs, inf = node_run(train, dev, cond, sources, "S", cal, variant)
    return {"train_oof": stats(oof, cuts, train), "dev": stats(effs, [inf["cutoff"]] * len(dev), dev)}


def cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound on an error rate (k errors in n); k=0 gives 1-alpha**(1/n) (~3/n)."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(100):                       # P(X <= k | n, p) = alpha, decreasing in p
        mid = (lo + hi) / 2
        lp, lq = math.log(mid), math.log1p(-mid)   # log space: math.comb(n, i) overflows float for n in the thousands
        cdf = sum(math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
                  for i in range(k + 1))
        lo, hi = (mid, hi) if cdf > alpha else (lo, mid)
    return hi


def selective(train, dev, cond, sources, variant, dev_eff, ysup_dev, targets=(1.0, 0.99, 0.98, 0.97), veto=None) -> dict:
    """Selective mode: the ThresholdLearner picks an abstain band on train OOF posteriors; it is applied to dev unchanged."""
    oof, _, _ = oof_effs(train, cond, sources, variant, veto=veto)
    ys_tr = [r["label3"] == "SUPPORT" for r in train]
    tmp = tempfile.mkdtemp(prefix="rwn_p3band_")
    out = {}
    for tg in targets:
        lo, hi, cov_tr, acc_tr = band_exact(oof, ys_tr, tg, min_decided=10)
        dec = [i for i, e in enumerate(dev_eff) if e < lo or e >= hi]
        acc_i = [i for i in dec if dev_eff[i] >= hi]
        ok = [float((dev_eff[i] >= hi) == bool(ysup_dev[i])) for i in dec]
        rng = random.Random(5); n = len(dev_eff); bs = []
        for _ in range(B):
            smp = [rng.randrange(n) for _ in range(n)]
            d = [i for i in smp if dev_eff[i] < lo or dev_eff[i] >= hi]
            if d:
                bs.append(sum((dev_eff[i] >= hi) == bool(ysup_dev[i]) for i in d) / len(d))
        bs.sort()
        out[f"{tg:.2f}"] = {"band": [lo, hi], "train_oof_coverage": cov_tr, "train_oof_accuracy": acc_tr,
                            "dev_coverage": len(dec) / n, "dev_decided": len(dec),
                            "dev_accuracy_decided": (sum(ok) / len(ok)) if ok else None,
                            "dev_accuracy_ci": [bs[int(0.025 * len(bs))], bs[int(0.975 * len(bs)) - 1]] if bs else None,
                            "dev_errors_decided": int(len(ok) - sum(ok)),
                            "dev_error_rate_upper95_clopper_pearson": cp_upper(int(len(ok) - sum(ok)), len(ok)),
                            "dev_accepted": len(acc_i),
                            "dev_false_accepts": sum(1 for i in acc_i if not ysup_dev[i]),
                            "dev_false_accept_rate_of_accepted": (sum(1 for i in acc_i if not ysup_dev[i]) / len(acc_i)) if acc_i else None,
                            "dev_unsupported_accepted_of_all_unsupported": sum(1 for i in acc_i if not ysup_dev[i]) / max(1, sum(1 for y in ysup_dev if not y))}
    shutil.rmtree(tmp)
    return out


def band_exact(oof: List[float], ys: List[bool], target: float, min_decided: int = 10):
    """Exact (lo, hi) abstain band on train OOF posteriors: reject eff < lo, accept eff >= hi, abstain in between.
    Unlike ThresholdLearner.band's 0.02 grid this searches every distinct posterior, which matters because the node's
    posteriors pile up near 0 and 1. Widest coverage with decided accuracy >= target; ties -> higher accuracy.
    No qualifying band -> (0, 1.001) = abstain on everything."""
    n = len(oof)
    o = sorted(range(n), key=lambda i: oof[i])
    e = [oof[i] for i in o]
    pt = [0]
    for i in o:
        pt.append(pt[-1] + int(ys[i]))
    cuts = [c for c in range(n + 1) if c in (0, n) or e[c - 1] < e[c]]     # cut between distinct posteriors only
    val = lambda c: e[c] if c < n else 1.001
    best = (0.0, 1.001, 0.0, 1.0)
    for a in cuts:                                   # reject o[:a]
        r_ok = a - pt[a]
        for b in cuts:                               # accept o[b:]
            if b < a:
                continue
            d = a + (n - b)
            if d < max(1, min_decided):
                continue
            acc = (r_ok + pt[n] - pt[b]) / d
            if acc >= target and (d / n > best[2] + 1e-12 or (abs(d / n - best[2]) <= 1e-12 and acc > best[3])):
                best = (val(a) if a < n else 1.001, val(b), d / n, acc)
    return best


def risk_coverage(p: List[float], y: List[int], cut: float = 0.5) -> List[List[float]]:
    """Dev risk-coverage curve, ordering claims by confidence |p - cut| (for plotting; no selection is made on it)."""
    order = sorted(range(len(p)), key=lambda i: (-abs(p[i] - cut), i))
    pts, err = [], 0
    for c, i in enumerate(order, 1):
        err += int((p[i] >= cut) != bool(y[i]))
        pts.append([c / len(p), err / c])
    return pts


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
        sel3c = json.load(open(SELECTION_3C)) if os.path.exists(SELECTION_3C) else {}
        vname = sel3c.get("verifier") if sel3c.get("verifier") in judges else None
        veto = vname if (vname and sel3c.get("veto", {}).get(cond)) else None
        if vname:                                  # report both, whatever train CV chose for the headline
            a, e, _ = node_run(train, dev, cond, srcsets[sel_s], "S", cal, sel_v)
            pol[f"{sel_name}, no veto"], prob[f"{sel_name}, no veto"] = a, e
            a, e, _ = node_run(train, dev, cond, srcsets[sel_s], "S", cal, sel_v, veto=vname)
            pol[f"{sel_name} + {vname} veto"], prob[f"{sel_name} + {vname} veto"] = a, e
            a, e, _ = node_run(train, dev, cond, allsrc, "S", cal, "ess_cal", veto=vname)
            pol[f"bayes-ess_cal + {vname} veto"], prob[f"bayes-ess_cal + {vname} veto"] = a, e
            pol[f"{vname} proof alone (argmax S + verbatim quote)"] = [bool(r["cond"][cond]["verify"][vname]) for r in dev]
        if veto:
            sel_name = f"{sel_name} + {veto} veto"
        a, e, inf = node_run(train, dev, cond, srcsets[sel_s], "S", cal, sel_v, veto=veto)
        pol[sel_name], prob[sel_name], info["selected"] = a, e, dict(inf, variant=sel_v, sources=sel_s, cv_accuracy=cv, veto=veto)
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
            cadm, _, _ = node_run(train, dev, cond, [s for s in allsrc if s not in S_ONLY], "C", cal, v)
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
        # selective mode (abstain band chosen on train OOF posteriors) and risk-coverage curves
        sel_mode = selective(train, dev, cond, srcsets[sel_s], sel_v, prob[sel_name], ysup, veto=veto)
        ess_cal_mode = selective(train, dev, cond, allsrc, "ess_cal", prob["bayes-ess_cal"], ysup)
        p3c = {"proof_or_not_sure (selected config)": proof_or_not_sure(train, dev, cond, srcsets[sel_s], sel_v, cal, vname)} if vname else {}
        ltt = {"selected": ltt_eval(train, dev, cond, srcsets[sel_s], sel_v, cal, veto=veto),
               "bayes-ess_cal": ltt_eval(train, dev, cond, allsrc, "ess_cal", cal)}
        rc = {"node (selected)": risk_coverage(prob[sel_name], ysup, info["selected"]["cutoff"]),
              "bayes-ess_cal": risk_coverage(prob["bayes-ess_cal"], ysup, info["bayes-ess_cal"]["cutoff"]),
              f"{best_j} alone (calibrated)": risk_coverage(prob[f"{best_j} alone"], ysup)}
        # sensitivity: accuracy with the human-review label-noise candidates removed (NOT a headline number)
        lr_path = os.path.join(OUT if not PARTIAL else os.path.join(ROOT, "results", "phase3"), "label_review.json")
        noisy = {x["key"] for x in json.load(open(lr_path))["dev"]} if os.path.exists(lr_path) else set()
        keep = [i for i, k in enumerate(dev_keys) if k not in noisy]
        excl = {name: sum(pol[name][i] == bool(ysup[i]) for i in keep) / len(keep)
                for name in (sel_name, "bayes-ess_cal", f"{best_j} unwrapped", "J1 unwrapped (raw argmax, no node)")}
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
                        suspects.append({"key": k, "claim_id": int(k.split(":")[1]), "gold": r["label3"],   # IDs only (SciFact is non-commercial)
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
                                      "judge_train_accuracy": tr_acc, "selective": {"selected": sel_mode, "bayes-ess_cal": ess_cal_mode}, "ltt_not_sure": ltt, "p3c_proof_or_not_sure": p3c, "p3c_verifier": vname, "p3c_veto": veto,
                                      "risk_coverage_dev": rc, "accuracy_excluding_label_review": {"n_excluded": len(dev) - len(keep), "acc": excl}, "error_analysis": {k: {"n": len(v), "keys": v} for k, v in cats.items()},
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
