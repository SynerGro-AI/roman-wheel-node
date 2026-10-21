"""Scenario benchmark for Roman Wheel Node (v3).

Every scenario is run with 5 seeds. In each run, claims arrive one at a time; every policy
decides on the claim using only what it has learned so far, then the truth is revealed and fed
back (append-only ledgers). Metrics per policy:

* accuracy      - right calls / all claims (ACCEPT on true, not-ACCEPT on false; HEDGE counts as not accepted)
* true-accept   - share of TRUE claims accepted   (sensitivity)
* false-block   - share of FALSE claims not accepted (specificity)
* train / test  - accuracy on the first vs second half of each run (online: weights keep learning)

``oracle`` is the Bayes-optimal ceiling: it is told each source's true accuracy (and, for the
copycat scenario, the copying structure) and the exact score distributions, and accepts when
P(true | scores) > 0.5. No policy can beat it in expectation.

The first five scenarios are unchanged from v2. ``copycats`` is new and deliberately hard.
Run: pip install -e .[bench] && python benchmarks/run_benchmark.py
(writes suite_results.json and suite_results.png in the current directory)
"""
from __future__ import annotations

import json, math, os, random, statistics as st, sys, tempfile
from multiprocessing import Pool
from roman_wheel_node import roman_wheel_node, LiveWeights, ThresholdLearner, give_feedback

SEEDS = range(5)
T975_DF4 = 2.776                                # Student t, 95% two-sided, 4 dof (5 seeds)


def score(acc, truth, rng):
    """Unchanged from v2: source says the right thing w.p. acc; score ~ Beta(8,2) if it says
    'true', Beta(2,8) if it says 'false'; floored at 0.01 (exact 0 would be a hard veto)."""
    says_true = truth if rng.random() < acc else not truth
    return max(0.01, rng.betavariate(8, 2) if says_true else rng.betavariate(2, 8))


def q(s):
    """Density ratio Beta(8,2)/Beta(2,8) at s = (s/(1-s))^6: evidence that the source SAID 'true'."""
    s = min(1 - 1e-9, s)
    return (s / (1 - s)) ** 6


def indep_llr(acc_map, scores):
    """Exact log P(scores|true)/P(scores|false) for independent sources with known accuracies."""
    z = 0.0
    for k, a in acc_map.items():
        x = q(scores[k])
        z += math.log((a * x + (1 - a)) / (a + (1 - a) * x))
    return z


def make_indep(src):
    def gen(i, truth, rng):
        accs = src(i)
        sc = {k: score(a, truth, rng) for k, a in accs.items()}
        return sc, indep_llr(accs, sc)
    return gen


# --- new hard scenario: three sources copy a mediocre leader ------------------------------
LEAD, COPY_P, COPY_INDEP, SOLO = 0.65, 0.9, 0.55, 0.80

def copycats(i, truth, rng):
    """'lead' is 65% right. c1..c3 copy lead's call w.p. 0.9 (else their own 55%-right call),
    each with its own score noise. 'solo' is an independent 80% source. Naive (independent)
    fusion counts lead's call ~4 times."""
    lead_says = truth if rng.random() < LEAD else not truth
    sc = {"lead": max(0.01, rng.betavariate(8, 2) if lead_says else rng.betavariate(2, 8))}
    for c in ("c1", "c2", "c3"):
        if rng.random() < COPY_P:
            says = lead_says
        else:
            says = truth if rng.random() < COPY_INDEP else not truth
        sc[c] = max(0.01, rng.betavariate(8, 2) if says else rng.betavariate(2, 8))
    sc["solo"] = score(SOLO, truth, rng)
    return sc, copycat_llr(sc)


def copycat_llr(sc):
    """Oracle that knows the copying structure: marginalise lead's (hidden) call exactly."""
    def lik(truth):
        tot = 0.0
        for L in (True, False):
            pL = LEAD if L == truth else 1 - LEAD
            hl = q(sc["lead"]) if L else 1.0
            prod = 1.0
            for c in ("c1", "c2", "c3"):
                p_ind_true = COPY_INDEP if truth else 1 - COPY_INDEP
                p_true = COPY_P * (1.0 if L else 0.0) + (1 - COPY_P) * p_ind_true
                prod *= p_true * q(sc[c]) + (1 - p_true)
            tot += pL * hl * prod
        a = SOLO if truth else 1 - SOLO
        x = q(sc["solo"])
        return tot * (a * x + (1 - a))
    return math.log(lik(True) / lik(False))


SCEN = {
  "baseline":      dict(n=3000, gen=make_indep(lambda i: {"drone":.92,"trial":.80,"draw":.65,"harmony":.50,"instr":.25})),
  "drift (drone goes bad at 1500)": dict(n=3000, gen=make_indep(lambda i: {"drone":.92 if i<1500 else .15,"trial":.75,"draw":.65,"harmony":.50})),
  "bad majority":  dict(n=3000, gen=make_indep(lambda i: {"good":.85,"bad1":.30,"bad2":.30,"bad3":.35})),
  "all noisy (8 weak sources)": dict(n=3000, gen=make_indep(lambda i: {f"w{j}":.58 for j in range(8)})),
  "cold start (first 50 claims)": dict(n=50, gen=make_indep(lambda i: {"drone":.92,"trial":.80,"draw":.65,"harmony":.50,"instr":.25})),
  "copycats (3 copy a 65% leader) NEW": dict(n=3000, gen=copycats),
}

# policy -> (live ledger: None|'none'|'150', node kwargs, uses threshold)
POL = {
  "static":          (None,   {}, False),
  "live":            ("none", {}, False),
  "live+flip":       ("none", {"flip": True}, False),
  "live+flip+decay": ("150",  {"flip": True}, False),
  "flip+decay+thr":  ("150",  {"flip": True}, True),
  "bayes-hard":      ("150",  {"mode": "bayes", "variant": "hard"}, False),
  "bayes-soft":      ("150",  {"mode": "bayes", "variant": "soft"}, False),
  "bayes-logit":     ("150",  {"mode": "bayes", "variant": "logit"}, False),
  "bayes-hard+thr":  ("150",  {"mode": "bayes", "variant": "hard"}, True),
  "oracle":          ("oracle", {}, False),
}


def run(args):
    name, seed = args
    sc = SCEN[name]
    rng = random.Random(seed); tmp = tempfile.mkdtemp()
    lws = {"none": LiveWeights(os.path.join(tmp, "w_none.jsonl"), half_life=None),
           "150": LiveWeights(os.path.join(tmp, "w_150.jsonl"), half_life=150)}
    tl = ThresholdLearner(os.path.join(tmp, "thr.jsonl"), half_life=500)
    acc = {p: [] for p in POL}; truths = []
    for i in range(sc["n"]):
        truth = rng.random() < .5
        scores, o_llr = sc["gen"](i, truth, rng)
        truths.append(truth)
        effs = {}
        for p, (lw, kw, thr) in POL.items():
            if lw == "oracle":
                accepted = o_llr > 0
            else:
                eff, _, s = roman_wheel_node(1, scores, lws.get(lw), threshold=tl if thr else None, **kw)
                accepted = s == "ACCEPT"; effs[p] = eff
            acc[p].append(accepted)
        for p, (lw, kw, thr) in POL.items():
            if thr:
                give_feedback(scores, truth, threshold=tl, eff=effs[p],
                              mode=kw.get("mode", "geomean"), variant=kw.get("variant", "hard"))
        for lw in lws.values():
            give_feedback(scores, truth, live=lw)
    return name, seed, acc, truths


def metrics(acc, truths, part=slice(None)):
    a, t = acc[part], truths[part]
    n_t = sum(t); n_f = len(t) - n_t
    return dict(accuracy=sum(x == y for x, y in zip(a, t)) / len(t),
                true_accept=sum(x and y for x, y in zip(a, t)) / max(1, n_t),
                false_block=sum((not x) and (not y) for x, y in zip(a, t)) / max(1, n_f))


def ci(xs):
    m = st.mean(xs)
    return m, (T975_DF4 * st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0)


def oracle_ceiling(name, n_claims=200_000, seed=12345):
    """Large-sample Bayes-optimal accuracy for a scenario (tight estimate of the ceiling)."""
    sc = SCEN[name]; rng = random.Random(seed); ok = 0
    n = sc["n"]
    for j in range(n_claims):
        truth = rng.random() < .5
        _, llr = sc["gen"](j % n, truth, rng)
        ok += (llr > 0) == truth
    return ok / n_claims


def main():
    jobs = [(n, s) for n in SCEN for s in SEEDS]
    with Pool(min(len(jobs), os.cpu_count() or 1)) as pool:
        out = pool.map(run, jobs)
        ceil = dict(zip(SCEN, pool.map(oracle_ceiling, list(SCEN))))
    res = {}
    for name, sc in SCEN.items():
        runs = [(a, t) for n, s, a, t in out if n == name]
        half = sc["n"] // 2
        res[name] = {"ceiling_200k": ceil[name]}
        for p in POL:
            row = {}
            for label, part in (("all", slice(None)), ("train", slice(0, half)), ("test", slice(half, None))):
                ms = [metrics(a[p], t, part) for a, t in runs]
                for k in ms[0]:
                    row[f"{label}_{k}"] = ci([m[k] for m in ms])
            if name.startswith("drift"):
                ms = [metrics(a[p], t, slice(2500, None)) for a, t in runs]
                row["late_accuracy"] = ci([m["accuracy"] for m in ms])
            res[name][p] = row
    with open("suite_results.json", "w") as f:
        json.dump(res, f, indent=1)

    print("accuracy (mean +/- 95% CI over 5 seeds) | true-accept | false-block")
    for name in SCEN:
        print(f"\n== {name}   [Bayes-optimal ceiling, 200k claims: {res[name]['ceiling_200k']:.1%}]")
        for p in POL:
            r = res[name][p]
            line = (f"  {p:16} acc {r['all_accuracy'][0]:6.1%} +/-{r['all_accuracy'][1]:4.1%}   "
                    f"TA {r['all_true_accept'][0]:6.1%}  FB {r['all_false_block'][0]:6.1%}   "
                    f"train {r['train_accuracy'][0]:6.1%}  test {r['test_accuracy'][0]:6.1%}")
            if "late_accuracy" in r: line += f"  late {r['late_accuracy'][0]:6.1%}"
            print(line)
    plot(res)


def plot(res, path="suite_results.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["static", "live", "live+flip+decay", "bayes-hard", "bayes-hard+thr", "oracle"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), sharey=True)
    w = 0.38
    for ax, name in zip(axes.flat, SCEN):
        xs = range(len(show))
        for j, (lab, col) in enumerate((("train", "#9db4d6"), ("test", "#2a5d9f"))):
            m = [res[name][p][f"{lab}_accuracy"][0] * 100 for p in show]
            e = [res[name][p][f"{lab}_accuracy"][1] * 100 for p in show]
            ax.bar([x + (j - .5) * w for x in xs], m, w, yerr=e, capsize=2, color=col,
                   label=f"{lab} ({'first' if lab == 'train' else 'second'} half)")
        ax.axhline(res[name]["ceiling_200k"] * 100, color="#c0392b", ls="--", lw=1.2, label="Bayes-optimal ceiling")
        ax.axhline(90, color="#555", ls=":", lw=1, label="90% target")
        ax.set_title(name, fontsize=10)
        ax.set_xticks(list(xs)); ax.set_xticklabels(show, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(40, 100); ax.grid(axis="y", alpha=.3)
    axes[0, 0].set_ylabel("correct calls (%)"); axes[1, 0].set_ylabel("correct calls (%)")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Roman Wheel Node v3 - accuracy, first vs second half of each run (mean, 95% CI over 5 seeds)")
    fig.tight_layout(); fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
