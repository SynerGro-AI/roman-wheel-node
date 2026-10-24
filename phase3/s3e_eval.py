"""Phase 3e: zero-cost fixes on the cached Phase 3d judge outputs (no xAI call: judges are read from cache only, and
the script refuses to run with XAI_API_KEY set).

Protocol (no test label is used for any choice):
* Calibration populations: SciFact train (350 + unused, gold evidence) and HealthVer train (internal, unlicensed).
* Calibration schemes: per-dataset (each population calibrates its own node) and pooled (Phase 3d), both evaluated on
  each population's own out-of-fold (5-fold) posteriors.
* Pre-specified family per population: 2 schemes x 3 nodes (bayes-hard [judges], bayes-logit [all], bayes-ess [judges])
  x accept rules (base, U5, U6, U6+NLIe, U6+noNLIc, U6/U0; SciFact per-dataset adds J11, J11+U6 because J11 is cached
  only there) x 4 asymmetry ratios rho (0 = accept-only, 0.5, 1, 2).
* LTT: fixed-sequence testing along each config's pre-ordered margin path (K = 40 steps, conservative first),
  Bonferroni over all (config, rho) of the population, delta = 0.05 per population; one-sided Clopper-Pearson on
  out-of-fold decided errors. Per alpha the certified config with the most decided calibration claims is used.
* Tests (frozen, fit on the whole scheme pool): SciFact dev with the SciFact certificate, HealthVer test with the
  HealthVer certificate, Climate-FEVER (no calibration data, transfer: measured, not certified) with the SciFact
  certificate restricted to configs that need no J11 (J11 is not cached on Climate-FEVER).
* Full coverage: per-dataset node chosen by 5-fold CV accuracy on each population (10 Phase 3 candidates).
Outputs: results/phase3/step3e.json + step3e.txt (aggregates only); per-item outputs in cache/step3e (gitignored).
"""

from __future__ import annotations

import collections
import json
import os
import sys

if os.environ.get("XAI_API_KEY"):
    sys.exit("refusing to run with XAI_API_KEY set: Phase 3e must make no API call (run with env -u XAI_API_KEY)")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as E  # noqa: E402
import s2_eval as S2  # noqa: E402
import s3e_rules as R  # noqa: E402
from factcheck.interfaces import SpendLedger  # noqa: E402
from factcheck.scifact import sha  # noqa: E402
from features import CACHE  # noqa: E402
from judges import make_judges  # noqa: E402

COND = "gold"
PANEL = list(S2.PANEL)
ALLSRC = PANEL + list(S2.LOCALS)
SRCSETS = {"all": ALLSRC, "judges": PANEL}
NODES = (("hard", "judges"), ("logit", "all"), ("ess", "judges"))
POPS = {"scifact": ("scifact_train350", "scifact_train_unused"), "healthver": ("healthver_train",)}
TEST_OF = {"scifact": "scifact_dev", "healthver": "healthver_test"}
ALPHAS = (0.10, 0.05, 0.02, 0.01)
RHOS = (0.0, 0.5, 1.0, 2.0)
K = 40
OUT = os.path.join(CACHE, "step3e")


def features(rows):
    """Per-row static features: gold, BLOCK, raw judge votes, local NLI, J11 verified (argmax SUPPORT + verbatim
    quote)."""
    items = {i["id"]: i for i in json.load(open(os.path.join(CACHE, "step2", "items.json")))}
    local = json.load(open(os.path.join(CACHE, "step2", "local.json")))
    J11 = make_judges(SpendLedger(os.path.join(CACHE, "spend.jsonl"), 1e-9))["J11"]  # cache reads only
    F = {}
    for r in rows:
        it = items[r["id"]]
        g = r["cond"][COND]
        lf = local[r["id"]]
        j11 = None
        if J11.cached(it["claim"], it["evidence"]):
            p, q = J11.judge(it["claim"], it["evidence"])
            hashes = {sha(u) for u in it["units"]}
            j11 = bool(
                max(("SUPPORT", "CONTRADICT", "NEI"), key=lambda x: p[x]) == "SUPPORT"
                and E.quote_verbatim(q or "", [S2.Doc("", tuple(it["units"]))], hashes)
            )
        F[r["id"]] = {
            "y": r["label3"] == "SUPPORT",
            "blocked": g["a"] == 0,
            "votes": R.judge_votes(g["src"], PANEL),
            "nli_contra": lf["nli_contra"],
            "nli2_contra": lf["nli2_contra"],
            "nli2_entail": lf["nli2_entail"],
            "j11": j11,
        }
    return F


def fit_frozen(pool, test, v, sn):
    """Node learned on the whole pool, applied to test: posteriors, cutoff, disagreement flags."""
    cal = E.fit_cal(pool, COND, SRCSETS[sn])
    _, effs, info = E.node_run(pool, test, COND, SRCSETS[sn], "S", cal, v)
    cut = info["cutoff"]
    dis = [
        e > 0.0 and r["cond"][COND]["a"] == 1 and E.disagree(r, COND, cal, SRCSETS[sn], e, cut)
        for r, e in zip(test, effs)
    ]
    return effs, [cut] * len(test), dis


def tstats(ys, dec):
    """dec: True/False/None per claim. Accuracy over decided (bootstrap CI), false accepts / unsupported (CP upper)."""
    di = [i for i, d in enumerate(dec) if d is not None]
    ok = [float(dec[i] == ys[i]) for i in di]
    acc = [i for i in di if dec[i]]
    fa = sum(1 for i in acc if not ys[i])
    uns = sum(1 for y in ys if not y)
    out = {
        "n": len(ys),
        "decided": len(di),
        "coverage": len(di) / max(1, len(ys)),
        "errors": int(len(ok) - sum(ok)),
        "error_upper95_cp": R.cp_upper(int(len(ok) - sum(ok)), len(di)),
        "accepted": len(acc),
        "false_accepts": fa,
        "unsupported": uns,
        "false_accept_rate_all": fa / max(1, uns),
        "false_accept_rate_all_upper95_cp": R.cp_upper(fa, uns),
        "false_accept_share_of_accepted_upper95_cp": R.cp_upper(fa, len(acc)),
    }
    if ok:
        out["accuracy_decided"] = list(E.boot_ci(ok))
    return out


RISKS = {  # name: (BLOCK counted as a decided REJECT?, risk = false accepts among accepted?, rhos)
    "selective_blocks_decided": (True, False, RHOS),  # Phase 3d accounting (primary, pre-registered)
    "selective_blocks_separate": (False, False, RHOS),  # BLOCK reported as its own fail-closed outcome
    "accept_precision": (False, True, (0.0,)),  # accept-only: of what it ACCEPTs, share unsupported
}


def certify_population(pop, pools, oof, F, tests, blocks_decided, precision, rhos, per_item, tag):
    prs = pools[pop]
    fam, meta = {}, {}
    for sch in (pop, "pooled"):
        rules = R.make_rules(j11=(pop == "scifact" and sch == "scifact"))
        for v, sn in NODES:
            o = oof[(sch, v, sn)]
            feats = [dict(F[r["id"]], dis=o[r["id"]][1]) for r in prs]
            for rn, rule in rules.items():
                name = f"{sch}|bayes-{v} [{sn}]|{rn}"
                fam[name] = R.to_items(
                    feats, [o[r["id"]][0] for r in prs], [o[r["id"]][2] for r in prs], rule, blocks_decided
                )
                meta[name] = (sch, v, sn, rn)
    cert = R.certify_family(fam, ALPHAS, 0.05, rhos, K, precision)
    allres = cert.pop("all")
    print(
        tag,
        pop,
        "tests",
        cert["n_tests"],
        {
            a: {k: x.get(k) for k in ("certified", "config", "rho", "decided", "errors", "upper")}
            for a, x in cert["alphas"].items()
        },
        "tight",
        cert["tightest_first_step"],
        flush=True,
    )
    P = {"calibration_n": len(prs), "certificates": cert, "tests": {}}
    targets = [(TEST_OF[pop], False)] + ([("climatefever", True)] if pop == "scifact" else [])
    cache_fit = {}
    for tds, no_j11 in targets:
        trs = tests[tds]
        ys = [F[r["id"]]["y"] for r in trs]
        T = {"n": len(trs), "labels": dict(collections.Counter(r["label3"] for r in trs)), "ltt": {}}
        for al in ALPHAS:
            c = cert["alphas"][f"{al}"]
            if no_j11:  # best certified config usable without J11 (J11 is not cached on Climate-FEVER)
                best = None
                for key, r_ in allres.items():
                    a_, rest = key.split("|", 1)
                    name, rho = rest.rsplit("|", 1)
                    if (
                        float(a_) == al
                        and r_["certified"]
                        and "J11" not in meta[name][3]
                        and (best is None or r_["decided"] > best["decided"])
                    ):
                        best = dict(r_, config=name, rho=float(rho))
                c = best or {"certified": False}
            if not c["certified"]:
                base = [False if (F[r["id"]]["blocked"] and blocks_decided) else None for r in trs]
                T["ltt"][f"{al}"] = dict(tstats(ys, base), certified=False)
                continue
            sch, v, sn, rn = meta[c["config"]]
            key = (tds, sch, v, sn)
            if key not in cache_fit:
                cache_fit[key] = fit_frozen(pools[sch], trs, v, sn)
            effs, cuts, dis = cache_fit[key]
            feats = [dict(F[r["id"]], dis=d) for r, d in zip(trs, dis)]
            its = R.to_items(feats, effs, cuts, R.make_rules(j11=True)[rn], blocks_decided)
            dec = [R.decide(it, c["m_a"], c["m_r"]) for it in its]
            T["ltt"][f"{al}"] = dict(tstats(ys, dec), certified=True, config=c["config"], rho=c["rho"])
            for r, d in zip(trs, dec):
                per_item.setdefault(r["id"], {})[f"{tag}:{al}"] = d
        P["tests"][tds] = T
    return P


def main() -> None:
    rows, skipped = S2.build_rows()
    F = features(rows)
    cal_all = [r for r in rows if r["ds"] in S2.CALIB]
    pools = {p: [r for r in cal_all if r["ds"] in d] for p, d in POPS.items()}
    pools["pooled"] = cal_all
    tests = {ds: [r for r in rows if r["ds"] == ds] for ds in S2.TESTS}
    oof = {}
    for sch, pool in pools.items():
        for v, sn in NODES:
            o, d, c = E.oof_effs(pool, COND, SRCSETS[sn], v)
            oof[(sch, v, sn)] = {r["id"]: (o[i], d[i], c[i]) for i, r in enumerate(pool)}
    res = {
        "protocol": __doc__,
        "skipped_incomplete": skipped,
        "calibration_counts": {p: len(x) for p, x in pools.items()},
        "test_counts": {k: len(v) for k, v in tests.items()},
        "internal_note": {"healthver": S2.NOTE},
        "calibration_diagnostics": {},
        "risks": {},
    }
    for pop in POPS:  # aggregate filter precision on calibration (counts only)
        prs = pools[pop]
        nb = [F[r["id"]] for r in prs if not F[r["id"]]["blocked"]]
        dg = {
            "blocks": sum(F[r["id"]]["blocked"] for r in prs),
            "blocks_on_supported": sum(F[r["id"]]["blocked"] and F[r["id"]]["y"] for r in prs),
        }
        for nm, fn in (
            ("U6", lambda f: f["votes"] == 6),
            ("U6+NLIe", lambda f: f["votes"] == 6 and f["nli2_entail"] >= 0.5),
            ("J11", lambda f: bool(f["j11"])),
            ("J11+U6", lambda f: bool(f["j11"]) and f["votes"] == 6),
            ("U0", lambda f: f["votes"] == 0),
        ):
            sel = [f for f in nb if fn(f)]
            dg[nm] = {"n": len(sel), "gold_support": sum(f["y"] for f in sel)}
        res["calibration_diagnostics"][pop] = dg
    per_item = {}
    for tag, (bd, prec, rhos) in RISKS.items():
        res["risks"][tag] = {
            pop: certify_population(pop, pools, oof, F, tests, bd, prec, rhos, per_item, tag) for pop in POPS
        }
    fc = {}
    for pop, tlist in (("scifact", ("scifact_dev", "climatefever")), ("healthver", ("healthver_test",))):
        (v, sn), cv = E.cv_select(pools[pop], COND, SRCSETS)
        cal = E.fit_cal(pools[pop], COND, ALLSRC)
        for tds in tlist:
            adm, _, _ = E.node_run(pools[pop], tests[tds], COND, SRCSETS[sn], "S", cal, v)
            ys = [F[r["id"]]["y"] for r in tests[tds]]
            fc[tds] = dict(
                tstats(ys, [bool(a) for a in adm]),
                node=f"bayes-{v} [{sn}]",
                calibrated_on=pop,
                cv_accuracy=cv[f"bayes-{v} [{sn}]"],
            )
    res["full_coverage_per_dataset"] = fc
    ref = json.load(open(os.path.join(E.ROOT, "results", "phase3", "step2.json")))["datasets"]
    res["phase3d_reference"] = {
        ds: {k: d[k] for k in ("node_full_coverage", "ltt", "ltt_family")} for ds, d in ref.items()
    }
    json.dump(per_item, open(os.path.join(OUT, "eval_items.json"), "w"))
    out = os.path.join(E.ROOT, "results", "phase3")
    json.dump(res, open(os.path.join(out, "step3e.json"), "w"), indent=1, default=float)
    open(os.path.join(out, "step3e.txt"), "w").write(report(res))
    print(report(res))


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def line(s):
    a = s.get("accuracy_decided", [None] * 3)
    return (
        f"coverage {pct(s['coverage'])} ({s['decided']}/{s['n']}), acc {pct(a[0])} [{pct(a[1])}, {pct(a[2])}], "
        f"errors {s['errors']} (CP UB {pct(s['error_upper95_cp'])}), false accepts "
        f"{s['false_accepts']}/{s['unsupported']} "
        f"unsupported ({pct(s['false_accept_rate_all'])}, UB {pct(s['false_accept_rate_all_upper95_cp'])}), "
        f"{s['accepted']} accepted"
    )


def report(res) -> str:
    L = [
        "Phase 3e: stricter accept rules, per-dataset calibration, fixed-sequence LTT (zero new API calls)",
        "",
        f"Calibration: {res['calibration_counts']}  (HealthVer: {S2.NOTE})",
        "Calibration filter diagnostics (non-BLOCK claims passing each filter, of which gold SUPPORT):",
    ]
    for pop, d in res["calibration_diagnostics"].items():
        L.append(
            f"  {pop}: BLOCKs {d['blocks']} ({d['blocks_on_supported']} on SUPPORT gold); "
            + "; ".join(f"{k} {v['gold_support']}/{v['n']}" for k, v in d.items() if isinstance(v, dict))
        )
    for tag, pops in res["risks"].items():
        L += ["", f"######## risk: {tag}"]
        for pop, P in pops.items():
            c = P["certificates"]
            L += [
                f"== population {pop} (calibration n={P['calibration_n']}; {c['n_tests']} (config, rho) tests, "
                f"Bonferroni delta_each={c['delta_each']:.2e}; fixed sequence K={K})"
            ]
            t = c["tightest_first_step"]
            L.append(
                f"  best single-step upper bound in the family (diagnostic): {pct(t['upper'])} ({t['config']}, rho "
                f"{t['rho']}: {t['decided']} decided, {t['errors']} errors)"
            )
            for al, x in c["alphas"].items():
                need = c["min_decided_zero_errors"][al]
                if x.get("certified"):
                    L.append(
                        f"  alpha {float(al):.0%}: CERTIFIED {x['config']} rho {x['rho']}; calib coverage "
                        f"{pct(x['decided'] / x['n'])} ({x['decided']} decided, {x['errors']} errors, UB "
                        f"{pct(x['upper'])}; {x['accepted']} accepted, {x['false_accepts']} false)  [0-error minimum "
                        f"n={need}]"
                    )
                else:
                    L.append(
                        f"  alpha {float(al):.0%}: none certifiable -> NOT_SURE on every non-BLOCK claim  [0-error "
                        f"minimum n={need}]"
                    )
            for tds, T in P["tests"].items():
                L.append(
                    f"  -- test {tds} (n={T['n']})"
                    + ("  [transfer: measured, not certified]" if tds == "climatefever" else "")
                )
                for al, s in T["ltt"].items():
                    L.append(f"     alpha {float(al):.0%}{'' if s['certified'] else ' (none)'}: {line(s)}")
    L += ["", "== full coverage (per-dataset CV-selected node) vs Phase 3d pooled node"]
    for tds, s in res["full_coverage_per_dataset"].items():
        r = res["phase3d_reference"][tds]["node_full_coverage"]
        L.append(f"  {tds}: {s['node']} calibrated on {s['calibrated_on']}: {line(s)}")
        L.append(
            f"     Phase 3d pooled: acc {pct(r['accuracy_decided'][0])}, false accepts "
            f"{r['false_accepts']}/{r['unsupported']}"
        )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
