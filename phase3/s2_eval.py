"""Phase 3d (step 2) evaluation: larger calibration set, out-of-sample tests, LTT certification at 1% / 2%.

Protocol (no test label is used for any choice):
* Calibration pool = SciFact train (350 + 449 unused) + HealthVer train (~2,000 pairs), gold evidence, full 6-judge
  panel (J1 J2 J4 J7 J9 J10) + local NLI (nli, nli2) + overlap. Source calibration (isotonic), node variant and source
  set (5-fold CV), reliabilities, dependence, cutoff and the LTT NOT_SURE margin are all fitted on the pool only.
* Task: ACCEPT = "the shown evidence supports the claim"; REJECT = contradicted or not enough info; NOT_SURE = abstain.
* LTT: Bonferroni over K=20 log-odds margins, delta=0.05, one-sided Clopper-Pearson on out-of-fold decided errors.
  The guarantee is for claims exchangeable with the calibration pool. Climate-FEVER has no calibration data at all, so
  it is a transfer test: its numbers are measured, not certified.
* Supplement: LTT over the whole pre-specified candidate family (all 10 CV candidates x 20 margins, Bonferroni
  delta/200), the certified candidate with the widest calibration coverage chosen per alpha on calibration OOF only.
* Tests (frozen node): HealthVer test (internal, unlicensed dataset, data not redistributed), Climate-FEVER
  unanimous-vote subset (CC BY-SA), SciFact dev (if J10 is complete there).
Outputs: results/phase3/step2.json + step2.txt (aggregates only); per-item outputs in cache/step2/eval_items.json
(gitignored).
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate as E  # noqa: E402
from factcheck.interfaces import SpendLedger  # noqa: E402
from factcheck.scifact import sha  # noqa: E402
from features import CACHE  # noqa: E402
from judges import make_judges  # noqa: E402

PANEL = ("J1", "J2", "J4", "J7", "J9", "J10")
LOCALS = ("nli", "nli2", "overlap")
COND = "gold"
CALIB = ("scifact_train350", "scifact_train_unused", "healthver_train")
TESTS = ("healthver_test", "climatefever", "scifact_dev")
INTERNAL = {"healthver_train", "healthver_test"}
NOTE = "internal, unlicensed dataset, data not redistributed"
ALPHAS = (0.10, 0.05, 0.02, 0.01)
Doc = collections.namedtuple("Doc", "title sentences")


def build_rows():
    items = json.load(open(os.path.join(CACHE, "step2", "items.json")))
    local = json.load(open(os.path.join(CACHE, "step2", "local.json")))
    J = make_judges(
        SpendLedger(os.path.join(CACHE, "spend.jsonl"), 1e-9)
    )  # cache reads only: any real call would raise
    rows, skipped = [], collections.Counter()
    for it in items:
        if it["id"] not in local or not all(J[n].cached(it["claim"], it["evidence"]) for n in PANEL):
            skipped[it["ds"]] += 1
            continue
        lf = local[it["id"]]
        src = {
            "nli": {"S": lf["nli_entail"], "C": lf["nli_contra"]},
            "nli2": {"S": lf["nli2_entail"], "C": lf["nli2_contra"]},
            "overlap": {"S": lf["overlap"]},
        }
        quotes = {}
        for n in PANEL:
            p, q = J[n].judge(it["claim"], it["evidence"])
            src[n] = {"S": p["SUPPORT"], "C": p["CONTRADICT"], "N": p["NEI"], "unparsed": "_unparsed" in p}
            quotes[n] = q
        hashes = {sha(u) for u in it["units"]}
        docs = [Doc("", tuple(it["units"]))]
        shown_ok = all(sha(s) in hashes for s in it["sents"])
        commits = max(("S", "C", "N"), key=lambda x: src["J1"][x]) != "N"
        quote_ok = (not commits) or E.quote_verbatim(quotes["J1"], docs, hashes)
        rows.append(
            {
                "id": it["id"],
                "ds": it["ds"],
                "split": it["role"],
                "label3": it["label3"],
                "claim": it["claim"],
                "cond": {
                    COND: {
                        "src": src,
                        "a": int(shown_ok and quote_ok),
                        "quote_ok": quote_ok,
                        "verify": {},
                        "j1_commits": commits,
                        "quote_given": bool(quotes["J1"]),
                    }
                },
            }
        )
    rows.sort(key=lambda r: hashlib.sha256(("fold" + r["id"]).encode()).hexdigest())  # mixes datasets across CV folds
    return rows, dict(skipped)


def stats(rows, status):
    """status: ACCEPT / REJECT / NOT_SURE per row. Accuracy over decided claims; false accept = ACCEPT on a
    non-SUPPORT."""
    ys = [r["label3"] == "SUPPORT" for r in rows]
    dec = [i for i, s in enumerate(status) if s != "NOT_SURE"]
    ok = [float((status[i] == "ACCEPT") == ys[i]) for i in dec]
    fa = [float(status[i] == "ACCEPT" and not ys[i]) for i in range(len(rows))]
    acc_i = [i for i in dec if status[i] == "ACCEPT"]
    nf = sum(1 for i in acc_i if not ys[i])
    err = int(len(ok) - sum(ok))
    out = {
        "n": len(rows),
        "decided": len(dec),
        "coverage": len(dec) / max(1, len(rows)),
        "errors": err,
        "error_rate": err / len(dec) if dec else None,
        "error_upper95_cp": E.cp_upper(err, len(dec)),
        "accepted": len(acc_i),
        "false_accepts": nf,
        "false_accept_upper95_cp_of_accepted": E.cp_upper(nf, len(acc_i)),
        "unsupported": sum(1 for y in ys if not y),
    }
    if ok:
        m, lo, hi = E.boot_ci(ok)
        out["accuracy_decided"] = [m, lo, hi]
        m, lo, hi = E.boot_ci(fa)
        out["false_accept_rate_all"] = [m, lo, hi]
    return out


def main() -> None:
    rows, skipped = build_rows()
    cal_rows = [r for r in rows if r["ds"] in CALIB]
    test = {ds: [r for r in rows if r["ds"] == ds] for ds in TESTS}
    test = {k: v for k, v in test.items() if v}
    print(
        "calibration",
        collections.Counter(r["ds"] for r in cal_rows),
        "tests",
        {k: len(v) for k, v in test.items()},
        "skipped (incomplete)",
        skipped,
        flush=True,
    )
    allsrc = list(PANEL) + list(LOCALS)
    srcsets = {"all": allsrc, "judges": list(PANEL)}
    (v, sn), cv = E.cv_select(cal_rows, COND, srcsets)
    srcs = srcsets[sn]
    print("CV selected", v, sn, {k: round(x, 4) for k, x in cv.items()}, flush=True)
    cal = E.fit_cal(cal_rows, COND, allsrc)
    oof, dis, cuts = E.oof_effs(cal_rows, COND, srcs, v)
    ys_c = [r["label3"] == "SUPPORT" for r in cal_rows]
    alltest = [r for ds in test for r in test[ds]]
    res = {
        "protocol": __doc__,
        "calibration_counts": dict(collections.Counter(r["ds"] for r in cal_rows)),
        "test_counts": {k: len(x) for k, x in test.items()},
        "skipped_incomplete": skipped,
        "selected": {"variant": v, "sources": sn, "cv_accuracy": cv},
        "internal_note": {d: NOTE for d in INTERNAL},
    }
    # full-coverage node + baselines (unwrapped J1, best single judge by calibration accuracy)
    adm, effs, info = E.node_run(cal_rows, alltest, COND, srcs, "S", cal, v)
    res["cutoff"] = info["cutoff"]
    res["reliability"] = info["reliability"]
    st_node = ["ACCEPT" if a else "REJECT" for a in adm]
    j_acc = {
        n: sum(
            (max(("S", "C", "N"), key=lambda x: r["cond"][COND]["src"][n][x]) == "S") == y
            for r, y in zip(cal_rows, ys_c)
        )
        / len(cal_rows)
        for n in PANEL
    }
    best = max(PANEL, key=lambda n: j_acc[n])
    res["calibration_judge_accuracy"] = j_acc
    res["best_judge_by_calibration"] = best

    def arg(r, n):
        return "ACCEPT" if max(("S", "C", "N"), key=lambda x: r["cond"][COND]["src"][n][x]) == "S" else "REJECT"

    # LTT per alpha: margin certified on calibration OOF, then the frozen node + NOT_SURE band on every test claim
    ltt = {}
    for al in ALPHAS:
        m, n_tr, k_tr, ub_tr = E.ltt_band(oof, ys_c, dis, cuts, al)
        a2, e2, inf2 = E.node_run(cal_rows, alltest, COND, srcs, "S", cal, v, ltt_margin=m)
        st = ["NOT_SURE" if s == "NOT_SURE" else ("ACCEPT" if a else "REJECT") for s, a in zip(inf2["status"], a2)]
        ltt[f"{al}"] = {
            "margin_logodds": m,
            "certified": m < 50,
            "calibration_oof": {
                "decided": n_tr,
                "errors": k_tr,
                "upper95_bonferroni": ub_tr,
                "coverage": n_tr / len(cal_rows),
                "disagreements": sum(dis),
            },
            "status": st,
        }
        print(
            f"LTT alpha={al}: margin {m}, calib decided {n_tr}/{len(cal_rows)} errors {k_tr} ub {ub_tr:.4f}", flush=True
        )
    # supplement: LTT over the whole candidate family (every CV candidate x 20 margins, Bonferroni delta/(20*|family|)),
    # chosen per alpha by calibration OOF coverage only; still valid because the family is fixed before looking at data
    fam = {}
    nf = len(E.CANDIDATES)
    for al in ALPHAS:
        bestf = None
        for fv, fsn in E.CANDIDATES:
            o2, d2, c2 = E.oof_effs(cal_rows, COND, srcsets[fsn], fv)
            m, n_tr, k_tr, ub_tr = E.ltt_band(o2, ys_c, d2, c2, al, delta=0.05 / nf)
            if m < 50 and (bestf is None or n_tr > bestf[2]):
                bestf = (fv, fsn, n_tr, k_tr, ub_tr, m)
        if bestf is None:
            fam[f"{al}"] = {"certified": False, "status": ["NOT_SURE"] * len(alltest)}
        else:
            fv, fsn, n_tr, k_tr, ub_tr, m = bestf
            a2, _, inf2 = E.node_run(cal_rows, alltest, COND, srcsets[fsn], "S", cal, fv, ltt_margin=m)
            st = ["NOT_SURE" if x == "NOT_SURE" else ("ACCEPT" if a else "REJECT") for x, a in zip(inf2["status"], a2)]
            fam[f"{al}"] = {
                "certified": True,
                "variant": fv,
                "sources": fsn,
                "margin_logodds": m,
                "calibration_oof": {
                    "decided": n_tr,
                    "errors": k_tr,
                    "upper95_bonferroni": ub_tr,
                    "coverage": n_tr / len(cal_rows),
                },
                "status": st,
            }
        print(f"family LTT alpha={al}:", {k: x for k, x in fam[f"{al}"].items() if k != "status"}, flush=True)
    res["datasets"] = {}
    off = 0
    per_item = {}
    for ds, rs in test.items():
        sl = slice(off, off + len(rs))
        off += len(rs)
        ys = [r["label3"] == "SUPPORT" for r in rs]
        node_ok = [(s == "ACCEPT") == y for s, y in zip(st_node[sl], ys)]
        j1 = [arg(r, "J1") for r in rs]
        bj = [arg(r, best) for r in rs]
        d = {
            "labels": dict(collections.Counter(r["label3"] for r in rs)),
            "node_full_coverage": stats(rs, st_node[sl]),
            "unwrapped_J1": stats(rs, j1),
            f"best_judge_{best}": stats(rs, bj),
            "mcnemar_node_vs_J1": E.mcnemar(node_ok, [(s == "ACCEPT") == y for s, y in zip(j1, ys)]),
            f"mcnemar_node_vs_{best}": E.mcnemar(node_ok, [(s == "ACCEPT") == y for s, y in zip(bj, ys)]),
            "ltt": {al: stats(rs, x["status"][sl]) for al, x in ltt.items()},
            "ltt_family": {al: stats(rs, x["status"][sl]) for al, x in fam.items() if x["certified"]},
        }
        if ds in INTERNAL:
            d["note"] = NOTE
        res["datasets"][ds] = d
        for k, r in enumerate(rs):
            per_item[r["id"]] = {
                "node": st_node[sl][k],
                "eff": effs[sl][k],
                **{f"ltt{al}": x["status"][sl][k] for al, x in ltt.items()},
            }
    # exchangeable part only (SciFact + HealthVer tests, same sources as calibration); Climate-FEVER is transfer
    exch = [ds for ds in ("healthver_test", "scifact_dev") if ds in test]
    if exch:
        idx = []
        off = 0
        for ds, rs in test.items():
            if ds in exch:
                idx += list(range(off, off + len(rs)))
            off += len(rs)
        rs = [alltest[i] for i in idx]
        res["pooled_in_distribution"] = {
            "datasets": exch,
            "node_full_coverage": stats(rs, [st_node[i] for i in idx]),
            "ltt": {al: stats(rs, [x["status"][i] for i in idx]) for al, x in ltt.items()},
        }
    res["ltt_certificates"] = {
        al: {k: x[k] for k in ("margin_logodds", "certified", "calibration_oof")} for al, x in ltt.items()
    }
    res["ltt_family_certificates"] = {al: {k: v_ for k, v_ in x.items() if k != "status"} for al, x in fam.items()}
    json.dump(per_item, open(os.path.join(CACHE, "step2", "eval_items.json"), "w"))
    out = os.path.join(E.ROOT, "results", "phase3")
    json.dump(res, open(os.path.join(out, "step2.json"), "w"), indent=1, default=float)
    open(os.path.join(out, "step2.txt"), "w").write(report(res))
    print(report(res))


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def report(res) -> str:
    L = [
        "Phase 3d (step 2): Roman Wheel Node, larger calibration, out-of-sample tests",
        "",
        f"Calibration pool: {res['calibration_counts']}  (HealthVer: {NOTE})",
        f"Selected on calibration CV: bayes-{res['selected']['variant']} [{res['selected']['sources']}]; cutoff "
        f"{res['cutoff']:.4f}",
        f"Best single judge on calibration: {res['best_judge_by_calibration']}",
        "",
        "LTT certificates (delta=0.05, Bonferroni K=20, calibration OOF):",
    ]
    for al, c in res["ltt_certificates"].items():
        o = c["calibration_oof"]
        L.append(
            f"  alpha {float(al):.0%}: {'CERTIFIED' if c['certified'] else 'none certifiable'}; margin "
            f"{c['margin_logodds']}; calib coverage {pct(o['coverage'])} ({o['decided']} decided, {o['errors']} "
            f"errors, UB {pct(o['upper95_bonferroni'])})"
        )
    L.append("LTT over the full candidate family (10 configs x 20 margins, Bonferroni delta/200, calibration OOF):")
    for al, c in res.get("ltt_family_certificates", {}).items():
        if c["certified"]:
            o = c["calibration_oof"]
            L.append(
                f"  alpha {float(al):.0%}: CERTIFIED bayes-{c['variant']} [{c['sources']}]; margin "
                f"{c['margin_logodds']}; calib coverage {pct(o['coverage'])} ({o['decided']} decided, {o['errors']} "
                f"errors, UB {pct(o['upper95_bonferroni'])})"
            )
        else:
            L.append(f"  alpha {float(al):.0%}: none certifiable (abstain on all non-BLOCK claims)")
    for ds, d in res["datasets"].items():
        L += [
            "",
            f"== {ds} (n={d['node_full_coverage']['n']}, labels {d['labels']})"
            + (f"  [{d['note']}]" if "note" in d else "")
            + ("  [transfer: no calibration data from this source]" if ds == "climatefever" else ""),
        ]
        for name in [k for k in d if k in ("node_full_coverage", "unwrapped_J1") or k.startswith("best_judge_")]:
            s = d[name]
            a = s.get("accuracy_decided", [None] * 3)
            f = s.get("false_accept_rate_all", [None] * 3)
            L.append(
                f"  {name:22s} acc {pct(a[0])} [{pct(a[1])}, {pct(a[2])}]  false accepts "
                f"{s['false_accepts']}/{s['unsupported']} "
                f"({pct(f[0])} [{pct(f[1])}, {pct(f[2])}])"
            )
        L.append(f"  McNemar node vs J1: {d['mcnemar_node_vs_J1']}")
        for al, s in d["ltt"].items():
            L.append(
                f"  LTT {float(al):.0%}: coverage {pct(s['coverage'])} ({s['decided']}/{s['n']}), errors {s['errors']} "
                f"(rate {pct(s['error_rate'])}, CP UB {pct(s['error_upper95_cp'])}), false accepts "
                f"{s['false_accepts']}/{s['accepted']} accepted"
            )
        for al, s in d.get("ltt_family", {}).items():
            L.append(
                f"  family LTT {float(al):.0%}: coverage {pct(s['coverage'])} ({s['decided']}/{s['n']}), errors "
                f"{s['errors']} (rate {pct(s['error_rate'])}, CP UB {pct(s['error_upper95_cp'])}), false accepts "
                f"{s['false_accepts']}/{s['accepted']} accepted"
            )
    if "pooled_in_distribution" in res:
        p = res["pooled_in_distribution"]
        L += ["", f"== pooled in-distribution tests {p['datasets']}"]
        for al, s in p["ltt"].items():
            L.append(
                f"  LTT {float(al):.0%}: coverage {pct(s['coverage'])} ({s['decided']}/{s['n']}), errors {s['errors']} "
                f"(CP UB {pct(s['error_upper95_cp'])}), false accepts {s['false_accepts']}"
            )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
