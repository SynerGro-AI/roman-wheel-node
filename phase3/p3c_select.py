"""Phase 3c, TRAIN ONLY: choose the strict verifier (J11 prompt V vs J12 prompt W) and whether to use it as a hard
ACCEPT veto, by 5-fold CV on the 350 training claims. Dev rows are never read for selection.
Writes results/phase3/p3c_selection.json, which evaluate.py reads.
J12 (prompt W) was dropped after 47 calls for budget (measured $0.010/call), so in practice only J11 is available.
Rule (fixed before running): verifier = higher best train-CV accuracy on rv3 (the realistic condition; ties -> gold,
then J11, the cheaper one). Veto per condition = on iff it raises train-CV accuracy of the CV-selected node config."""
import json
import os
import sys

os.environ["RWN_P3_PARTIAL"] = "1"            # dev rows may still lack verifier calls; they are dropped below anyway
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as E                           # noqa: E402
from judges import JUDGES                      # noqa: E402


def cv_acc(train, cond, sources, variant, veto, folds=5):
    ok = 0
    for f in range(folds):
        tr = [r for i, r in enumerate(train) if i % folds != f]
        te = [r for i, r in enumerate(train) if i % folds == f]
        cal = E.fit_cal(tr, cond, sources)
        adm, _, _ = E.node_run(tr, te, cond, sources, "S", cal, variant, veto=veto)
        ok += sum(a == (r["label3"] == "SUPPORT") for a, r in zip(adm, te))
    return ok / len(train)


def main():
    rows = E.load_rows()
    train = [r for k, r in sorted(rows.items()) if r["split"] == "train"]
    assert len(train) == 350 and all("gold" in r["cond"] and "rv3" in r["cond"] for r in train)
    out = {"protocol": __doc__.strip(), "conditions": {}}
    for cond in ("gold", "rv3"):
        base_j = [j for j in JUDGES if cond in JUDGES[j]["conditions"] and j not in E.VERIFIERS]
        loc = [s for s in E.LOCAL if s in train[0]["cond"][cond]["src"]]
        C = {}
        avail = [v for v in E.VERIFIERS if all(v in r["cond"][cond]["verify"] for r in train)]
        for v in avail:
            diag = {}
            for lab in ("SUPPORT", "CONTRADICT", "NEI"):
                rs = [r for r in train if r["label3"] == lab]
                diag[lab] = {"n": len(rs), "proof_rate": sum(r["cond"][cond]["verify"][v] for r in rs) / len(rs)}
            srcsets = {"all": loc + base_j + [v], "judges": base_j + [v]}
            (sv, ss), cv = E.cv_select(train, cond, srcsets)
            nov = cv[f"bayes-{sv} [{ss}]"]
            wv = cv_acc(train, cond, srcsets[ss], sv, v)
            C[v] = {"proof_rate_by_label": diag, "config": f"bayes-{sv} [{ss}]", "cv_no_veto": nov, "cv_veto": wv,
                    "cv_all_configs": cv}
            print(cond, v, C[v]["config"], f"no-veto {nov:.4f} veto {wv:.4f}", {k: round(d["proof_rate"], 3) for k, d in diag.items()}, flush=True)
        srcsets0 = {"all": loc + base_j, "judges": base_j}
        (sv, ss), cv0 = E.cv_select(train, cond, srcsets0)
        C["without_verifier"] = {"config": f"bayes-{sv} [{ss}]", "cv": cv0[f"bayes-{sv} [{ss}]"]}
        print(cond, "no verifier", C["without_verifier"], flush=True)
        out["conditions"][cond] = C
    best = lambda c, v: max(out["conditions"][c][v]["cv_no_veto"], out["conditions"][c][v]["cv_veto"])
    ver = max([v for v in E.VERIFIERS if v in out["conditions"]["rv3"]], key=lambda v: (best("rv3", v), best("gold", v), v == "J11"))
    out["verifier"] = ver
    out["veto"] = {c: out["conditions"][c][ver]["cv_veto"] > out["conditions"][c][ver]["cv_no_veto"] for c in ("gold", "rv3")}
    json.dump(out, open(E.SELECTION_3C, "w"), indent=1, sort_keys=True)
    print("selected", ver, out["veto"])


if __name__ == "__main__":
    main()
