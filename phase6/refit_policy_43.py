"""Policy 4.3.0: refit the shipped profiles from local open-model features only (no xAI/Grok output anywhere).

Why: the owner's decision is that xAI/Grok judge outputs may be kept only as a repo-only performance ledger and must
not feed any shipped weight. Policy 4.2.0's ``panel`` profile was fitted on xAI judge outputs, and even its free
``local`` / ``local-lite`` margins depended on them: the fit pool was restricted to claims with cached judge outputs,
and the gate bit of every calibration row included the J1 judge's verbatim-quote check. 4.3.0 removes both.

Protocol (fixed before any SciFact dev number was computed for 4.3.0):
* Fit pool: SciFact train with gold evidence, every claim that has local features (scifact_train350 +
  scifact_train_unused), ordered by sha256("fold" + id) as in every earlier refit. Features: the cached outputs of
  the local open models only (nli = DeBERTa-v3-small int8 ONNX, nli2 = DeBERTa-v3-base mnli-fever-anli, overlap).
  Gate bit a = 1 for every row (gold evidence is shown verbatim; no judge quote).
* Fitting: the public API ``roman_wheel_node.fit.fit_profile`` (isotonic calibrators, Beta reliabilities, learned
  cut-off; accept-only margin pre-registered from 5-fold out-of-fold posteriors, K = 40). It is checked here to give
  exactly the research pipeline's numbers (phase4/export_policy.fit, phase3/evaluate.oof_effs) on the same rows.
* Certificate attempt: LTT fixed-sequence testing (one-sided Clopper-Pearson, delta 0.05 split over the family of
  the 2 profiles, accept-precision risk) at alpha 10%, 5%, 2%, 1% on the out-of-fold posteriors. A profile uses the
  margin of its tightest certified alpha; if none certifies it keeps the pre-registered margin and says so.
* Disagreement rule: none (as the shipped local profiles of 4.0.0-4.2.0); the margin is chosen with the same rule.
* Held-out evaluation: SciFact dev (300 claims, gold evidence, never used to fit; its numbers were reported in
  earlier phases, so it is not pristine), through the shipped runtime (roman_wheel_node.shadow.FrozenNode), a = 1.
Outputs: src/roman_wheel_node/policies/rwn-policy-4.3.0.json, the pin in policy.py, results/phase6/refit_43.{json,txt}
(aggregates only).

    env -u XAI_API_KEY python3 phase6/refit_policy_43.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from roman_wheel_node import fit as FIT  # noqa: E402
from roman_wheel_node.fresh import cp_upper  # noqa: E402
from roman_wheel_node.policy import load_policy  # noqa: E402
from roman_wheel_node.shadow import FrozenNode  # noqa: E402

CACHE = os.path.join(ROOT, "phase3", "cache", "step2")
FIT_DS = ("scifact_train350", "scifact_train_unused")
TEST_DS = "scifact_dev"
PROFILES = {"local": ("nli2", "overlap"), "local-lite": ("nli", "overlap")}
ALPHAS = (0.10, 0.05, 0.02, 0.01)
K = 40
VERSION = "4.3.0"
OUT = os.path.join(ROOT, "src", "roman_wheel_node", "policies", f"rwn-policy-{VERSION}.json")
RES = os.path.join(ROOT, "results", "phase6")
PREV_COMMIT = "69aa8db"
PREV_PATH = "src/roman_wheel_node/policies/rwn-policy-4.2.0.json"
PREV_SHA = "6f7bb9e3762754dc6ca6343c44f000c7caf05a819a1f63c857471856d3946c2a"
POOL_NOTE = (
    "SciFact train, gold evidence ({n} claims), local open-model features only (no hosted-model output). Claims and "
    "evidence annotations CC BY 4.0, abstracts ODC-By 1.0 (github.com/allenai/scifact LICENSE.md, a5254a1). See "
    "DATA_LICENSES.md."
)
DATA_LICENCES = (
    "Fitted only on SciFact train (CC BY 4.0 claims, ODC-By 1.0 abstracts) with local open-model features. No "
    "HealthVer, Climate-FEVER or COVID-Fact data, and no hosted-model (LLM API) output. See DATA_LICENSES.md."
)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_rows():
    items = _load(os.path.join(CACHE, "items.json"))
    local = _load(os.path.join(CACHE, "local.json"))
    rows = []
    for it in items:
        if it["ds"] not in (*FIT_DS, TEST_DS):
            continue
        lf = local[it["id"]]  # KeyError = missing local features: the cache is incomplete, stop
        rows.append(
            {
                "id": it["id"],
                "ds": it["ds"],
                "claim": it["claim"],
                "evidence": list(it["sents"]),
                "label": it["label3"],
                "scores": {
                    "nli": {"S": lf["nli_entail"], "C": lf["nli_contra"]},
                    "nli2": {"S": lf["nli2_entail"], "C": lf["nli2_contra"]},
                    "overlap": {"S": lf["overlap"]},
                },
            }
        )
    rows.sort(key=lambda r: hashlib.sha256(("fold" + r["id"]).encode()).hexdigest())
    return [r for r in rows if r["ds"] in FIT_DS], [r for r in rows if r["ds"] == TEST_DS]


def check_against_research(pool):
    """The public fit API must reproduce the research pipeline exactly on the same rows (a = 1, no judges)."""
    for p in (os.path.join(ROOT, "phase4"), os.path.join(ROOT, "phase3")):
        sys.path.insert(0, p)
    import evaluate as E  # noqa: E402
    import export_policy as X4  # noqa: E402

    rr = [
        {"id": r["id"], "claim": r["claim"], "label3": r["label"], "cond": {"gold": {"src": r["scores"], "a": 1}}}
        for r in pool
    ]
    out = {}
    for name, srcs in PROFILES.items():
        calj, rel, cut, hw = X4.fit(rr, list(srcs))
        m = FIT._fit(pool, srcs)
        mine = {s: {t: [list(v[0]), list(v[1])] for (s2, t), v in m.cal.items() if s2 == s} for s in srcs}
        assert mine == calj and m.rel == rel and m.cut == cut and m.hedge_width == hw, name
        o, _, c = E.oof_effs(rr, "gold", list(srcs), "logit")
        o2, c2, _ = FIT._oof(pool, srcs, 5, {})
        assert o == o2 and c == c2, name
        out[name] = "identical (calibrators, reliabilities, cut-off, hedge width, out-of-fold posteriors and cut-offs)"
    return out


def evaluate(policy, rows, profile):
    """Every held-out row through the shipped FrozenNode, a = 1 (gold evidence shown verbatim)."""
    node = FrozenNode(os.path.join(tempfile.mkdtemp(prefix="rwn_p6_"), "trace.jsonl"), policy=policy, profile=profile)
    srcs = node.prof.sources
    out = {"n": len(rows), "ACCEPT": 0, "NOT_SURE": 0, "BLOCK": 0, "accepted_true": 0, "false_accepts": 0}
    out.update(blocked_true=0, supported=0, errors_or_mismatch=0)
    for r in rows:
        v = node.decide(r["claim"], {s: r["scores"][s] for s in srcs}, 1)
        y = r["label"] == "SUPPORT"
        out["errors_or_mismatch"] += v.reason.startswith("error") or v.reason == "internal_mismatch"
        out["supported"] += y
        out[v.outcome] += 1
        if v.outcome == "ACCEPT":
            out["accepted_true" if y else "false_accepts"] += 1
        if v.outcome == "BLOCK" and y:
            out["blocked_true"] += 1
    acc, dec = out["ACCEPT"], out["ACCEPT"] + out["BLOCK"]
    correct = out["accepted_true"] + (out["BLOCK"] - out["blocked_true"])
    out["accept_rate"] = acc / out["n"]
    out["accept_precision"] = out["accepted_true"] / acc if acc else None
    out["false_accept_share_upper95_cp"] = cp_upper(out["false_accepts"], acc) if acc else None
    out["coverage_of_supported"] = out["accepted_true"] / out["supported"] if out["supported"] else None
    out["decided"] = dec
    out["accuracy_decided"] = correct / dec if dec else None
    out["accuracy_all_not_sure_as_reject"] = (
        out["accepted_true"] + (out["n"] - out["supported"] - out["false_accepts"])
    ) / out["n"]
    return out


def build_doc(pool):
    """The shipped policy document, fitted with the public fit API (deterministic for the same rows)."""
    profiles = {
        name: FIT.fit_profile(
            pool,
            srcs,
            folds=5,
            K=K,
            alphas=ALPHAS,
            delta=0.05,
            n_tests=len(PROFILES),
            pool_note=POOL_NOTE.format(n=len(pool)),
        )
        for name, srcs in PROFILES.items()
    }
    doc = FIT.build_policy(
        profiles,
        version=VERSION,
        default_profile="local",
        data_licences=DATA_LICENCES,
        fit_script="phase6/refit_policy_43.py (roman_wheel_node.fit)",
    )
    return doc


def main() -> None:
    if os.environ.get("XAI_API_KEY"):
        sys.exit("refusing to run with XAI_API_KEY set: the refit makes no API call (use env -u XAI_API_KEY)")
    pool, test = load_rows()
    if not pool or not test:
        sys.exit("SciFact rows missing from phase3/cache/step2: run phase3/s2_data.py and the local feature step")
    port = check_against_research(pool)
    doc = build_doc(pool)
    FIT.write_policy(doc, OUT)
    pp = os.path.join(ROOT, "src", "roman_wheel_node", "policy.py")
    s = open(pp, encoding="utf-8").read()
    s = re.sub(r'DEFAULT_POLICY_SHA256 = "[^"]*"', f'DEFAULT_POLICY_SHA256 = "{doc["sha256"]}"', s)
    open(pp, "w", encoding="utf-8").write(s)
    new = load_policy(OUT, expected_sha256=doc["sha256"])

    prev_txt = subprocess.run(
        ["git", "-C", ROOT, "show", f"{PREV_COMMIT}:{PREV_PATH}"], check=True, capture_output=True, text=True
    ).stdout
    prev = json.loads(prev_txt)
    res = {
        "protocol": __doc__,
        "fit_pool_n": len(pool),
        "fit_pool_by_ds": {d: sum(r["ds"] == d for r in pool) for d in FIT_DS},
        "test": TEST_DS,
        "test_n": len(test),
        "policy_4_3_0_sha256": doc["sha256"],
        "port_check": port,
        "previous": f"rwn-policy-4.2.0 {PREV_SHA} (git {PREV_COMMIT}); its numbers: results/phase6/refit_42.json",
        "profiles": {},
    }
    prev42 = _load(os.path.join(RES, "refit_42.json"))["profiles"]
    for name in PROFILES:
        p = new.profile(name)
        res["profiles"][name] = {
            "4.3.0": evaluate(new, test, name),
            "4.2.0": prev42[name]["4.2.0"],
            "params": {
                "4.3.0": {"cutoff": p.cutoff, "margin": p.accept_margin, "threshold": p.accept_threshold()},
                "4.2.0": {
                    "cutoff": prev["profiles"][name]["cutoff"],
                    "margin": prev["profiles"][name]["accept_margin"],
                },
            },
            "preregistration": p.raw["preregistration"],
            "ltt": p.raw["ltt"],
            "certified": p.certified,
        }
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "refit_43.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
        f.write("\n")
    txt = report(res)
    with open(os.path.join(RES, "refit_43.txt"), "w") as f:
        f.write(txt)
    print(txt)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def line(s) -> str:
    return (
        f"ACCEPT {s['ACCEPT']} (rate {pct(s['accept_rate'])}), false accepts {s['false_accepts']}, precision "
        f"{pct(s['accept_precision'])} (FA upper95 {pct(s['false_accept_share_upper95_cp'])}), coverage of supported "
        f"{pct(s['coverage_of_supported'])}, BLOCK {s['BLOCK']}, NOT_SURE {s['NOT_SURE']}, handled correctly "
        f"(NOT_SURE=reject) {pct(s['accuracy_all_not_sure_as_reject'])}"
    )


def report(res) -> str:
    L = [
        f"Policy 4.3.0 (local open-model features only; SciFact train n={res['fit_pool_n']}; no hosted-model output)",
        f"held-out: {res['test']} n={res['test_n']}; policy 4.3.0 sha256 {res['policy_4_3_0_sha256']}",
        "port check vs research pipeline: " + "; ".join(f"{k}: {v}" for k, v in res["port_check"].items()),
        "",
    ]
    for name, d in res["profiles"].items():
        L.append(f"[{name}]")
        L.append(f"  4.3.0: {line(d['4.3.0'])}")
        L.append(f"  4.2.0 (withdrawn; margin depended on a hosted-model quote check): {line(d['4.2.0'])}")
        p3, p2 = d["params"]["4.3.0"], d["params"]["4.2.0"]
        L.append(
            f"  params: 4.3.0 cutoff {p3['cutoff']:.3f} margin {p3['margin']:.3f} threshold {p3['threshold']:.4f}; "
            f"4.2.0 cutoff {p2['cutoff']:.3f} margin {p2['margin']:.3f}"
        )
        pr = d["preregistration"]
        L.append(
            f"  preregistration (out-of-fold, train): step {pr['step']}, {pr['accepted_oof']} accepted / "
            f"{pr['false_accepts_oof']} false, upper95 {pct(pr['upper95'])}"
        )
        for a, c in d["ltt"]["alphas"].items():
            L.append(
                f"  LTT alpha {float(a):.0%} (delta {d['ltt']['delta']}/{d['ltt']['n_tests']}): "
                + (
                    f"certified, {c['accepted']} accepted / {c['false_accepts']} false, upper {pct(c['upper'])}"
                    if c["certified"]
                    else "not certified"
                )
            )
        L.append(f"  certified: {d['certified'].get('alpha') or d['certified'].get('note')}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
