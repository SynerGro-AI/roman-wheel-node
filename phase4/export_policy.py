"""Phase 4: export the frozen deployable policy (zero API calls; judges read from cache only).

Profiles (all fitted on the Phase 3e POOLED calibration pool: SciFact train 799 + HealthVer train 1,999, gold evidence):
* panel      - the Phase 3e certified configuration: pooled | bayes-logit [all = J1 J2 J4 J7 J9 J10 + nli nli2 overlap]
               | base rule | accept-only, margin = the 3e alpha-5% certificate (SciFact population, accept-precision).
               Needs the paid xAI panel (opt-in).
* local      - free, deterministic: bayes-logit over nli2 (DeBERTa-v3-base mnli-fever-anli, torch) + overlap. The ONNX
               int8 nli source is left out because it is not deterministic on this CPU (same input, different outputs).
* local-lite - free, ONNX only: bayes-logit over nli + overlap.
For the local profiles the accept-only margin is PRE-REGISTERED from calibration only: the step of the fixed K=40
margin path (top q/40 of out-of-fold accepts) with the lowest one-sided 95% Clopper-Pearson upper bound on the share of
false accepts, on the pooled out-of-fold posteriors. No fresh-set label exists when this runs.
The policy body contains fitted parameters only (isotonic breakpoints, reliabilities, cut-offs), no claims or labels.
Output: src/roman_wheel_node/policies/rwn-policy-4.0.0.json and the pinned hash in src/roman_wheel_node/policy.py.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import tempfile

if os.environ.get("XAI_API_KEY"):
    sys.exit("refusing to run with XAI_API_KEY set: the export makes no API call (use env -u XAI_API_KEY)")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "phase3"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import evaluate as E  # noqa: E402
import s2_eval as S2  # noqa: E402
import s3e_rules as R  # noqa: E402

from roman_wheel_node import LiveWeights, ThresholdLearner, give_feedback, roman_wheel_node  # noqa: E402
from roman_wheel_node.policy import policy_sha256  # noqa: E402

HISTORICAL = (
    "superseded: rwn-policy-4.0.0/4.1.0 were fitted partly on HealthVer and no longer ship (policy 4.2.0, "
    "phase6/refit_policy_42.py). Reproduce this historical run from git commit 3857154."
)

COND = "gold"
PANEL = list(S2.PANEL)
PROFILES = {"panel": PANEL + ["nli", "nli2", "overlap"], "local": ["nli2", "overlap"], "local-lite": ["nli", "overlap"]}
OUT = os.path.join(ROOT, "src", "roman_wheel_node", "policies", "rwn-policy-4.0.0.json")
K = 40


def fit(pool, srcs):
    cal = E.fit_cal(pool, COND, srcs)

    def sc(r):
        return {s: E.platt(cal[(COND, s, "S")], r["cond"][COND]["src"][s]["S"]) for s in srcs}

    tmp = tempfile.mkdtemp(prefix="rwn_p4_")
    lw = LiveWeights(os.path.join(tmp, "w.jsonl"), half_life=None)
    for r in pool:
        give_feedback(sc(r), r["label3"] == "SUPPORT", live=lw)
    thr = ThresholdLearner(os.path.join(tmp, "t.jsonl"), half_life=None)
    for r in pool:
        eff, _, _ = roman_wheel_node(1, sc(r), live=lw, mode="bayes", variant="logit")
        thr.record(eff, r["label3"] == "SUPPORT", "bayes:logit")
    calj = {}
    for s in srcs:
        calj[s] = {"S": [list(v) for v in cal[(COND, s, "S")]]}
        if (COND, s, "C") in cal:
            calj[s]["C"] = [list(v) for v in cal[(COND, s, "C")]]
    return calj, {s: lw.reliability(s) for s in srcs}, thr.cutoff("bayes:logit"), thr.hedge_width


def prereg_margin(pool, srcs):
    o, d, c = E.oof_effs(pool, COND, srcs, "logit")
    feats = [
        {"y": r["label3"] == "SUPPORT", "blocked": r["cond"][COND]["a"] == 0, "dis": d[i]} for i, r in enumerate(pool)
    ]
    its = R.to_items(feats, o, c, (lambda f: not f["dis"], lambda f: not f["dis"]), False)
    best = None
    for q, (ma, _) in enumerate(R.path(its, 0.0, K), 1):
        n, k, _, _ = R.risk(its, ma, R.INF, True)
        ub = R.cp_upper(k, n, 0.05)
        if math.isfinite(ma) and (best is None or ub <= best["upper95"]):
            best = {
                "step": q,
                "margin": ma,
                "accepted_oof": n,
                "false_accepts_oof": k,
                "upper95": ub,
                "n_pool": len(pool),
            }
    return best


def main() -> None:
    sys.exit(HISTORICAL)
    rows, _ = S2.build_rows()
    pool = [r for r in rows if r["ds"] in S2.CALIB]
    cert = json.load(open(os.path.join(ROOT, "results", "phase3", "step3e.json"), encoding="utf-8"))["risks"][
        "accept_precision"
    ]["scifact"]["certificates"]["alphas"]
    c5, c10 = cert["0.05"], cert["0.1"]
    assert c5["certified"] and c5["config"] == "pooled|bayes-logit [all]|base" and c5["rho"] == 0.0
    import judges as JD

    profiles = {}
    for name, srcs in PROFILES.items():
        calj, rel, cut, hw = fit(pool, srcs)
        p = {
            "sources": srcs,
            "variant": "logit",
            "prior": 0.5,
            "reliability": rel,
            "cutoff": cut,
            "hedge_width": hw,
            "calibrators": calj,
            "needs_judges": name == "panel",
            "calibration": {
                "pool": "pooled: SciFact train (799) + HealthVer train (1,999; internal, unlicensed)",
                "n": len(pool),
                "min_isotonic_block_note": "isotonic blocks may summarise few calibration items",
            },
        }
        if name == "panel":
            p.update(
                accept_margin=c5["m_a"],
                disagreement={"judge_split": True, "nli2_contradiction": True},
                gate_judge_quote="J1",
                certified={
                    "risk": "accept_precision (false accepts among ACCEPTs; BLOCK reported separately)",
                    "population": "SciFact (calibration out-of-fold, n=799)",
                    "alpha": 0.05,
                    "delta": 0.05,
                    "phase": "3e",
                    "calibration_accepted": c5["accepted"],
                    "calibration_false_accepts": c5["false_accepts"],
                    "upper95": c5["upper"],
                    "not_certified": ["alpha 0.02", "alpha 0.01", "HealthVer", "any other population"],
                    "alpha_0.10_margin": c10["m_a"],
                },
                judges={
                    n: {k: JD.JUDGES[n][k] for k in ("model", "style", "max_tokens", "temperature", "samples")}
                    for n in PANEL
                },
            )
        else:
            pr = prereg_margin(pool, srcs)
            p.update(
                accept_margin=pr["margin"],
                disagreement={},
                gate_judge_quote=None,
                certified={
                    "alpha": None,
                    "note": "NOT certified at any level: free local sources are too weak on the "
                    "calibration pool (best out-of-fold upper bound on the false-accept share shown in "
                    "preregistration)",
                },
                preregistration=dict(
                    pr,
                    rule="argmin one-sided 95% CP upper bound of the false-accept share over the "
                    "fixed K=40 accept-only margin path, pooled out-of-fold posteriors; chosen before any fresh label",
                ),
            )
        profiles[name] = p
    doc = {
        "format": "rwn-policy",
        "version": "4.0.0",
        "outcomes": ["ACCEPT", "NOT_SURE", "BLOCK"],
        "source_commit": "7466880 (Phase 3e)",
        "default_profile": "local",
        "fail_closed": "missing or erroring source, malformed score, disagreement flag, or posterior below the accept "
        "margin -> NOT_SURE (never admitted); exact-copy gate failure -> BLOCK",
        "honesty": "Certified only: profile 'panel', SciFact population, accept-precision <= 5% (delta 0.05, Phase "
        "3e). "
        "Nothing is certified at 1% or 2%. The free local profiles are not certified at any level.",
        "profiles": profiles,
    }
    doc["sha256"] = policy_sha256(doc)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")
    pp = os.path.join(ROOT, "src", "roman_wheel_node", "policy.py")
    s = open(pp, encoding="utf-8").read()
    s = re.sub(r'DEFAULT_POLICY_SHA256 = "[^"]*"', f'DEFAULT_POLICY_SHA256 = "{doc["sha256"]}"', s)
    open(pp, "w", encoding="utf-8").write(s)
    for n, p in profiles.items():
        print(
            n,
            "cutoff",
            round(p["cutoff"], 3),
            "margin",
            round(p["accept_margin"], 4),
            p.get("preregistration", p["certified"]),
        )
    print("wrote", OUT, doc["sha256"])


if __name__ == "__main__":
    main()
