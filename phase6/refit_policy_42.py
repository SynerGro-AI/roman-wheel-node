"""Policy 4.2.0: refit every shipped profile on commercial-use data only (zero API calls; judges read from cache).

Why: rwn-policy-4.0.0 / 4.1.0 were fitted on SciFact train + HealthVer train. The owner's rule for anything that ships
is "fitted only on data whose licence permits commercial use, verified at the primary source" (DATA_LICENSES.md).
Of the datasets this project has used, SciFact (claims CC BY 4.0, abstracts ODC-By 1.0) is the only one in scope:
HealthVer is excluded by the owner's decision, Climate-FEVER and COVID-Fact have no licence at their primary source.

Protocol (fixed before any SciFact dev number was computed for 4.2.0):
* Fit pool: SciFact train with gold evidence (the 799 claims of the Phase 3e SciFact population; the 10 J9 few-shot
  claims stay excluded). Same fitting code as 4.0.0 (phase4/export_policy.fit): isotonic calibrators, Beta
  reliabilities, learned cut-off.
* local / local-lite: accept margin pre-registered exactly as in 4.0.0 (phase4/export_policy.prereg_margin), on the
  SciFact-only out-of-fold posteriors.
* panel: the same structural config as 4.0.0's panel (bayes-logit over all nine sources, base rule, accept-only,
  rho = 0), calibrated on SciFact only. Its margin comes from LTT fixed-sequence testing (K = 40, one-sided
  Clopper-Pearson, delta = 0.05, accept-precision risk) on the SciFact-only out-of-fold posteriors. The family is this
  one config (n_tests = 1), declared in advance. The 5% certificate is used if it exists, else 10%; if neither
  certifies, the panel profile is not shipped.
* Gate: fp2 exact-copy fingerprint, as 4.1.0.
* Held-out evaluation: SciFact dev (300 claims, gold evidence; never used to fit 4.0.0 or 4.2.0, but its numbers were
  reported in Phase 3, so it is not pristine). Both 4.0.0 (from git, 3857154) and 4.2.0 are run through the same
  runtime (roman_wheel_node.shadow.FrozenNode) on the same cached source outputs.
Outputs: src/roman_wheel_node/policies/rwn-policy-4.2.0.json, the pin in policy.py, results/phase6/refit_42.{json,txt}
(aggregates only).

    env -u XAI_API_KEY python3 phase6/refit_policy_42.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

if os.environ.get("XAI_API_KEY"):
    sys.exit("refusing to run with XAI_API_KEY set: the refit makes no API call (use env -u XAI_API_KEY)")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "phase4"), os.path.join(ROOT, "phase3"), os.path.join(ROOT, "src")):
    sys.path.insert(0, p)
import evaluate as E  # noqa: E402  (research scripts live outside the package)
import export_policy as X4  # noqa: E402
import judges as JD  # noqa: E402
import s2_eval as S2  # noqa: E402
import s3e_eval as S3E  # noqa: E402
import s3e_rules as R  # noqa: E402

from roman_wheel_node.fingerprint import NORMALIZE_V2_TABLE_SHA256  # noqa: E402
from roman_wheel_node.policy import load_policy, policy_sha256  # noqa: E402
from roman_wheel_node.shadow import FrozenNode  # noqa: E402

COND = "gold"
FIT_DS = ("scifact_train350", "scifact_train_unused")
TEST_DS = "scifact_dev"
PANEL = list(S2.PANEL)
PROFILES = {"panel": [*PANEL, "nli", "nli2", "overlap"], "local": ["nli2", "overlap"], "local-lite": ["nli", "overlap"]}
ALPHAS = (0.10, 0.05, 0.02, 0.01)
K = 40
BASELINE_COMMIT = "3857154"
BASELINE_PATH = "src/roman_wheel_node/policies/rwn-policy-4.0.0.json"
BASELINE_SHA = "254804f8ac9b500ef1951879fb64328b0f54303b8bef20e38b1684545f876eb1"
OUT = os.path.join(ROOT, "src", "roman_wheel_node", "policies", "rwn-policy-4.2.0.json")
RES = os.path.join(ROOT, "results", "phase6")
POOL_NOTE = (
    "SciFact train, gold evidence ({n} claims). Claims and evidence annotations CC BY 4.0, abstracts ODC-By 1.0 "
    "(github.com/allenai/scifact LICENSE.md, a5254a1). See DATA_LICENSES.md."
)


def panel_certificate(pool, F):
    o, d, c = E.oof_effs(pool, COND, PROFILES["panel"], "logit")
    feats = [dict(F[r["id"]], dis=d[i]) for i, r in enumerate(pool)]
    items = R.to_items(feats, o, c, R.make_rules(j11=False)["base"], False)
    return R.certify_family({"scifact|bayes-logit [all]|base": items}, ALPHAS, 0.05, (0.0,), K, True)


def evaluate(policy, rows, profile):
    """Run every held-out row through FrozenNode. Panel uses the row's gate bit (shown units + J1 verbatim quote);
    the local profiles use the shown-units gate only (gold evidence is shown verbatim, so a = 1)."""
    node = FrozenNode(os.path.join(tempfile.mkdtemp(prefix="rwn_p6_"), "trace.jsonl"), policy=policy, profile=profile)
    srcs = node.prof.sources
    out = {
        "n": len(rows),
        "ACCEPT": 0,
        "NOT_SURE": 0,
        "BLOCK": 0,
        "accepted_true": 0,
        "false_accepts": 0,
        "blocked_true": 0,
        "supported": 0,
        "errors_or_mismatch": 0,
    }
    for r in rows:
        g = r["cond"][COND]
        raw = {s: g["src"][s] for s in srcs}
        a = g["a"] if profile == "panel" else 1
        v = node.decide(r["claim"], raw, a)
        y = r["label3"] == "SUPPORT"
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
    out["false_accept_share_upper95_cp"] = R.cp_upper(out["false_accepts"], acc) if acc else None
    out["coverage_of_supported"] = out["accepted_true"] / out["supported"] if out["supported"] else None
    out["decided"] = dec
    out["accuracy_decided"] = correct / dec if dec else None
    out["accuracy_all_not_sure_as_reject"] = (
        out["accepted_true"] + (out["n"] - out["supported"] - out["false_accepts"])
    ) / out["n"]
    return out


HISTORICAL_43 = (
    "superseded: policy 4.2.0 no longer ships (its panel profile was fitted on xAI judge outputs and its local margins "
    "used the J1 quote gate). Shipped policy: 4.3.0 (phase6/refit_policy_43.py). This run's aggregates are kept as a "
    "benchmark ledger only (results/phase6/xai_panel_ledger.md); reproduce it from git commit 69aa8db."
)


def main() -> None:
    sys.exit(HISTORICAL_43)
    rows, skipped = S2.build_rows()
    F = S3E.features(rows)
    pool = [r for r in rows if r["ds"] in FIT_DS]
    test = [r for r in rows if r["ds"] == TEST_DS]
    if not pool or not test:
        sys.exit(
            "SciFact rows missing from the cache: run phase3/s2_data.py and the Phase 3d feature/judge steps first"
        )
    cert = panel_certificate(pool, F)
    chosen = next((al for al in ("0.05", "0.1") if cert["alphas"][al].get("certified")), None)
    profiles = {}
    for name, srcs in PROFILES.items():
        if name == "panel" and chosen is None:
            continue
        calj, rel, cut, hw = X4.fit(pool, srcs)
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
                "pool": POOL_NOTE.format(n=len(pool)),
                "n": len(pool),
                "min_isotonic_block_note": "isotonic blocks may summarise few calibration items",
            },
        }
        if name == "panel":
            c = cert["alphas"][chosen]
            c10 = cert["alphas"]["0.1"]
            p.update(
                accept_margin=c["m_a"],
                disagreement={"judge_split": True, "nli2_contradiction": True},
                gate_judge_quote="J1",
                certified={
                    "risk": "accept_precision (false accepts among ACCEPTs; BLOCK reported separately)",
                    "population": f"SciFact train (out-of-fold, n={len(pool)})",
                    "alpha": float(chosen),
                    "delta": 0.05,
                    "phase": "6 (4.2.0 refit)",
                    "family": "1 config (scifact|bayes-logit [all]|base, rho 0), declared before fitting",
                    "calibration_accepted": c["accepted"],
                    "calibration_false_accepts": c["false_accepts"],
                    "upper95": c["upper"],
                    "not_certified": [f"alpha {a}" for a in ALPHAS if not cert["alphas"][f"{a}"].get("certified")]
                    + ["any population other than SciFact"],
                    "alpha_0.10_margin": c10["m_a"] if c10.get("certified") else None,
                },
                judges={
                    n: {k: JD.JUDGES[n][k] for k in ("model", "style", "max_tokens", "temperature", "samples")}
                    for n in PANEL
                },
            )
        else:
            pr = X4.prereg_margin(pool, srcs)
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
                    "fixed K=40 accept-only margin path, SciFact-only out-of-fold posteriors",
                ),
            )
        profiles[name] = p
    cert_line = (
        f"Certified only: profile 'panel', SciFact population, accept-precision <= {float(chosen):.0%} (delta 0.05, "
        "4.2.0 refit on SciFact train only)."
        if chosen
        else "No profile is certified at any level."
    )
    doc = {
        "format": "rwn-policy",
        "version": "4.2.0",
        "outcomes": ["ACCEPT", "NOT_SURE", "BLOCK"],
        "default_profile": "local",
        "fit_script": "phase6/refit_policy_42.py",
        "data_licences": "Fitted only on SciFact train (CC BY 4.0 claims, ODC-By 1.0 abstracts). No HealthVer, "
        "Climate-FEVER or COVID-Fact data. See DATA_LICENSES.md.",
        "replaces": {
            "4.0.0": BASELINE_SHA,
            "4.1.0": "2156f5844d0d1fdad9d415b0aba0226c8c6b17a8a9b2294546b166dff4c3c82d",
            "why": "fitted partly on HealthVer, excluded from shipped files",
        },
        "gate": {
            "fingerprint": "fp2",
            "normalize": "NFC; drop U+200B-200D,2060,FEFF,00AD; curly/typographic quotes "
            "-> ' and \"; hyphen/dash/minus variants -> -; whitespace runs -> one space; strip. Not NFKC.",
            "normalize_table_sha256": NORMALIZE_V2_TABLE_SHA256,
            "superset_of": "fp1 (a unit passes if its raw sha256 or its fp2 fingerprint is trusted)",
        },
        "fail_closed": "missing or erroring source, malformed score, disagreement flag, or posterior below the accept "
        "margin -> NOT_SURE (never admitted); exact-copy gate failure -> BLOCK",
        "honesty": cert_line
        + " Nothing is certified at 1% or 2%. The free local profiles are not certified at any level.",
        "profiles": profiles,
    }
    doc["sha256"] = policy_sha256(doc)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    pp = os.path.join(ROOT, "src", "roman_wheel_node", "policy.py")
    s = open(pp, encoding="utf-8").read()
    s = re.sub(r'DEFAULT_POLICY_SHA256 = "[^"]*"', f'DEFAULT_POLICY_SHA256 = "{doc["sha256"]}"', s)
    open(pp, "w", encoding="utf-8").write(s)

    new = load_policy(OUT, expected_sha256=doc["sha256"])
    base_txt = subprocess.run(
        ["git", "-C", ROOT, "show", f"{BASELINE_COMMIT}:{BASELINE_PATH}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    bp = os.path.join(tempfile.mkdtemp(prefix="rwn_p6_base_"), "rwn-policy-4.0.0.json")
    open(bp, "w", encoding="utf-8").write(base_txt)
    old = load_policy(bp, expected_sha256=BASELINE_SHA)
    res = {
        "protocol": __doc__,
        "skipped_incomplete": skipped,
        "fit_pool_n": len(pool),
        "test": TEST_DS,
        "test_n": len(test),
        "policy_4_2_0_sha256": doc["sha256"],
        "baseline": f"rwn-policy-4.0.0 {BASELINE_SHA} (git {BASELINE_COMMIT})",
        "panel_certificate_4_2_0": {a: {k: v for k, v in x.items() if k != "path"} for a, x in cert["alphas"].items()},
        "panel_certificate_tests": cert["n_tests"],
        "profiles": {},
    }
    for name in PROFILES:
        res["profiles"][name] = {
            "4.0.0": evaluate(old, test, name),
            "4.2.0": evaluate(new, test, name) if name in profiles else "not shipped (no certificate)",
        }
        if name in profiles:
            res["profiles"][name]["params"] = {
                "4.0.0": {
                    "cutoff": old.profile(name).cutoff,
                    "margin": old.profile(name).accept_margin,
                    "threshold": old.profile(name).accept_threshold(),
                },
                "4.2.0": {
                    "cutoff": new.profile(name).cutoff,
                    "margin": new.profile(name).accept_margin,
                    "threshold": new.profile(name).accept_threshold(),
                },
            }
    os.makedirs(RES, exist_ok=True)
    json.dump(res, open(os.path.join(RES, "refit_42.json"), "w", encoding="utf-8"), indent=1, default=float)
    txt = report(res)
    open(os.path.join(RES, "refit_42.txt"), "w", encoding="utf-8").write(txt)
    print(txt)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def report(res) -> str:
    L = [
        f"Policy 4.2.0 refit (SciFact train only, n={res['fit_pool_n']}) vs 4.0.0 (SciFact + HealthVer train)",
        f"held-out: {res['test']} n={res['test_n']}; policy 4.2.0 sha256 {res['policy_4_2_0_sha256']}",
        "",
    ]
    L.append(f"4.2.0 panel certificate (SciFact train out-of-fold, {res['panel_certificate_tests']} test, delta 0.05):")
    for a, c in res["panel_certificate_4_2_0"].items():
        L.append(
            f"  alpha {float(a):.0%}: "
            + (
                f"certified, {c['decided']} accepted / {c['errors']} false, upper95 {pct(c['upper'])}"
                if c.get("certified")
                else "not certified"
            )
        )
    L.append("")
    for name, d in res["profiles"].items():
        L.append(f"[{name}]")
        for ver in ("4.0.0", "4.2.0"):
            s = d[ver]
            if isinstance(s, str):
                L.append(f"  {ver}: {s}")
                continue
            L.append(
                f"  {ver}: ACCEPT {s['ACCEPT']} (rate {pct(s['accept_rate'])}), false accepts {s['false_accepts']}, "
                f"precision {pct(s['accept_precision'])} (FA upper95 {pct(s['false_accept_share_upper95_cp'])}), "
                f"coverage of supported {pct(s['coverage_of_supported'])}, BLOCK {s['BLOCK']}, NOT_SURE "
                f"{s['NOT_SURE']}, accuracy decided {pct(s['accuracy_decided'])}, accuracy all (NOT_SURE=reject) "
                f"{pct(s['accuracy_all_not_sure_as_reject'])}"
            )
        if "params" in d:
            L.append(
                "  params: "
                + "; ".join(
                    f"{v} cutoff {p['cutoff']:.3f} margin {p['margin']:.3f} threshold {p['threshold']:.4f}"
                    for v, p in d["params"].items()
                )
            )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
