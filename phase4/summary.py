"""Phase 4: side-by-side table of the fresh-set certificates (aggregates only) -> results/phase4/SUMMARY.md."""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "phase4")
NAMES = {
    "local": "local (free: nli2 + overlap)",
    "local-lite": "local-lite (free: ONNX nli + overlap)",
    "panel": "panel (frozen Phase 3e rule: 6 xAI judges + local)",
}


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def main():
    L = [
        "# Phase 4 fresh-set results (aggregates only)",
        "",
        "Fresh = never used to select or tune any rule: HealthVer test pairs left unjudged in Phase 3d "
        "(`hv_test_unjudged`), COVID-Fact (`covidfact`), HealthVer dev (`hv_dev`, local profiles only). HealthVer "
        "and COVID-Fact are internal, unlicensed datasets: no claim, evidence or per-item output is committed. "
        "Labels: dataset gold (SUPPORT vs not).",
        "",
        "Risk = false accepts among ACCEPTs; one-sided Clopper-Pearson, delta = 0.05; one pre-registered fixed rule "
        "per profile (no search). Precision CI = two-sided 95% Clopper-Pearson. 'needs' = labelled claims needed at "
        "the observed ACCEPT rate if every accept were correct (299 / 149 / 59 error-free accepts for 1% / 2% / 5%).",
        "",
        "| profile | set | labelled | ACCEPT (coverage) | false accepts | precision [95% CI] | UB on false-accept "
        "share | 1% | 2% | 5% | needs for 1% / 2% |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in ("local", "local-lite", "panel"):
        f = os.path.join(R, f"fresh_{p}.json")
        if not os.path.exists(f):
            continue
        res = json.load(open(f, encoding="utf-8"))
        for g, s in res["groups"].items():
            ci = s["precision_ci95"]
            a = s["alphas"]

            def cert(al):
                return "**yes**" if a[al]["certified"] else "no"

            def need(al):
                return a[al]["labelled_needed_at_observed_coverage_zero_errors"] or "n/a"

            L.append(
                f"| {NAMES[p]} | {g.replace('ds=', '')} | {s['labelled']} | {s['accepted']} ({pct(s['coverage'])}) | "
                f"{s['false_accepts']} | {pct(s['precision'])}"
                + (f" [{pct(ci[0])}, {pct(ci[1])}]" if ci else "")
                + f" | {pct(s['false_accept_share_upper'])} | {cert('0.01')} | {cert('0.02')} | {cert('0.05')} | "
                f"{need('0.01')} / {need('0.02')} |"
            )
    open(os.path.join(R, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
