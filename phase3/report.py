"""Stage 4: text summary + charts from results/phase3/scifact.json."""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "phase3")
ORDER0 = [
    "nli alone",
    "overlap alone",
    "J1 alone",
    "J2 alone",
    "J3 alone",
    "J1 unwrapped (raw argmax, no node)",
    "judge majority (raw argmax votes)",
    "static node (geomean, equal trust)",
    "bayes-hard",
    "bayes-ess",
    "bayes-ess live (prequential)",
    "bayes-ess, gate off (ablation)",
    "bayes-ess, judges only",
    "bayes-ess, local only (no API)",
    "reference: logistic stack (not gated)",
]


def order(C) -> list:
    ks = list(C["binary"])
    sel = C["selected"]
    singles = sorted(k for k in ks if k.endswith(" alone"))
    unwr = sorted(k for k in ks if k.endswith(" unwrapped"))
    rest = [k for k in ORDER0 if k in ks and k not in singles and k not in unwr and k != sel]
    rest += sorted(
        k
        for k in ks
        if k not in rest
        and k not in singles
        and k not in unwr
        and k != sel
        and k != "J1 unwrapped (raw argmax, no node)"
    )
    return singles + unwr + rest + [sel]


def pct(v):
    return f"{100 * v[0]:5.1f} [{100 * v[1]:.1f}, {100 * v[2]:.1f}]"


def text(R) -> str:
    L = [
        f"SciFact dev (n={R['n_dev']}: {R['dev_label_counts']}), learned on {R['n_train']} train claims only.",
        f"Judges: {R['judges']}",
        "BM25 recall@3 (dev SUPPORT+CONTRADICT claims): "
        f"{100 * R['retrieval']['recall_at_3_supported_or_contradicted']:.1f}%",
        "",
    ]
    for cond, C in R["conditions"].items():
        L.append(
            f"=== condition: {cond} ===  binary task (supported vs not); accuracy and false-accept rate with 95% "
            "bootstrap CIs"
        )
        L.append(f"{'policy':42s} {'accuracy %':22s} {'false-accept %':22s} {'TA %':>6s} {'ECE':>6s}")
        for k in order(C):
            t = C["binary"][k]
            e = f"{t['ece']:.3f}" if t["ece"] is not None else "  -  "
            L.append(
                f"{k:42s} {pct(t['accuracy']):22s} {pct(t['false_accept']):22s} {100 * t['true_accept']:6.1f} {e:>6s}"
            )
        L.append("3-way (SUPPORT / CONTRADICT / NEI): accuracy [95% CI], macro-F1")
        for k, t in C["three_way"].items():
            L.append(f"  {k:40s} {pct(t['accuracy']):22s} F1 {t['macro_f1']:.3f}")
        L.append(
            "Safety: unsupported dev claims accepted, vs J1 unwrapped (exact McNemar; paired bootstrap of the drop)"
        )
        for k, s in C["safety_vs_unwrapped_J1"].items():
            d = s["fa_drop_paired_boot"]
            g = s["accuracy_gain_vs_baseline"]
            L.append(
                f"  {k:40s} FA {s['false_accepts_baseline']:3d} -> {s['false_accepts_policy']:3d} of "
                f"{s['n_unsupported']}  (only-baseline {s['only_baseline_false_accepts']}, only-policy "
                f"{s['only_policy_false_accepts']}, McNemar p={s['mcnemar_p']:.4f}; drop {100 * d[0]:+.1f} pts "
                f"[{100 * d[1]:+.1f}, {100 * d[2]:+.1f}])  accuracy {100 * g[0]:+.1f} pts [{100 * g[1]:+.1f}, "
                f"{100 * g[2]:+.1f}] p={s['accuracy_mcnemar_p']:.4f}"
            )
        L.append("Score correlation (dev, Pearson on calibrated logits) / error phi:")
        for k, v in C["score_correlation_dev"].items():
            e = C["error_overlap_dev"][k]
            L.append(
                f"  {k:14s} r={v:+.2f}  both wrong {e['both_wrong']:3d} (a {e['a_wrong']}, b {e['b_wrong']})  "
                f"phi_err={e['phi_errors']:+.2f}"
            )
        L.append(
            "Train-CV accuracy of candidate node configurations: "
            f"{ {k: round(v, 4) for k, v in C['cv_train'].items()} }"
        )
        L.append(f"Selected (by train CV): {C['selected']}")
        L.append("Node vs best single judge (chosen on train accuracy), paired on dev:")
        for k, v in C["vs_best_single_judge"].items():
            d = v["diff_paired_boot"]
            L.append(
                f"  {k:44s} vs {v['judge']} (train {100 * v['judge_train_acc']:.1f}%): {100 * v['node_acc']:.1f}% vs "
                f"{100 * v['judge_acc']:.1f}%  diff {100 * d[0]:+.1f} pts [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]  "
                f"node-only-right {v['node_only_right']}, judge-only-right {v['judge_only_right']}, McNemar "
                f"p={v['mcnemar_p']:.3f}"
            )
        L.append(
            f"Error analysis of the selected node (dev errors): { {k: v['n'] for k, v in C['error_analysis'].items()} }"
        )
        if C.get("label_noise_candidates"):
            L.append(
                "Label-noise candidates (every judge confidently disagrees with gold, gold evidence): "
                f"{len(C['label_noise_candidates'])}"
            )
            for x in C["label_noise_candidates"]:
                # SciFact IDs only, never claim text (SciFact is licensed for non-commercial use)
                L.append(
                    f"  {x['key']:12s} gold={x['gold']:10s} panel={x['panel']:10s}".rstrip()
                    + f"  (SciFact {x['key'].split(':')[0]} claim id {x['claim_id']}; cited docs "
                    f"{','.join(x['cited'])})"
                )
        if C.get("selective"):
            L.append("Selective mode (abstain band picked on TRAIN out-of-fold posteriors, applied to dev unchanged):")
            for pol_, S in C["selective"].items():
                for tg, v in S.items():
                    acc = v["dev_accuracy_decided"]
                    L.append(
                        f"  {pol_:14s} target {tg}: band [{v['band'][0]:.3f}, {v['band'][1]:.3f}) train-OOF cov "
                        f"{100 * v['train_oof_coverage']:.1f}% -> dev coverage {100 * v['dev_coverage']:.1f}% "
                        f"({v['dev_decided']} decided), errors {v['dev_errors_decided']}, acc "
                        f"{'-' if acc is None else f'{100 * acc:.1f}%'}, error-rate 95% upper (Clopper-Pearson) "
                        f"{100 * v['dev_error_rate_upper95_clopper_pearson']:.2f}%, false accepts "
                        f"{v['dev_false_accepts']}/{v['dev_accepted']} accepted"
                    )
        if C.get("ltt_not_sure"):
            L.append(
                "Fail-closed NOT_SURE (Learn-Then-Test margin on train OOF + disagreement rule; through ClaimTrace):"
            )
            for pol_, S in C["ltt_not_sure"].items():
                for al, v in S.items():
                    t = v["train_oof"]
                    L.append(
                        f"  {pol_:14s} alpha {al}: log-odds margin {v['margin_logodds']:.2f}; train OOF "
                        f"{t['errors']}/{t['decided']} err, UB {100 * t['upper95']:.2f}% -> dev coverage "
                        f"{100 * v['dev_coverage']:.1f}% ({v['dev_decided']}), errors {v['dev_errors_decided']}, UB95 "
                        f"{100 * v['dev_error_rate_upper95_clopper_pearson']:.2f}%, FA "
                        f"{v['dev_false_accepts']}/{v['dev_accepted']}, {v['dev_status_counts']}"
                    )
        for nm, v in C.get("p3c_proof_or_not_sure", {}).items():
            for part in ("train_oof", "dev"):
                x = v[part]
                L.append(
                    f"  Phase 3c {nm} [{part}]: coverage {100 * x['coverage']:.1f}% ({x['decided']}), errors "
                    f"{x['errors_decided']}, UB95 {100 * x['error_rate_upper95_clopper_pearson']:.2f}%, false accepts "
                    f"{x['false_accepts']}/{x['accepted']} (UB95 {100 * x['false_accept_upper95_of_accepted']:.2f}%), "
                    f"{x['status_counts']}"
                )
        if C.get("accuracy_excluding_label_review"):
            x = C["accuracy_excluding_label_review"]
            L.append(
                f"Sensitivity, {x['n_excluded']} label_review dev claims excluded (not a headline): "
                + ", ".join(f"{k} {100 * v:.1f}%" for k, v in x["acc"].items())
            )
        L.append(f"Gate: {C['gate']}")
        L.append(f"Learned reliabilities (train): {C['node_info'].get('bayes-ess', {}).get('reliability')}")
        L.append("")
    return "\n".join(L)


def charts(R) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    nc = len(R["conditions"])
    fig, axes = plt.subplots(1, nc, figsize=(7.5 * nc, 8))
    for ax, (cond, C) in zip(axes, R["conditions"].items()):
        show = [k for k in order(C) if "ablation" not in k]
        acc = [C["binary"][k]["accuracy"] for k in show]
        cols = [
            "#c44" if (k.startswith("bayes-ess") or k.startswith("node:")) else ("#888" if "alone" in k else "#48a")
            for k in show
        ]
        ax.barh(
            range(len(show)),
            [100 * a[0] for a in acc],
            color=cols,
            xerr=[[100 * (a[0] - a[1]) for a in acc], [100 * (a[2] - a[0]) for a in acc]],
            capsize=3,
        )
        ax.axvline(90, ls="--", c="k", lw=1)
        ax.set_xlim(40, 100)
        ax.set_yticks(range(len(show)))
        ax.set_yticklabels(show, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"SciFact dev, {cond} evidence: binary accuracy (95% CI)")
        ax.set_xlabel("% correct")
    fig.tight_layout()
    p = os.path.join(OUT, "scifact_accuracy.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    fig, axes = plt.subplots(1, nc, figsize=(6 * nc, 4.8), sharey=True)
    for ax, (cond, C) in zip(axes, R["conditions"].items()):
        keys = [
            "J1 unwrapped (raw argmax, no node)",
            "judge majority (raw argmax votes)",
            "static node (geomean, equal trust)",
            "bayes-hard",
            "bayes-ess",
            "bayes-ess_cal",
            "bayes-ess live (prequential)",
            C["selected"],
        ]
        fa = [C["binary"][k]["false_accept"] for k in keys]
        ax.bar(
            range(len(keys)),
            [100 * f[0] for f in fa],
            color=["#888", "#888", "#48a", "#48a", "#c44", "#c44", "#c44", "#a22"],
            yerr=[[100 * (f[0] - f[1]) for f in fa], [100 * (f[2] - f[0]) for f in fa]],
            capsize=3,
        )
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([k.split(" (")[0] for k in keys], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{cond}: unsupported claims accepted (%)")
    fig.tight_layout()
    p = os.path.join(OUT, "scifact_false_accept.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    fig, axes = plt.subplots(1, nc, figsize=(6.5 * nc, 5.5))
    for ax, (cond, C) in zip(axes, R["conditions"].items()):
        src = sorted(
            {x for k in C["score_correlation_dev"] for x in k.split("~")}, key=lambda x: (x.startswith("J"), x)
        )
        M = [
            [
                1.0
                if a == b
                else C["score_correlation_dev"].get(f"{a}~{b}", C["score_correlation_dev"].get(f"{b}~{a}"))
                for b in src
            ]
            for a in src
        ]
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        n = len(src)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i][j]:.2f}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(n))
        ax.set_xticklabels(src, fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(src, fontsize=7)
        ax.set_title(f"{cond}: score correlation (dev)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    p = os.path.join(OUT, "scifact_correlation.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    fig, axes = plt.subplots(1, nc, figsize=(6 * nc, 4.8), sharey=True)
    for ax, (cond, C) in zip(axes, R["conditions"].items()):
        for (name, pts), col in zip(C.get("risk_coverage_dev", {}).items(), ("#a22", "#c84", "#48a")):
            ax.plot([100 * p[0] for p in pts], [100 * p[1] for p in pts], color=col, label=name)
        for pol_, mk in (("selected", "o"), ("bayes-ess_cal", "s")):
            for tg, v in C.get("selective", {}).get(pol_, {}).items():
                ax.plot(
                    100 * v["dev_coverage"],
                    100 * (v["dev_errors_decided"] / max(1, v["dev_decided"])),
                    mk,
                    color="k",
                    ms=5,
                )
                ax.annotate(
                    tg, (100 * v["dev_coverage"], 100 * v["dev_errors_decided"] / max(1, v["dev_decided"])), fontsize=6
                )
            for al, v in C.get("ltt_not_sure", {}).get(pol_, {}).items():
                ax.plot(
                    100 * v["dev_coverage"],
                    100 * v["dev_errors_decided"] / max(1, v["dev_decided"]),
                    "^",
                    color="g",
                    ms=6,
                )
                ax.annotate(
                    f"LTT {al}",
                    (100 * v["dev_coverage"], 100 * v["dev_errors_decided"] / max(1, v["dev_decided"])),
                    fontsize=6,
                    color="g",
                )
        ax.set_xlabel("coverage (% of dev claims decided)")
        ax.set_ylabel("error rate on decided (%)")
        ax.set_title(f"{cond}: risk-coverage (dev); markers = train-chosen bands")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 20)
    fig.tight_layout()
    p = os.path.join(OUT, "risk_coverage.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths


if __name__ == "__main__":
    R = json.load(open(os.path.join(OUT, "scifact.json")))
    t = text(R)
    open(os.path.join(OUT, "scifact.txt"), "w").write(t + "\n")
    print(t)
    print(charts(R))
