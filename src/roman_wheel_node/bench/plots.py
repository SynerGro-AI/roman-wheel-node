"""Charts for the benchmark (requires matplotlib: pip install roman-wheel-node[bench])."""
from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHOW = ["static-geomean", "live+flip+decay", "flip+decay+thr", "bayes-hard", "bayes-hard+thr", "oracle"]


def plot_all(res: dict, out: str) -> List[str]:
    paths = [_accuracy(res, out), _h2h(res, out), _calibration(res, out), _tracking(res, out)]
    plt.close("all")
    return paths


def _grid(n):
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(17, 4.6 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    return fig, axes.flat


def _accuracy(res, out):
    sc = res["scenarios"]; fig, axes = _grid(len(sc))
    for ax, (k, d) in zip(axes, sc.items()):
        m = [100 * d["policies"][p]["accuracy"]["mean"] for p in SHOW]
        e = [100 * d["policies"][p]["accuracy"]["ci95"] for p in SHOW]
        ax.bar(range(len(SHOW)), m, yerr=e, capsize=3, color=["#9aa"] + ["#2a5d9f"] * 4 + ["#7a7"])
        ax.axhline(100 * d["ceiling"], color="#c0392b", ls="--", lw=1.2, label="Bayes-optimal ceiling")
        ax.axhline(90, color="#555", ls=":", lw=1, label="90%")
        for i, v in enumerate(m):
            ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=7)
        ax.set_title(d["spec"]["title"], fontsize=10); ax.set_ylim(40, 100)
        ax.set_xticks(range(len(SHOW))); ax.set_xticklabels(SHOW, rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=7, loc="lower right"); axes[0].set_ylabel("correct calls (%)")
    fig.suptitle(f"Accuracy, mean and 95% CI over {res['config']['seeds']} seeds")
    fig.tight_layout(); p = os.path.join(out, "bench_accuracy.png"); fig.savefig(p, dpi=120); return p


def _h2h(res, out):
    sc = res["scenarios"]; fig, ax = plt.subplots(figsize=(12, 5))
    pairs = [h["geomean"] for h in next(iter(sc.values()))["head_to_head"]]
    w = 0.8 / len(pairs)
    for j, name in enumerate(pairs):
        m = [100 * d["head_to_head"][j]["geomean_minus_product"]["mean"] for d in sc.values()]
        e = [100 * d["head_to_head"][j]["geomean_minus_product"]["ci95"] for d in sc.values()]
        ax.bar([i + (j - (len(pairs) - 1) / 2) * w for i in range(len(sc))], m, w, yerr=e, capsize=3,
               label=f"{name} vs its product twin")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(sc))); ax.set_xticklabels([d["spec"]["title"] for d in sc.values()], rotation=15, fontsize=8)
    ax.set_ylabel("accuracy: geomean − product (pts)"); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Product vs geomean on identical streams (paired, 95% CI); > 0 means geomean is better")
    fig.tight_layout(); p = os.path.join(out, "bench_product_vs_geomean.png"); fig.savefig(p, dpi=120); return p


def _calibration(res, out):
    sc = res["scenarios"]; fig, axes = _grid(len(sc))
    styles = {"bayes-hard": ("#2a5d9f", "o"), "bayes-soft": ("#8e44ad", "s"),
              "live+flip+decay": ("#e67e22", "^"), "oracle": ("#27ae60", "d")}
    for ax, (k, d) in zip(axes, sc.items()):
        ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
        for p, c in d["eff_calibration"].items():
            pts = [(b[1] / b[0], b[2] / b[0]) for b in c["pooled_bins"] if b[0] >= 20]
            col, mk = styles.get(p, ("k", "."))
            ax.plot([x for x, _ in pts], [y for _, y in pts], marker=mk, color=col, lw=1,
                    label=f"{p} (ECE {c['ece']['mean']:.3f})")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(d["spec"]["title"], fontsize=10)
        ax.set_xlabel("mean eff in bin"); ax.set_ylabel("fraction actually true"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.suptitle("Calibration of eff (pooled over seeds, bins with >= 20 claims). Only bayes eff is meant to be a probability")
    fig.tight_layout(); p = os.path.join(out, "bench_calibration.png"); fig.savefig(p, dpi=120); return p


def _tracking(res, out):
    keys = [k for k in ("baseline", "drift", "copycats") if k in res["scenarios"]]
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4.4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        tr = res["scenarios"][k]["tracking_seed0"]
        xs = [t["i"] for t in tr]
        for j, s in enumerate(tr[0]["learned"]):
            col = f"C{j}"
            ax.plot(xs, [t["learned"][s] for t in tr], color=col, lw=1.4, label=s)
            ax.plot(xs, [t["true"][s] for t in tr], color=col, lw=1, ls="--")
        ax.set_title(f"{res['scenarios'][k]['spec']['title']} (seed 0)", fontsize=10)
        ax.set_xlabel("claim"); ax.set_ylabel("reliability"); ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Learned reliability (solid, half_life=150) vs true accuracy (dashed)")
    fig.tight_layout(); p = os.path.join(out, "bench_reliability_tracking.png"); fig.savefig(p, dpi=120); return p
