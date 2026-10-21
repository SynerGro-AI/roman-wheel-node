"""Give the live weights real work: simulated claims with ground truth.
Each source has a hidden true accuracy. Live weights must discover it.
Compare: static (equal trust) vs live (learned trust)."""
import random, os, json
from roman_wheel_node import roman_wheel_node, LiveWeights

random.seed(7)
SOURCES = {"drone": 0.92, "trial": 0.80, "draw": 0.65, "harmony": 0.50, "instr": 0.25}
N, WINDOW = 3000, 250

def source_score(acc, truth):
    right = random.random() < acc
    says_true = truth if right else not truth
    s = random.betavariate(8, 2) if says_true else random.betavariate(2, 8)
    return max(0.01, s)  # real zero is a hard veto; keep sim scores > 0

def accepted(status):
    return status == "ACCEPT"

ledger = "train_ledger.jsonl"
if os.path.exists(ledger): os.remove(ledger)
live = LiveWeights(ledger, half_life=None)  # v2 run: no forgetting (v3 default is 150)

rows, stat_ok, live_ok = [], 0, 0
stat_fa = live_fa = 0  # false accepts (ACCEPTed a false claim)
for i in range(1, N + 1):
    truth = random.random() < 0.5
    scores = {k: source_score(acc, truth) for k, acc in SOURCES.items()}
    _, _, s_stat = roman_wheel_node(1, scores)
    _, _, s_live = roman_wheel_node(1, scores, live)
    stat_ok += accepted(s_stat) == truth
    live_ok += accepted(s_live) == truth
    stat_fa += accepted(s_stat) and not truth
    live_fa += accepted(s_live) and not truth
    for k, v in scores.items():           # feedback once truth is known
        live.observe(k, 1.0 if (v > 0.5) == truth else 0.0)
    if i % WINDOW == 0:
        rows.append({"claims": i, "static_acc": stat_ok / WINDOW, "live_acc": live_ok / WINDOW,
                     "static_false_accepts": stat_fa, "live_false_accepts": live_fa,
                     "rel": {k: round(live.reliability(k), 3) for k in SOURCES}})
        stat_ok = live_ok = stat_fa = live_fa = 0

json.dump(rows, open("train_results.json", "w"), indent=1)
print(f"{'claims':>6} {'static':>7} {'live':>6} {'FA st':>6} {'FA lv':>6}  reliabilities")
for r in rows:
    print(f"{r['claims']:>6} {r['static_acc']:>7.1%} {r['live_acc']:>6.1%} "
          f"{r['static_false_accepts']:>6} {r['live_false_accepts']:>6}  {r['rel']}")
print("true accuracies:", SOURCES)

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    x = [r["claims"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(x, [r["static_acc"] for r in rows], "o-", label="static (equal trust)")
    ax[0].plot(x, [r["live_acc"] for r in rows], "o-", label="live (learned trust)")
    ax[0].set_title(f"Decision accuracy per {WINDOW} claims"); ax[0].set_xlabel("claims seen"); ax[0].legend(); ax[0].grid(alpha=.3)
    for k, acc in SOURCES.items():
        l, = ax[1].plot(x, [r["rel"][k] for r in rows], label=f"{k} (true {acc})")
        ax[1].axhline(acc, color=l.get_color(), ls=":", alpha=.6)
    ax[1].set_title("Learned reliability vs hidden true accuracy"); ax[1].set_xlabel("claims seen"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig("train_results.png", dpi=120); print("chart saved")
except ImportError as e:
    print("no matplotlib:", e)
