# Roman Wheel Node

[![CI](https://github.com/SynerGro-AI/roman-wheel-node/actions/workflows/ci.yml/badge.svg)](https://github.com/SynerGro-AI/roman-wheel-node/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/SynerGro-AI/roman-wheel-node/blob/main/LICENSE)

An exact-copy gate plus learned, auditable evidence fusion that decides whether a claim
can enter an agent's trace. Pure Python (3.9+), standard library only, zero dependencies.

**Real data (SciFact, Phase 3e): certified at a ≤ 5% false-accept bound with 19.0% coverage.** With the 6-judge panel,
the accept-only rule accepted 152 of 799 SciFact calibration claims with 0 false accepts (one-sided Clopper-Pearson
upper bound 4.3%, δ = 0.05). On held-out SciFact dev the frozen rule accepted 63 of 300 claims (21%) with 1 false
accept. At full coverage the node scores **89.7%** [86.0, 93.0] on SciFact dev (300 claims, cited abstracts).
On fresh claims it was never tuned on (Phase 4: COVID-Fact and HealthVer, 3,829 labelled), the same frozen panel rule
reaches **73.5% precision** [69.7, 77.1] **at 14.6% coverage**. That is not certified at any level. **Nothing is
certified at 1% or 2% anywhere.** Sources and caveats: [Phase 3](#phase-3-real-claims-scifact-and-safety),
[Phase 4](#phase-4-shadow-mode-deployment-and-fresh-set-certification), `results/phase3/step3e.txt`,
`results/phase3/RESULTS.md` and `results/phase4/SUMMARY.md`.

*Synthetic benchmark (simulated claim streams, not real data):* 92.2% ± 0.3% correct calls on the baseline scenario
(20 seeds), against a 92.4% Bayes-optimal ceiling. The other scenarios have ceilings below 90%; see
[Synthetic benchmark](#synthetic-benchmark-phase-2-python--m-roman_wheel_nodebench---seeds-20).

## Install

Not on PyPI yet. Install from GitHub:

```bash
pip install "git+https://github.com/SynerGro-AI/roman-wheel-node@v1.0.0"
# or, for development
git clone https://github.com/SynerGro-AI/roman-wheel-node && cd roman-wheel-node
pip install -e ".[test,bench]" && python -m pytest -q
```

## Design from first principles

**1. Gate first, then weigh.** Some checks are binary facts (hash match, binary selection,
drawing invariant). Weighing can't make up for a failed fact, so `a = 0` BLOCKs before anything
else. A source score of exactly 0 is a veto: it means "this is impossible", not "weak evidence",
so it BLOCKs no matter how much that source is trusted. No scores means no evidence, so that BLOCKs too.

**2. Why log-odds (`mode="bayes"`).** We want P(claim true | what the sources said). If the sources
are independent, Bayes' rule says the evidence *adds* in log-odds space:

    logit P(true | s1..sn) = logit(prior) + Σ_k log[ P(s_k | true) / P(s_k | false) ]

If source k is right with probability r_k, its vote is worth ±logit(r_k) = ±log(r_k / (1 − r_k)). This gives three
things for free that the geometric mean only handled with patches:
* r > 0.5 counts as positive evidence, r = 0.5 (a coin flip) counts as exactly zero, and r < 0.5 counts as
  *negative* evidence, so a liar is inverted automatically with no `flip` flag.
* More agreeing sources raise confidence. Product mode does the opposite: more evidence lowers the score.
* `eff` is a real probability, so the natural ACCEPT boundary is 0.5 rather than a hand-picked 0.35.

There are three likelihood variants, and all were benchmarked: `hard` (the source's vote: score > 0.5), `soft`
(the score treated as P(source says true)) and `logit` ((2r−1)·logit(score)). **`hard` was the best or tied for best
everywhere, so it is the default.** In this data the score's distance from 0.5 carries little information beyond which
side it falls on. Each source's r is capped to [0.01, 0.99], which limits any one source to ±4.6 log-odds.

**3. Why append-only ledgers.** Every piece of feedback is a new JSONL line. State is a pure function of the ledger,
so it can be audited ("why did you trust drone?"), replayed deterministically, and re-run under different settings.
Nothing is ever overwritten, which means a bug or a bad setting can't destroy past learning.

**4. Why decay (half-life 150 by default).** Sources change. Without forgetting, a source that was right
1,500 times needs hundreds of failures before its trust drops. Decay is applied *while replaying*, so the
ledger itself stays complete and you can replay with `half_life=None` at any time.

**5. Why a learned threshold.** A cut-off is only right if `eff` is calibrated. When the model's assumptions are
broken (for example, correlated sources), the posterior becomes over-confident. `ThresholdLearner` keeps an
append-only ledger of `(eff, truth)` for each mode and picks the cut-off that maximised decayed accuracy over recent
outcomes (half-life 500, 0.001 grid, `default` until `min_obs`).

## Limits (stated honestly)
* **The independence assumption** (naive `bayes-hard`). Sources that copy each other get counted several times: on `copycats`,
  bayes-hard reaches 67.7% against a 79.1% optimum. **Fixed by `variant="ess"` with a `DependenceLearner`, which reaches 79.0%** (see
  [Correlation-aware combining](#correlation-aware-combining-variantess-with-a-dependencelearner)).
  A latent common cause (`partial_corr`) is only partly captured.
* **The information ceiling.** No fusion rule can beat the Bayes-optimal oracle. Eight independent 58%-accurate
  sources cap out at about 67% (exact majority-vote value: 67.1%). Reaching 90% would take about 63 such sources.
* **Feedback is required.** Reliabilities come from right/wrong feedback. With no feedback, the node uses
  `assumed_reliability` (0.75) for bayes mode, or equal trust for geomean.
* **The simulations are only simulations.** The real accuracy of your sources is what sets the ceiling.

## API

```python
from roman_wheel_node import roman_wheel_node, LiveWeights, ThresholdLearner, give_feedback

live = LiveWeights("weights_ledger.jsonl")           # half_life=150 by default (None = never forget)
thr  = ThresholdLearner("threshold_ledger.jsonl")    # half_life=500, min_obs=30, default cut-off 0.5

scores = {"drone": 0.95, "trial": 0.81, "instr": 0.12}
eff, op, status = roman_wheel_node(1, scores, live, mode="bayes", threshold=thr)  # variant="hard"
# ... later, when the truth is known:
give_feedback(scores, truth=True, live=live, threshold=thr, eff=eff, mode="bayes")
```

`roman_wheel_node(a, scores, live=None, mode="geomean", log_path=None, flip=False, threshold=None,
variant="hard", prior=0.5, assumed_reliability=0.75, cutoff=None) -> (eff, opcode, status)`

* `mode`: `"geomean"` (default, v1/v2), `"product"`, or `"bayes"` (new).
* Opcodes without a learner: geomean/product use the fixed v1 bands (≥0.85 `111`, ≥0.65 `110`, ≥0.35 `101`,
  >0.05 `100` HEDGE). Bayes uses cut-off 0.5. `cutoff=<float>` sets a fixed cut-off, and `cutoff="fixed"` forces the v1 bands.
* Opcodes with a learner: ACCEPT iff `eff ≥ c`; the span [c, 1] is split into thirds for `101`/`110`/`111`; HEDGE `100`
  covers `[c − 0.15, c)` but only above 0.05; anything lower is BLOCK.
* Backward compatible: every v2 call gives bit-identical results, and the 6 original tests are unchanged
  (checked by pytest against a frozen v2 reference, `tests/_v2_reference.py`, on 1,600 random claims).
* **Behaviour change:** the `LiveWeights` default `half_life` is now 150. Pass `half_life=None` to get v2 behaviour
  (`benchmarks/train_sim.py` pins this so its old results still reproduce).

## Gated agent trace (Phase 1): `roman_wheel_node.trace`

`ClaimTrace` is the only way a claim can enter an agent's context or become a parent, and every entry goes through `roman_wheel_node`.

```python
from roman_wheel_node.trace import ClaimTrace, replay

trace = ClaimTrace("audit.jsonl", live=live, mode="bayes")      # allow_hedge=False by default
root = trace.submit("sha matches release", {"drone": 0.95, "trial": 0.9}, a=1)
if root.admitted:
    child = trace.submit("so the build is reproducible", {"trial": 0.8}, a=1, parents=[root.claim])
context = trace.context()        # immutable tuple of admitted Claim objects
replay("audit.jsonl")            # verifies the hash chain and recomputes every decision
trace.verify()                   # also checks the in-memory claims against the log
```

Invariants (all covered by tests, including property-based tests with hypothesis):
- **One entry point.** `Claim` can't be constructed outside the trace (private token). Claims are immutable:
  their slots are frozen, `scores` is a read-only mapping, and they can't be pickled into new objects. The trace has no
  append, insert or remove, and `context()` returns a tuple.
- **Zero-propagation.** `a` must be exactly 1. `a = 0`, BLOCK and any score of exactly 0 never enter. A claim whose parent
  is not an admitted claim *of this trace* is evaluated with `a = 0`. That covers a blocked parent, an unknown or
  forged one, and one from another trace (checked by object identity). So no descendant of a blocked claim can enter.
- **Containment.** A blocked claim is invisible. `get()` treats its id exactly like an id that never existed, and
  `context()` never shows it. HEDGE is admitted only with `allow_hedge=True`.
- **Tamper-evident audit.** Every call, admitted or not, appends one JSONL record with the full audit tuple: content,
  raw scores, `a` and effective `a`, parent ids, the reliabilities and cut-off used, mode, eff, opcode, status,
  admitted/reason, and a timestamp. Each record also stores `prev` and `hash = sha256(prev | canonical record)`.
  Pass `hmac_key=` to make it an HMAC chain. `replay()` rejects edited, deleted, reordered or garbage records, and any
  record whose decision doesn't recompute from its own inputs. A trace refuses to resume from a log that doesn't verify.
- **Threat model (honest):** this is an API-level guarantee. In-process Python can always reach private state.
  Such tampering is caught by `verify()` and by the parent identity checks. Without `hmac_key`, someone who rewrites
  the *whole* log consistently can't be detected unless you anchor `trace.head()` somewhere else.

## Synthetic benchmark (Phase 2): `python -m roman_wheel_node.bench --seeds 20`

**Everything in this section and in [Correlation-aware combining](#correlation-aware-combining-variantess-with-a-dependencelearner) is synthetic: simulated sources on
simulated claims, not real data.** The real-data results are in Phase 3 and Phase 4.

These are synthetic claim streams from `roman_wheel_node.bench.generator`. The generator is documented and has a CLI
(`python -m roman_wheel_node.bench.generator --list` prints every scenario spec). Each source makes the right call with
its stated accuracy. Its score is drawn from Beta(8,2) when it says "true" and Beta(2,8) when it says "false", with a
floor of 0.01. Seeds 0–19 are used, so each scenario has 20 independent streams. Every policy sees the same stream,
decides with what it has learned so far, and then receives the truth as feedback.
- **Accuracy** is the share of right calls. A HEDGE counts as not accepted.
- **TA / FB** are the true-accept rate (share of true claims accepted) and the false-block rate (share of false claims blocked).
- **The ceiling** is the Bayes-optimal oracle. It knows the true accuracies, the drift schedule, the copy structure and
  the noise model, and decides by sign(log-likelihood ratio). Its value is measured on ≥200k claims. Nothing can beat it in expectation.
- Intervals are 95% Student-t intervals over the 20 seeds.

| Scenario | Ceiling | static geomean | v2 best (flip+decay) | **bayes-hard** | bayes-hard TA / FB | bayes-hard + learned threshold | oracle (same streams) |
|---|---|---|---|---|---|---|---|
| all noisy (8 weak sources) | **66.9** | 65.1 ± 0.5 | 64.2 ± 0.5 | **66.3 ± 0.5** | 67.0 / 65.6 | 66.1 ± 0.5 | 67.1 ± 0.5 |
| bad majority | **86.1** | 42.4 ± 0.4 | 81.4 ± 0.4 | **85.7 ± 0.3** | 85.9 / 85.5 | 85.4 ± 0.3 | 85.9 ± 0.3 |
| baseline | **92.4** | 69.5 ± 0.5 | 88.3 ± 0.3 | **92.2 ± 0.3** | 92.3 / 92.1 | 92.0 ± 0.3 | 92.6 ± 0.3 |
| cold start (first 50 claims) | **92.6** | 69.8 ± 2.3 | 82.9 ± 2.7 | **87.4 ± 2.0** | 86.9 / 87.8 | 87.2 ± 2.2 | 92.0 ± 1.7 |
| copycats (3 copy a 65% leader) | **79.1** | 65.8 ± 0.4 | 68.4 ± 0.3 | **67.7 ± 0.3** | 67.8 / 67.7 | 71.6 ± 0.5 | 79.2 ± 0.4 |
| drift (drone goes bad at 1500) | **87.8** | 65.0 ± 0.3 | 81.7 ± 0.4 | **85.1 ± 0.3** | 85.5 / 84.8 | 84.9 ± 0.3 | 87.6 ± 0.2 |

**Where it reaches 90%, and why it doesn't elsewhere.** 90% is reached only on the baseline. Every other scenario has a
ceiling below 90%, so no method could reach it there. bayes-hard is within 0.6 points of the ceiling on baseline,
bad majority and all-noisy. It is 2.7 points short on drift, because it takes time to unlearn the drone after it goes bad.
On cold start it is 5.2 points short of the ceiling (4.6 short of the oracle on the same streams): 50 claims are not enough to learn from. On copycats it is 11.4 points short
(7.5 with the learned threshold), because the independence assumption counts the leader four times.

![accuracy](https://raw.githubusercontent.com/SynerGro-AI/roman-wheel-node/main/results/bench_accuracy.png)

### Product vs geomean, head to head on identical streams
The difference is geomean accuracy minus product accuracy in points, paired per seed, with geomean/tie/product wins over 20 seeds.

| Scenario | static | live+flip+decay | flip+decay + learned threshold |
|---|---|---|---|
| all noisy (8 weak sources) | +15.2 ± 0.7 (20/0/0) | +1.3 ± 0.5 (18/0/2) | +0.8 ± 0.3 (17/0/3) |
| bad majority | -6.9 ± 0.7 (0/0/20) | +5.3 ± 0.6 (20/0/0) | +0.2 ± 0.1 (13/1/6) |
| baseline | +18.3 ± 0.6 (20/0/0) | +12.3 ± 0.5 (20/0/0) | +0.2 ± 0.1 (15/1/4) |
| cold start (first 50 claims) | +21.7 ± 4.2 (20/0/0) | +11.6 ± 3.5 (19/0/1) | +11.9 ± 2.0 (20/0/0) |
| copycats (3 copy a 65% leader) | +8.3 ± 0.5 (20/0/0) | -1.5 ± 0.3 (0/0/20) | -0.2 ± 0.2 (6/0/14) |
| drift (drone goes bad at 1500) | +11.5 ± 0.6 (20/0/0) | +1.8 ± 0.5 (19/0/1) | +0.6 ± 0.2 (20/0/0) |

**Verdict:** with the fixed v1 bands, **product mode is broken**. It multiplies weights, so its eff shrinks with every
added source and it ends up rejecting almost everything: on baseline it accepts 2.4% of true claims. It only "wins" when
rejecting everything happens to be better than the alternative. That is the case in bad majority with static weights,
where static geomean is worse than a coin flip, and in copycats with live weights.
With a learned threshold the gap mostly closes, because product is close to a monotone transform of geomean once the
reliabilities settle. Even so, geomean is equal or better in 5 of 6 scenarios, and it is far better on cold start
(+11.9 points), where the threshold has too little data. Keep `geomean` (or `bayes`) as the default. `product` stays
available only for backward compatibility.

![product vs geomean](https://raw.githubusercontent.com/SynerGro-AI/roman-wheel-node/main/results/bench_product_vs_geomean.png)

### Calibration
- **Learned reliability error:** mean over sources of |r − true accuracy|, averaged over the run / at the last claim.
- **ECE of eff:** expected calibration error over 10 bins, the mean of per-seed values. Only bayes eff is meant to be a probability.

| Scenario | r error, half_life 150 | r error, no decay | ECE bayes-hard | ECE bayes-soft | ECE geomean (flip+decay) | ECE oracle |
|---|---|---|---|---|---|---|
| all noisy (8 weak sources) | 0.021 / 0.018 | 0.014 / 0.008 | 0.025 | 0.039 | 0.119 | 0.020 |
| bad majority | 0.022 / 0.018 | 0.016 / 0.011 | 0.017 | 0.083 | 0.139 | 0.014 |
| baseline | 0.022 / 0.019 | 0.015 / 0.011 | 0.011 | 0.081 | 0.181 | 0.012 |
| cold start (first 50 claims) | 0.075 / 0.046 | 0.074 / 0.045 | 0.106 | 0.146 | 0.204 | 0.074 |
| copycats (3 copy a 65% leader) | 0.021 / 0.016 | 0.014 / 0.009 | 0.172 | 0.098 | 0.087 | 0.015 |
| drift (drone goes bad at 1500) | 0.034 / 0.021 | 0.079 / 0.102 | 0.015 | 0.086 | 0.123 | 0.013 |

Reading the calibration table:
- **Stable sources:** forgetting costs a little accuracy in the learned reliabilities (0.02 error vs 0.01).
- **Drift:** forgetting is what makes the reliabilities track the change (0.021 error at the end, vs 0.102 without decay).
- **bayes-hard eff is a calibrated probability** whenever the sources really are independent (ECE 0.011–0.025, the same as the oracle).
- **Copycats:** it is badly over-confident (ECE 0.17). This is the double-counting problem, and the learned threshold only partly offsets it.
- **Cold start:** the ECE figures are inflated by small-sample binning. 50 claims per run gives noisy bins; even the oracle shows 0.074.
- **geomean:** its eff is a score, not a probability (ECE 0.09–0.20).

![calibration](https://raw.githubusercontent.com/SynerGro-AI/roman-wheel-node/main/results/bench_calibration.png)
![reliability tracking](https://raw.githubusercontent.com/SynerGro-AI/roman-wheel-node/main/results/bench_reliability_tracking.png)

### Reproduce
```bash
pip install -e ".[bench]"
python -m roman_wheel_node.bench --seeds 20                  # writes results/ (~1 min on 8 cores)
python -m roman_wheel_node.bench --seeds 20 --check results  # prints IDENTICAL if byte-for-byte equal
RWN_FULL_BENCH=1 python -m pytest tests/test_bench.py         # the same check as a test
```
`benchmark.json` and `benchmark.txt` are byte-identical across runs and worker counts. This was verified on Python
3.9, 3.12 and 3.13. The charts are not byte-compared, because PNG bytes depend on the matplotlib version.
The default test suite also re-derives a subset of the committed per-seed numbers: all 20 cold-start seeds, baseline seed 0 and copycats seed 7.

## Correlation-aware combining: `variant="ess"` with a `DependenceLearner`

Naive Bayes assumes sources are independent given the truth. Sources that copy each other then get counted several times
(the copycats scenario lost 11 points to this). `roman_wheel_node.dependence.DependenceLearner` learns the dependence from
feedback. Like every other learner here, it uses an append-only, replayable ledger with decay applied on replay.

**From first principles:** if two sources are independent given the truth, whether one is *right* tells you nothing
about whether the other is right. So the learner tracks, for every pair, the decayed 2×2 table of their correctness
(right/wrong). The G-test (`2·N·MI > 6.63`, p < 0.01) decides whether a pair is significantly dependent, and
φ (the correlation of correctness) measures how much evidence they duplicate. Three combiners were tried, all on hard votes:

| variant | idea | verdict |
|---|---|---|
| **`ess`** | effective-sample-size discount: weight each vote `logit(r_k)` by `1 / (1 + Σ_j max(0, φ_jk))` over significant partners, so a cluster of near-duplicates counts about once | **chosen**: best accuracy and calibration on every dependence scenario |
| `tan` | tree-augmented naive Bayes: a Chow-Liu forest over significant pairs, with `P(c_k | c_parent)` shrunk toward `r_k` | close second (−0.1 to −0.4 points vs ess); one parent per source can't represent a cluster as well |
| `stack` | online logistic regression on signed votes, L2-anchored to naive-Bayes weights | wins only on drift (+0.4); loses 1.0 point on 8 weak sources and is noisier at cold start |

With no significant pairs, `ess` and `tan` reduce **exactly** to `variant="hard"`. That is why they tie bayes-hard on the
independent scenarios: the paired difference is ≤ 0.01 points, with 19–20 of 20 seeds identical.

```python
from roman_wheel_node import DependenceLearner, roman_wheel_node, give_feedback
dep = DependenceLearner("dependence_ledger.jsonl")          # half_life=150 claims by default
eff, op, status = roman_wheel_node(1, scores, mode="bayes", variant="ess", dependence=dep)
give_feedback(scores, truth, dependence=dep)                # once the truth is known
# ClaimTrace(..., mode="bayes", variant="ess", dependence=dep) logs the exact model snapshot, so replay() is exact
```

### Synthetic dependence benchmark: `python -m roman_wheel_node.bench --suite dependence --seeds 20`
This covers the six standard scenarios (definitions unchanged) plus three new dependence scenarios, each with an oracle
that knows the true joint model:
- `partial_corr`: four sources repeat a *hidden* shared call half the time.
- `two_clusters`: two independent leader-plus-two-copier clusters.
- `leader_switch`: three copiers switch from a 75% leader to a 60% one at claim 1500.

Each table cell is accuracy, mean ± 95% CI over 20 seeds. TA / FB are the true-accept and false-block rates. The paired
column is (ess wins / ties / hard wins) over 20 seeds. ECE is the expected calibration error of eff.

| Scenario | Ceiling | bayes-hard | **bayes-ess** | ess TA / FB | bayes-tan | bayes-stack | ess − hard, paired | ECE hard → ess |
|---|---|---|---|---|---|---|---|---|
| baseline | 92.4 | 92.2 ± 0.3 | **92.2 ± 0.3** | 92.3 / 92.1 | 92.2 ± 0.3 | 92.0 ± 0.3 | -0.00 ± 0.00 (0/19/1) | 0.011 → **0.011** |
| drift (drone goes bad at 1500) | 87.8 | 85.1 ± 0.3 | **85.1 ± 0.3** | 85.5 / 84.8 | 85.1 ± 0.3 | 85.5 ± 0.3 | +0.00 ± 0.00 (1/19/0) | 0.015 → **0.015** |
| bad majority | 86.1 | 85.7 ± 0.3 | **85.7 ± 0.3** | 85.9 / 85.5 | 85.7 ± 0.3 | 85.5 ± 0.3 | +0.00 ± 0.00 (0/20/0) | 0.017 → **0.017** |
| all noisy (8 weak sources) | 66.9 | 66.3 ± 0.5 | **66.3 ± 0.5** | 67.0 / 65.6 | 66.3 ± 0.5 | 65.3 ± 0.5 | -0.00 ± 0.01 (1/16/3) | 0.025 → **0.025** |
| cold start (first 50 claims) | 92.6 | 87.4 ± 2.0 | **87.4 ± 2.0** | 86.9 / 87.8 | 87.4 ± 2.0 | 86.9 ± 1.6 | +0.00 ± 0.00 (0/20/0) | 0.106 → **0.106** |
| copycats (3 copy a 65% leader) | 79.1 | 67.7 ± 0.3 | **79.0 ± 0.4** | 79.2 / 78.8 | 78.8 ± 0.4 | 78.1 ± 0.5 | +11.23 ± 0.33 (20/0/0) | 0.172 → **0.021** |
| partial correlation (latent shared call) (new) | 79.7 | 77.9 ± 0.3 | **78.3 ± 0.4** | 78.3 / 78.4 | 78.2 ± 0.3 | 78.3 ± 0.4 | +0.46 ± 0.29 (15/1/4) | 0.081 → **0.027** |
| two copy clusters (new) | 78.0 | 75.9 ± 0.4 | **77.7 ± 0.3** | 77.8 / 77.7 | 77.4 ± 0.3 | 77.1 ± 0.3 | +1.87 ± 0.24 (20/0/0) | 0.108 → **0.018** |
| copiers switch leader at 1500 (new) | 80.6 | 76.2 ± 0.6 | **79.5 ± 0.3** | 79.9 / 79.2 | 79.4 ± 0.3 | 79.0 ± 0.3 | +3.35 ± 0.43 (20/0/0) | 0.138 → **0.018** |

- **Copycats:** the gap to the ceiling went from **11.4 points to 0.1**, and ECE went from 0.172 to 0.021.
- **leader_switch:** the gap went from 4.4 to 1.1 points. Decay lets the learned dependence follow the switch.
- **two_clusters:** the gap went from 2.1 to 0.3 points.
- **partial_corr:** the gap only went from 1.8 to 1.4 points. The shared cause is *latent*, so pairwise statistics
  capture it only partly. Calibration still improves (0.081 to 0.027).
- **Remaining limits:** dependence must be learned, so a cluster costs a few hundred claims before it is discounted
  (cold start can't benefit). Pairwise φ approximates higher-order structure. The combiners use hard votes only.

![dependence accuracy](https://raw.githubusercontent.com/SynerGro-AI/roman-wheel-node/main/results/dependence/dep_accuracy.png)
![paired](https://raw.githubusercontent.com/SynerGro-AI/roman-wheel-node/main/results/dependence/dep_paired.png)
![calibration](https://raw.githubusercontent.com/SynerGro-AI/roman-wheel-node/main/results/dependence/dep_calibration.png)

The standard suite (`results/`) is frozen and still byte-identical. The dependence suite (`results/dependence/`)
is byte-reproducible too (`--suite dependence --check results/dependence`). The dependence code uses
`math.fsum`, because Python 3.12 changed the builtin float `sum()`, which made one near-threshold edge differ on 3.9 before the fix.

## Phase 3: real claims (SciFact) and safety
**Safety guarantees and their limits: see [SAFETY.md](https://github.com/SynerGro-AI/roman-wheel-node/blob/main/SAFETY.md).** It separates what is guaranteed by construction
(fail-closed, NOT_SURE never accepted, zero-propagation, non-bypassable gate, hash-chained audit, each mapped to its
tests) from what is only measured on data.

Held-out SciFact dev (300 claims; everything learned on 350 train claims), binary "supported vs not":

| Evidence | Node accuracy [95% CI] | Best single judge | False accepts, unwrapped J1 → node (of 176) |
|---|---|---|---|
| gold abstracts | **91.3%** [88.0, 94.3] | J4 90.0% | 19 → 6, p = 0.0002 |
| rv3 retrieval (RRF + MedCPT cross-encoder, top 5) | **87.3%** [83.3, 91.0] | J4 85.7% (calibrated J9 88.7%) | 29 → 12, p < 0.0001 |

On retrieved evidence the node ties the best judges. It does not reach 90%. Its gain there is the false-accept rate,
which falls by 59%. Selective mode on gold evidence decides 56/300 claims with 0 errors (95% upper bound 5.21%). A
risk-controlled ≤ 1% NOT_SURE band cannot be certified with 350 calibration claims, so the node abstains (fails closed).
Phase 3c added a strict proof verifier (J11). As a hard veto it cuts rv3 false accepts from 12 to 4 but costs about 9
accuracy points, so train CV leaves it off. As a source it leaves rv3 at 87.3%. See SAFETY.md §b5.
Details: [results/phase3/RESULTS.md](https://github.com/SynerGro-AI/roman-wheel-node/blob/main/results/phase3/RESULTS.md), `results/phase3/scifact.txt`, and the charts in `results/phase3/`
(`scifact_accuracy.png`, `scifact_false_accept.png`, `scifact_correlation.png`, `risk_coverage.png`).

## Phase 4: shadow-mode deployment and fresh-set certification
**This is shadow mode: the node watches an agent's claims and changes nothing.** A frozen, hash-pinned policy decides
every claim and logs the decision. The agent's output is returned untouched. The fresh claims it logs can later earn a
certificate on data the node was never tuned on.

```python
from roman_wheel_node.shadow import ShadowRunner
rwn = ShadowRunner("rwn_shadow/", nli2="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", corpus=trusted_docs)
respond = rwn.wrap(respond, extract=lambda out: [(c, ev) for c, ev in my_claims(out)])   # same return value
```

* **Policy.** `src/roman_wheel_node/policies/rwn-policy-4.0.0.json`, loaded by `roman_wheel_node.policy.load_policy()`.
  The loader pins its sha256 and refuses an edited or swapped file.
  * There are three outcomes: `ACCEPT`, `NOT_SURE` and `BLOCK`. The policy is accept-only, so it never rejects on its
    posterior.
  * Anything uncertain or erroring becomes `NOT_SURE`: a missing source, a malformed score, the disagreement rule, or a
    posterior below the margin.
  * An exact-copy gate failure is `BLOCK`.
  * It has three profiles:
    * `local` (default, free): DeBERTa-v3-base NLI + lexical overlap.
    * `local-lite` (free): ONNX DeBERTa-v3-small + overlap.
    * `panel` (paid, opt-in): the Phase 3e certified configuration. It reproduces 3e exactly: 63 accepted and 1 false on SciFact dev, and 36 accepted and 0 false on Climate-FEVER.
* **Judges are off by default.** The paid panel runs only with `XAIPanel(enable=True, cap_usd=...)` or
  `--enable-xai-judges --cap-usd N`. Before that, `XAI_API_KEY` is never read, and it is never printed or logged.
* **Fresh-data pipeline.** `python -m roman_wheel_node.fresh prereg | label | certify`.
  * Pre-register one fixed rule before any label.
  * Label each claim SUPPORT / NOT / UNSURE.
  * Certify with one-sided Clopper-Pearson at 1/2/5% (δ = 0.05). With no search over rules, 299 / 149 / 59 error-free
    accepts certify 1% / 2% / 5%.
* **Honesty.** The only certificate behind any profile is Phase 3e's: false accepts ≤ 5% of accepts, on SciFact, panel
  profile. **Nothing is certified at 1% or 2%.** The local profiles are not certified at any level.
* **Fresh-set run on the box:**
  * Sets: COVID-Fact, HealthVer dev, and the HealthVer test pairs left unjudged in 3d. All are internal, and only
    aggregates are committed.
  * Results are in [results/phase4/SUMMARY.md](https://github.com/SynerGro-AI/roman-wheel-node/blob/main/results/phase4/SUMMARY.md). The free local profile accepts 10% of 6,311
    fresh claims at 67.8% precision. COVID-Fact's one-word counterfactual claims fool it, so it is far from any
    certificate.
* **Desktop / GPU runbook:** [PHASE4_DESKTOP.md](https://github.com/SynerGro-AI/roman-wheel-node/blob/main/PHASE4_DESKTOP.md), with Windows and PowerShell instructions, the
  Seraphina hook, and where the logs go.

### Test-only smoke flags (Phase 5)
The Phase 5 scripts have flags that exist only to smoke-test the pipeline on a CPU box. **They are test-only, and no
reported number comes from them.**
* `phase5/train_pair_checker.py --tiny-random` trains a tiny, randomly initialised DeBERTa-v2 instead of the real base
  model. `--limit N` trains on only the first N items. Either flag marks the model `smoke` in its metadata, and a smoke
  model's training record is never written to `results/`.
* `phase5/eval_pair_checker.py --smoke` replaces the held-out set B with a stand-in made from two disjoint halves of
  A_dev (B is never read). It writes only to `phase5/runs/smoke/`, never to `results/`, and labels its output
  "SMOKE ... numbers mean nothing".
* Without `--smoke`, `eval_pair_checker.py` refuses a smoke model and exits, so a tiny-random model can't produce a real
  result.

## Layout
* `src/roman_wheel_node/core.py`: the node, `LiveWeights` and `ThresholdLearner`. The public API is re-exported from `roman_wheel_node`.
* `src/roman_wheel_node/dependence.py`: `DependenceLearner`, `FrozenDependence`, and the `ess` / `tan` / `stack` combiners.
* `src/roman_wheel_node/trace.py`: `ClaimTrace`, `Claim`, `Decision`, `replay`, `TamperError` (Phase 1).
* `src/roman_wheel_node/policy.py`, `policies/`, `shadow.py`, `sources.py`, `fresh.py`: Phase 4, the frozen policy, shadow runtime, sources, and the label/certify CLI.
* `phase4/`: the policy export, fresh-set builders, the paid fresh-set judge runner and `run_fresh.py` (not part of the package).
* `src/roman_wheel_node/bench/`: `generator.py` (the claim-stream generator and its CLI), `runner.py` (policies,
  confidence intervals, calibration, head-to-heads), `plots.py`, and `__main__.py` (`python -m roman_wheel_node.bench`).
* `results/`: the committed 20-seed standard suite (frozen). `results/dependence/`: the 20-seed dependence suite.
* `phase3/`: the Phase 3 SciFact pipeline (retrieval, NLI, xAI judges, evaluation, reports; not part of the package).
* `SAFETY.md`: guarantees by construction (with tests) vs measured guarantees.
* `phase5/`: the Phase 5 pair-checker pipeline (split, bundle, train, evaluate; not part of the package). Its
  `--tiny-random`, `--limit` and `--smoke` flags are test-only (see [Test-only smoke flags](#test-only-smoke-flags-phase-5)).
* `tests/`: 221 tests (`PYTHONPATH=src python -m pytest -q`: 218 pass. Two full-benchmark byte checks run only with
  `RWN_FULL_BENCH=1`, and one Phase 5 test runs only where the internal COVID-Fact cache is present).
* `benchmarks/train_sim.py` and `benchmarks/results_v1.0.0.*`: the original demo and the 1.0.0 five-seed results, kept for history.

## Support development

Roman Wheel Node is built in the open by **SynerGro AI Corp**, a for-profit company. If the project is useful to you
or your team, you can sponsor its development. Sponsorships go to SynerGro AI Corp and fund maintenance, benchmarks
and new releases.

* **GitHub Sponsors:** `<GITHUB_SPONSORS_LINK_TBD>` (placeholder; the owner will add the link once the account exists)
* **Ko-fi or another sponsor page:** `<SPONSOR_PAGE_LINK_TBD>` (placeholder)
* **Commercial support or a custom integration:** `<CONTACT_EMAIL_TBD>` (placeholder)

Once the links are live, the commented template in `.github/FUNDING.yml` turns on GitHub's **Sponsor** button.

## Next steps
* A larger calibration set (and reviewed labels) so a ≤ 1% decided-error band can actually be certified.
* Learned score calibration per source: histogram P(truth | score bin) from the ledger, replacing the fixed soft/hard models.
