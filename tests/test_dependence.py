"""Correlation-aware combining: learning, invariants, determinism and replay of the dependence state."""

import json
import math
import random
from pathlib import Path

import pytest

from roman_wheel_node import LiveWeights, give_feedback, roman_wheel_node
from roman_wheel_node.bench.generator import ALL_SCENARIOS, stream
from roman_wheel_node.dependence import DependenceLearner, FrozenDependence, votes_of
from roman_wheel_node.trace import ClaimTrace, TamperError, replay

DEP = ("ess", "tan", "stack")


def feed(dl, key, n, seed=0, lw=None):
    for c in stream(ALL_SCENARIOS[key], seed, n):
        dl.record(c.scores, c.truth)
        if lw is not None:
            give_feedback(c.scores, c.truth, live=lw)


def test_votes():
    assert votes_of({"a": 0.9, "b": 0.1, "c": 0.5, "d": 7, "e": -1}) == {"a": 1, "b": -1, "c": 0, "d": 1, "e": -1}


@pytest.mark.parametrize("variant", DEP)
@pytest.mark.parametrize("mode_case", ["gate", "empty", "veto"])
def test_gate_empty_and_veto_still_block(tmp_path, variant, mode_case):
    dl = DependenceLearner(str(tmp_path / "d.jsonl"))
    a, sc = {"gate": (0, {"x": 0.9}), "empty": (1, {}), "veto": (1, {"x": 0.9, "y": 0.0})}[mode_case]
    assert roman_wheel_node(a, sc, mode="bayes", variant=variant, dependence=dl) == (0.0, "000", "BLOCK")


def test_dependence_variant_requires_learner():
    with pytest.raises(ValueError):
        roman_wheel_node(1, {"x": 0.9}, mode="bayes", variant="ess")


def test_independent_sources_reduce_to_naive_bayes(tmp_path):
    dl = DependenceLearner(str(tmp_path / "d.jsonl"))
    lw = LiveWeights(str(tmp_path / "w.jsonl"))
    feed(dl, "baseline", 1500, lw=lw)
    ks = ["drone", "trial", "draw", "harmony", "instr"]
    n_sig = sum(dl.dependence(a, b)[1] > dl.g_threshold for i, a in enumerate(ks) for b in ks[i + 1 :])
    assert n_sig <= 1  # at p < 0.01 over 10 pairs, ~0.1 false edges expected
    for k in ks:
        assert dl.reliability(k) == pytest.approx(lw.reliability(k), abs=1e-9)
    rng = random.Random(0)
    hits = 0
    for _ in range(50):
        sc = {k: rng.random() for k in ks}
        hard = roman_wheel_node(1, sc, lw, mode="bayes", variant="hard")
        m = dl.model([k for k, v in votes_of(sc).items() if v], "ess")
        if all(w == 1.0 for w in m["w"].values()):
            hits += 1
            got = roman_wheel_node(1, sc, mode="bayes", variant="ess", dependence=dl)
            assert got[1:] == hard[1:] and got[0] == pytest.approx(hard[0], abs=1e-3)
    assert hits >= 25


def test_copies_are_detected_and_discounted(tmp_path):
    dl = DependenceLearner(str(tmp_path / "d.jsonl"))
    feed(dl, "copycats", 1500)
    _, g, phi = dl.dependence("lead", "c1")
    assert g > 100 and phi > 0.6
    assert dl.dependence("solo", "c1")[1] < dl.g_threshold
    m = dl.model(["lead", "c1", "c2", "c3", "solo"], "ess")
    assert m["w"]["solo"] == 1.0 and max(m["w"][k] for k in ("lead", "c1", "c2", "c3")) < 0.5
    tan = dl.model(["lead", "c1", "c2", "c3", "solo"], "tan")
    parents = {f["source"]: f["parent"] for f in tan["factors"]}
    assert parents["solo"] is None and sum(p is not None for p in parents.values()) == 3


def test_cluster_counts_about_once(tmp_path):
    """Four perfect copies of a source must not outvote one equally reliable independent source."""
    dl = DependenceLearner(str(tmp_path / "d.jsonl"), half_life=None)
    rng = random.Random(1)
    for _ in range(800):
        t = rng.random() < 0.5
        a = t if rng.random() < 0.8 else not t
        b = t if rng.random() < 0.8 else not t
        sc = {
            "a": 0.9 if a else 0.1,
            "a2": 0.9 if a else 0.1,
            "a3": 0.9 if a else 0.1,
            "a4": 0.9 if a else 0.1,
            "b": 0.9 if b else 0.1,
        }
        dl.record(sc, t)
    sc = {"a": 0.9, "a2": 0.9, "a3": 0.9, "a4": 0.9, "b": 0.1}  # cluster says true, independent says false
    for v in ("ess", "tan"):
        eff = roman_wheel_node(1, sc, mode="bayes", variant=v, dependence=dl)[0]
        assert 0.3 < eff < 0.7, v  # roughly a tie, not 4:1
    naive = LiveWeights(str(tmp_path / "w.jsonl"), half_life=None)
    for k in sc:
        for _ in range(8):
            naive.observe(k, 1)
        for _ in range(2):
            naive.observe(k, 0)
    assert roman_wheel_node(1, sc, naive, mode="bayes")[0] > 0.95


def test_leader_switch_is_tracked_with_decay(tmp_path):
    dl = DependenceLearner(str(tmp_path / "d.jsonl"))
    feed(dl, "leader_switch", 3000)
    assert dl.dependence("f1", "L2")[1] > dl.g_threshold
    assert dl.dependence("f1", "L2")[2] > dl.dependence("f1", "L1")[2]


def test_ledger_append_only_and_replay_identical(tmp_path):
    p = str(tmp_path / "d.jsonl")
    dl = DependenceLearner(p)
    feed(dl, "two_clusters", 400)
    first = Path(p).read_text()
    feed(dl, "two_clusters", 200, seed=9)
    assert Path(p).read_text().startswith(first) and len(Path(p).read_text().splitlines()) == 600
    dl2 = DependenceLearner(p)
    ks = ["a0", "a1", "a2", "b0", "b1", "b2", "solo"]
    for v in DEP:
        assert dl2.model(ks, v) == dl.model(ks, v)
    for c in stream(ALL_SCENARIOS["two_clusters"], 5, 50):
        for v in DEP:
            assert dl2.llr(c.scores, v) == dl.llr(c.scores, v)


def test_replay_with_other_half_life_differs_but_ledger_is_same(tmp_path):
    p = str(tmp_path / "d.jsonl")
    dl = DependenceLearner(p, half_life=150)
    feed(dl, "leader_switch", 3000)
    dl_inf = DependenceLearner(p, half_life=None)
    assert dl_inf.dependence("f1", "L1")[2] > dl.dependence("f1", "L1")[2]


def test_frozen_snapshot_reproduces_llr_exactly(tmp_path):
    dl = DependenceLearner(str(tmp_path / "d.jsonl"))
    feed(dl, "copycats", 600)
    for c in stream(ALL_SCENARIOS["copycats"], 4, 40):
        voting = [k for k, v in votes_of(c.scores).items() if v]
        for v in DEP:
            m = json.loads(json.dumps(dl.model(voting, v)))  # through JSON, as in the audit log
            assert FrozenDependence(m).llr(c.scores, v) == dl.llr(c.scores, v)


def test_abstaining_and_unknown_sources(tmp_path):
    dl = DependenceLearner(str(tmp_path / "d.jsonl"))
    feed(dl, "copycats", 300)
    for v in DEP:
        assert math.isfinite(dl.llr({"lead": 0.5, "c1": 0.9, "new_source": 0.9}, v))
        assert dl.llr({"lead": 0.5}, v) == pytest.approx(0.0 if v != "stack" else dl.model(["lead"], v)["bias"])


def test_eff_is_monotone_in_a_single_independent_vote(tmp_path):
    dl = DependenceLearner(str(tmp_path / "d.jsonl"))
    feed(dl, "copycats", 1000)
    base = {"lead": 0.9, "c1": 0.9, "c2": 0.1, "c3": 0.9}
    for v in ("ess", "tan"):
        lo = roman_wheel_node(1, {**base, "solo": 0.1}, mode="bayes", variant=v, dependence=dl)[0]
        hi = roman_wheel_node(1, {**base, "solo": 0.9}, mode="bayes", variant=v, dependence=dl)[0]
        assert hi > lo


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1
        return self.t


@pytest.mark.parametrize("variant", DEP)
def test_trace_logs_and_replays_dependence_model(tmp_path, variant):
    dl = DependenceLearner(str(tmp_path / "d.jsonl"))
    t = ClaimTrace(str(tmp_path / "audit.jsonl"), mode="bayes", variant=variant, dependence=dl, clock=Clock())
    for c in stream(ALL_SCENARIOS["copycats"], 2, 150):
        t.submit("claim", c.scores, a=1)
        dl.record(c.scores, c.truth)
    rep = t.verify()
    assert rep.n_records == 150 and rep.n_admitted == len(t)
    rec = json.loads(Path(t._log_path).read_text().splitlines()[-1])
    assert rec["dependence_model"]["variant"] == variant


def test_trace_tampered_dependence_model_detected(tmp_path):
    from roman_wheel_node.trace import _canon, _chain_hash

    dl = DependenceLearner(str(tmp_path / "d.jsonl"))
    feed(dl, "copycats", 500)
    t = ClaimTrace(str(tmp_path / "audit.jsonl"), mode="bayes", variant="ess", dependence=dl, clock=Clock())
    t.submit("x", {"lead": 0.9, "c1": 0.9, "c2": 0.9, "c3": 0.9, "solo": 0.1}, a=1)
    r = json.loads(Path(t._log_path).read_text())
    r.pop("hash")
    r["dependence_model"]["w"] = {k: 1.0 for k in r["dependence_model"]["w"]}  # undo the discount
    r["hash"] = _chain_hash(r["prev"], r, None)
    Path(t._log_path).write_text(_canon(r).decode() + "\n")
    with pytest.raises(TamperError):
        replay(t._log_path)
    r.pop("hash")
    r.pop("dependence_model")
    r["hash"] = _chain_hash(r["prev"], r, None)
    Path(t._log_path).write_text(_canon(r).decode() + "\n")
    with pytest.raises(TamperError):
        replay(t._log_path)


def test_trace_requires_learner_for_dependence_variant(tmp_path):
    with pytest.raises(ValueError):
        ClaimTrace(str(tmp_path / "a.jsonl"), mode="bayes", variant="ess")


# ---- ess_cal: ESS discount on calibrated confidences ------------------------------------------------------------
def test_ess_cal_reduces_to_calibrated_naive_bayes_without_dependence(tmp_path):
    import math

    from roman_wheel_node import DependenceLearner, roman_wheel_node

    rng = random.Random(7)
    dep = DependenceLearner(str(tmp_path / "d.jsonl"), half_life=None)
    for _ in range(400):  # two independent sources, base rate 0.5
        t = rng.random() < 0.5
        dep.record({"a": 0.8 if (rng.random() < 0.8) == t else 0.2, "b": 0.7 if (rng.random() < 0.7) == t else 0.3}, t)
    m = dep.model(["a", "b"], "ess_cal")
    assert m["w"] == {"a": 1.0, "b": 1.0} and abs(m["prior"] - 0.5) < 0.06
    lp = math.log(m["prior"] / (1 - m["prior"]))
    want = lp + (math.log(0.9 / 0.1) - lp) + (math.log(0.6 / 0.4) - lp)
    assert abs(dep.llr({"a": 0.9, "b": 0.6}, "ess_cal") - want) < 1e-12
    eff, _, _ = roman_wheel_node(1, {"a": 0.9, "b": 0.6}, mode="bayes", variant="ess_cal", dependence=dep)
    assert eff == round(1 / (1 + math.exp(-want)), 3)


def test_ess_cal_discounts_copies_and_replays_in_trace(tmp_path):
    from roman_wheel_node import DependenceLearner
    from roman_wheel_node.trace import ClaimTrace, replay

    rng = random.Random(3)
    dep = DependenceLearner(str(tmp_path / "d.jsonl"), half_life=None)
    for _ in range(500):
        t = rng.random() < 0.5
        lead = (rng.random() < 0.7) == t
        dep.record(
            {
                "l": 0.8 if lead else 0.2,
                "c1": 0.8 if lead else 0.2,
                "c2": 0.8 if lead else 0.2,
                "x": 0.8 if (rng.random() < 0.7) == t else 0.2,
            },
            t,
        )
    m = dep.model(["l", "c1", "c2", "x"], "ess_cal")
    assert m["w"]["l"] < 0.5 and m["w"]["x"] > 0.9
    tr = ClaimTrace(str(tmp_path / "a.jsonl"), mode="bayes", variant="ess_cal", dependence=dep)
    d1 = tr.submit("c", {"l": 0.8, "c1": 0.8, "c2": 0.8, "x": 0.2}, a=1)
    assert 0.5 < d1.eff < 0.8  # the copies count about once against x
    rep = replay(str(tmp_path / "a.jsonl"))
    assert rep.n_records == 1 and tr.verify().head == rep.head
