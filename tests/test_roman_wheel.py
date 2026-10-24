"""Unit tests for roman_wheel_node. Run: python -m pytest -q"""

import importlib.util
import json
import os
import random
import re

import pytest

from roman_wheel_node import (
    LiveWeights,
    ThresholdLearner,
    clamp01,
    give_feedback,
    opcode,
    opcode_learned,
    roman_wheel_node,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_SIX = [
    ((1, [0.9, 0.92, 0.88]), (0.9, "111", "ACCEPT")),
    ((1, [0.7] * 4), (0.7, "110", "ACCEPT")),
    ((0, [0.95, 0.9]), (0.0, "000", "BLOCK")),
    ((1, [0.8, 0.0, 0.9]), (0.0, "000", "BLOCK")),
    ((1, [1.5, 0.9]), (0.949, "111", "ACCEPT")),
    ((1, []), (0.0, "000", "BLOCK")),
]


def lw(tmp_path, name="w.jsonl", **kw):
    return LiveWeights(str(tmp_path / name), **kw)


# --- backward compatibility ---------------------------------------------------------------
@pytest.mark.parametrize("args,expected", ORIGINAL_SIX)
def test_original_six_identical(args, expected):
    assert roman_wheel_node(*args) == expected


def _load_v2():
    p = os.path.join(HERE, "_v2_reference.py")  # frozen v2 implementation, for bit-compat checks
    spec = importlib.util.spec_from_file_location("rw_v2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("mode", ["geomean", "product"])
@pytest.mark.parametrize("flip", [False, True])
def test_v2_paths_bit_identical_on_random_claims(tmp_path, mode, flip):
    v2 = _load_v2()
    rng = random.Random(1)
    old = v2.LiveWeights(str(tmp_path / "old.jsonl"), half_life=None)
    new = LiveWeights(str(tmp_path / "new.jsonl"), half_life=None)
    for _ in range(400):
        sc = {
            k: rng.choice([0.0, rng.random(), 1.0, 1.3, -0.2]) if rng.random() < 0.05 else rng.random()
            for k in ("a", "b", "c", "d")
        }
        assert roman_wheel_node(1, sc, new, mode=mode, flip=flip) == v2.roman_wheel_node(
            1, sc, old, mode=mode, flip=flip
        )
        assert roman_wheel_node(1, sc, mode=mode) == v2.roman_wheel_node(1, sc, mode=mode)
        truth = rng.random() < 0.5
        for k, v in sc.items():
            o = 1.0 if (v > 0.5) == truth else 0.0
            new.observe(k, o)
            old.observe(k, o)


# --- gate, empty, veto, clamping ----------------------------------------------------------
@pytest.mark.parametrize("mode", ["geomean", "product", "bayes"])
def test_gate_blocks_non_exact_copy(mode):
    assert roman_wheel_node(0, {"x": 0.99}, mode=mode) == (0.0, "000", "BLOCK")
    assert roman_wheel_node(-1, {"x": 0.99}, mode=mode) == (0.0, "000", "BLOCK")


@pytest.mark.parametrize("mode", ["geomean", "product", "bayes"])
@pytest.mark.parametrize("empty", [[], {}, ()])
def test_empty_scores_block(mode, empty):
    assert roman_wheel_node(1, empty, mode=mode) == (0.0, "000", "BLOCK")


@pytest.mark.parametrize("mode", ["geomean", "product", "bayes"])
@pytest.mark.parametrize("zero", [0.0, 0, -0.5])
def test_hard_zero_veto_even_from_trusted_or_flipped_source(tmp_path, mode, zero):
    w = lw(tmp_path, half_life=None)
    for _ in range(50):
        w.observe("trusted", 1)
        w.observe("liar", 0)
    assert roman_wheel_node(1, {"trusted": zero, "x": 0.99}, w, mode=mode)[2] == "BLOCK"
    assert roman_wheel_node(1, {"liar": zero, "x": 0.99}, w, mode=mode, flip=True)[2] == "BLOCK"


def test_clamping():
    assert clamp01(1.7) == 1.0 and clamp01(-3) == 0.0 and clamp01("0.25") == 0.25
    assert roman_wheel_node(1, [1.5, 0.9]) == roman_wheel_node(1, [1.0, 0.9])
    assert roman_wheel_node(1, [5.0, 5.0], mode="bayes") == roman_wheel_node(1, [1.0, 1.0], mode="bayes")


def test_fixed_opcode_bands():
    assert [opcode(x)[0] for x in (0.9, 0.7, 0.4, 0.2, 0.05)] == ["111", "110", "101", "100", "000"]


def test_learned_opcode_bands():
    c = 0.6
    assert opcode_learned(0.99, c) == ("111", "ACCEPT")
    assert opcode_learned(0.75, c) == ("110", "ACCEPT")
    assert opcode_learned(0.61, c) == ("101", "ACCEPT")
    assert opcode_learned(0.50, c) == ("100", "HEDGE")
    assert opcode_learned(0.30, c) == ("000", "BLOCK")
    assert opcode_learned(0.04, 0.1)[1] == "BLOCK"  # never HEDGE at or below 0.05


# --- ledger: append-only, replay determinism, decay ----------------------------------------
def test_ledger_is_append_only(tmp_path):
    w = lw(tmp_path, half_life=None)
    path = tmp_path / "w.jsonl"
    w.observe("a", 1)
    first = path.read_text(encoding="utf-8")
    w.observe("a", 0)
    w.observe("b", 1)
    after = path.read_text(encoding="utf-8")
    assert after.startswith(first) and len(after.splitlines()) == 3
    assert all(set(json.loads(line)) >= {"t", "source", "outcome", "d_alpha", "d_beta"} for line in after.splitlines())


@pytest.mark.parametrize("hl", [None, 150, 7])
def test_replay_is_deterministic(tmp_path, hl):
    w = lw(tmp_path, half_life=hl)
    rng = random.Random(3)
    for _ in range(500):
        w.observe(rng.choice("abc"), rng.random() < 0.7)
    w2 = lw(tmp_path, half_life=hl)
    for s in "abc":
        assert w2.reliability(s) == w.reliability(s)


def test_default_half_life_is_150(tmp_path):
    assert lw(tmp_path).half_life == 150
    assert lw(tmp_path, "n.jsonl", half_life=None).decay == 1.0


def test_decay_tracks_change_faster_than_no_decay(tmp_path):
    keep, forget = lw(tmp_path, "k.jsonl", half_life=None), lw(tmp_path, "f.jsonl", half_life=50)
    for _ in range(1000):
        keep.observe("s", 1)
        forget.observe("s", 1)
    for _ in range(100):
        keep.observe("s", 0)
        forget.observe("s", 0)
    assert keep.reliability("s") > 0.85 and forget.reliability("s") < 0.3


def test_decay_half_life_math(tmp_path):
    w = lw(tmp_path, half_life=10)
    w.observe("s", 1)  # alpha excess = 1
    for _ in range(10):
        w.observe("s", 0.5, strength=0)  # 10 empty observations
    a, b = w.state["s"]
    assert a - 1.0 == pytest.approx(0.5)


def test_same_ledger_replayed_with_other_half_life(tmp_path):
    w = lw(tmp_path, half_life=None)
    for _ in range(300):
        w.observe("s", 1)
    for _ in range(60):
        w.observe("s", 0)
    assert lw(tmp_path, half_life=None).reliability("s") > lw(tmp_path, half_life=20).reliability("s")


# --- flip -------------------------------------------------------------------------------------
def test_flip_inverts_reliably_wrong_source(tmp_path):
    w = lw(tmp_path, half_life=None)
    for _ in range(40):
        w.observe("liar", 0)
        w.observe("ok", 1)
    sc = {"liar": 0.1, "ok": 0.9}
    assert roman_wheel_node(1, sc, w, flip=True)[0] > roman_wheel_node(1, sc, w, flip=False)[0]


# --- bayes ------------------------------------------------------------------------------------
def test_bayes_posterior_matches_closed_form(tmp_path):
    w = lw(tmp_path, half_life=None)
    for s in ("a", "b"):
        for _ in range(7):
            w.observe(s, 1)
        w.observe(s, 0)  # r = 8/10 = 0.8
    eff, _, st = roman_wheel_node(1, {"a": 0.9, "b": 0.9}, w, mode="bayes", variant="hard")
    assert eff == round(0.64 / (0.64 + 0.04), 3) and st == "ACCEPT"
    eff2, *_ = roman_wheel_node(1, {"a": 0.9, "b": 0.1}, w, mode="bayes", variant="hard")
    assert eff2 == 0.5  # equal and opposite evidence


@pytest.mark.parametrize("variant", ["hard", "soft", "logit"])
def test_bayes_auto_flips_and_ignores_coin_flips(tmp_path, variant):
    w = lw(tmp_path, half_life=None)
    for i in range(200):
        w.observe("liar", 0)
        w.observe("coin", i % 2)
    assert roman_wheel_node(1, {"liar": 0.1}, w, mode="bayes", variant=variant)[0] > 0.85
    assert roman_wheel_node(1, {"coin": 0.95}, w, mode="bayes", variant=variant)[0] == pytest.approx(0.5, abs=0.02)


def test_bayes_symmetry_and_monotonicity():
    lo = roman_wheel_node(1, {"a": 0.3, "b": 0.4}, mode="bayes", variant="soft")[0]
    hi = roman_wheel_node(1, {"a": 0.7, "b": 0.6}, mode="bayes", variant="soft")[0]
    assert lo == pytest.approx(1 - hi, abs=1e-3)
    assert (
        roman_wheel_node(1, {"a": 0.95}, mode="bayes", variant="soft")[0]
        > roman_wheel_node(1, {"a": 0.6}, mode="bayes", variant="soft")[0]
    )


def test_bayes_more_agreeing_evidence_increases_confidence():
    """Unlike product mode, extra agreeing sources never lower eff."""
    effs = [roman_wheel_node(1, [0.7] * n, mode="bayes")[0] for n in (1, 2, 4, 10)]
    assert effs == sorted(effs) and effs[-1] > effs[0]


def test_bayes_default_cutoff_is_half_and_fixed_still_available():
    assert roman_wheel_node(1, {"a": 0.6}, mode="bayes", prior=0.4, assumed_reliability=0.5)[2] != "ACCEPT"
    assert (
        roman_wheel_node(1, {"a": 0.6}, mode="bayes", prior=0.4, assumed_reliability=0.5, cutoff="fixed")[2] == "ACCEPT"
    )


def test_bad_variant_raises():
    with pytest.raises(ValueError):
        roman_wheel_node(1, {"a": 0.6}, mode="bayes", variant="nope")


# --- learned threshold ------------------------------------------------------------------------
def test_threshold_default_before_min_obs(tmp_path):
    t = ThresholdLearner(str(tmp_path / "t.jsonl"), min_obs=30, default=0.5)
    for _ in range(29):
        t.record(0.9, True)
    assert t.cutoff() == 0.5


def test_threshold_finds_separating_cutoff(tmp_path):
    t = ThresholdLearner(str(tmp_path / "t.jsonl"), half_life=None)
    rng = random.Random(0)
    for _ in range(2000):
        truth = rng.random() < 0.5
        t.record(rng.uniform(0.7, 1) if truth else rng.uniform(0, 0.7), truth)
    c = t.cutoff()
    assert 0.68 <= c <= 0.72


def test_threshold_modes_are_separate_and_replay_deterministic(tmp_path):
    p = str(tmp_path / "t.jsonl")
    t = ThresholdLearner(p, half_life=100)
    rng = random.Random(5)
    for _ in range(500):
        tr = rng.random() < 0.5
        t.record(rng.uniform(0.3, 1) if tr else rng.uniform(0, 0.3), tr, mode="geomean")
        t.record(rng.uniform(0.8, 1) if tr else rng.uniform(0, 0.8), tr, mode="bayes:hard")
    assert abs(t.cutoff("geomean") - t.cutoff("bayes:hard")) > 0.3
    t2 = ThresholdLearner(p, half_life=100)
    assert t2.cutoff("geomean") == t.cutoff("geomean") and t2.cutoff("bayes:hard") == t.cutoff("bayes:hard")


def test_threshold_decay_follows_shift(tmp_path):
    t = ThresholdLearner(str(tmp_path / "t.jsonl"), half_life=100)
    rng = random.Random(9)
    for phase_cut in (0.3, 0.8):
        for _ in range(1500):
            tr = rng.random() < 0.5
            t.record(rng.uniform(phase_cut, 1) if tr else rng.uniform(0, phase_cut), tr)
    assert t.cutoff() == pytest.approx(0.8, abs=0.03)


def test_node_uses_learned_cutoff_and_feedback_roundtrip(tmp_path):
    w = lw(tmp_path)
    t = ThresholdLearner(str(tmp_path / "t.jsonl"))
    for _ in range(200):
        give_feedback({"x": 0.9}, True, live=w)
    eff0 = roman_wheel_node(1, {"x": 0.9}, w, mode="bayes")[0]
    for _ in range(200):
        t.record(eff0, False, mode="bayes:hard")  # pretend claims at this eff keep being false
    eff, _, st = roman_wheel_node(1, {"x": 0.9}, w, mode="bayes", threshold=t)
    assert t.cutoff("bayes:hard") > eff and st != "ACCEPT"
    give_feedback({"x": 0.9}, True, threshold=t, eff=eff)
    assert t.n("bayes:hard") == 201 and eff == eff0


def test_audit_log(tmp_path):
    p = tmp_path / "audit.jsonl"
    r = roman_wheel_node(1, {"a": 0.8}, mode="bayes", log_path=str(p))
    ev = json.loads(p.read_text(encoding="utf-8"))
    assert (ev["eff"], ev["opcode"], ev["status"]) == r and ev["mode"] == "bayes:hard"


def test_package_metadata():
    import roman_wheel_node as pkg
    from roman_wheel_node import _version, core

    assert pkg.__version__ == core.__version__ == _version.__version__ and callable(pkg.roman_wheel_node)
    pyproject = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pyproject.toml")
    if os.path.exists(pyproject):  # repo / sdist: the version is read from _version.py, never duplicated
        with open(pyproject, encoding="utf-8") as f:
            text = f.read()
        assert 'dynamic = ["version"]' in text and "roman_wheel_node._version.__version__" in text
        assert not re.search(r'(?m)^version\s*=\s*"', text)  # no static version string
