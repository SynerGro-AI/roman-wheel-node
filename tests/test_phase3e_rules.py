"""Phase 3e selective rules: fixed-sequence LTT, accept filters, BLOCK accounting (offline, no network, no key)."""
import os, random, subprocess, sys

import pytest  # noqa: E402
if not os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase3")):
    pytest.skip("phase3/ research scripts are repo-only (not in the sdist/wheel)", allow_module_level=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase3"))
import s3e_rules as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def it(side, z, y, acc_ok=True, rej_ok=True):
    return {"side": side, "z": z, "y": y, "acc_ok": acc_ok, "rej_ok": rej_ok}


def test_cp_upper_zero_errors_closed_form():
    for n in (10, 150, 800):
        assert abs(R.cp_upper(0, n, 0.05) - (1 - 0.05 ** (1 / n))) < 1e-6
    assert R.cp_upper(0, 0) == 1.0 and R.cp_upper(5, 5) == 1.0


def test_min_decided_is_tight():
    for al, d in ((0.01, 0.05), (0.02, 3e-4), (0.05, 1e-3), (0.10, 0.05)):
        n = R.min_decided_for(al, d)
        assert R.cp_upper(0, n, d) <= al < R.cp_upper(0, n - 1, d)


def test_judge_votes_counts_raw_top_label():
    src = {"J1": {"S": .8, "C": .1, "N": .1}, "J2": {"S": .1, "C": .1, "N": .8}, "J3": {"S": .5, "C": .5, "N": 0}}
    assert R.judge_votes(src, ["J1", "J2"]) == 1


def test_decide_block_notsure_and_filters():
    assert R.decide(it("B", R.INF, True), 0, 0) is False            # gate BLOCK is never accepted
    assert R.decide(it("X", R.INF, True), 0, 0) is None             # BLOCK reported as its own outcome
    assert R.decide(it("A", 9.0, True, acc_ok=False), 0, 0) is None  # failed accept filter -> NOT_SURE, not ACCEPT
    assert R.decide(it("A", 1.0, True), 2.0, 0) is None
    assert R.decide(it("A", 3.0, True), 2.0, 0) is True
    assert R.decide(it("R", 3.0, False, rej_ok=False), 0, 0) is None
    assert R.decide(it("R", 3.0, False), 0, R.INF) is None          # rho = 0: accept-only


def test_fixed_sequence_stops_at_first_failure():
    # accepts ordered by z: the top 100 clean, then a block of 100 errors, then 800 clean (later steps would pass alone)
    items = [it("A", 1000 - i, not (100 <= i < 200)) for i in range(1000)]
    r = R.ltt_fixed_sequence(items, 0.10, 0.05, rho=0.0, K=10)
    assert r["certified"] and r["step"] == 1 and r["decided"] == 100 and r["errors"] == 0


def test_small_steps_are_skipped_not_failed():
    # 40 steps over 400 clean accepts: early steps (< min n) must be skipped, so the full set is certified at 1%
    items = [it("A", 400 - i, True) for i in range(400)]
    n_min = R.min_decided_for(0.01, 0.05)
    r = R.ltt_fixed_sequence(items, 0.01, 0.05, rho=0.0, K=40)
    assert n_min < 400 and r["certified"] and r["decided"] == 400


def test_precision_risk_ignores_rejects():
    items = [it("A", 10 - i, True) for i in range(5)] + [it("R", 10, True) for _ in range(5)]   # 5 wrong rejects
    assert R.risk(items, 0, 0) == (10, 5, 5, 0)
    assert R.risk(items, 0, 0, precision=True) == (5, 0, 5, 0)


def test_to_items_block_accounting_and_sides():
    rule = (lambda f: f["votes"] == 6, lambda f: True)
    feats = [{"y": True, "blocked": True, "votes": 6}, {"y": True, "blocked": False, "votes": 6},
             {"y": False, "blocked": False, "votes": 2}]
    a = R.to_items(feats, [0.0, 0.9, 0.1], [0.5] * 3, rule)
    b = R.to_items(feats, [0.0, 0.9, 0.1], [0.5] * 3, rule, blocks_decided=False)
    assert [x["side"] for x in a] == ["B", "A", "R"] and b[0]["side"] == "X"
    assert a[1]["acc_ok"] and a[1]["z"] > 0


def test_certify_family_bonferroni_and_choice():
    clean = [it("A", 500 - i, True) for i in range(500)]
    noisy = [it("A", 500 - i, i % 5 != 0) for i in range(500)]
    out = R.certify_family({"clean": clean, "noisy": noisy}, (0.05,), 0.05, rhos=(0.0, 1.0), K=10)
    assert out["n_tests"] == 4 and abs(out["delta_each"] - 0.0125) < 1e-12
    assert out["alphas"]["0.05"]["config"] == "clean"


def test_certified_bounds_hold_and_filters_respected_random():
    rng = random.Random(0)
    for _ in range(20):
        items = [it(rng.choice("ARB"), rng.random() * 10, rng.random() < 0.6, rng.random() < 0.7, rng.random() < 0.7)
                 for _ in range(300)]
        r = R.ltt_fixed_sequence(items, 0.2, 0.05, rho=1.0, K=20)
        if r["certified"]:
            assert r["upper"] <= 0.2
            for x in items:
                d = R.decide(x, r["m_a"], r["m_r"])
                assert not (d is True and not x["acc_ok"])


def test_ltt_false_certification_rate_below_delta():
    # every decided claim has true error 15%: certifying alpha = 10% is a false certificate; must happen <~ delta
    rng = random.Random(1)
    bad = 0
    T = 300
    for _ in range(T):
        items = [it("A", rng.random(), rng.random() >= 0.15) for _ in range(400)]
        bad += R.ltt_fixed_sequence(items, 0.10, 0.05, rho=0.0, K=20)["certified"]
    assert bad / T <= 0.05 + 0.03


def test_s3e_eval_refuses_with_api_key():
    env = dict(os.environ, XAI_API_KEY="not-a-real-key", PYTHONPATH=os.path.join(ROOT, "src"))
    p = subprocess.run([sys.executable, os.path.join(ROOT, "phase3", "s3e_eval.py")], env=env, capture_output=True, text=True)
    assert p.returncode != 0 and "refusing to run" in p.stderr
