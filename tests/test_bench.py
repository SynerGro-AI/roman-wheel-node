"""Phase 2: the generator is exact and the benchmark is byte-reproducible."""
import json, os, subprocess, sys

import pytest

from roman_wheel_node.bench import SCENARIOS, stream, run, run_one, to_json, to_text
from roman_wheel_node.bench.generator import main as gen_main, oracle_llr
from roman_wheel_node.bench.runner import R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMITTED = os.path.join(ROOT, "results", "benchmark.json")


def test_six_standard_scenarios():
    assert list(SCENARIOS) == ["baseline", "drift", "bad_majority", "all_noisy", "cold_start", "copycats"]
    assert SCENARIOS["cold_start"].n == 50 and all(SCENARIOS[k].n == 3000 for k in SCENARIOS if k != "cold_start")


def test_stream_is_deterministic_and_seed_dependent():
    a = list(stream(SCENARIOS["baseline"], 7, 200)); b = list(stream(SCENARIOS["baseline"], 7, 200))
    c = list(stream(SCENARIOS["baseline"], 8, 200))
    assert a == b and a != c


def test_noise_model_and_floor():
    cs = list(stream(SCENARIOS["all_noisy"], 0, 2000))
    vals = [v for c in cs for v in c.scores.values()]
    assert min(vals) >= 0.01 and max(vals) < 1.0
    # sources say the right thing ~58% of the time (vote = score > 0.5)
    right = sum((v > .5) == c.truth for c in cs for v in c.scores.values()) / len(vals)
    assert 0.55 < right < 0.61


def test_drift_schedule():
    sc = SCENARIOS["drift"]
    assert sc.true_accuracy(1499)["drone"] == 0.92 and sc.true_accuracy(1500)["drone"] == 0.15


def test_copycat_marginal_accuracy_and_correlation():
    sc = SCENARIOS["copycats"]
    assert sc.true_accuracy(0)["c1"] == pytest.approx(0.9 * 0.65 + 0.1 * 0.55)
    cs = list(stream(sc, 0, 3000))
    agree = sum((c.scores["lead"] > .5) == (c.scores["c1"] > .5) for c in cs) / len(cs)
    assert agree > 0.88                       # copies, not independent (independent would be ~0.47)


def test_oracle_beats_chance_and_is_symmetric():
    sc = SCENARIOS["baseline"]
    cs = list(stream(sc, 1, 3000))
    assert sum((c.oracle_llr > 0) == c.truth for c in cs) / len(cs) > 0.9
    s = {k: 0.5 for k in ("drone", "trial", "draw", "harmony", "instr")}
    assert oracle_llr(sc, 0, s) == pytest.approx(0.0, abs=1e-6)


def test_generator_cli(tmp_path, capsys):
    out = tmp_path / "s.jsonl"
    assert gen_main(["--scenario", "copycats", "--seed", "3", "--n", "10", "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    first = json.loads(lines[0])
    c0 = next(stream(SCENARIOS["copycats"], 3, 1))
    assert len(lines) == 10 and first["scores"] == c0.scores and first["truth"] == c0.truth
    assert gen_main(["--list"]) == 0
    assert set(json.loads(capsys.readouterr().out)) == set(SCENARIOS)


def test_output_identical_across_runs_and_worker_counts(tmp_path):
    r1 = run(2, ["cold_start", "all_noisy"], ceiling_claims=500, jobs=1)
    r2 = run(2, ["cold_start", "all_noisy"], ceiling_claims=500, jobs=3)
    assert to_json(r1) == to_json(r2) and to_text(r1) == to_text(r2)


def test_cli_byte_identical(tmp_path):
    args = [sys.executable, "-m", "roman_wheel_node.bench", "--seeds", "2", "--scenarios", "cold_start",
            "--ceiling-claims", "500", "--no-plots", "--jobs", "1"]
    for d in ("a", "b"):
        subprocess.run(args + ["--out", str(tmp_path / d)], check=True, capture_output=True)
    for f in ("benchmark.json", "benchmark.txt"):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()
    chk = subprocess.run(args + ["--check", str(tmp_path / "a")], capture_output=True, text=True)
    assert chk.returncode == 0 and "IDENTICAL" in chk.stdout


@pytest.mark.skipif(not os.path.exists(COMMITTED), reason="committed results not present")
@pytest.mark.parametrize("key,seeds", [("cold_start", range(20)), ("baseline", [0]), ("copycats", [7])])
def test_committed_results_reproduce(key, seeds):
    res = json.load(open(COMMITTED))
    assert res["config"]["seeds"] >= 20
    pol = res["scenarios"][key]["policies"]
    for s in seeds:
        got = run_one(key, s)
        for p, metrics in got["policies"].items():
            for m, v in metrics.items():
                assert R(v) == pol[p][m]["per_seed"][s], (key, s, p, m)


@pytest.mark.skipif(os.environ.get("RWN_FULL_BENCH") != "1", reason="set RWN_FULL_BENCH=1 (~1 min on 8 cores)")
def test_full_benchmark_byte_identical_to_committed():
    chk = subprocess.run([sys.executable, "-m", "roman_wheel_node.bench", "--seeds", "20",
                          "--check", os.path.join(ROOT, "results")], capture_output=True, text=True)
    assert chk.returncode == 0, chk.stdout + chk.stderr
