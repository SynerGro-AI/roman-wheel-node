"""ThresholdLearner.band: the selective-mode abstain band (new in Phase 3b; existing behaviour unchanged)."""
import random

from roman_wheel_node import ThresholdLearner


def test_band_reaches_target_and_maximises_coverage(tmp_path):
    rng = random.Random(0)
    th = ThresholdLearner(str(tmp_path / "t.jsonl"), half_life=None)
    for _ in range(2000):
        truth = rng.random() < 0.5
        eff = min(1.0, max(0.0, rng.gauss(0.75 if truth else 0.25, 0.18)))
        th.record(eff, truth, "m")
    lo, hi, cov, acc = th.band("m", target=0.99)
    assert lo < 0.5 < hi and acc >= 0.99 and 0.2 < cov < 0.95
    lo2, hi2, cov2, acc2 = th.band("m", target=0.95)
    assert cov2 > cov and lo2 >= lo and hi2 <= hi          # looser target -> narrower band, more coverage
    # reproducible from the ledger alone
    assert ThresholdLearner(str(tmp_path / "t.jsonl"), half_life=None).band("m", 0.99) == (lo, hi, cov, acc)


def test_band_unknown_mode_abstains_everywhere(tmp_path):
    th = ThresholdLearner(str(tmp_path / "t.jsonl"))
    lo, hi, cov, acc = th.band("nothing")
    assert cov == 0.0 and hi > 1.0
