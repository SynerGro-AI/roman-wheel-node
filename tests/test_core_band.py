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


# ---- NOT_SURE: a distinct, fail-closed gate outcome in ClaimTrace -----------------------------------------------
import json  # noqa: E402

import pytest  # noqa: E402

from roman_wheel_node.trace import ClaimTrace, TamperError, replay  # noqa: E402


def _tr(tmp_path, **kw):
    return ClaimTrace(str(tmp_path / "audit.jsonl"), mode="bayes", variant="hard", **kw)


def test_not_sure_band_is_never_admitted_even_with_hedge(tmp_path):
    tr = _tr(tmp_path, not_sure_band=(0.2, 0.95), allow_hedge=True)
    d = tr.submit("mid", {"a": 0.9, "b": 0.9}, a=1)          # eff 0.9 (0.75 * 0.75 odds) is inside the band
    assert d.status == "NOT_SURE" and not d.admitted and d.reason == "not_sure" and d.claim is None
    d2 = tr.submit("sure", {"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9}, a=1)
    assert d2.eff >= 0.95 and d2.admitted and d2.status == "ACCEPT"
    assert len(tr) == 1 and tr.get(f"{d.seq}") is None


def test_forced_not_sure_and_zero_propagation(tmp_path):
    tr = _tr(tmp_path)
    d = tr.submit("judges split", {"a": 0.99, "b": 0.99, "c": 0.99}, a=1, not_sure=True)
    assert d.status == "NOT_SURE" and not d.admitted
    child = tr.submit("built on it", {"a": 0.99, "b": 0.99, "c": 0.99}, a=1, parents=["not-an-admitted-id"])
    assert not child.admitted and child.reason == "parent_not_admitted"
    blocked = tr.submit("gate closed", {"a": 0.99}, a=0, not_sure=True)
    assert blocked.status == "BLOCK"                         # BLOCK stays BLOCK


def test_not_sure_replays_and_tampering_is_caught(tmp_path):
    tr = _tr(tmp_path, not_sure_band=(0.3, 0.97))
    tr.submit("x", {"a": 0.9, "b": 0.9}, a=1)
    tr.submit("y", {"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9}, a=1, not_sure=True)
    rep = tr.verify()
    assert rep.n_records == 2 and rep.n_admitted == 0
    p = tmp_path / "audit.jsonl"
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1]); rec.pop("not_sure_forced"); rec["status"] = "ACCEPT"; rec["admitted"] = True; rec["reason"] = "ok"
    lines[1] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(TamperError):
        replay(str(p))


def test_old_logs_unchanged_without_band(tmp_path):
    tr = _tr(tmp_path)
    tr.submit("z", {"a": 0.9}, a=1)
    rec = json.loads((tmp_path / "audit.jsonl").read_text().splitlines()[0])
    assert "not_sure_band" not in rec and "not_sure_forced" not in rec


def test_bad_band_rejected(tmp_path):
    with pytest.raises(ValueError):
        _tr(tmp_path, not_sure_band=(0.9, 0.1))
