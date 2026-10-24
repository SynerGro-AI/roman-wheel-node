"""Phase 5 Task A: split, manifest, veto, threshold rule and fixed-sequence LTT (no torch needed)."""
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase5")):
    pytest.skip("phase5/ research scripts are repo-only (not in the sdist/wheel)", allow_module_level=True)
sys.path.insert(0, os.path.join(ROOT, "phase5"))
import pairs_common as C  # noqa: E402

ITEMS = os.path.join(ROOT, "phase3", "cache", "phase4", "items.json")


def _toy():
    out = []
    for g in range(40):
        ev = f"Evidence sentence number {g} about topic {g * 7}."
        out.append({"id": f"t{g}s", "claim": f"claim {g} true", "label3": "SUPPORT", "evidence": ev, "units": [ev]})
        out.append({"id": f"t{g}r", "claim": f"claim {g} fake", "label3": "CONTRADICT", "evidence": ev, "units": [ev]})
    return out


def test_split_groups_never_straddle_and_match_phase4b():
    items = _toy()
    roles = C.build_split(items)
    for g, r in roles.items():
        assert (r == "B_test") == (C.p4b_half(g) == "B")
    ids = C.check_roles(items, C.manifest_doc(items, roles))
    by_group = {}
    for it in items:
        by_group.setdefault(C.group_of(it["evidence"]), set()).add(ids[it["id"]])
    assert all(len(v) == 1 for v in by_group.values())


def test_quarantine_shared_sentence():
    items = _toy()
    roles = C.build_split(items)
    a = next(g for g, r in roles.items() if r != "B_test")
    b_ev = next(it["evidence"] for it in items if roles[C.group_of(it["evidence"])] == "B_test")
    for it in items:
        if C.group_of(it["evidence"]) == a:
            it["units"] = it["units"] + [b_ev]
    assert C.build_split(items)[a] == "A_quarantine"


def test_manifest_has_no_text_and_is_self_hashed(tmp_path):
    m = C.load_manifest()
    assert all(re.fullmatch(r"[0-9a-f]{20}", k) and v in C.ROLES for k, v in m["groups"].items())
    assert sum(c["items"] for c in m["counts"].values()) == 4086 and len(m["groups"]) == 1290
    assert m["counts"]["B_test"] == {"groups": 680, "items": 2143, "supported": 682, "refuted": 1461}
    blob = json.dumps(m)
    assert len(blob) < 80000
    bad = dict(m); bad["groups"] = dict(m["groups"]); k = next(iter(bad["groups"])); bad["groups"][k] = "A_train"
    p = tmp_path / "m.json"; p.write_text(json.dumps(bad))
    if m["groups"][k] != "A_train":
        with pytest.raises(SystemExit):
            C.load_manifest(str(p))


@pytest.mark.skipif(not os.path.exists(ITEMS), reason="internal COVID-Fact cache not present")
def test_manifest_reproduces_from_data():
    items = [i for i in json.load(open(ITEMS, encoding="utf-8")) if i["ds"] == "covidfact"]
    assert C.manifest_doc(items, C.build_split(items))["sha256"] == C.load_manifest()["sha256"]
    blob = open(C.MANIFEST, encoding="utf-8").read()
    assert not any(it["claim"] in blob for it in items if len(it["claim"]) > 15)
    assert not any(C.group_of(it["evidence"]) in blob for it in items)          # raw group keys are salted away


def test_protocol_frozen():
    p = json.load(open(C.PROTOCOL))
    assert p["sha256"] == C.canon_sha({k: v for k, v in p.items() if k != "sha256"})
    assert p["split_manifest_sha256"] == C.load_manifest()["sha256"]
    assert [t["retention"] for t in p["tau_sequence"]] == [0.80, 0.90, 0.95]


def test_veto_is_accept_only():
    assert C.apply_veto("ACCEPT", 0.2, 0.5) == "NOT_SURE"
    assert C.apply_veto("ACCEPT", 0.5, 0.5) == "ACCEPT"
    assert C.apply_veto("ACCEPT", None, 0.5) == "NOT_SURE" and C.apply_veto("ACCEPT", float("nan"), 0.1) == "NOT_SURE"
    for o in ("BLOCK", "NOT_SURE", None):
        assert C.apply_veto(o, 0.99, 0.5) == o and C.apply_veto(o, 0.0, 0.5) == o


def test_tau_rule():
    xs = [i / 100 for i in range(100)]
    assert C.tau_for_retention(xs, 0.90) == 0.10 and C.tau_for_retention(xs, 0.95) == 0.05
    assert C.tau_for_retention(xs, 0.80) >= C.tau_for_retention(xs, 0.90) >= C.tau_for_retention(xs, 0.95)
    kept = sum(x >= C.tau_for_retention(xs, 0.9) for x in xs) / len(xs)
    assert kept >= 0.9
    with pytest.raises(ValueError):
        C.tau_for_retention([], 0.9)


def test_fixed_sequence_stops_at_first_failure():
    ok = {"accepted": 400, "false_accept_share_upper": 0.004}
    bad = {"accepted": 400, "false_accept_share_upper": 0.3}
    r = C.fixed_sequence_ltt([("a", ok), ("b", bad), ("c", ok)], [0.01, 0.05])
    assert r["0.01"] == {"certified": ["a"], "loosest_certified": "a"}
    r = C.fixed_sequence_ltt([("a", bad), ("b", ok)], [0.05])
    assert r["0.05"]["certified"] == []
