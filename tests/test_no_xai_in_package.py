"""Guard: nothing shipped depends on xAI/Grok (or any hosted-model) output.

The owner's rule: xAI/Grok judge results may be kept only as a repo-only benchmark ledger
(results/phase6/xai_panel_ledger.md), never in shipped weights, code paths or defaults. This test scans every file of
the installed package (so it checks the wheel too when run against an installed wheel), the shipped policy, and the
loader, and fails if anything xAI-derived can come back.
"""

import json
import os
import re
from pathlib import Path

import pytest

import roman_wheel_node
from roman_wheel_node.policy import DEFAULT_POLICY_PATH, LOCAL_SOURCES, PolicyError, load_policy, policy_sha256

PKG = Path(roman_wheel_node.__file__).resolve().parent
ROOT = Path(__file__).resolve().parent.parent
# Withdrawn policy files whose profiles or margins were fitted with xAI judge outputs (4.0.0-4.2.0), by sha256.
WITHDRAWN = (
    "254804f8ac9b500ef1951879fb64328b0f54303b8bef20e38b1684545f876eb1",
    "2156f5844d0d1fdad9d415b0aba0226c8c6b17a8a9b2294546b166dff4c3c82d",
    "6f7bb9e3762754dc6ca6343c44f000c7caf05a819a1f63c857471856d3946c2a",
)
FORBIDDEN = [
    r"xai",
    r"grok",
    r"x\.ai",
    r"panel",
    r"judge",
    r"\bJ\d{1,2}\b",
    r"needs_judges",
    r"gate_judge_quote",
    r"quote_verbatim",
    r"enable[-_]xai",
    r"cap[-_]usd",
    r"rwn-policy-4\.[0-2]\.0",
    r"\b4\.[0-2]\.0\b",
] + [h[:16] for h in WITHDRAWN]


def shipped_files():
    out = [p for p in PKG.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    assert any(p.name == "fit.py" for p in out) and any(p.suffix == ".json" for p in out), out
    return out


@pytest.mark.parametrize("pattern", FORBIDDEN)
def test_no_shipped_file_mentions_xai_derived_fits(pattern):
    rx = re.compile(pattern, re.IGNORECASE)
    hits = []
    for p in shipped_files():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{p.relative_to(PKG)}:{i}: {line.strip()[:100]}")
    assert not hits, "\n".join(hits)


def test_only_the_local_policy_ships():
    assert sorted(os.listdir(os.path.dirname(DEFAULT_POLICY_PATH))) == ["rwn-policy-4.3.0.json"]
    for p in shipped_files():
        if p.suffix == ".json":
            doc = json.loads(p.read_text(encoding="utf-8"))
            assert doc["sha256"] not in WITHDRAWN
            for name, prof in doc["profiles"].items():
                assert set(prof["sources"]) <= set(LOCAL_SOURCES), (p.name, name)
                assert set(prof["calibrators"]) <= set(LOCAL_SOURCES) and set(prof["reliability"]) <= set(LOCAL_SOURCES)
    pol = load_policy()
    assert sorted(pol.profiles) == ["local", "local-lite"] and pol.default_profile == "local"
    for name in ("panel", "J1", "xai"):
        with pytest.raises(PolicyError):
            pol.profile(name)


def test_loader_refuses_hosted_model_profiles(tmp_path):
    base = json.loads(Path(DEFAULT_POLICY_PATH).read_text(encoding="utf-8"))
    local = base["profiles"]["local"]

    def refused(mutate):
        doc = json.loads(json.dumps(base))
        mutate(doc)
        doc["sha256"] = policy_sha256(doc)
        p = tmp_path / "p.json"
        p.write_text(json.dumps(doc))
        with pytest.raises(PolicyError):
            load_policy(str(p), expected_sha256=doc["sha256"])

    def add_judge(d):
        prof = d["profiles"]["local"]
        prof["sources"] = prof["sources"] + ["J1"]
        prof["reliability"]["J1"] = 0.9
        prof["calibrators"]["J1"] = local["calibrators"]["overlap"]

    def panel_profile(d):
        p = json.loads(json.dumps(local))
        p["sources"] = ["J2", "nli2", "overlap"]
        p["reliability"]["J2"] = 0.9
        p["calibrators"]["J2"] = local["calibrators"]["overlap"]
        d["profiles"]["panel"] = p
        d["default_profile"] = "panel"

    refused(add_judge)
    refused(panel_profile)
    refused(lambda d: d["profiles"]["local"].update(needs_judges=True))
    refused(lambda d: d["profiles"]["local"].update(gate_judge_quote="J1"))
    refused(lambda d: d["profiles"]["local"].update(disagreement={"judge_split": True}))


def test_ledger_stays_out_of_the_distribution():
    """results/ (which holds the xAI ledger) is repo-only: the sdist never includes it and the wheel has only the
    package. Run from a checkout this reads MANIFEST.in; run from an unpacked sdist it checks the tree itself."""
    if (ROOT / "PKG-INFO").is_file():  # an unpacked sdist
        assert not (ROOT / "results").exists() and not (ROOT / "phase6").exists()
        return
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    included = [ln for ln in manifest.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert not [ln for ln in included if "results" in ln or "phase" in ln or "graft" in ln], included
    assert (ROOT / "results" / "phase6" / "xai_panel_ledger.md").is_file()
    ledger = (ROOT / "results" / "phase6" / "xai_panel_ledger.md").read_text(encoding="utf-8")
    assert "benchmark comparison" in ledger.lower() and "not used for fitting" in ledger.lower()
