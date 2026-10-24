"""Phase 4 (SAFETY.md G6-G9): the frozen policy, shadow mode never alters agent outputs, everything fails closed,
paid judges stay off by default, and the fresh-set certificate arithmetic."""

import copy
import json
import os
import uuid
from pathlib import Path

import pytest

from roman_wheel_node import fresh as FR
from roman_wheel_node.policy import (
    ACCEPT,
    BLOCK,
    DEFAULT_POLICY_PATH,
    NOT_SURE,
    PolicyError,
    load_policy,
    policy_sha256,
)
from roman_wheel_node.shadow import FrozenNode, ShadowRunner
from roman_wheel_node.sources import BM25, XAIPanel, exact_gate, sha256

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    HAVE_HYP = True
except ImportError:  # pragma: no cover
    HAVE_HYP = False

EV = ["Vitamin D supplementation reduced the incidence of acute respiratory infection in the trial."]
TRUST = {sha256(s) for s in EV}


def fake_sources(nli2=0.99, overlap=0.95, contra=0.01):
    return lambda claim, sents: {"nli2": {"S": nli2, "C": contra}, "overlap": {"S": overlap}}


def runner(tmp_path, **kw):
    kw.setdefault("sources", fake_sources())
    kw.setdefault("trusted_hashes", TRUST)
    return ShadowRunner(str(tmp_path / uuid.uuid4().hex), **kw)


# ---- policy -------------------------------------------------------------------------------------------------------
def test_default_policy_is_pinned_and_honest():
    p = load_policy()
    assert p.default_profile == "local" and not p.profile().needs_judges
    assert p.doc["outcomes"] == [ACCEPT, NOT_SURE, BLOCK]
    assert "Nothing is certified at 1% or 2%" in p.doc["honesty"]
    c = p.profile("panel").certified
    assert c["alpha"] == 0.10 and "SciFact" in c["population"] and p.profile("panel").needs_judges
    for name in ("local", "local-lite"):
        assert p.profile(name).certified["alpha"] is None
    for prof in p.profiles.values():  # accept-only: positive margin over the learned cut-off
        assert prof.accept_margin > 0 and prof.accept_threshold() > prof.cutoff


def test_tampered_policy_refused(tmp_path):
    doc = json.loads(Path(DEFAULT_POLICY_PATH).read_text())
    doc["profiles"]["local"]["accept_margin"] = 0.01
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(PolicyError):  # body no longer matches its own hash
        load_policy(str(bad))
    doc["sha256"] = policy_sha256(doc)  # re-hashed forgery: refused against the pinned hash
    bad.write_text(json.dumps(doc))
    with pytest.raises(PolicyError):
        load_policy(str(bad), expected_sha256=load_policy().sha256)
    with pytest.raises(PolicyError):
        load_policy(str(tmp_path / "missing.json"))


def test_policy_never_rejects_on_posterior(tmp_path):
    n = FrozenNode(str(tmp_path / "a.jsonl"))
    outs = set()
    for s in (0.01, 0.2, 0.5, 0.8, 0.99):
        for o in (0.05, 0.5, 0.95):
            outs.add(n.decide("c", {"nli2": {"S": s, "C": 0.0}, "overlap": {"S": o}}, 1).outcome)
    assert outs <= {ACCEPT, NOT_SURE} and ACCEPT in outs and NOT_SURE in outs
    n.trace.verify()


# ---- shadow never alters outputs ----------------------------------------------------------------------------------
class Out:
    def __init__(self, v):
        self.v = v


if HAVE_HYP:
    outputs = st.one_of(
        st.none(),
        st.integers(),
        st.text(),
        st.lists(st.text(max_size=20), max_size=4),
        st.dictionaries(st.text(max_size=5), st.floats(allow_nan=True), max_size=3),
        st.builds(Out, st.text()),
    )
    raws = st.dictionaries(
        st.sampled_from(["nli", "nli2", "overlap", "J1", "junk"]),
        st.dictionaries(
            st.sampled_from(["S", "C", "N"]),
            st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.text(max_size=3), st.none()),
            max_size=3,
        ),
        max_size=4,
    )

    @settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        out=outputs,
        raw=raws,
        ev=st.lists(st.text(max_size=30), max_size=3),
        boom=st.booleans(),
        trusted=st.booleans(),
        claims=st.lists(st.text(max_size=30), max_size=3),
    )
    def test_shadow_returns_the_agent_output_unchanged(tmp_path, out, raw, ev, boom, trusted, claims):
        def src(claim, sents):
            if boom:
                raise RuntimeError("source crashed")
            return copy.deepcopy(raw)

        evidence = list(ev) + (EV if trusted else [])
        r = runner(tmp_path, sources=src, trusted_hashes={sha256(s) for s in evidence} if trusted else None)
        before = repr(out)

        def agent(x):
            return out

        got = r.wrap(agent, extract=lambda o: [(c, evidence) for c in claims] + [{"claim": "x", "evidence": evidence}])(
            "q"
        )
        assert got is out and repr(out) == before  # same object, untouched
        assert sum(r.counts.values()) == len(claims) + 1
        for rec in FR.load_jsonl(r.fresh_path):  # fail closed: ACCEPT only with a passed gate and full sources
            if rec["outcome"] == ACCEPT:
                assert rec["a"] == 1 and not boom and {"nli2", "overlap"} <= set(rec["raw"])
            if rec["a"] != 1:
                assert rec["outcome"] == BLOCK
        r.verify()

    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(nli2=st.floats(0, 1), ov=st.floats(0, 1), contra=st.floats(0, 1))
    def test_enforce_releases_only_accepts(tmp_path, nli2, ov, contra):
        r = runner(tmp_path, sources=fake_sources(nli2, ov, contra), enforce=True, on_withhold=lambda o, v: "WITHHELD")
        res = r.wrap(lambda: "answer", extract=lambda o: [("claim", EV)])()
        v = json.loads(Path(r.fresh_path).read_text().splitlines()[-1])["outcome"]
        assert res == ("answer" if v == ACCEPT else "WITHHELD")


def test_agent_exceptions_propagate_unchanged(tmp_path):
    r = runner(tmp_path)
    err = ValueError("agent failed")

    def agent():
        raise err

    with pytest.raises(ValueError) as e:
        r.wrap(agent)()
    assert e.value is err and r.counts == {ACCEPT: 0, NOT_SURE: 0, BLOCK: 0}


def test_extract_crash_does_not_touch_output(tmp_path):
    r = runner(tmp_path)
    o = Out("x")
    assert r.wrap(lambda: o, extract=lambda _: 1 / 0)() is o
    r2 = runner(tmp_path, enforce=True)
    assert r2.wrap(lambda: o, extract=lambda _: 1 / 0)() is None  # enforce fails closed


def test_fail_closed_cases(tmp_path):
    r = runner(tmp_path)
    assert r.observe("c", EV).outcome == ACCEPT  # strong, gated, all sources
    assert r.observe("c", ["not in the trusted store"]).outcome == BLOCK  # exact-copy gate
    assert r.observe("c", []).outcome == BLOCK  # no evidence
    for raw in (
        {},
        {"nli2": {"S": 0.99}},
        {"nli2": {"S": float("nan"), "C": 0}, "overlap": {"S": 0.9}},
        {"nli2": {"S": 1.5, "C": 0}, "overlap": {"S": 0.9}},
        {"nli2": {"S": "x"}, "overlap": {"S": 0.9}},
    ):
        rr = runner(tmp_path, sources=lambda c, s, raw=raw: raw)
        assert rr.observe("c", EV).outcome == NOT_SURE, raw
    rr = runner(tmp_path, sources=fake_sources(contra=0.99))
    assert rr.observe("c", EV).outcome in (ACCEPT, NOT_SURE)  # local profile has no disagreement rule
    rn = runner(tmp_path, trusted_hashes=None)
    assert rn.observe("c", EV).outcome == BLOCK  # no trusted store: unverifiable -> BLOCK
    rp = runner(tmp_path, profile="panel")  # panel without judges: never ACCEPT
    assert rp.observe("c", EV).outcome == NOT_SURE


def test_retrieval_feeds_the_gate(tmp_path):
    corpus = {"d1": EV, "d2": ["Unrelated sentence about climate models and sea level."]}
    r = runner(tmp_path, corpus=corpus, trusted_hashes=None)
    v = r.observe("vitamin D reduced respiratory infection")
    rec = json.loads(Path(r.fresh_path).read_text().splitlines()[-1])
    assert rec["retrieved"] and rec["evidence"] == EV and rec["a"] == 1 and v.outcome == ACCEPT
    assert exact_gate(EV, BM25(corpus).hashes()) == 1 and exact_gate(["x"], BM25(corpus).hashes()) == 0


# ---- judges off by default ----------------------------------------------------------------------------------------
class SpyEnv(dict):
    def __init__(self, base):
        super().__init__(base)
        self.reads = []

    def get(self, k, d=None):
        self.reads.append(k)
        return super().get(k, d)

    def __getitem__(self, k):
        self.reads.append(k)
        return super().__getitem__(k)


def test_judges_off_by_default_and_key_never_read(tmp_path, monkeypatch):
    spy = SpyEnv(os.environ)
    spy["XAI_API_KEY"] = "test-not-a-key"
    monkeypatch.setattr(os, "environ", spy)
    r = ShadowRunner(str(tmp_path / "d"), sources=fake_sources(), trusted_hashes=TRUST)
    assert r.judges is None and r.node.prof.name == "local"
    r.observe("c", EV)
    r.wrap(lambda: "o", extract=lambda o: [("c", EV)])()
    assert "XAI_API_KEY" not in spy.reads
    with open(r.fresh_path, encoding="utf-8") as f:
        for line in f:
            assert "test-not-a-key" not in line and not any(k.startswith("J") for k in json.loads(line)["raw"])
    with pytest.raises(PermissionError):
        XAIPanel()
    with pytest.raises(ValueError):
        XAIPanel(enable=True)  # a spend cap is mandatory
    assert "XAI_API_KEY" not in spy.reads


def test_cli_default_has_no_judges(tmp_path, capsys):
    from roman_wheel_node.shadow import _main

    f = tmp_path / "c.jsonl"
    f.write_text(json.dumps({"claim": "c", "evidence": EV}) + "\n")
    assert _main([str(f), "--log-dir", str(tmp_path / "L"), "--trust-given-evidence"]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out.splitlines()[0])["claim"] == "c"  # shadow echoes every item unchanged
    assert json.loads(out.err.splitlines()[-1])["profile"] == "local"


# ---- fresh-data certification -------------------------------------------------------------------------------------
def test_certificate_arithmetic():
    assert (FR.min_accepts_for(0.01), FR.min_accepts_for(0.02), FR.min_accepts_for(0.05)) == (299, 149, 59)
    assert FR.cp_upper(0, 299) <= 0.01 < FR.cp_upper(0, 298)
    assert FR.labelled_needed(0.01, 0.25) == 1196
    assert FR.accepts_needed_at(0.02, 0.01) is None
    assert FR.cp_lower(0, 10) == 0.0 and 0 < FR.cp_lower(5, 10) < 0.5 < FR.cp_upper(5, 10) < 1


def test_prereg_label_certify_flow(tmp_path):
    r = runner(tmp_path)
    for i in range(320):
        r.observe(f"claim {i}", EV, meta={"k": i})
    d = r.log_dir
    FR.prereg(d, r.node.policy.sha256, r.node.prof.name)
    with pytest.raises(FileExistsError):
        FR.prereg(d, r.node.policy.sha256, r.node.prof.name)
    with open(os.path.join(d, "labels.jsonl"), "w") as f:
        for x in FR.load_jsonl(r.fresh_path):
            f.write(json.dumps({"id": x["id"], "label": "UNSURE" if x["meta"]["k"] < 10 else "SUPPORT"}) + "\n")
    res = FR.certify(d)
    s = res["groups"]["all"]
    assert res["skipped"]["unsure"] == 10 and s["labelled"] == 310 and s["accepted"] == 310 and s["false_accepts"] == 0
    assert s["alphas"]["0.01"]["certified"] and s["alphas"]["0.05"]["certified"]
    assert "CERTIFIED" in FR.report(res)
