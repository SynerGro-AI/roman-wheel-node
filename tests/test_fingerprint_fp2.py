"""Task B: the fp2 (NFC-normalized, versioned) exact-copy fingerprint and policy 4.1.0."""
import json
import os
import unicodedata

import pytest
from hypothesis import given, settings, strategies as st

from roman_wheel_node import fingerprint as FP
from roman_wheel_node.policy import (DEFAULT_POLICY_PATH, DEFAULT_POLICY_SHA256, POLICY_4_1_PATH, POLICY_4_1_SHA256,
                                     PolicyError, load_policy, policy_sha256)
from roman_wheel_node.shadow import ShadowRunner, quote_verbatim
from roman_wheel_node.sources import exact_gate, sha256

S = "Vitamin D reduced infections by 12% in adults."


def test_normalize_cases():
    n = FP.normalize_v2
    assert n("cafe\u0301 is open") == n("caf\u00e9 is open") == "caf\u00e9 is open"            # NFC
    assert n("a\u00a0b\u2009c\t\n d  ") == "a b c d"                                          # whitespace
    assert n("\u201cyes\u201d it\u2019s") == "\"yes\" it's"                                      # quotes
    assert n("COVID\u201119 \u2013 5\u20147 \u22125") == "COVID-19 - 5-7 -5"                     # dashes, minus
    assert n("zero\u200bwidth\u00ad soft\ufeff") == "zerowidth soft"                              # invisible
    assert n("10\u00b2 cells") == "10\u00b2 cells"                                                # NOT NFKC
    assert n("Not") != n("not") and n("12%") != n("12 %")                                         # case, tokens kept


def test_versioned_and_distinct():
    assert FP.fp2(S).startswith("rwn-fp2:") and FP.fp1(S) == sha256(S)
    assert FP.fp1(S) != FP.fp2(S) and FP.fingerprint(S, "fp1") == FP.fp1(S)
    with pytest.raises(ValueError):
        FP.fingerprint(S, "fp9")


def test_gate_v2_superset_and_formatting_only():
    hs = FP.trusted_set([S])
    assert FP.exact_gate_v2([S], hs) == 1
    assert FP.exact_gate_v2(["Vitamin  D reduced infections by 12%\u00a0in adults."], hs) == 1
    assert exact_gate(["Vitamin  D reduced infections by 12%\u00a0in adults."], {sha256(S)}) == 0      # fp1 blocked it
    assert FP.exact_gate_v2(["Vitamin D reduced infections by 21% in adults."], hs) == 0               # content change
    assert FP.exact_gate_v2(["vitamin D reduced infections by 12% in adults."], hs) == 0               # case kept
    assert FP.exact_gate_v2([], hs) == 0 and FP.exact_gate_v2([S], None) == 0
    assert FP.exact_gate_v2([S], {sha256(S)}) == 1                                                      # fp1-only store


def test_quote_v2():
    unit = "If the risk is confirmed,  then the crisis will shift to heart failure, they wrote."
    hs = FP.trusted_set([unit])
    q = "If the risk is confirmed, then the crisis will shift to heart failure, they wrote."
    assert not quote_verbatim(q, [unit], hs) and FP.quote_verbatim_v2(q, [unit], hs)
    assert not FP.quote_verbatim_v2("If the risk is confirmed ... heart failure, they wrote.", [unit], hs)  # elision
    assert not FP.quote_verbatim_v2("If the risk is denied, then the crisis will shift.", [unit], hs)
    assert not FP.quote_verbatim_v2("short", [unit], hs)
    assert not FP.quote_verbatim_v2(q, [unit], set())                                                 # untrusted unit


txt = st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=0x2100), max_size=60)


@settings(max_examples=300, deadline=None)
@given(txt)
def test_normalize_idempotent_and_nfc(s):
    n = FP.normalize_v2(s)
    assert FP.normalize_v2(n) == n and unicodedata.is_normalized("NFC", n)


@settings(max_examples=300, deadline=None)
@given(st.lists(txt.filter(lambda x: x.strip()), min_size=1, max_size=4), st.lists(txt, max_size=4))
def test_v2_never_blocks_what_v1_admits(ev, extra):
    store = list(ev) + list(extra)
    if exact_gate(ev, {sha256(u) for u in store}) == 1:
        assert FP.exact_gate_v2(ev, FP.trusted_set(store)) == 1


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet="abcde .!?", min_size=20, max_size=60), st.text(alphabet="abcde .!?XY", max_size=40))
def test_quote_v2_superset_on_ascii(q, pad):
    unit = pad + q + pad
    hs = FP.trusted_set([unit])
    if quote_verbatim(q, [unit], {sha256(unit)}):
        assert FP.quote_verbatim_v2(q, [unit], hs)


def test_policy_400_untouched_and_410_pinned():
    p0, p1 = load_policy(), load_policy(POLICY_4_1_PATH)
    assert p0.sha256 == DEFAULT_POLICY_SHA256 == "254804f8ac9b500ef1951879fb64328b0f54303b8bef20e38b1684545f876eb1"
    assert p0.version == "4.0.0" and p0.fingerprint == "fp1"
    assert p1.sha256 == POLICY_4_1_SHA256 and p1.version == "4.1.0" and p1.fingerprint == "fp2"
    assert p1.doc["parent_policy"]["sha256"] == DEFAULT_POLICY_SHA256
    assert all(p0.profiles[k].raw == p1.profiles[k].raw for k in p0.profiles) and set(p0.profiles) == set(p1.profiles)
    assert p1.doc["gate"]["normalize_table_sha256"] == FP.NORMALIZE_V2_TABLE_SHA256


def test_policy_410_tamper_refused(tmp_path):
    doc = json.load(open(POLICY_4_1_PATH))
    doc["gate"]["normalize_table_sha256"] = "0" * 64
    doc["sha256"] = policy_sha256(doc)
    p = tmp_path / "x.json"; p.write_text(json.dumps(doc))
    with pytest.raises(PolicyError):
        load_policy(str(p))
    doc = json.load(open(POLICY_4_1_PATH)); doc["gate"]["fingerprint"] = "fp7"; doc["sha256"] = policy_sha256(doc)
    p.write_text(json.dumps(doc))
    with pytest.raises(PolicyError):
        load_policy(str(p))
    doc = json.load(open(POLICY_4_1_PATH)); doc["profiles"]["local"]["cutoff"] = 0.5      # edited body, old hash
    p.write_text(json.dumps(doc))
    with pytest.raises(PolicyError):
        load_policy(str(p))


def test_shadow_runner_fp2_vs_fp1(tmp_path):
    src = lambda c, s: {"nli2": {"S": 0.999, "C": 0.0}, "overlap": {"S": 1.0}}
    messy = ["Vitamin  D reduced infections by 12%\u00a0in adults."]
    r0 = ShadowRunner(str(tmp_path / "a"), sources=src, trusted_units=[S])
    r1 = ShadowRunner(str(tmp_path / "b"), policy=load_policy(POLICY_4_1_PATH), sources=src, trusted_units=[S])
    assert r0.observe("claim", messy).outcome == "BLOCK"
    assert r1.observe("claim", messy).outcome != "BLOCK"
    assert r1.observe("claim", ["Vitamin D reduced infections by 21% in adults."]).outcome == "BLOCK"
    r0.verify(); r1.verify()
