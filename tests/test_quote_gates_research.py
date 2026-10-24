"""Research only: the Phase 3 judge-quote gates (phase5/quote_gates.py), which are not part of the package."""

import os
import sys

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isdir(os.path.join(ROOT, "phase5")):
    pytest.skip("phase5/ research scripts are repo-only (not in the sdist/wheel)", allow_module_level=True)
sys.path.insert(0, os.path.join(ROOT, "phase5"))
from quote_gates import quote_verbatim, quote_verbatim_v2  # noqa: E402

from roman_wheel_node import fingerprint as FP  # noqa: E402
from roman_wheel_node.sources import sha256  # noqa: E402


def test_quote_v2():
    unit = "If the risk is confirmed,  then the crisis will shift to heart failure, they wrote."
    hs = FP.trusted_set([unit])
    q = "If the risk is confirmed, then the crisis will shift to heart failure, they wrote."
    assert not quote_verbatim(q, [unit], hs) and quote_verbatim_v2(q, [unit], hs)
    assert not quote_verbatim_v2("If the risk is confirmed ... heart failure, they wrote.", [unit], hs)  # elision
    assert not quote_verbatim_v2("If the risk is denied, then the crisis will shift.", [unit], hs)
    assert not quote_verbatim_v2("short", [unit], hs)
    assert not quote_verbatim_v2(q, [unit], set())  # untrusted unit


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet="abcde .!?", min_size=20, max_size=60), st.text(alphabet="abcde .!?XY", max_size=40))
def test_quote_v2_superset_on_ascii(q, pad):
    unit = pad + q + pad
    hs = FP.trusted_set([unit])
    if quote_verbatim(q, [unit], {sha256(unit)}):
        assert quote_verbatim_v2(q, [unit], hs)
