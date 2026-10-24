"""Fail-closed guarantee (SAFETY.md G1/G2): an unverified claim is never emitted as verified, and NOT_SURE never
propagates as accepted. Property-based (hypothesis) and a seeded fuzz loop, over random bands, flags, gates,
scores, parents and modes."""

import os
import random
import uuid

from roman_wheel_node.trace import ClaimTrace, replay

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st

    HAVE_HYP = True
except ImportError:  # pragma: no cover
    HAVE_HYP = False


def _case(tmp, ops, band, allow_hedge, mode):
    t = ClaimTrace(
        os.path.join(tmp, f"fc_{uuid.uuid4().hex}.jsonl"), mode=mode, allow_hedge=allow_hedge, not_sure_band=band
    )
    held, all_seq_status = [], []
    for a, scores, ns, pick in ops:
        parents = [held[i % len(held)] for i in pick] if (held and pick) else (["never-admitted"] if pick else [])
        d = t.submit("c", scores, a=a, parents=parents, not_sure=ns)
        all_seq_status.append(d.status)
        # G1: fail-closed. Admission requires an exact gate pass, admitted parents, no NOT_SURE, ACCEPT (or opt-in
        # HEDGE)
        if d.admitted:
            assert a == 1 and not ns and d.status in (("ACCEPT", "HEDGE") if allow_hedge else ("ACCEPT",))
            assert band is None or not (band[0] <= d.eff < band[1])
            assert all(t.get(p) is not None for p in d.claim.parents)
            held.append(d.claim)
        else:
            assert d.claim is None and t.get(str(d.seq)) is None
        # G2: NOT_SURE is never admitted, whatever allow_hedge says
        if d.status == "NOT_SURE":
            assert not d.admitted and d.reason == "not_sure"
        if ns and a == 1 and scores:
            assert not d.admitted
    assert list(t.context()) == held
    rep = replay(t._log_path)  # every NOT_SURE/ACCEPT/BLOCK recomputes from the log
    assert rep.n_admitted == len(held) and rep.n_records == len(ops)


band_st = (
    st.one_of(st.none(), st.tuples(st.floats(0, 1), st.floats(0, 1)).map(lambda x: (min(x), max(x))))
    if HAVE_HYP
    else None
)

if HAVE_HYP:
    op_st = st.tuples(
        st.sampled_from([0, 1, 1, 1]),
        st.dictionaries(st.sampled_from(list("abcd")), st.floats(-1, 2, allow_nan=False), max_size=4),
        st.booleans(),
        st.lists(st.integers(0, 40), max_size=2),
    )

    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(
        ops=st.lists(op_st, max_size=20),
        band=band_st,
        allow_hedge=st.booleans(),
        mode=st.sampled_from(["geomean", "product", "bayes"]),
    )
    def test_property_fail_closed(tmp_path, ops, band, allow_hedge, mode):
        _case(str(tmp_path), ops, band, allow_hedge, mode)


def test_fail_closed_fuzz_loop(tmp_path):
    rng = random.Random(2026)
    for _ in range(80):
        ops = [
            (
                rng.choice([0, 1, 1, 1]),
                {k: rng.uniform(-0.5, 1.5) for k in rng.sample("abcd", rng.randint(0, 4))},
                rng.random() < 0.3,
                [rng.randint(0, 40) for _ in range(rng.randint(0, 2))],
            )
            for _ in range(rng.randint(1, 25))
        ]
        lo = rng.random()
        band = rng.choice([None, (lo, min(1.0, lo + rng.random()))])
        _case(str(tmp_path), ops, band, rng.random() < 0.5, rng.choice(["geomean", "product", "bayes"]))
