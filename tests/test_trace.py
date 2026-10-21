"""Phase 1 invariants: the gated trace is the only way in, blocks propagate, the log is tamper-evident."""
import copy, json, math, os, pickle, random, uuid

import pytest

from roman_wheel_node import LiveWeights, ThresholdLearner, roman_wheel_node
from roman_wheel_node.trace import Claim, ClaimTrace, Decision, TamperError, replay, _TOKEN

try:
    from hypothesis import given, settings, strategies as st, HealthCheck
    HAVE_HYP = True
except ImportError:  # pragma: no cover
    HAVE_HYP = False


class Clock:
    def __init__(self): self.t = 1_700_000_000.0
    def __call__(self):
        self.t += 1.0
        return self.t


@pytest.fixture
def trace(tmp_path):
    return ClaimTrace(str(tmp_path / "audit.jsonl"), clock=Clock())


GOOD = {"x": 0.9, "y": 0.92}


# --- one entry point / bypass attempts -----------------------------------------------------------
def test_claim_cannot_be_constructed_directly():
    with pytest.raises(TypeError):
        Claim(_token=object(), id="x", seq=0, content="c", scores={}, a=1, parents=(), eff=1.0,
              opcode="111", status="ACCEPT", record_hash="0" * 64)
    with pytest.raises(TypeError):
        Claim()  # type: ignore[call-arg]


def test_claim_is_immutable(trace):
    c = trace.submit("fact", GOOD, a=1).claim
    for attr, val in (("status", "BLOCK"), ("eff", 0.0), ("parents", ("z",)), ("new", 1)):
        with pytest.raises(AttributeError):
            setattr(c, attr, val)
    with pytest.raises(AttributeError):
        del c.status
    with pytest.raises(TypeError):
        c.scores["x"] = 0.0           # MappingProxyType is read-only


def test_claim_cannot_be_pickled_or_copied_into_a_new_object(trace):
    c = trace.submit("fact", GOOD, a=1).claim
    with pytest.raises(TypeError):
        pickle.dumps(c)
    assert copy.copy(c) is c and copy.deepcopy(c) is c


def test_trace_has_no_public_mutators(trace):
    public = {n for n in dir(trace) if not n.startswith("_")}
    assert public == {"context", "get", "head", "submit", "verify"}
    ctx = trace.context()
    assert isinstance(ctx, tuple)
    with pytest.raises(AttributeError):
        ctx.append("raw string")      # type: ignore[attr-defined]
    _ = ctx + ("raw string",)          # builds a new tuple only
    assert len(trace) == 0 and "raw string" not in trace.context()


def test_raw_string_content_must_be_str_and_scored(trace):
    with pytest.raises(TypeError):
        trace.submit(b"bytes", GOOD, a=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        trace.submit("nan", {"x": float("nan")}, a=1)
    assert len(trace) == 0


def test_forged_claim_object_cannot_become_parent(trace):
    real = trace.submit("real", GOOD, a=1).claim
    forged = object.__new__(Claim)
    for f in Claim.__slots__:
        object.__setattr__(forged, f, getattr(real, f))          # perfect copy of an admitted claim
    d = trace.submit("child of forgery", GOOD, a=1, parents=[forged])
    assert not d.admitted and d.reason == "parent_not_admitted"
    # even with the private token, a new object is not "of this trace"
    tok = Claim(_token=_TOKEN, id=real.id, seq=real.seq, content=real.content, scores=real.scores, a=1,
                parents=(), eff=real.eff, opcode=real.opcode, status=real.status, record_hash=real.record_hash)
    assert not trace.submit("child", GOOD, a=1, parents=[tok]).admitted


def test_claim_from_another_trace_is_not_a_parent(tmp_path):
    t1 = ClaimTrace(str(tmp_path / "a.jsonl"), clock=Clock()); t2 = ClaimTrace(str(tmp_path / "b.jsonl"), clock=Clock())
    c = t1.submit("fact", GOOD, a=1).claim
    assert not t2.submit("child", GOOD, a=1, parents=[c]).admitted


def test_in_memory_tampering_is_caught_by_verify(trace):
    c = trace.submit("fact", GOOD, a=1).claim
    trace.submit("other", GOOD, a=1)
    trace.verify()
    object.__setattr__(c, "content", "something else")
    with pytest.raises(TamperError):
        trace.verify()


def test_deleted_log_is_caught_by_verify(trace):
    trace.submit("fact", GOOD, a=1)
    os.remove(trace._log_path)
    with pytest.raises(TamperError):
        trace.verify()


def test_private_list_injection_is_caught_by_verify(trace):
    c = trace.submit("fact", GOOD, a=1).claim
    trace._claims.append(c)                    # reaching into private state
    with pytest.raises(TamperError):
        trace.verify()


# --- gate + zero-propagation + containment ---------------------------------------------------------
@pytest.mark.parametrize("a", [0, 0.0, 0.5, 2, -1, False, None, "1"])
def test_only_exact_a_equal_1_passes(trace, a):
    d = trace.submit("c", GOOD, a=a)  # type: ignore[arg-type]
    assert not d.admitted and d.status == "BLOCK" and d.claim is None


def test_a_true_or_1_passes(trace):
    assert trace.submit("c1", GOOD, a=1).admitted and trace.submit("c2", GOOD, a=True).admitted
    assert trace.submit("c3", GOOD, a=1.0).admitted


def test_block_and_zero_score_never_enter(trace):
    assert not trace.submit("veto", {"x": 0.9, "z": 0.0}, a=1).admitted
    assert not trace.submit("empty", {}, a=1).admitted
    assert not trace.submit("weak", {"x": 0.02}, a=1).admitted
    assert len(trace) == 0


def test_zero_propagation_through_generations(trace):
    root = trace.submit("root", GOOD, a=1).claim
    bad = trace.submit("bad", GOOD, a=0, parents=[root])
    assert not bad.admitted
    # the blocked claim's would-be id is never handed out; children by guessed id, by root+missing id
    kid = trace.submit("kid", GOOD, a=1, parents=[root, "deadbeefdeadbeef"])
    assert not kid.admitted and kid.reason == "parent_not_admitted"
    good_kid = trace.submit("good kid", GOOD, a=1, parents=[root])
    assert good_kid.admitted
    grandkid = trace.submit("grandkid", GOOD, a=1, parents=[good_kid.claim.id])
    assert grandkid.admitted
    # a blocked middle generation stops everything below it
    mid = trace.submit("mid", {"x": 0.9, "z": 0.0}, a=1, parents=[grandkid.claim])
    assert not mid.admitted
    assert [c.content for c in trace.context()] == ["root", "good kid", "grandkid"]


def test_blocked_is_invisible(trace):
    d = trace.submit("secret bad claim", {"x": 0.9, "z": 0.0}, a=1)
    rec = json.loads(open(trace._log_path).read().splitlines()[-1])
    blocked_id = rec["hash"][:16]
    assert trace.get(blocked_id) is None and trace.get("never-existed") is None
    assert all("secret" not in c.content for c in trace.context())
    via_blocked = trace.submit("c", GOOD, a=1, parents=[blocked_id])
    via_unknown = trace.submit("c", GOOD, a=1, parents=["never-existed0000"])
    assert (via_blocked.status, via_blocked.reason) == (via_unknown.status, via_unknown.reason)


def test_hedge_off_by_default_and_configurable(tmp_path):
    sc = {"x": 0.2}                                     # geomean 0.2 -> 100 HEDGE
    assert roman_wheel_node(1, sc)[2] == "HEDGE"
    off = ClaimTrace(str(tmp_path / "off.jsonl"), clock=Clock())
    d = off.submit("h", sc, a=1)
    assert d.status == "HEDGE" and not d.admitted and d.reason == "hedge_not_allowed"
    on = ClaimTrace(str(tmp_path / "on.jsonl"), allow_hedge=True, clock=Clock())
    d2 = on.submit("h", sc, a=1)
    assert d2.admitted and d2.claim.status == "HEDGE"
    off_child = off.submit("child", GOOD, a=1, parents=["x" * 16])
    assert not off_child.admitted


def test_clamping_in_trace(trace):
    d1 = trace.submit("c1", {"x": 1.7, "y": 0.9}, a=1)
    d2 = trace.submit("c2", {"x": 1.0, "y": 0.9}, a=1)
    assert d1.eff == d2.eff == 0.949
    assert not trace.submit("neg", {"x": -0.3, "y": 0.9}, a=1).admitted      # clamps to 0 -> veto
    assert trace.submit("inf", {"x": float("inf"), "y": 0.9}, a=1).eff == 0.949


def test_geomean_vs_product(tmp_path):
    g = ClaimTrace(str(tmp_path / "g.jsonl"), mode="geomean", clock=Clock())
    p = ClaimTrace(str(tmp_path / "p.jsonl"), mode="product", clock=Clock())
    sc = {f"s{i}": 0.7 for i in range(4)}
    dg, dp = g.submit("c", sc, a=1), p.submit("c", sc, a=1)
    assert dg.eff == 0.7 and dg.opcode == "110" and dg.admitted
    assert dp.eff == round(0.7 ** 4, 3) and dp.status == "HEDGE" and not dp.admitted   # product punishes more sources


# --- audit log + replay ------------------------------------------------------------------------------
def _populate(trace, n=60, seed=0):
    rng = random.Random(seed)
    ids = []
    for i in range(n):
        sc = {k: rng.choice([0.0, rng.random(), rng.uniform(.6, 1)]) for k in rng.sample("abcdef", rng.randint(0, 4))}
        parents = rng.sample(ids, min(len(ids), rng.randint(0, 2))) + (["bogus"] if rng.random() < .1 else [])
        d = trace.submit(f"claim {i}", sc, a=rng.choice([1, 1, 1, 0]), parents=parents)
        if d.admitted:
            ids.append(d.claim.id)
    return ids


def test_every_call_logged_with_full_audit_tuple(trace):
    _populate(trace, 30)
    lines = open(trace._log_path).read().splitlines()
    assert len(lines) == 30
    need = {"seq", "ts", "content", "scores", "a", "a_eff", "parents", "reliability", "cutoff", "mode",
            "eff", "opcode", "status", "admitted", "reason", "prev", "hash"}
    assert all(need <= set(json.loads(l)) for l in lines)


def test_replay_reconstructs_exactly(trace):
    _populate(trace, 80)
    rep = replay(trace._log_path)
    assert rep.n_records == 80 and rep.head == trace.head()
    assert [c.id for c in rep.claims] == [c.id for c in trace.context()]
    trace.verify()


def test_replay_with_learning_state(tmp_path):
    live = LiveWeights(str(tmp_path / "w.jsonl")); thr = ThresholdLearner(str(tmp_path / "t.jsonl"), min_obs=5)
    t = ClaimTrace(str(tmp_path / "audit.jsonl"), live=live, threshold=thr, mode="bayes", clock=Clock())
    rng = random.Random(1)
    for i in range(120):
        truth = rng.random() < .5
        sc = {"good": .9 if (truth if rng.random() < .9 else not truth) else .1,
              "liar": .9 if (not truth if rng.random() < .8 else truth) else .1}
        d = t.submit(f"c{i}", sc, a=1)
        for k, v in sc.items(): live.observe(k, 1.0 if (v > .5) == truth else 0.0)
        thr.record(d.eff, truth, "bayes:hard")
    rep = t.verify()
    assert rep.n_records == 120 and rep.n_admitted == len(t)


def test_resume_from_log(tmp_path):
    p = str(tmp_path / "audit.jsonl")
    t = ClaimTrace(p, clock=Clock()); ids = _populate(t, 40)
    t2 = ClaimTrace(p, clock=Clock())
    assert [c.id for c in t2.context()] == ids and t2.head() == t.head()
    d = t2.submit("after resume", GOOD, a=1, parents=ids[:1])
    assert d.admitted
    t2.verify()


def _rewrite(path, fn):
    lines = open(path).read().splitlines()
    lines = fn(lines)
    open(path, "w").write("\n".join(lines) + "\n")


def test_editing_a_logged_decision_is_detected(trace):
    _populate(trace, 20)
    def edit(lines):
        r = json.loads(lines[5]); r["status"] = "ACCEPT"; r["admitted"] = True
        lines[5] = json.dumps(r, sort_keys=True, separators=(",", ":")); return lines
    _rewrite(trace._log_path, edit)
    with pytest.raises(TamperError):
        replay(trace._log_path)


def test_deleting_or_reordering_records_is_detected(trace):
    _populate(trace, 20)
    orig = open(trace._log_path).read()
    _rewrite(trace._log_path, lambda L: L[:3] + L[4:])
    with pytest.raises(TamperError): replay(trace._log_path)
    open(trace._log_path, "w").write(orig)
    _rewrite(trace._log_path, lambda L: [L[1], L[0]] + L[2:])
    with pytest.raises(TamperError): replay(trace._log_path)


def test_forged_entry_with_recomputed_hash_but_wrong_decision_is_detected(trace):
    """An attacker who recomputes the (unkeyed) chain still can't make an inconsistent decision verify."""
    from roman_wheel_node.trace import _chain_hash, _canon, GENESIS
    trace.submit("weak", {"x": 0.03}, a=1)                      # BLOCK
    lines = open(trace._log_path).read().splitlines()
    r = json.loads(lines[0]); r.pop("hash")
    r.update(status="ACCEPT", opcode="111", admitted=True, reason="ok")
    r["hash"] = _chain_hash(GENESIS, r, None)
    open(trace._log_path, "w").write(_canon(r).decode() + "\n")
    with pytest.raises(TamperError, match="recompute"):
        replay(trace._log_path)


def test_forged_consistent_entry_detected_with_hmac(tmp_path):
    from roman_wheel_node.trace import _chain_hash, _canon
    key = b"secret-key"
    t = ClaimTrace(str(tmp_path / "a.jsonl"), hmac_key=key, clock=Clock())
    t.submit("real", GOOD, a=1)
    lines = open(t._log_path).read().splitlines()
    r = json.loads(lines[0]); prev = r["hash"]
    fake = dict(r); fake.pop("hash"); fake.update(seq=1, content="injected", prev=prev)
    fake["hash"] = _chain_hash(prev, fake, None)                  # attacker without the key
    open(t._log_path, "a").write(_canon(fake).decode() + "\n")
    with pytest.raises(TamperError):
        replay(t._log_path, hmac_key=key)
    with pytest.raises(TamperError):
        ClaimTrace(t._log_path, hmac_key=key)                     # refuses to resume from a bad log


def test_truncated_or_garbage_line_detected(trace):
    _populate(trace, 5)
    open(trace._log_path, "a").write("{not json\n")
    with pytest.raises(TamperError):
        replay(trace._log_path)


# --- property-based / fuzz -----------------------------------------------------------------------------
def _fuzz_case(tmp, ops, mode, allow_hedge):
    t = ClaimTrace(os.path.join(tmp, f"fuzz_{uuid.uuid4().hex}.jsonl"),
                   mode=mode, allow_hedge=allow_hedge, clock=Clock())
    admitted = []
    for a, scores, pick in ops:
        parents = [admitted[i % len(admitted)] for i in pick] if admitted else (["ghost"] if pick else [])
        d = t.submit("x", scores, a=a, parents=parents)
        eff, op, st_ = roman_wheel_node(a if a == 1 else 0, scores, mode=mode)
        if d.reason != "parent_not_admitted":
            assert (d.eff, d.opcode, d.status) == (eff, op, st_)
        # invariants
        if a != 1 or any(min(1.0, max(0.0, v)) == 0.0 for v in scores.values()) or not scores:
            assert not d.admitted
        if d.admitted:
            assert d.status == "ACCEPT" or (allow_hedge and d.status == "HEDGE")
            assert all(t.get(p) is not None for p in d.claim.parents)
            admitted.append(d.claim)
        else:
            assert d.claim is None
    assert list(t.context()) == admitted
    t.verify()


if HAVE_HYP:
    score_st = st.floats(min_value=-2, max_value=3, allow_nan=False)
    op_st = st.tuples(st.sampled_from([0, 1, 1, 1]),
                      st.dictionaries(st.sampled_from(list("abcde")), score_st, max_size=4),
                      st.lists(st.integers(0, 50), max_size=3))

    @settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(ops=st.lists(op_st, max_size=25), mode=st.sampled_from(["geomean", "product", "bayes"]),
           allow_hedge=st.booleans())
    def test_property_invariants(tmp_path, ops, mode, allow_hedge):
        _fuzz_case(str(tmp_path), ops, mode, allow_hedge)

    @settings(max_examples=200, deadline=None)
    @given(st.lists(st.floats(min_value=-5, max_value=5, allow_nan=False), max_size=8),
           st.sampled_from(["geomean", "product"]))
    def test_property_node_bounds(ws, mode):
        eff, op, status = roman_wheel_node(1, ws, mode=mode)
        assert 0.0 <= eff <= 1.0
        if not ws or any(min(1, max(0, w)) == 0 for w in ws):
            assert status == "BLOCK" and eff == 0.0
        if ws and all(w >= 1 for w in ws):
            assert eff == 1.0


def test_randomized_fuzz_loop(tmp_path):
    rng = random.Random(42)
    for case in range(60):
        ops = [(rng.choice([0, 1, 1, 1]),
                {k: rng.uniform(-1, 2) if rng.random() < .2 else rng.random() for k in rng.sample("abcde", rng.randint(0, 4))},
                [rng.randint(0, 50) for _ in range(rng.randint(0, 2))]) for _ in range(rng.randint(1, 30))]
        _fuzz_case(str(tmp_path), ops, rng.choice(["geomean", "product", "bayes"]), rng.random() < .5)
