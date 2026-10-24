"""Phase 4b structural veto: determinism, vector (not scalar), veto only on ACCEPT. Synthetic text only."""

from roman_wheel_node import structure as S

EV = ["The novel compound XR-201 reduced viral load in treated hamsters by 40 percent within two days."]


def test_vector_is_deterministic_and_normalised():
    a = S.vector("XR-201 reduced viral load by 40%")
    b = S.vector("ＸＲ-201 REDUCED viral load by 40%")  # NFKC + casefold
    assert a == b and a.digest() == b.digest()
    assert a.numbers == ("40",) and "xr-201" in a.identifiers and a.directions == ((0, -1),)


def test_vector_fields_do_not_collide_like_a_scalar_sum():
    one = S.vector("not 5 cases")  # 1 negation, 1 number
    two = S.vector("no never cases")  # 2 negations, 0 numbers
    r1, r2 = S.rule24_fields(one), S.rule24_fields(two)
    assert r1["fields"] != r2["fields"] and r1["digest"] != r2["digest"]
    assert r1["names"] == ("sides", "points", "dots", "intersections", "spirals")
    assert "score" not in r1 and "sum" not in r1


def test_verbatim_and_paraphrase_claims_are_not_vetoed():
    for claim in (
        "XR-201 reduced viral load in treated hamsters",
        "The novel compound XR-201 reduced viral load in treated hamsters",
    ):
        out, c = S.veto_accept("ACCEPT", claim, EV)
        assert out == "ACCEPT" and not c.veto, c


def test_identifier_mutation_is_vetoed():
    out, c = S.veto_accept("ACCEPT", "The novel compound QR-201 reduced viral load in treated hamsters", EV)
    assert out == "NOT_SURE" and "identifier_mutation" in c.reasons


def test_single_word_swap_in_near_copy_is_vetoed():
    out, c = S.veto_accept("ACCEPT", "The novel compound XR-201 boosted viral load in treated hamsters", EV)
    assert out == "NOT_SURE" and "near_copy_swap" in c.reasons
    assert c.near_copy_swap[0][:2] == ("boosted", "reduced")


def test_fragment_token_is_vetoed():
    out, c = S.veto_accept("ACCEPT", "The novel compound k reduced viral load", EV)
    assert out == "NOT_SURE" and c.fragment == ["k"]


def test_veto_never_touches_non_accepts():
    for o in ("BLOCK", "NOT_SURE"):
        assert S.veto_accept(o, "The novel compound QR-201 boosted viral load", EV) == (o, None)


def test_quotes_are_optional_extra_evidence():
    claim = "The novel compound XR-201 reduced viral load in treated hamsters"
    assert S.check(claim, [], quotes=EV).veto is False
    assert S.check(claim, EV, quotes=None).reasons == S.check(claim, EV, quotes=[""]).reasons


def test_unvetoing_fields_are_still_reported():
    c = S.check("XR-201 never increased viral load by 90 percent", EV)
    assert c.numbers_missing == ["90"] and c.direction_conflict == [(0, 1)] and c.negation == (1, 0)
    assert set(c.pattern()) >= {"numbers_missing", "direction_conflict"}


def test_frozen_rule_hash_is_stable():
    assert S.frozen_rule_sha256() == S.frozen_rule_sha256() and len(S.frozen_rule_sha256()) == 64
    assert S.FROZEN["on_veto"] == "NOT_SURE"
